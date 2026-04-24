"""Practical ACH example for OpenSpiel.

This trainer follows the paper's practical Actor-Critic Hedge (ACH)
implementation more closely than ``ach_pytorch.py``:

- one shared actor-value network is used for all players;
- trajectories are sampled from the current self-play policy;
- only sampled state-action pairs are optimized;
- advantages are estimated with GAE(lambda);
- the policy loss follows the clipped ACH objective from Appendix E.

Compared with the previous version, this implementation now preserves
multi-dimensional observation tensors instead of flattening them by default.
The trainer always uses a residual CNN trunk inspired by the paper's Mahjong
architecture. Flat vector games are reshaped into a single-row spatial tensor
so they can use the same backbone. When a game also exposes separate one-hot
information-state features, those auxiliary features are read separately and
concatenated after the CNN flattening step.

It is still a synchronous trainer rather than the paper's full decoupled
actor/learner system:

- no asynchronous actors or learner;
- one shared optimizer update loop per self-play batch;
- optional multiprocessing for self-play collection on Linux;
- only turn-based sequential games are supported.

Paper:
https://openreview.net/pdf?id=DTXZqTNV5nW
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import pyspiel
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


INVALID_ACTION_LOGIT = -1e9


def module_init(module: nn.Module,
                std: float = np.sqrt(2.0),
                bias_const: float = 0.0) -> nn.Module:
  """Orthogonally initializes a linear or convolution layer."""
  if not hasattr(module, "weight") or module.weight is None:
    raise ValueError(f"Cannot initialize module without weights: {module!r}")
  torch.nn.init.orthogonal_(module.weight, std)
  if hasattr(module, "bias") and module.bias is not None:
    torch.nn.init.constant_(module.bias, bias_const)
  return module


def _safe_game_name(game_name: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", game_name).strip("_") or "game"


def _parse_int_tuple(values: str,
                     allow_empty: bool = False) -> Tuple[int, ...]:
  parsed = tuple(int(value.strip()) for value in values.split(",")
                 if value.strip())
  if not parsed and not allow_empty:
    raise ValueError("Expected at least one integer.")
  return parsed


def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
  return tuple(int(dim) for dim in shape)


def _canonical_observation_shape(observation_shape: Sequence[int]) -> Tuple[int, ...]:
  observation_shape = _shape_tuple(observation_shape)
  if len(observation_shape) == 1:
    return (1, 1, observation_shape[0])
  if len(observation_shape) == 2:
    return (1, observation_shape[0], observation_shape[1])
  if len(observation_shape) == 3:
    return observation_shape
  raise ValueError(
      "Observation tensors must be 1D, 2D, or 3D to be used with the residual "
      f"CNN, got {observation_shape}")


def get_state_shape(game: pyspiel.Game) -> Tuple[int, ...]:
  """Returns the preferred spatial input shape for a game."""
  game_type = game.get_type()
  if game_type.provides_observation_tensor:
    return _shape_tuple(game.observation_tensor_shape())
  if game_type.provides_information_state_tensor:
    return _shape_tuple(game.information_state_tensor_shape())
  raise ValueError(
      f"Game {game_type.short_name} provides neither information_state_tensor "
      "nor observation_tensor.")


def get_state_tensor(state: pyspiel.State,
                     player: int,
                     game: pyspiel.Game) -> np.ndarray:
  """Returns the preferred spatial state tensor for one player."""
  if game.get_type().provides_observation_tensor:
    tensor = state.observation_tensor(player)
  else:
    tensor = state.information_state_tensor(player)
  return np.asarray(tensor, dtype=np.float32).reshape(get_state_shape(game))


def infer_aux_feature_channels(game_name: str, game: pyspiel.Game) -> int:
  """Returns the number of auxiliary info-state features to read separately."""
  game_type = game.get_type()
  if (not game_type.provides_observation_tensor or
      not game_type.provides_information_state_tensor):
    return 0

  information_state_shape = _shape_tuple(game.information_state_tensor_shape())
  if (_safe_game_name(game_name).startswith("eren_yifang") and
      len(information_state_shape) == 1):
    return information_state_shape[0]
  return 0


def get_auxiliary_features(state: pyspiel.State,
                           player: int,
                           game: pyspiel.Game,
                           aux_feature_channels: int = 0) -> np.ndarray:
  """Returns separate one-hot auxiliary features from InformationStateTensor."""
  if aux_feature_channels <= 0:
    return np.zeros((0,), dtype=np.float32)
  if not game.get_type().provides_information_state_tensor:
    raise ValueError("Requested auxiliary features, but the game has none.")

  features = np.asarray(
      state.information_state_tensor(player), dtype=np.float32).reshape(-1)
  if features.shape[0] != aux_feature_channels:
    raise ValueError(
        "InformationStateTensor size does not match aux_feature_channels: "
        f"{features.shape[0]} vs {aux_feature_channels}")
  return features


def legal_actions_mask(legal_actions: Sequence[int],
                       num_actions: int) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=bool)
  mask[list(legal_actions)] = True
  return mask


def masked_softmax(logits: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
  masked_logits = logits.masked_fill(~mask, INVALID_ACTION_LOGIT)
  return F.softmax(masked_logits, dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  mask_f = mask.float()
  denom = mask_f.sum(dim=-1).clamp_min(1.0)
  return (values * mask_f).sum(dim=-1) / denom


@dataclass
class PendingTransition:
  """A transition for one player before the next own decision is reached."""

  state: np.ndarray
  auxiliary_features: np.ndarray
  legal_mask: np.ndarray
  action: int
  old_prob: float
  value: float
  reward: float = 0.0
  bootstrap_discount: float = 1.0
  next_value: float = 0.0
  done: bool = False


@dataclass
class BatchTransition:
  """A fully processed training sample."""

  state: np.ndarray
  auxiliary_features: np.ndarray
  legal_mask: np.ndarray
  action: int
  old_prob: float
  advantage: float
  return_: float


def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  """Returns a detached CPU snapshot that is safe to ship to worker processes."""
  return {
      key: value.detach().cpu().clone()
      for key, value in state_dict.items()
  }


def _split_sample_count(total_samples: int, num_workers: int) -> List[int]:
  """Splits a target sample count across workers as evenly as possible."""
  base = total_samples // num_workers
  remainder = total_samples % num_workers
  counts = [
      base + (1 if worker_idx < remainder else 0)
      for worker_idx in range(num_workers)
  ]
  return [count for count in counts if count > 0]


def _resolve_num_workers(num_workers: Optional[int]) -> int:
  """Resolves the requested worker count, defaulting to Linux parallelism."""
  if num_workers is not None:
    return max(1, num_workers)
  cpu_count = os.cpu_count() or 1
  if sys.platform.startswith("linux"):
    return min(cpu_count, 4)
  return 1


def _resolve_start_method(device: str,
                          mp_start_method: Optional[str]) -> str:
  """Chooses a multiprocessing start method that is safe for the platform."""
  if mp_start_method is not None:
    return mp_start_method
  if sys.platform.startswith("linux") and not str(device).startswith("cuda"):
    return "fork"
  return "spawn"


def _configure_torch_multiprocessing() -> None:
  """Reduces file-descriptor pressure when sharing tensors with workers."""
  if not sys.platform.startswith("linux"):
    return
  try:
    torch.multiprocessing.set_sharing_strategy("file_system")
  except (AttributeError, RuntimeError):
    pass


def _latest_checkpoint_path(checkpoint_dir: Path,
                            checkpoint_prefix: str,
                            game_name: str) -> Optional[Path]:
  """Returns the newest matching checkpoint path in the checkpoint dir."""
  if not checkpoint_dir.exists():
    return None

  pattern = f"{checkpoint_prefix}_{_safe_game_name(game_name)}_*.pth"
  candidates = [
      path for path in checkpoint_dir.glob(pattern)
      if path.is_file()
  ]
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


class ResidualConvBlock(nn.Module):
  """A simple residual block for image-like information-state tensors."""

  def __init__(self, channels: int):
    super().__init__()
    self.conv1 = module_init(nn.Conv2d(channels, channels, kernel_size=3,
                                       padding=1))
    self.bn1 = nn.BatchNorm2d(channels)
    self.conv2 = module_init(nn.Conv2d(channels, channels, kernel_size=3,
                                       padding=1))
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    hidden = self.conv1(inputs)
    hidden = self.bn1(hidden)
    hidden = F.relu(hidden)
    hidden = self.conv2(hidden)
    hidden = self.bn2(hidden)
    hidden = hidden + inputs
    return F.relu(hidden)


class TransitionConvBlock(nn.Module):
  """A 1x1 transition block used between residual stages."""

  def __init__(self, input_channels: int, output_channels: int):
    super().__init__()
    self.conv = module_init(nn.Conv2d(input_channels, output_channels,
                                      kernel_size=1))
    self.bn = nn.BatchNorm2d(output_channels)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    hidden = self.conv(inputs)
    hidden = self.bn(hidden)
    return F.relu(hidden)


def _build_head(input_size: int,
                hidden_sizes: Sequence[int],
                output_size: int,
                final_std: float) -> nn.Sequential:
  layers: List[nn.Module] = []
  prev_size = input_size
  for hidden_size in hidden_sizes:
    layers.extend([
        module_init(nn.Linear(prev_size, hidden_size)),
        nn.ReLU(),
    ])
    prev_size = hidden_size
  layers.append(module_init(nn.Linear(prev_size, output_size), std=final_std))
  return nn.Sequential(*layers)


class SharedActorCritic(nn.Module):
  """Shared torso with separate policy and value heads."""

  def __init__(self,
               input_shape: Sequence[int],
               num_actions: int,
               aux_feature_channels: int = 0,
               conv_channels: Sequence[int] = (64, 128, 32),
               residual_blocks_per_stage: int = 3,
               shared_feature_size: int = 1024,
               head_hidden_sizes: Sequence[int] = (512, 512)):
    super().__init__()

    self.input_shape = tuple(int(dim) for dim in input_shape)
    if not self.input_shape:
      raise ValueError("input_shape must contain at least one dimension.")
    self.aux_feature_channels = int(aux_feature_channels)
    if self.aux_feature_channels < 0:
      raise ValueError("aux_feature_channels must be non-negative.")
    self.image_input_shape = _canonical_observation_shape(self.input_shape)

    conv_layers: List[nn.Module] = []
    input_channels = self.image_input_shape[0]
    for output_channels in conv_channels:
      conv_layers.append(
          TransitionConvBlock(input_channels, int(output_channels)))
      for _ in range(residual_blocks_per_stage):
        conv_layers.append(ResidualConvBlock(int(output_channels)))
      input_channels = int(output_channels)
    self.encoder = nn.Sequential(*conv_layers)

    with torch.no_grad():
      dummy = torch.zeros(1, *self.image_input_shape)
      encoded = self.encoder(dummy)
      encoded_size = int(np.prod(encoded.shape[1:]))
    projection_input_size = encoded_size + self.aux_feature_channels
    self.shared_projection = nn.Sequential(
        module_init(nn.Linear(projection_input_size, shared_feature_size)),
        nn.ReLU(),
    )
    head_input_size = shared_feature_size
    resolved_head_sizes = tuple(int(size) for size in head_hidden_sizes)

    self.policy_head = _build_head(
        input_size=head_input_size,
        hidden_sizes=resolved_head_sizes,
        output_size=num_actions,
        final_std=0.01)
    self.value_head = _build_head(
        input_size=head_input_size,
        hidden_sizes=resolved_head_sizes,
        output_size=1,
        final_std=1.0)

  def _reshape_states(self, states: torch.Tensor) -> torch.Tensor:
    if len(self.input_shape) == 1:
      if states.dim() == 2:
        return states.unsqueeze(1).unsqueeze(1)
      if states.dim() == 3 and states.shape[1] == 1:
        return states.unsqueeze(1)
      if states.dim() == 4 and states.shape[1] == 1 and states.shape[2] == 1:
        return states
      raise ValueError(
          "Expected batched 1D states with shape [B, N] for the residual CNN, "
          f"got {tuple(states.shape)}")

    if len(self.input_shape) == 2:
      if states.dim() == 3:
        return states.unsqueeze(1)
      if states.dim() == 4 and states.shape[1] == 1:
        return states
      raise ValueError(
          "Expected batched 2D states with shape [B, H, W] for the residual "
          f"encoder, got {tuple(states.shape)}")

    if states.dim() != 4:
      raise ValueError(
          "Expected batched 3D states with shape [B, C, H, W] for the residual "
          f"encoder, got {tuple(states.shape)}")

    return states

  def forward(self,
              states: torch.Tensor,
              auxiliary_features: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    image_inputs = self._reshape_states(states)
    hidden = self.encoder(image_inputs)
    hidden = hidden.reshape(hidden.shape[0], -1)
    if self.aux_feature_channels:
      if auxiliary_features is None:
        auxiliary_features = hidden.new_zeros(
            (hidden.shape[0], self.aux_feature_channels))
      if auxiliary_features.dim() != 2:
        raise ValueError(
            "Expected auxiliary features with shape [B, F], got "
            f"{tuple(auxiliary_features.shape)}")
      if auxiliary_features.shape[1] != self.aux_feature_channels:
        raise ValueError(
            "Auxiliary feature width does not match the network config: "
            f"{auxiliary_features.shape[1]} vs {self.aux_feature_channels}")
      hidden = torch.cat([hidden, auxiliary_features], dim=-1)
    hidden = self.shared_projection(hidden)
    logits = self.policy_head(hidden)
    values = self.value_head(hidden).squeeze(-1)
    return logits, values


class PracticalACHAgent:
  """A practical ACH learner based on Appendix E."""

  def __init__(
      self,
      action_size: int,
      input_shape: Optional[Sequence[int]] = None,
      state_size: Optional[int] = None,
      aux_feature_channels: int = 0,
      conv_channels: Sequence[int] = (64, 128, 32),
      residual_blocks_per_stage: int = 3,
      shared_feature_size: int = 1024,
      head_hidden_sizes: Sequence[int] = (512, 512),
      learning_rate: float = 3e-4,
      gamma: float = 1.0,
      gae_lambda: float = 0.95,
      clip_ratio: float = 0.1,
      logit_threshold: float = 2.0,
      entropy_coef: float = 0.01,
      value_coef: float = 0.5,
      eta: float = 1.0,
      max_grad_norm: float = 0.5,
      update_epochs: int = 1,
      num_minibatches: int = 1,
      normalize_advantages: bool = False,
      device: Optional[str] = None,
  ):
    if input_shape is None:
      if state_size is None:
        raise ValueError("Either input_shape or state_size must be provided.")
      input_shape = (int(state_size),)

    self.input_shape = tuple(int(dim) for dim in input_shape)
    self.state_size = int(np.prod(self.input_shape))
    self.action_size = action_size
    self.aux_feature_channels = int(aux_feature_channels)
    self.conv_channels = tuple(int(size) for size in conv_channels)
    self.residual_blocks_per_stage = int(residual_blocks_per_stage)
    self.shared_feature_size = int(shared_feature_size)
    self.head_hidden_sizes = tuple(int(size) for size in head_hidden_sizes)
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.clip_ratio = clip_ratio
    self.logit_threshold = logit_threshold
    self.entropy_coef = entropy_coef
    self.value_coef = value_coef
    self.eta = eta
    self.max_grad_norm = max_grad_norm
    self.update_epochs = update_epochs
    self.num_minibatches = num_minibatches
    self.normalize_advantages = normalize_advantages
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    self.network = SharedActorCritic(
        input_shape=self.input_shape,
        num_actions=action_size,
        aux_feature_channels=self.aux_feature_channels,
        conv_channels=self.conv_channels,
        residual_blocks_per_stage=self.residual_blocks_per_stage,
        shared_feature_size=self.shared_feature_size,
        head_hidden_sizes=self.head_hidden_sizes).to(self.device)
    self.optimizer = optim.Adam(
        self.network.parameters(),
        lr=learning_rate,
        eps=1e-5)

  def export_config(self) -> Dict[str, object]:
    """Returns the constructor config needed to rebuild the agent."""
    return {
        "action_size": self.action_size,
        "input_shape": self.input_shape,
        "aux_feature_channels": self.aux_feature_channels,
        "conv_channels": self.conv_channels,
        "residual_blocks_per_stage": self.residual_blocks_per_stage,
        "shared_feature_size": self.shared_feature_size,
        "head_hidden_sizes": self.head_hidden_sizes,
        "learning_rate": self.learning_rate,
        "gamma": self.gamma,
        "gae_lambda": self.gae_lambda,
        "clip_ratio": self.clip_ratio,
        "logit_threshold": self.logit_threshold,
        "entropy_coef": self.entropy_coef,
        "value_coef": self.value_coef,
        "eta": self.eta,
        "max_grad_norm": self.max_grad_norm,
        "update_epochs": self.update_epochs,
        "num_minibatches": self.num_minibatches,
        "normalize_advantages": self.normalize_advantages,
    }

  def _validate_checkpoint_config(self,
                                  checkpoint_config: Dict[str, object]) -> None:
    expected = self.export_config()
    structural_keys = (
        "action_size",
        "input_shape",
        "aux_feature_channels",
        "conv_channels",
        "residual_blocks_per_stage",
        "shared_feature_size",
        "head_hidden_sizes",
    )
    mismatches = []
    for key in structural_keys:
      if key not in checkpoint_config:
        continue
      checkpoint_value = checkpoint_config[key]
      expected_value = expected[key]
      if isinstance(expected_value, tuple):
        matches = tuple(checkpoint_value) == expected_value  # type: ignore[arg-type]
      else:
        matches = checkpoint_value == expected_value
      if not matches:
        mismatches.append(
            f"{key}: checkpoint={checkpoint_value!r}, "
            f"current={expected_value!r}")
    if mismatches:
      raise ValueError(
          "Checkpoint architecture does not match the current agent:\n" +
          "\n".join(mismatches))

  def _to_tensor(self, array: np.ndarray,
                 dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=self.device)

  def _policy_and_value(
      self,
      states: torch.Tensor,
      auxiliary_features: Optional[torch.Tensor],
      legal_masks: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = self.network(states, auxiliary_features)
    probs = masked_softmax(logits, legal_masks)
    return logits, probs, values

  def _inference_policy_and_value(
      self,
      states: torch.Tensor,
      auxiliary_features: Optional[torch.Tensor],
      legal_masks: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = self.network.training
    self.network.eval()
    try:
      with torch.no_grad():
        _, probs, values = self._policy_and_value(
            states, auxiliary_features, legal_masks)
    finally:
      if was_training:
        self.network.train()
    return probs, values

  def act(self,
          state: np.ndarray,
          legal_mask: np.ndarray,
          auxiliary_features: Optional[np.ndarray] = None,
          training: bool = True) -> Tuple[int, float, float]:
    """Samples or greedily selects an action from the current policy."""
    state_t = self._to_tensor(state[None, ...])
    auxiliary_features_t = None
    if auxiliary_features is not None and len(auxiliary_features):
      auxiliary_features_t = self._to_tensor(auxiliary_features[None, ...])
    mask_t = self._to_tensor(legal_mask[None, :], dtype=torch.bool)

    probs, values = self._inference_policy_and_value(
        state_t, auxiliary_features_t, mask_t)
    if training:
      action = torch.distributions.Categorical(probs).sample()[0]
    else:
      action = probs.argmax(dim=-1)[0]
    chosen_prob = probs[0, action].clamp_min(1e-8)
    return int(action.item()), float(chosen_prob.item()), float(values[0].item())

  def _accumulate_rewards(self,
                          state: pyspiel.State,
                          pending: List[Optional[PendingTransition]]) -> None:
    rewards = np.asarray(state.rewards(), dtype=np.float32)
    for player, transition in enumerate(pending):
      if transition is None:
        continue
      transition.reward += float(transition.bootstrap_discount * rewards[player])
      transition.bootstrap_discount *= self.gamma

  def _finalize_episode(self,
                        player_steps: List[List[PendingTransition]]
                        ) -> List[BatchTransition]:
    batch: List[BatchTransition] = []
    for steps in player_steps:
      if not steps:
        continue

      advantages = np.zeros(len(steps), dtype=np.float32)
      returns = np.zeros(len(steps), dtype=np.float32)
      gae = 0.0
      for index in reversed(range(len(steps))):
        step = steps[index]
        next_value = 0.0 if step.done else step.next_value
        next_gae = 0.0 if step.done else gae
        delta = step.reward + step.bootstrap_discount * next_value - step.value
        gae = delta + step.bootstrap_discount * self.gae_lambda * next_gae
        advantages[index] = gae
        returns[index] = gae + step.value

      for step, advantage, return_ in zip(steps, advantages, returns):
        batch.append(
            BatchTransition(
                state=step.state,
                auxiliary_features=step.auxiliary_features,
                legal_mask=step.legal_mask,
                action=step.action,
                old_prob=step.old_prob,
                advantage=float(advantage),
                return_=float(return_)))
    return batch

  def collect_batch(self,
                    game: pyspiel.Game,
                    min_samples: int) -> List[BatchTransition]:
    """Collects one self-play batch with the current policy."""
    transitions: List[BatchTransition] = []
    num_players = game.num_players()

    while len(transitions) < min_samples:
      state = game.new_initial_state()
      pending: List[Optional[PendingTransition]] = [None] * num_players
      player_steps: List[List[PendingTransition]] = [[] for _ in range(num_players)]

      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(np.random.choice(actions, p=probs))
          self._accumulate_rewards(state, pending)
          continue

        player = state.current_player()
        model_input = get_state_tensor(state, player, game)
        auxiliary_features = get_auxiliary_features(
            state, player, game, self.aux_feature_channels)
        legal_actions = state.legal_actions(player)
        mask = legal_actions_mask(legal_actions, self.action_size)
        action, old_prob, value = self.act(
            model_input,
            mask,
            auxiliary_features=auxiliary_features,
            training=True)

        if pending[player] is not None:
          pending[player].next_value = value
          pending[player].done = False
          player_steps[player].append(pending[player])

        pending[player] = PendingTransition(
            state=model_input.copy(),
            auxiliary_features=auxiliary_features.copy(),
            legal_mask=mask.copy(),
            action=action,
            old_prob=old_prob,
            value=value)

        state.apply_action(action)
        self._accumulate_rewards(state, pending)

      for player, transition in enumerate(pending):
        if transition is None:
          continue
        transition.next_value = 0.0
        transition.bootstrap_discount = 0.0
        transition.done = True
        player_steps[player].append(transition)

      transitions.extend(self._finalize_episode(player_steps))

    return transitions

  def collect_batch_parallel(
      self,
      game_name: str,
      min_samples: int,
      num_workers: Optional[int] = None,
      pool: Optional[mp.pool.Pool] = None,
      worker_seed: Optional[int] = None,
      mp_start_method: Optional[str] = None,
  ) -> List[BatchTransition]:
    """Collects one self-play batch with multiprocessing."""
    resolved_workers = min(_resolve_num_workers(num_workers), max(1, min_samples))
    if resolved_workers <= 1 or min_samples <= 1:
      game = pyspiel.load_game(game_name)
      return self.collect_batch(game, min_samples=min_samples)

    _configure_torch_multiprocessing()

    worker_counts = _split_sample_count(min_samples, resolved_workers)
    agent_config = {
        "input_shape": self.input_shape,
        "action_size": self.action_size,
        "aux_feature_channels": self.aux_feature_channels,
        "conv_channels": self.conv_channels,
        "residual_blocks_per_stage": self.residual_blocks_per_stage,
        "shared_feature_size": self.shared_feature_size,
        "head_hidden_sizes": self.head_hidden_sizes,
        "gamma": self.gamma,
        "gae_lambda": self.gae_lambda,
    }
    network_state = _cpu_state_dict(self.network.state_dict())
    worker_args = []
    for worker_idx, worker_count in enumerate(worker_counts):
      seed = None if worker_seed is None else worker_seed + worker_idx
      worker_args.append(
          (game_name, worker_count, seed, agent_config, network_state))

    if pool is not None:
      results = pool.map(_collect_batch_worker, worker_args)
    else:
      start_method = _resolve_start_method(self.device, mp_start_method)
      ctx = mp.get_context(start_method)
      with ctx.Pool(processes=len(worker_counts)) as tmp_pool:
        results = tmp_pool.map(_collect_batch_worker, worker_args)

    transitions: List[BatchTransition] = []
    for worker_result in results:
      transitions.extend(worker_result)
    return transitions[:min_samples]

  def _build_minibatches(self, batch_size: int) -> List[np.ndarray]:
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    if self.num_minibatches <= 1:
      return [indices]
    return list(np.array_split(indices, self.num_minibatches))

  def update(self, transitions: Sequence[BatchTransition]) -> Dict[str, float]:
    """Updates the shared network with the practical ACH loss."""
    if not transitions:
      return {
          "loss": 0.0,
          "policy_loss": 0.0,
          "value_loss": 0.0,
          "entropy_term": 0.0,
      }

    self.network.train()

    states = self._to_tensor(np.asarray([t.state for t in transitions]))
    auxiliary_features = None
    if self.aux_feature_channels:
      auxiliary_features = self._to_tensor(
          np.asarray([t.auxiliary_features for t in transitions],
                     dtype=np.float32))
    legal_masks = self._to_tensor(
        np.asarray([t.legal_mask for t in transitions]), dtype=torch.bool)
    actions = self._to_tensor(
        np.asarray([t.action for t in transitions]), dtype=torch.long)
    old_probs = self._to_tensor(
        np.asarray([t.old_prob for t in transitions], dtype=np.float32))
    advantages = self._to_tensor(
        np.asarray([t.advantage for t in transitions], dtype=np.float32))
    returns = self._to_tensor(
        np.asarray([t.return_ for t in transitions], dtype=np.float32))

    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_term": 0.0,
    }
    num_updates = 0

    for _ in range(self.update_epochs):
      for minibatch_indices in self._build_minibatches(len(transitions)):
        mb_states = states[minibatch_indices]
        mb_auxiliary_features = None
        if auxiliary_features is not None:
          mb_auxiliary_features = auxiliary_features[minibatch_indices]
        mb_masks = legal_masks[minibatch_indices]
        mb_actions = actions[minibatch_indices]
        mb_old_probs = old_probs[minibatch_indices].clamp_min(1e-8)
        mb_advantages = advantages[minibatch_indices]
        mb_returns = returns[minibatch_indices]

        if self.normalize_advantages and len(mb_advantages) > 1:
          mb_advantages = ((mb_advantages - mb_advantages.mean()) /
                           (mb_advantages.std() + 1e-8))

        logits, policy_probs, values = self._policy_and_value(
            mb_states, mb_auxiliary_features, mb_masks)
        chosen_probs = policy_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
        chosen_probs = chosen_probs.clamp_min(1e-8)

        taken_logits = logits.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
        mean_logits = masked_mean(logits, mb_masks)
        centered_logits = taken_logits - mean_logits
        clipped_centered_logits = centered_logits.clamp(
            -self.logit_threshold, self.logit_threshold)

        ratio = chosen_probs / mb_old_probs
        positive_adv = mb_advantages >= 0
        positive_gate = ((ratio < (1.0 + self.clip_ratio)) &
                         (centered_logits < self.logit_threshold))
        negative_gate = ((ratio > (1.0 - self.clip_ratio)) &
                         (centered_logits > -self.logit_threshold))
        gate = torch.where(positive_adv, positive_gate, negative_gate).float()

        policy_loss = (
            -self.eta * gate * clipped_centered_logits * mb_advantages /
            mb_old_probs).mean()

        value_loss = 0.5 * F.mse_loss(values, mb_returns)
        entropy_term = (
            policy_probs * torch.log(policy_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_term)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        metrics["loss"] += float(total_loss.item())
        metrics["policy_loss"] += float(policy_loss.item())
        metrics["value_loss"] += float(value_loss.item())
        metrics["entropy_term"] += float(entropy_term.item())
        num_updates += 1

    for key in metrics:
      metrics[key] /= max(num_updates, 1)
    return metrics

  def save(self, path: str, iteration: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(_checkpoint_payload(self, iteration), path)

  def load(self, path: str) -> int:
    checkpoint = torch.load(path, map_location=self.device)
    checkpoint_config = checkpoint.get("agent_config")
    if checkpoint_config is not None:
      self._validate_checkpoint_config(checkpoint_config)
    self.network.load_state_dict(checkpoint["network"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", 0))

  @classmethod
  def from_checkpoint(
      cls,
      path: str,
      device: Optional[str] = None,
  ) -> Tuple["PracticalACHAgent", int]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    checkpoint_config = checkpoint.get("agent_config")
    if checkpoint_config is None:
      raise ValueError(
          "Checkpoint does not contain agent_config metadata. "
          "Rebuild the agent manually before loading this older checkpoint.")
    agent_kwargs = dict(checkpoint_config)
    agent_kwargs.pop("hidden_sizes", None)
    agent_kwargs.pop("encoder_type", None)
    if device is not None:
      agent_kwargs["device"] = device
    agent = cls(**agent_kwargs)
    agent.network.load_state_dict(checkpoint["network"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    return agent, int(checkpoint.get("iteration", 0))


def _checkpoint_payload(agent: PracticalACHAgent,
                        iteration: int) -> Dict[str, object]:
  """Builds a serializable checkpoint payload."""
  return {
      "iteration": iteration,
      "agent_config": agent.export_config(),
      "network": agent.network.state_dict(),
      "optimizer": agent.optimizer.state_dict(),
  }


def _collect_batch_worker(worker_args) -> List[BatchTransition]:
  """Collects one self-play shard in a separate process."""
  (game_name, min_samples, worker_seed, agent_config,
   network_state) = worker_args

  if worker_seed is not None:
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
  torch.set_num_threads(1)

  game = pyspiel.load_game(game_name)
  worker_agent = PracticalACHAgent(
      input_shape=agent_config["input_shape"],
      action_size=agent_config["action_size"],
      aux_feature_channels=agent_config["aux_feature_channels"],
      conv_channels=agent_config["conv_channels"],
      residual_blocks_per_stage=agent_config["residual_blocks_per_stage"],
      shared_feature_size=agent_config["shared_feature_size"],
      head_hidden_sizes=agent_config["head_hidden_sizes"],
      gamma=agent_config["gamma"],
      gae_lambda=agent_config["gae_lambda"],
      device="cpu")
  worker_agent.network.load_state_dict(network_state)
  return worker_agent.collect_batch(game, min_samples=min_samples)


def evaluate_vs_random(agent: PracticalACHAgent,
                       game_name: str,
                       num_games: int = 200) -> Tuple[float, float]:
  """Evaluates player 0 against a random opponent in a two-player game."""
  game = pyspiel.load_game(game_name)
  if game.num_players() != 2:
    raise ValueError("evaluate_vs_random only supports two-player games.")

  wins = 0
  total_return = 0.0
  num_actions = game.num_distinct_actions()

  for _ in range(num_games):
    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(np.random.choice(actions, p=probs))
        continue

      player = state.current_player()
      legal_actions = state.legal_actions(player)
      if player == 0:
        model_input = get_state_tensor(state, player, game)
        auxiliary_features = get_auxiliary_features(
            state, player, game, agent.aux_feature_channels)
        mask = legal_actions_mask(legal_actions, num_actions)
        action, _, _ = agent.act(
            model_input,
            mask,
            auxiliary_features=auxiliary_features,
            training=False)
      else:
        action = int(np.random.choice(legal_actions))
      state.apply_action(action)

    returns = state.returns()
    total_return += returns[0]
    if returns[0] > returns[1]:
      wins += 1

  return wins / num_games, total_return / num_games


def train_practical_ach(
    game_name: str = "eren_yifang",
    num_iterations: int = 1000,
    batch_size: int = 512,
    eval_freq: int = 50,
    eval_games: int = 200,
    save_interval_seconds: float = 3600.0,
    checkpoint_prefix: str = "ach_practical",
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
    seed: Optional[int] = None,
    num_workers: Optional[int] = None,
    mp_start_method: Optional[str] = None,
    **agent_kwargs,
) -> PracticalACHAgent:
  """Trains the practical ACH approximation on one OpenSpiel game."""
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  game = pyspiel.load_game(game_name)
  game_type = game.get_type()
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("This example only supports turn-based sequential games.")

  state_shape = get_state_shape(game)
  action_size = game.num_distinct_actions()
  if agent_kwargs.get("aux_feature_channels") is None:
    agent_kwargs["aux_feature_channels"] = infer_aux_feature_channels(
        game_name, game)

  agent = PracticalACHAgent(
      input_shape=state_shape,
      action_size=action_size,
      **agent_kwargs)

  checkpoint_dir_path = Path(checkpoint_dir)
  checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

  start_iteration = 0
  latest_checkpoint = None
  if resume:
    latest_checkpoint = _latest_checkpoint_path(
        checkpoint_dir=checkpoint_dir_path,
        checkpoint_prefix=checkpoint_prefix,
        game_name=game_name)
    if latest_checkpoint is not None:
      start_iteration = agent.load(str(latest_checkpoint))
      print(
          f"Resumed from latest checkpoint: {latest_checkpoint} "
          f"(iteration {start_iteration})")

  last_save_time = time.time()
  last_saved_iteration = start_iteration
  last_completed_iteration = start_iteration

  resolved_workers = _resolve_num_workers(num_workers)
  use_parallel = resolved_workers > 1
  pool = None

  if use_parallel:
    _configure_torch_multiprocessing()
    start_method = _resolve_start_method(agent.device, mp_start_method)
    print(
        f"Using {resolved_workers} worker processes for self-play collection "
        f"({start_method} start method)")
    ctx = mp.get_context(start_method)
    pool = ctx.Pool(processes=resolved_workers)

  try:
    for iteration in range(start_iteration + 1, num_iterations + 1):
      if use_parallel:
        iteration_seed = None
        if seed is not None:
          iteration_seed = seed + (iteration - 1) * resolved_workers
        batch = agent.collect_batch_parallel(
            game_name=game_name,
            min_samples=batch_size,
            num_workers=resolved_workers,
            pool=pool,
            worker_seed=iteration_seed)
      else:
        batch = agent.collect_batch(game, min_samples=batch_size)

      metrics = agent.update(batch)
      last_completed_iteration = iteration

      if eval_freq and iteration % eval_freq == 0:
        if game.num_players() == 2:
          win_rate, avg_return = evaluate_vs_random(
              agent, game_name, num_games=eval_games)
          print(
              f"Iter {iteration:>5}/{num_iterations}  "
              f"Samples={len(batch):>4}  "
              f"WinRate={win_rate:.0%}  "
              f"AvgRet={avg_return:+.3f}  "
              f"Loss={metrics['loss']:.4f}  "
              f"PiL={metrics['policy_loss']:.4f}  "
              f"VL={metrics['value_loss']:.4f}")
        else:
          print(
              f"Iter {iteration:>5}/{num_iterations}  "
              f"Samples={len(batch):>4}  "
              f"Loss={metrics['loss']:.4f}  "
              f"PiL={metrics['policy_loss']:.4f}  "
              f"VL={metrics['value_loss']:.4f}")

      if save_interval_seconds > 0:
        now = time.time()
        if now - last_save_time >= save_interval_seconds:
          checkpoint_path = checkpoint_dir_path / (
              f"{checkpoint_prefix}_{_safe_game_name(game_name)}_"
              f"{iteration}.pth")
          agent.save(str(checkpoint_path), iteration=iteration)
          last_save_time = now
          last_saved_iteration = iteration
          print(f"Saved checkpoint: {checkpoint_path}")
  finally:
    if pool is not None:
      pool.close()
      pool.join()

  if last_completed_iteration > start_iteration and (
      last_saved_iteration != last_completed_iteration):
    checkpoint_path = checkpoint_dir_path / (
        f"{checkpoint_prefix}_{_safe_game_name(game_name)}_"
        f"{last_completed_iteration}.pth")
    agent.save(str(checkpoint_path), iteration=last_completed_iteration)
    print(f"Saved final checkpoint: {checkpoint_path}")

  return agent


def play_interactive_game(game_name: str = "eren_yifang",
                          checkpoint_dir: str = "checkpoints",
                          checkpoint_prefix: str = "ach_practical",
                          device: Optional[str] = None) -> None:
  """Plays an interactive game against the most recent trained checkpoint."""
  latest_checkpoint = _latest_checkpoint_path(
      checkpoint_dir=Path(checkpoint_dir),
      checkpoint_prefix=checkpoint_prefix,
      game_name=game_name)
  if latest_checkpoint is None:
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir!r} for game {game_name!r}.")

  agent, loaded_iteration = PracticalACHAgent.from_checkpoint(
      str(latest_checkpoint), device=device)
  print(f"Loaded checkpoint iteration: {loaded_iteration}")

  game = pyspiel.load_game(game_name)
  state = game.new_initial_state()

  print(f"\n{'=' * 50}")
  print(f"Playing {game_name}")
  print(f"{'=' * 50}\n")

  while not state.is_terminal():
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      actions, probs = zip(*outcomes)
      state.apply_action(np.random.choice(actions, p=probs))
      continue

    print(f"\nCurrent state:\n{state}")
    current_player = state.current_player()
    legal_actions = state.legal_actions(current_player)

    if current_player == 0:
      model_input = get_state_tensor(state, current_player, game)
      auxiliary_features = get_auxiliary_features(
          state, current_player, game, agent.aux_feature_channels)
      mask = legal_actions_mask(legal_actions, game.num_distinct_actions())
      action, _, _ = agent.act(
          model_input,
          mask,
          auxiliary_features=auxiliary_features,
          training=False)
      print(f"Agent chooses action: {action}")
    else:
      print(f"Legal actions: {legal_actions}")
      while True:
        try:
          action = int(input("Your action: "))
          if action in legal_actions:
            break
          print(f"Invalid action! Choose from {legal_actions}")
        except ValueError:
          print("Please enter a number.")

    state.apply_action(action)

  print("\nGame over!")
  print(f"Returns: {state.returns()}")
  if state.returns()[1] > state.returns()[0]:
    print("You win!")
  elif state.returns()[0] > state.returns()[1]:
    print("Agent wins!")
  else:
    print("It's a draw!")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train or play with the practical PyTorch ACH trainer.")
  parser.add_argument(
      "--mode",
      choices=("train", "play"),
      default="train",
      help="Whether to train a model or play against the latest checkpoint.")
  parser.add_argument(
      "--game_name",
      "--game",
      default="eren_yifang",
      help="OpenSpiel game string. Defaults to eren_yifang.")
  parser.add_argument("--num_iterations", type=int, default=1000000)
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--eval_freq", type=int, default=1000)
  parser.add_argument("--eval_games", type=int, default=200)
  parser.add_argument("--save_interval_seconds", type=float, default=3600.0)
  parser.add_argument("--checkpoint_dir", default="checkpoints")
  parser.add_argument("--checkpoint_prefix", default="ach_practical")
  parser.add_argument(
      "--no_resume",
      action="store_true",
      help="Start fresh instead of loading the newest matching checkpoint.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--mp_start_method", default=None)
  parser.add_argument(
      "--aux_feature_channels",
      type=int,
      default=None,
      help=(
          "Number of auxiliary features read from InformationStateTensor and "
          "concatenated after the CNN. By default this is inferred for "
          "supported games such as eren_yifang."))

  parser.add_argument(
      "--conv_channels",
      default="64,128,32",
      help="Comma-separated residual stage widths for image-like inputs.")
  parser.add_argument(
      "--residual_blocks_per_stage",
      type=int,
      default=3,
      help="Number of residual blocks in each CNN stage.")
  parser.add_argument(
      "--shared_feature_size",
      type=int,
      default=1024,
      help="Shared feature width after the CNN trunk.")
  parser.add_argument(
      "--head_hidden_sizes",
      default="512,512",
      help="Comma-separated hidden sizes for policy/value heads after the CNN.")

  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--gae_lambda", type=float, default=0.95)
  parser.add_argument("--clip_ratio", type=float, default=0.05)
  parser.add_argument("--logit_threshold", type=float, default=1.5)
  parser.add_argument("--entropy_coef", type=float, default=0.02)
  parser.add_argument("--value_coef", type=float, default=0.5)
  parser.add_argument("--eta", type=float, default=1.0)
  parser.add_argument("--max_grad_norm", type=float, default=0.5)
  parser.add_argument("--update_epochs", type=int, default=1)
  parser.add_argument("--num_minibatches", type=int, default=1)
  parser.add_argument(
      "--normalize_advantages",
      action="store_true",
      help="Normalize GAE advantages inside each minibatch.")
  parser.add_argument(
      "--device",
      default=None,
      help="Torch device, e.g. cpu or cuda. Defaults to CUDA if available.")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()

  if args.mode == "play":
    play_interactive_game(
        game_name=args.game_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        device=args.device)
    return

  print(f"Training practical ACH on {args.game_name}...")
  train_practical_ach(
      game_name=args.game_name,
      num_iterations=args.num_iterations,
      batch_size=args.batch_size,
      eval_freq=args.eval_freq,
      eval_games=args.eval_games,
      save_interval_seconds=args.save_interval_seconds,
      checkpoint_prefix=args.checkpoint_prefix,
      checkpoint_dir=args.checkpoint_dir,
      resume=not args.no_resume,
      seed=args.seed,
      num_workers=args.num_workers,
      mp_start_method=args.mp_start_method,
      aux_feature_channels=args.aux_feature_channels,
      conv_channels=_parse_int_tuple(args.conv_channels),
      residual_blocks_per_stage=args.residual_blocks_per_stage,
      shared_feature_size=args.shared_feature_size,
      head_hidden_sizes=_parse_int_tuple(args.head_hidden_sizes, allow_empty=True),
      learning_rate=args.learning_rate,
      gamma=args.gamma,
      gae_lambda=args.gae_lambda,
      clip_ratio=args.clip_ratio,
      logit_threshold=args.logit_threshold,
      entropy_coef=args.entropy_coef,
      value_coef=args.value_coef,
      eta=args.eta,
      max_grad_norm=args.max_grad_norm,
      update_epochs=args.update_epochs,
      num_minibatches=args.num_minibatches,
      normalize_advantages=args.normalize_advantages,
      device=args.device)


if __name__ == "__main__":
  main()
