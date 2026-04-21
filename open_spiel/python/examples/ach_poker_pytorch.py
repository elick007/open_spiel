"""PyTorch port of the ACH_poker ACH trainer for OpenSpiel.

This example mirrors the ACH variant used in
https://github.com/Liuweiming/ACH_poker without modifying the existing
``ach_pytorch.py`` or ``ach_paper_pytorch.py`` implementations in this
repository.

What is preserved from ``ACH_poker``:
  - one policy/value network per player;
  - target-player external-sampling traversals from ``algorithms/ach_solver.cc``;
  - the reverse-pass target construction from ``algorithms/ach.cc``;
  - the clipped ACH objective from ``algorithms/deep_cfr_model.py``.

What is intentionally adapted:
  - the TensorFlow model is replaced with PyTorch;
  - the poker-specific card/bet encoder is replaced with a generic MLP over
    OpenSpiel information-state (or observation) tensors so it works for
    standard OpenSpiel sequential games.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import os
from pathlib import Path
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


INVALID_ACTION_LOGIT = -1e9
pyspiel = None


def _ensure_pyspiel():
  """Imports pyspiel lazily so CLI help works before OpenSpiel is built."""
  global pyspiel
  if pyspiel is None:
    import pyspiel as pyspiel_module  # pylint: disable=g-import-not-at-top
    pyspiel = pyspiel_module
  return pyspiel


def _safe_game_name(game_name: str) -> str:
  """Returns a filesystem-safe game name for checkpoint filenames."""
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", game_name).strip("_") or "game"


def layer_init(layer: nn.Linear,
               std: float = np.sqrt(2.0),
               bias_const: float = 0.0) -> nn.Linear:
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


def get_state_shape(game: pyspiel.Game) -> Tuple[int, ...]:
  """Returns the preferred state tensor shape for a game."""
  game_type = game.get_type()
  if game_type.provides_information_state_tensor:
    return tuple(game.information_state_tensor_shape())
  if game_type.provides_observation_tensor:
    return tuple(game.observation_tensor_shape())
  raise ValueError(
      f"Game {game_type.short_name} provides neither information_state_tensor "
      "nor observation_tensor.")


def get_state_tensor(state: pyspiel.State,
                     player: int,
                     game: pyspiel.Game) -> np.ndarray:
  """Returns a flattened state tensor for one player."""
  if game.get_type().provides_information_state_tensor:
    tensor = state.information_state_tensor(player)
  else:
    tensor = state.observation_tensor(player)
  return np.asarray(tensor, dtype=np.float32).reshape(-1)


def legal_actions_mask(legal_actions: Sequence[int],
                       num_actions: int) -> np.ndarray:
  mask = np.zeros(num_actions, dtype=bool)
  mask[list(legal_actions)] = True
  return mask


def get_state_key(state: pyspiel.State,
                  player: int,
                  game: pyspiel.Game) -> str:
  """Returns an information-state key for tabular average-policy tracking."""
  game_type = game.get_type()
  if getattr(game_type, "provides_information_state_string", False):
    return state.information_state_string(player)
  if getattr(game_type, "provides_observation_string", False):
    return state.observation_string(player)
  tensor = get_state_tensor(state, player, game)
  return tensor.tobytes().hex()


def _renormalize(probs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
  """Clips to legal actions and renormalizes a policy vector."""
  probs = np.asarray(probs, dtype=np.float64)
  legal_probs = np.where(legal_mask, probs, 0.0)
  total = legal_probs.sum()
  if total <= 0:
    legal_probs = legal_mask.astype(np.float64)
    total = legal_probs.sum()
  return (legal_probs / max(total, 1.0)).astype(np.float32)


def uniform_policy_vector(legal_actions: Sequence[int],
                          num_actions: int) -> np.ndarray:
  probs = np.zeros(num_actions, dtype=np.float32)
  if legal_actions:
    probs[list(legal_actions)] = 1.0 / len(legal_actions)
  return probs


def sample_action_from_policy(legal_actions: Sequence[int],
                              policy_probs: np.ndarray) -> int:
  legal_probs = np.asarray([policy_probs[action] for action in legal_actions],
                           dtype=np.float64)
  legal_probs = legal_probs / legal_probs.sum()
  return int(np.random.choice(list(legal_actions), p=legal_probs))


def masked_softmax(logits: torch.Tensor,
                   legal_mask: torch.Tensor) -> torch.Tensor:
  masked_logits = logits.masked_fill(~legal_mask, INVALID_ACTION_LOGIT)
  return F.softmax(masked_logits, dim=-1)


def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]
                    ) -> Dict[str, torch.Tensor]:
  """Returns a detached CPU copy safe to send to worker processes."""
  return {
      key: value.detach().cpu().clone()
      for key, value in state_dict.items()
  }


def _split_sample_count(total_samples: int, num_workers: int) -> List[int]:
  base = total_samples // num_workers
  remainder = total_samples % num_workers
  counts = [
      base + (1 if worker_idx < remainder else 0)
      for worker_idx in range(num_workers)
  ]
  return [count for count in counts if count > 0]


def _resolve_num_workers(num_workers: Optional[int]) -> int:
  if num_workers is not None:
    return max(1, num_workers)
  cpu_count = os.cpu_count() or 1
  if sys.platform.startswith("linux"):
    return min(cpu_count, 4)
  return 1


def _resolve_start_method(device: str,
                          mp_start_method: Optional[str]) -> str:
  if mp_start_method is not None:
    return mp_start_method
  if sys.platform.startswith("linux") and not str(device).startswith("cuda"):
    return "fork"
  return "spawn"


def _latest_checkpoint_path(checkpoint_dir: Path,
                            checkpoint_prefix: str,
                            game_name: str) -> Optional[Path]:
  """Returns the newest all-player checkpoint for this game."""
  if not checkpoint_dir.exists():
    return None

  pattern = f"{checkpoint_prefix}_{_safe_game_name(game_name)}_*.pth"
  candidates = []
  for path in checkpoint_dir.glob(pattern):
    if not path.is_file():
      continue
    if not path.stem.rsplit("_", maxsplit=1)[-1].isdigit():
      continue
    candidates.append(path)
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _save_agents_checkpoint(path: Path,
                            agents: Sequence["ACHPokerAgent"],
                            iteration: int,
                            average_policy: Optional[
                                "AveragePolicyAccumulator"] = None) -> None:
  """Saves every player network in one synchronized checkpoint."""
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "iteration": iteration,
      "agents": [
          {
              "network": agent.network.state_dict(),
              "optimizer": agent.optimizer.state_dict(),
          }
          for agent in agents
      ],
  }
  if average_policy is not None:
    payload["average_policy"] = average_policy.state_dict()
  torch.save(payload, path)


def _load_agents_checkpoint(path: Path,
                            agents: Sequence["ACHPokerAgent"],
                            average_policy: Optional[
                                "AveragePolicyAccumulator"] = None) -> int:
  """Loads a synchronized checkpoint and returns its iteration."""
  checkpoint = torch.load(path, map_location=agents[0].device)
  saved_agents = checkpoint["agents"]
  if len(saved_agents) != len(agents):
    raise ValueError(
        f"Checkpoint has {len(saved_agents)} agents, expected {len(agents)}.")

  for agent, saved_agent in zip(agents, saved_agents):
    agent.network.load_state_dict(saved_agent["network"])
    agent.optimizer.load_state_dict(saved_agent["optimizer"])
  if average_policy is not None and "average_policy" in checkpoint:
    average_policy.load_state_dict(checkpoint["average_policy"])
  return int(checkpoint.get("iteration", 0))


@dataclass
class TrajectoryStep:
  """One sampled decision point for a single player."""

  state: np.ndarray
  legal_mask: np.ndarray
  action: int
  old_prob: float
  current_prob: float
  value: float


@dataclass
class ACHSample:
  """A processed ACH training sample."""

  state: np.ndarray
  legal_mask: np.ndarray
  action: int
  old_prob: float
  advantage: float
  value_target: float


@dataclass
class CollectionResult:
  """Samples and optional average-policy increments from one collection pass."""

  samples: List[List[ACHSample]]
  average_policy_state: Optional[List[Dict[str, np.ndarray]]]
  num_trajectories: int
  num_states: int


class PolicyValueNet(nn.Module):
  """Shared torso with policy and value heads."""

  def __init__(self,
               input_size: int,
               num_actions: int,
               hidden_sizes: Sequence[int] = (256, 256)):
    super().__init__()
    layers: List[nn.Module] = []
    prev_size = input_size
    for hidden_size in hidden_sizes:
      layers.extend([
          layer_init(nn.Linear(prev_size, hidden_size)),
          nn.Tanh(),
      ])
      prev_size = hidden_size
    self.torso = nn.Sequential(*layers)
    self.policy_head = layer_init(nn.Linear(prev_size, num_actions), std=0.01)
    self.value_head = layer_init(nn.Linear(prev_size, 1), std=1.0)

  def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden = self.torso(states)
    logits = self.policy_head(hidden)
    values = self.value_head(hidden).squeeze(-1)
    return logits, values


class ACHPokerAgent:
  """PyTorch ACH agent aligned with the ACH_poker objective."""

  def __init__(
      self,
      state_size: int,
      action_size: int,
      hidden_sizes: Sequence[int] = (256, 256),
      learning_rate: float = 1e-4,
      weight_decay: float = 0.0,
      gamma: float = 0.995,
      gae_lambda: float = 0.95,
      ach_eta: float = 1.0,
      ach_alpha: float = 2.0,
      ach_beta: float = 0.01,
      ach_thres: float = 2.0,
      ach_epsilon: float = 0.05,
      ach_reward_scale: float = 1.0,
      update_epochs: int = 1,
      num_minibatches: int = 1,
      train_batch_size: int = 1024,
      max_grad_norm: float = 1.0,
      device: Optional[str] = None,
  ):
    self.state_size = state_size
    self.action_size = action_size
    self.hidden_sizes = tuple(hidden_sizes)
    self.gamma = gamma
    self.gae_lambda = gae_lambda
    self.ach_eta = ach_eta
    self.ach_alpha = ach_alpha
    self.ach_beta = ach_beta
    self.ach_thres = ach_thres
    self.ach_epsilon = ach_epsilon
    self.ach_reward_scale = ach_reward_scale
    self.update_epochs = update_epochs
    self.num_minibatches = num_minibatches
    self.train_batch_size = train_batch_size
    self.max_grad_norm = max_grad_norm
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    self.network = PolicyValueNet(
        input_size=state_size,
        num_actions=action_size,
        hidden_sizes=self.hidden_sizes).to(self.device)
    self.optimizer = optim.Adam(
        self.network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-5)

  def _to_tensor(self, array: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=self.device)

  def _policy_and_value(
      self,
      states: torch.Tensor,
      legal_masks: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, values = self.network(states)
    probs = masked_softmax(logits, legal_masks)
    return logits, probs, values

  def act(self,
          state: np.ndarray,
          legal_mask: np.ndarray,
          training: bool = True) -> Tuple[int, float, float]:
    """Samples or greedily selects an action from the current policy."""
    state_t = self._to_tensor(state[None, :])
    mask_t = self._to_tensor(legal_mask[None, :], dtype=torch.bool)

    with torch.no_grad():
      _, probs, values = self._policy_and_value(state_t, mask_t)
      if training:
        action = torch.distributions.Categorical(probs).sample()[0]
      else:
        action = probs.argmax(dim=-1)[0]
      chosen_prob = probs[0, action].clamp_min(1e-8)

    return int(action.item()), float(chosen_prob.item()), float(values[0].item())

  def policy_value(self,
                   state: np.ndarray,
                   legal_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Returns the masked current policy and scalar value for one state."""
    state_t = self._to_tensor(state[None, :])
    mask_t = self._to_tensor(legal_mask[None, :], dtype=torch.bool)

    with torch.no_grad():
      _, probs, values = self._policy_and_value(state_t, mask_t)
    return probs[0].detach().cpu().numpy(), float(values[0].item())

  def preprocess_trajectory(self,
                            steps: Sequence[TrajectoryStep],
                            terminal_return: float) -> List[ACHSample]:
    """Matches ACH_poker's reverse target construction."""
    if not steps:
      return []

    samples: List[ACHSample] = []
    cum_reward = 0.0
    last_importance = 1.0
    next_reward = float(terminal_return)
    next_value = 0.0
    next_advantage = 0.0

    for step in reversed(steps):
      old_prob = max(step.old_prob, 1e-8)
      importance = step.current_prob / old_prob
      scaled_reward = next_reward * self.ach_reward_scale

      cum_reward = importance * (scaled_reward + self.gamma * cum_reward)
      delta = scaled_reward + self.gamma * next_value - step.value
      advantage = (
          delta +
          self.gamma * self.gae_lambda * last_importance * next_advantage)

      samples.append(
          ACHSample(
              state=step.state,
              legal_mask=step.legal_mask,
              action=step.action,
              old_prob=step.old_prob,
              advantage=float(advantage),
              value_target=float(cum_reward)))

      next_reward = 0.0
      next_value = step.value
      next_advantage = advantage
      last_importance = importance

    samples.reverse()
    return samples

  @staticmethod
  def collect_ach_traversal_samples(
      game: pyspiel.Game,
      agents: Sequence["ACHPokerAgent"],
      num_traversals: int,
      traversal_eta: float = 1.0,
      average_policy: Optional["AveragePolicyAccumulator"] = None,
      start_step: int = 1,
      return_average_policy_state: bool = False,
  ) -> CollectionResult:
    """Collects ACHSolver-style target-player traversal samples."""
    collector = ACHExternalSampler(
        game=game,
        agents=agents,
        traversal_eta=traversal_eta,
        average_policy=average_policy)
    return collector.collect(
        num_traversals=num_traversals,
        start_step=start_step,
        return_average_policy_state=return_average_policy_state)

  @staticmethod
  def collect_ach_traversal_samples_parallel(
      game_name: str,
      agents: Sequence["ACHPokerAgent"],
      num_traversals: int,
      traversal_eta: float = 1.0,
      collect_average_policy: bool = False,
      num_workers: Optional[int] = None,
      pool: Optional[mp.pool.Pool] = None,
      worker_seed: Optional[int] = None,
      mp_start_method: Optional[str] = None,
      start_step: int = 1,
  ) -> CollectionResult:
    """Parallel ACHSolver-style traversal collection."""
    resolved_workers = min(_resolve_num_workers(num_workers),
                           max(1, num_traversals))
    if resolved_workers <= 1 or num_traversals <= 1:
      pyspiel_module = _ensure_pyspiel()
      game = pyspiel_module.load_game(game_name)
      return ACHPokerAgent.collect_ach_traversal_samples(
          game=game,
          agents=agents,
          num_traversals=num_traversals,
          traversal_eta=traversal_eta,
          start_step=start_step)

    worker_counts = _split_sample_count(num_traversals, resolved_workers)
    agent_configs = [{
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_sizes": agent.hidden_sizes,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "ach_reward_scale": agent.ach_reward_scale,
    } for agent in agents]
    network_states = [_cpu_state_dict(agent.network.state_dict())
                      for agent in agents]

    worker_args = []
    for worker_idx, worker_count in enumerate(worker_counts):
      seed = None if worker_seed is None else worker_seed + worker_idx
      worker_args.append(
          (game_name, worker_count, seed, agent_configs, network_states,
           traversal_eta, collect_average_policy, start_step))

    num_players = len(agents)
    all_samples = [[] for _ in range(num_players)]
    average_policy_state = (
        [dict() for _ in range(num_players)]
        if collect_average_policy else None)
    num_trajectories = 0
    num_states = 0

    def merge_result(result: CollectionResult) -> None:
      nonlocal num_trajectories, num_states
      for player in range(num_players):
        all_samples[player].extend(result.samples[player])
      if (average_policy_state is not None and
          result.average_policy_state is not None):
        _merge_average_policy_states(
            average_policy_state, result.average_policy_state)
      num_trajectories += result.num_trajectories
      num_states += result.num_states

    if pool is not None:
      for result in pool.imap_unordered(
          _collect_traversal_worker, worker_args, chunksize=1):
        merge_result(result)
    else:
      start_method = _resolve_start_method(agents[0].device, mp_start_method)
      ctx = mp.get_context(start_method)
      with ctx.Pool(processes=len(worker_counts)) as tmp_pool:
        for result in tmp_pool.imap_unordered(
            _collect_traversal_worker, worker_args, chunksize=1):
          merge_result(result)

    return CollectionResult(
        samples=all_samples,
        average_policy_state=average_policy_state,
        num_trajectories=num_trajectories,
        num_states=num_states)

  @staticmethod
  def collect_training_samples(
      game: pyspiel.Game,
      agents: Sequence["ACHPokerAgent"],
      num_episodes: int,
  ) -> List[List[ACHSample]]:
    """Collects self-play trajectories and processes them per player."""
    num_players = game.num_players()
    action_size = game.num_distinct_actions()
    all_samples = [[] for _ in range(num_players)]

    for _ in range(num_episodes):
      state = game.new_initial_state()
      player_steps: List[List[TrajectoryStep]] = [[] for _ in range(num_players)]

      while not state.is_terminal():
        if state.is_chance_node():
          actions, probs = zip(*state.chance_outcomes())
          state.apply_action(np.random.choice(actions, p=probs))
          continue

        player = state.current_player()
        info_state = get_state_tensor(state, player, game)
        legal_actions = state.legal_actions(player)
        mask = legal_actions_mask(legal_actions, action_size)
        action, action_prob, value = agents[player].act(
            info_state, mask, training=True)

        player_steps[player].append(
            TrajectoryStep(
                state=info_state.copy(),
                legal_mask=mask.copy(),
                action=action,
                old_prob=action_prob,
                current_prob=action_prob,
                value=value))
        state.apply_action(action)

      returns = state.returns()
      for player, steps in enumerate(player_steps):
        all_samples[player].extend(
            agents[player].preprocess_trajectory(steps, returns[player]))

    return all_samples

  @staticmethod
  def collect_training_samples_parallel(
      game_name: str,
      agents: Sequence["ACHPokerAgent"],
      num_episodes: int,
      num_workers: Optional[int] = None,
      pool: Optional[mp.pool.Pool] = None,
      worker_seed: Optional[int] = None,
      mp_start_method: Optional[str] = None,
  ) -> List[List[ACHSample]]:
    """Parallel self-play collection using detached CPU snapshots."""
    resolved_workers = min(_resolve_num_workers(num_workers), max(1, num_episodes))
    if resolved_workers <= 1 or num_episodes <= 1:
      pyspiel_module = _ensure_pyspiel()
      game = pyspiel_module.load_game(game_name)
      return ACHPokerAgent.collect_training_samples(game, agents, num_episodes)

    worker_counts = _split_sample_count(num_episodes, resolved_workers)
    agent_configs = [{
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_sizes": agent.hidden_sizes,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "ach_reward_scale": agent.ach_reward_scale,
    } for agent in agents]
    network_states = [_cpu_state_dict(agent.network.state_dict())
                      for agent in agents]

    worker_args = []
    for worker_idx, worker_count in enumerate(worker_counts):
      seed = None if worker_seed is None else worker_seed + worker_idx
      worker_args.append(
          (game_name, worker_count, seed, agent_configs, network_states))

    num_players = len(agents)
    all_samples = [[] for _ in range(num_players)]
    if pool is not None:
      result_iter = pool.imap_unordered(
          _collect_samples_worker, worker_args, chunksize=1)
      for worker_result in result_iter:
        for player in range(num_players):
          all_samples[player].extend(worker_result[player])
    else:
      start_method = _resolve_start_method(agents[0].device, mp_start_method)
      ctx = mp.get_context(start_method)
      with ctx.Pool(processes=len(worker_counts)) as tmp_pool:
        result_iter = tmp_pool.imap_unordered(
            _collect_samples_worker, worker_args, chunksize=1)
        for worker_result in result_iter:
          for player in range(num_players):
            all_samples[player].extend(worker_result[player])
    return all_samples

  @staticmethod
  def iter_ach_traversal_results_parallel(
      game_name: str,
      agents: Sequence["ACHPokerAgent"],
      num_traversals: int,
      traversal_eta: float = 1.0,
      collect_average_policy: bool = False,
      num_workers: Optional[int] = None,
      pool: Optional[mp.pool.Pool] = None,
      worker_seed: Optional[int] = None,
      mp_start_method: Optional[str] = None,
      start_step: int = 1,
  ):
    """Yields ACH traversal worker results one at a time."""
    resolved_workers = min(_resolve_num_workers(num_workers),
                           max(1, num_traversals))
    if resolved_workers <= 1 or num_traversals <= 1:
      pyspiel_module = _ensure_pyspiel()
      game = pyspiel_module.load_game(game_name)
      yield ACHPokerAgent.collect_ach_traversal_samples(
          game=game,
          agents=agents,
          num_traversals=num_traversals,
          traversal_eta=traversal_eta,
          start_step=start_step)
      return

    worker_counts = _split_sample_count(num_traversals, resolved_workers)
    agent_configs = [{
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_sizes": agent.hidden_sizes,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "ach_reward_scale": agent.ach_reward_scale,
    } for agent in agents]
    network_states = [_cpu_state_dict(agent.network.state_dict())
                      for agent in agents]
    worker_args = []
    for worker_idx, worker_count in enumerate(worker_counts):
      seed = None if worker_seed is None else worker_seed + worker_idx
      worker_args.append(
          (game_name, worker_count, seed, agent_configs, network_states,
           traversal_eta, collect_average_policy, start_step))

    if pool is not None:
      for result in pool.imap_unordered(
          _collect_traversal_worker, worker_args, chunksize=1):
        yield result
    else:
      start_method = _resolve_start_method(agents[0].device, mp_start_method)
      ctx = mp.get_context(start_method)
      with ctx.Pool(processes=len(worker_counts)) as tmp_pool:
        for result in tmp_pool.imap_unordered(
            _collect_traversal_worker, worker_args, chunksize=1):
          yield result

  @staticmethod
  def iter_training_sample_results_parallel(
      game_name: str,
      agents: Sequence["ACHPokerAgent"],
      num_episodes: int,
      num_workers: Optional[int] = None,
      pool: Optional[mp.pool.Pool] = None,
      worker_seed: Optional[int] = None,
      mp_start_method: Optional[str] = None,
  ):
    """Yields generic self-play worker samples one result at a time."""
    resolved_workers = min(_resolve_num_workers(num_workers), max(1, num_episodes))
    if resolved_workers <= 1 or num_episodes <= 1:
      pyspiel_module = _ensure_pyspiel()
      game = pyspiel_module.load_game(game_name)
      yield ACHPokerAgent.collect_training_samples(game, agents, num_episodes)
      return

    worker_counts = _split_sample_count(num_episodes, resolved_workers)
    agent_configs = [{
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "hidden_sizes": agent.hidden_sizes,
        "gamma": agent.gamma,
        "gae_lambda": agent.gae_lambda,
        "ach_reward_scale": agent.ach_reward_scale,
    } for agent in agents]
    network_states = [_cpu_state_dict(agent.network.state_dict())
                      for agent in agents]
    worker_args = []
    for worker_idx, worker_count in enumerate(worker_counts):
      seed = None if worker_seed is None else worker_seed + worker_idx
      worker_args.append(
          (game_name, worker_count, seed, agent_configs, network_states))

    if pool is not None:
      for result in pool.imap_unordered(
          _collect_samples_worker, worker_args, chunksize=1):
        yield result
    else:
      start_method = _resolve_start_method(agents[0].device, mp_start_method)
      ctx = mp.get_context(start_method)
      with ctx.Pool(processes=len(worker_counts)) as tmp_pool:
        for result in tmp_pool.imap_unordered(
            _collect_samples_worker, worker_args, chunksize=1):
          yield result

  @staticmethod
  def _merge_sample_results(
      results: Sequence[List[List[ACHSample]]],
      num_players: int) -> List[List[ACHSample]]:
    all_samples = [[] for _ in range(num_players)]
    for worker_result in results:
      for player in range(num_players):
        all_samples[player].extend(worker_result[player])
    return all_samples

  def _build_minibatches(self, batch_size: int) -> List[np.ndarray]:
    indices = np.arange(batch_size)
    np.random.shuffle(indices)
    if self.train_batch_size and self.train_batch_size > 0:
      return [
          indices[start:start + self.train_batch_size]
          for start in range(0, batch_size, self.train_batch_size)
      ]
    if self.num_minibatches <= 1:
      return [indices]
    return list(np.array_split(indices, self.num_minibatches))

  def update(self, samples: Sequence[ACHSample]) -> Dict[str, float]:
    """Updates the policy/value net with the ACH_poker loss."""
    if not samples:
      return {
          "loss": 0.0,
          "policy_loss": 0.0,
          "value_loss": 0.0,
          "entropy_loss": 0.0,
      }

    metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
    }
    num_updates = 0

    for _ in range(self.update_epochs):
      for minibatch_indices in self._build_minibatches(len(samples)):
        minibatch = [samples[index] for index in minibatch_indices]
        mb_states = self._to_tensor(
            np.asarray([sample.state for sample in minibatch]))
        mb_masks = self._to_tensor(
            np.asarray([sample.legal_mask for sample in minibatch]),
            dtype=torch.bool)
        mb_actions = self._to_tensor(
            np.asarray([sample.action for sample in minibatch]),
            dtype=torch.long)
        mb_old_probs = self._to_tensor(
            np.asarray([sample.old_prob for sample in minibatch],
                       dtype=np.float32)).clamp_min(1e-8)
        mb_advantages = self._to_tensor(
            np.asarray([sample.advantage for sample in minibatch],
                       dtype=np.float32))
        mb_value_targets = self._to_tensor(
            np.asarray([sample.value_target for sample in minibatch],
                       dtype=np.float32))

        logits, probs, values = self._policy_and_value(mb_states, mb_masks)
        chosen_logits = logits.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
        chosen_probs = probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
        chosen_probs = chosen_probs.clamp_min(1e-8)
        ratios = chosen_probs / mb_old_probs

        positive_gate = (
            (ratios < (1.0 + self.ach_epsilon)) &
            (chosen_logits < self.ach_thres))
        negative_gate = (
            (ratios > (1.0 - self.ach_epsilon)) &
            (chosen_logits > -self.ach_thres))
        gate = torch.where(
            mb_advantages >= 0,
            positive_gate,
            negative_gate).float()

        policy_loss = -(
            gate * chosen_logits * mb_advantages / mb_old_probs).mean()
        value_loss = 0.5 * F.mse_loss(values, mb_value_targets)
        entropy_loss = (
            probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        total_loss = (
            self.ach_eta * policy_loss +
            self.ach_alpha * value_loss +
            self.ach_beta * entropy_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        metrics["loss"] += float(total_loss.item())
        metrics["policy_loss"] += float(policy_loss.item())
        metrics["value_loss"] += float(value_loss.item())
        metrics["entropy_loss"] += float(entropy_loss.item())
        num_updates += 1

    for key in metrics:
      metrics[key] /= max(num_updates, 1)
    return metrics

  def save(self, path: str) -> None:
    torch.save({
        "network": self.network.state_dict(),
        "optimizer": self.optimizer.state_dict(),
    }, path)

  def load(self, path: str) -> None:
    checkpoint = torch.load(path, map_location=self.device)
    self.network.load_state_dict(checkpoint["network"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])


def _merge_average_policy_states(
    target: List[Dict[str, np.ndarray]],
    source: List[Dict[str, np.ndarray]]) -> None:
  """Adds one serialized average-policy table into another."""
  for player, player_table in enumerate(source):
    for state_key, weights in player_table.items():
      if state_key not in target[player]:
        target[player][state_key] = np.asarray(weights, dtype=np.float64).copy()
      else:
        target[player][state_key] += np.asarray(weights, dtype=np.float64)


class ACHNetworkPolicy:
  """OpenSpiel policy wrapper around current ACH player networks."""

  def __init__(self, game: pyspiel.Game, agents: Sequence[ACHPokerAgent]):
    self.game = game
    self.agents = agents
    self.action_size = game.num_distinct_actions()

  def action_probabilities(
      self,
      state: pyspiel.State,
      player_id: Optional[int] = None,
  ) -> Dict[int, float]:
    if state.is_chance_node():
      return dict(state.chance_outcomes())
    player = state.current_player() if player_id is None else player_id
    legal_actions = state.legal_actions(player)
    legal_mask = legal_actions_mask(legal_actions, self.action_size)
    info_state = get_state_tensor(state, player, self.game)
    probs, _ = self.agents[player].policy_value(info_state, legal_mask)
    probs = _renormalize(probs, legal_mask)
    return {action: float(probs[action]) for action in legal_actions}


class AveragePolicyAccumulator:
  """Reach-weighted tabular policy accumulator used by ACH_poker evaluation."""

  def __init__(self,
               game: pyspiel.Game,
               action_size: int,
               fallback_agents: Optional[Sequence[ACHPokerAgent]] = None):
    self.game = game
    self.action_size = action_size
    self.fallback_agents = fallback_agents
    self.tables: List[Dict[str, np.ndarray]] = [
        {} for _ in range(game.num_players())
    ]

  def add(self,
          state: pyspiel.State,
          player: int,
          legal_actions: Sequence[int],
          policy_probs: np.ndarray,
          weight: float) -> None:
    state_key = get_state_key(state, player, self.game)
    if state_key not in self.tables[player]:
      self.tables[player][state_key] = np.zeros(
          self.action_size, dtype=np.float64)
    for action in legal_actions:
      self.tables[player][state_key][action] += (
          float(weight) * float(policy_probs[action]))

  def merge_state(self, state: List[Dict[str, np.ndarray]]) -> None:
    _merge_average_policy_states(self.tables, state)

  def state_dict(self) -> List[Dict[str, np.ndarray]]:
    return [
        {key: value.copy() for key, value in player_table.items()}
        for player_table in self.tables
    ]

  def load_state_dict(self, state: List[Dict[str, np.ndarray]]) -> None:
    self.tables = [
        {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in player_table.items()
        }
        for player_table in state
    ]

  def action_probabilities(
      self,
      state: pyspiel.State,
      player_id: Optional[int] = None,
  ) -> Dict[int, float]:
    if state.is_chance_node():
      return dict(state.chance_outcomes())
    player = state.current_player() if player_id is None else player_id
    legal_actions = state.legal_actions(player)
    legal_mask = legal_actions_mask(legal_actions, self.action_size)
    state_key = get_state_key(state, player, self.game)
    weights = self.tables[player].get(state_key)
    if weights is None or weights.sum() <= 0:
      if self.fallback_agents is not None:
        info_state = get_state_tensor(state, player, self.game)
        probs, _ = self.fallback_agents[player].policy_value(
            info_state, legal_mask)
      else:
        probs = uniform_policy_vector(legal_actions, self.action_size)
    else:
      probs = weights
    probs = _renormalize(probs, legal_mask)
    return {action: float(probs[action]) for action in legal_actions}


class ACHExternalSampler:
  """External-sampling trajectory generator modeled on ACH_poker's ACHSolver."""

  def __init__(
      self,
      game: pyspiel.Game,
      agents: Sequence[ACHPokerAgent],
      traversal_eta: float = 1.0,
      average_policy: Optional[AveragePolicyAccumulator] = None,
  ):
    self.game = game
    self.agents = agents
    self.traversal_eta = traversal_eta
    self.average_policy = average_policy
    self.num_players = game.num_players()
    self.action_size = game.num_distinct_actions()
    self.num_trajectories = 0
    self.num_states = 0

  def collect(
      self,
      num_traversals: int,
      start_step: int = 1,
      return_average_policy_state: bool = False,
  ) -> CollectionResult:
    all_samples = [[] for _ in range(self.num_players)]
    for _ in range(num_traversals):
      step = start_step
      for player in range(self.num_players):
        trajectory: List[TrajectoryStep] = []
        terminal_return = self._traverse(
            state=self.game.new_initial_state(),
            target_player=player,
            player_reach=1.0,
            opponent_reach=1.0,
            sampling_reach=1.0,
            step=step,
            trajectory=trajectory)
        all_samples[player].extend(
            self.agents[player].preprocess_trajectory(
                trajectory, terminal_return))
        self.num_trajectories += 1
        self.num_states += len(trajectory)

    average_policy_state = None
    if self.average_policy is not None and return_average_policy_state:
      average_policy_state = self.average_policy.state_dict()
    return CollectionResult(
        samples=all_samples,
        average_policy_state=average_policy_state,
        num_trajectories=self.num_trajectories,
        num_states=self.num_states)

  def _policy_value(self,
                    state: pyspiel.State,
                    player: int,
                    step: int) -> Tuple[np.ndarray, float]:
    legal_actions = state.legal_actions(player)
    legal_mask = legal_actions_mask(legal_actions, self.action_size)
    if step <= 1:
      return uniform_policy_vector(legal_actions, self.action_size), 0.0
    info_state = get_state_tensor(state, player, self.game)
    probs, value = self.agents[player].policy_value(info_state, legal_mask)
    return _renormalize(probs, legal_mask), value

  def _traverse(
      self,
      state: pyspiel.State,
      target_player: int,
      player_reach: float,
      opponent_reach: float,
      sampling_reach: float,
      step: int,
      trajectory: List[TrajectoryStep],
  ) -> float:
    if state.is_terminal():
      return float(state.returns()[target_player])

    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      action = int(np.random.choice(actions, p=probs))
      next_state = state.clone()
      next_state.apply_action(action)
      return self._traverse(
          next_state,
          target_player,
          player_reach,
          opponent_reach,
          sampling_reach,
          step,
          trajectory)

    if state.is_simultaneous_node():
      raise ValueError("ACH_poker-style traversal only supports turn-based games.")

    player = state.current_player()
    legal_actions = state.legal_actions(player)
    legal_mask = legal_actions_mask(legal_actions, self.action_size)
    current_probs, value = self._policy_value(state, player, step)
    use_current_for_target = self.traversal_eta != 0.0
    if player == target_player and not use_current_for_target:
      sampling_probs = uniform_policy_vector(legal_actions, self.action_size)
    else:
      sampling_probs = current_probs

    action = sample_action_from_policy(legal_actions, sampling_probs)
    current_prob = max(float(current_probs[action]), 1e-8)
    old_prob = max(float(sampling_probs[action]), 1e-8)

    if player == target_player:
      info_state = get_state_tensor(state, player, self.game)
      trajectory.append(
          TrajectoryStep(
              state=info_state.copy(),
              legal_mask=legal_mask.copy(),
              action=action,
              old_prob=old_prob,
              current_prob=current_prob,
              value=value))

      if self.average_policy is not None and use_current_for_target:
        self.average_policy.add(
            state=state,
            player=player,
            legal_actions=legal_actions,
            policy_probs=current_probs,
            weight=player_reach / max(sampling_reach, 1e-8))

      next_player_reach = player_reach * current_prob
      next_opponent_reach = opponent_reach
    else:
      next_player_reach = player_reach
      next_opponent_reach = opponent_reach * current_prob

    next_state = state.clone()
    next_state.apply_action(action)
    return self._traverse(
        state=next_state,
        target_player=target_player,
        player_reach=next_player_reach,
        opponent_reach=next_opponent_reach,
        sampling_reach=sampling_reach * old_prob,
        step=step,
        trajectory=trajectory)


def _collect_traversal_worker(worker_args) -> CollectionResult:
  """Collects ACHSolver-style traversal data inside one worker process."""
  (game_name, num_traversals, worker_seed, agent_configs, network_states,
   traversal_eta, collect_average_policy, start_step) = worker_args

  if worker_seed is not None:
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
  torch.set_num_threads(1)

  pyspiel_module = _ensure_pyspiel()
  game = pyspiel_module.load_game(game_name)
  worker_agents = []
  for config, state_dict in zip(agent_configs, network_states):
    agent = ACHPokerAgent(
        state_size=config["state_size"],
        action_size=config["action_size"],
        hidden_sizes=config["hidden_sizes"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ach_reward_scale=config["ach_reward_scale"],
        device="cpu")
    agent.network.load_state_dict(state_dict)
    worker_agents.append(agent)

  average_policy = None
  if collect_average_policy:
    average_policy = AveragePolicyAccumulator(
        game=game,
        action_size=game.num_distinct_actions(),
        fallback_agents=worker_agents)
  return ACHPokerAgent.collect_ach_traversal_samples(
      game=game,
      agents=worker_agents,
      num_traversals=num_traversals,
      traversal_eta=traversal_eta,
      average_policy=average_policy,
      start_step=start_step,
      return_average_policy_state=collect_average_policy)


def _collect_samples_worker(worker_args) -> List[List[ACHSample]]:
  """Collects self-play data inside one worker process."""
  game_name, num_episodes, worker_seed, agent_configs, network_states = worker_args

  if worker_seed is not None:
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
  torch.set_num_threads(1)

  pyspiel_module = _ensure_pyspiel()
  game = pyspiel_module.load_game(game_name)
  worker_agents = []
  for config, state_dict in zip(agent_configs, network_states):
    agent = ACHPokerAgent(
        state_size=config["state_size"],
        action_size=config["action_size"],
        hidden_sizes=config["hidden_sizes"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ach_reward_scale=config["ach_reward_scale"],
        device="cpu")
    agent.network.load_state_dict(state_dict)
    worker_agents.append(agent)

  return ACHPokerAgent.collect_training_samples(game, worker_agents, num_episodes)


def evaluate_vs_random(agent: ACHPokerAgent,
                       game_name: str,
                       num_games: int = 200) -> Tuple[float, float]:
  """Evaluates player 0 against a random opponent in a two-player game."""
  pyspiel_module = _ensure_pyspiel()
  game = pyspiel_module.load_game(game_name)
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
        info_state = get_state_tensor(state, player, game)
        mask = legal_actions_mask(legal_actions, num_actions)
        action, _, _ = agent.act(info_state, mask, training=False)
      else:
        action = int(np.random.choice(legal_actions))
      state.apply_action(action)

    returns = state.returns()
    total_return += returns[0]
    if returns[0] > returns[1]:
      wins += 1

  return wins / num_games, total_return / num_games


def train_ach_poker_style(
    game_name: str = "eren_yifang",
    num_iterations: int = 1000,
    cfr_batch_size: int = 256,
    eval_freq: int = 50,
    save_freq: int = 0,
    checkpoint_prefix: str = "ach_poker_style",
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
    seed: Optional[int] = None,
    num_workers: Optional[int] = None,
    mp_start_method: Optional[str] = None,
    traversal_eta: float = 1.0,
    use_external_sampling: bool = True,
    use_average_policy: bool = True,
    compute_nash_conv: bool = False,
    parallel_average_policy: bool = False,
    stream_worker_updates: bool = True,
    **agent_kwargs,
) -> List[ACHPokerAgent]:
  """Trains the ACH_poker-style ACH variant on an OpenSpiel game."""
  if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

  pyspiel_module = _ensure_pyspiel()
  game = pyspiel_module.load_game(game_name)
  game_type = game.get_type()
  if game_type.dynamics != pyspiel_module.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("This example only supports turn-based sequential games.")
  if use_external_sampling and game.num_players() != 2:
    raise ValueError(
        "The ACH_poker traversal is defined here for two-player games. "
        "Use --self_play_collection for the generic self-play fallback.")
  if (use_external_sampling and
      game_type.reward_model != pyspiel_module.GameType.RewardModel.TERMINAL):
    raise ValueError(
        "The ACH_poker traversal assumes terminal rewards. "
        "Use --self_play_collection for the generic self-play fallback.")

  state_shape = get_state_shape(game)
  state_size = int(np.prod(state_shape))
  action_size = game.num_distinct_actions()
  num_players = game.num_players()

  agents = [
      ACHPokerAgent(
          state_size=state_size,
          action_size=action_size,
          **agent_kwargs)
      for _ in range(num_players)
  ]
  resolved_workers = _resolve_num_workers(num_workers)
  use_parallel = resolved_workers > 1
  collect_average_policy = (
      use_external_sampling and use_average_policy and compute_nash_conv and (
          not use_parallel or parallel_average_policy))
  average_policy = None
  if collect_average_policy:
    average_policy = AveragePolicyAccumulator(
        game=game,
        action_size=action_size,
        fallback_agents=agents)

  checkpoint_dir_path = Path(checkpoint_dir)
  checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

  start_iteration = 0
  if resume:
    latest_checkpoint = _latest_checkpoint_path(
        checkpoint_dir=checkpoint_dir_path,
        checkpoint_prefix=checkpoint_prefix,
        game_name=game_name)
    if latest_checkpoint is not None:
      start_iteration = _load_agents_checkpoint(
          latest_checkpoint, agents, average_policy)
      print(
          f"Resumed from latest checkpoint: {latest_checkpoint} "
          f"(iteration {start_iteration})")

  pool = None
  last_iteration = start_iteration

  if use_parallel:
    start_method = _resolve_start_method(agents[0].device, mp_start_method)
    collection_name = "ACH traversal" if use_external_sampling else "self-play"
    print(
        f"Using {resolved_workers} worker processes for {collection_name} "
        "collection "
        f"({start_method} start method)")
    if use_external_sampling and use_average_policy and not parallel_average_policy:
      print(
          "Parallel average-policy table collection is disabled to avoid large "
          "worker IPC payloads. Use --parallel_average_policy to enable it.")
    ctx = mp.get_context(start_method)
    pool = ctx.Pool(processes=resolved_workers)

  try:
    for iteration in range(start_iteration + 1, num_iterations + 1):
      last_iteration = iteration
      iteration_seed = None
      if seed is not None:
        iteration_seed = seed + (iteration - 1) * resolved_workers

      metric_totals = [
          {
              "loss": 0.0,
              "policy_loss": 0.0,
              "value_loss": 0.0,
              "entropy_loss": 0.0,
          } for _ in range(num_players)
      ]
      metric_weights = [0 for _ in range(num_players)]

      def update_from_samples(sample_lists: Sequence[Sequence[ACHSample]]) -> None:
        for player, player_samples in enumerate(sample_lists):
          sample_count = len(player_samples)
          player_metrics = agents[player].update(player_samples)
          weight = max(sample_count, 1)
          metric_weights[player] += weight
          for key, value in player_metrics.items():
            metric_totals[player][key] += value * weight

      collection = CollectionResult(
          samples=[[] for _ in range(num_players)],
          average_policy_state=None,
          num_trajectories=0,
          num_states=0)

      if use_external_sampling:
        if use_parallel and stream_worker_updates:
          result_iter = ACHPokerAgent.iter_ach_traversal_results_parallel(
              game_name=game_name,
              agents=agents,
              num_traversals=cfr_batch_size,
              traversal_eta=traversal_eta,
              collect_average_policy=collect_average_policy,
              num_workers=resolved_workers,
              pool=pool,
              worker_seed=iteration_seed,
              mp_start_method=mp_start_method,
              start_step=iteration)
          for worker_collection in result_iter:
            if (average_policy is not None and
                worker_collection.average_policy_state is not None):
              average_policy.merge_state(worker_collection.average_policy_state)
            update_from_samples(worker_collection.samples)
            collection.num_trajectories += worker_collection.num_trajectories
            collection.num_states += worker_collection.num_states
        elif use_parallel:
          collection = ACHPokerAgent.collect_ach_traversal_samples_parallel(
              game_name=game_name,
              agents=agents,
              num_traversals=cfr_batch_size,
              traversal_eta=traversal_eta,
              collect_average_policy=collect_average_policy,
              num_workers=resolved_workers,
              pool=pool,
              worker_seed=iteration_seed,
              mp_start_method=mp_start_method,
              start_step=iteration)
          if (average_policy is not None and
              collection.average_policy_state is not None):
            average_policy.merge_state(collection.average_policy_state)
          update_from_samples(collection.samples)
        else:
          collection = ACHPokerAgent.collect_ach_traversal_samples(
              game=game,
              agents=agents,
              num_traversals=cfr_batch_size,
              traversal_eta=traversal_eta,
              average_policy=average_policy,
              start_step=iteration)
          update_from_samples(collection.samples)
      else:
        if use_parallel and stream_worker_updates:
          result_iter = ACHPokerAgent.iter_training_sample_results_parallel(
              game_name=game_name,
              agents=agents,
              num_episodes=cfr_batch_size,
              num_workers=resolved_workers,
              pool=pool,
              worker_seed=iteration_seed,
              mp_start_method=mp_start_method)
          for worker_samples in result_iter:
            update_from_samples(worker_samples)
            collection.num_states += sum(
                len(samples) for samples in worker_samples)
          collection.num_trajectories = cfr_batch_size
        elif use_parallel:
          all_samples = ACHPokerAgent.collect_training_samples_parallel(
              game_name=game_name,
              agents=agents,
              num_episodes=cfr_batch_size,
              num_workers=resolved_workers,
              pool=pool,
              worker_seed=iteration_seed,
              mp_start_method=mp_start_method)
          collection = CollectionResult(
              samples=all_samples,
              average_policy_state=None,
              num_trajectories=cfr_batch_size,
              num_states=sum(len(samples) for samples in all_samples))
          update_from_samples(all_samples)
        else:
          all_samples = ACHPokerAgent.collect_training_samples(
              game=game,
              agents=agents,
              num_episodes=cfr_batch_size)
          collection = CollectionResult(
              samples=all_samples,
              average_policy_state=None,
              num_trajectories=cfr_batch_size,
              num_states=sum(len(samples) for samples in all_samples))
          update_from_samples(all_samples)

      metrics = []
      for player in range(num_players):
        weight = max(metric_weights[player], 1)
        metrics.append({
            key: value / weight
            for key, value in metric_totals[player].items()
        })
      gc.collect()
      if any(str(agent.device).startswith("cuda") for agent in agents):
        torch.cuda.empty_cache()

      if eval_freq and iteration % eval_freq == 0:
        if num_players == 2:
          win_rate, avg_return = evaluate_vs_random(
              agents[0], game_name, num_games=1000)
          nash_conv_text = ""
          if compute_nash_conv:
            try:
              from open_spiel.python.algorithms import exploitability
              eval_policy = average_policy or ACHNetworkPolicy(game, agents)
              nash_conv = exploitability.nash_conv(game, eval_policy)
              nash_conv_text = f"  NashConv={nash_conv:.4f}"
            except (ValueError, RuntimeError, NotImplementedError) as exc:
              nash_conv_text = f"  NashConv=unavailable({exc})"
          player0_metrics = metrics[0]
          print(
              f"Iter {iteration:>5}/{num_iterations}  "
              f"Traversals={collection.num_trajectories:>4}  "
              f"States={collection.num_states:>4}  "
              f"WinRate={win_rate:.0%}  "
              f"AvgRet={avg_return:+.3f}  "
              f"Loss={player0_metrics['loss']:.4f}  "
              f"PiL={player0_metrics['policy_loss']:.4f}  "
              f"VL={player0_metrics['value_loss']:.4f}"
              f"{nash_conv_text}")
        else:
          player0_metrics = metrics[0]
          print(
              f"Iter {iteration:>5}/{num_iterations}  "
              f"Traversals={collection.num_trajectories:>4}  "
              f"States={collection.num_states:>4}  "
              f"Loss={player0_metrics['loss']:.4f}  "
              f"PiL={player0_metrics['policy_loss']:.4f}  "
              f"VL={player0_metrics['value_loss']:.4f}")

      if save_freq and iteration % save_freq == 0:
        checkpoint_path = checkpoint_dir_path / (
            f"{checkpoint_prefix}_{_safe_game_name(game_name)}_{iteration}.pth")
        _save_agents_checkpoint(
            checkpoint_path, agents, iteration, average_policy)
        print(f"Saved checkpoint: {checkpoint_path}")

    if save_freq and last_iteration > start_iteration and (
        last_iteration % save_freq != 0):
      checkpoint_path = checkpoint_dir_path / (
          f"{checkpoint_prefix}_{_safe_game_name(game_name)}_{last_iteration}.pth")
      _save_agents_checkpoint(
          checkpoint_path, agents, last_iteration, average_policy)
      print(f"Saved final checkpoint: {checkpoint_path}")
  finally:
    if pool is not None:
      if sys.exc_info()[0] is None:
        pool.close()
      else:
        pool.terminate()
      pool.join()

  return agents


def _parse_hidden_sizes(hidden_sizes: str) -> Tuple[int, ...]:
  values = tuple(
      int(value.strip()) for value in hidden_sizes.split(",")
      if value.strip())
  if not values:
    raise ValueError("--hidden_sizes must contain at least one integer.")
  return values


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Train the ACH_poker-style PyTorch ACH trainer on an OpenSpiel game."))
  parser.add_argument(
      "--game_name",
      "--game",
      default="eren_yifang",
      help="OpenSpiel game string. Defaults to eren_yifang.")
  parser.add_argument("--num_iterations", type=int, default=1000000)
  parser.add_argument(
      "--cfr_batch_size",
      type=int,
      default=128,
      help="ACH traversals per player and learner iteration.")
  parser.add_argument("--eval_freq", type=int, default=5000)
  parser.add_argument("--save_freq", type=int, default=20000)
  parser.add_argument("--checkpoint_dir", default="checkpoints")
  parser.add_argument("--checkpoint_prefix", default="ach_poker_style")
  parser.add_argument(
      "--no_resume",
      action="store_true",
      help="Start fresh instead of loading the newest matching checkpoint.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_workers", type=int, default=1)
  parser.add_argument("--mp_start_method", default=None)
  parser.add_argument(
      "--traversal_eta",
      type=float,
      default=1.0,
      help=(
          "Matches ACH_poker's nfsp_eta traversal switch: nonzero samples the "
          "target player from the current policy; zero samples it uniformly."))
  parser.add_argument(
      "--self_play_collection",
      action="store_true",
      help="Use the older generic self-play collector instead of ACH traversal.")
  parser.add_argument(
      "--no_average_policy",
      action="store_true",
      help=(
          "Do not accumulate the ACH_poker-style tabular average policy when "
          "NashConv is enabled."))
  parser.add_argument(
      "--compute_nash_conv",
      action="store_true",
      help="Compute NashConv for small two-player games during evaluation.")
  parser.add_argument(
      "--parallel_average_policy",
      action="store_true",
      help=(
          "Also return average-policy tables from worker processes. This can "
          "be very large; leave it off unless you need averaged-policy eval."))
  parser.add_argument(
      "--no_stream_worker_updates",
      action="store_true",
      help=(
          "Collect all worker samples before training. Uses more memory but "
          "keeps the older whole-iteration update behavior."))

  parser.add_argument("--hidden_sizes", default="256,256")
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--gamma", type=float, default=0.995)
  parser.add_argument("--gae_lambda", type=float, default=0.95)
  parser.add_argument("--ach_eta", type=float, default=1.0)
  parser.add_argument("--ach_alpha", type=float, default=2.0)
  parser.add_argument("--ach_beta", type=float, default=0.01)
  parser.add_argument("--ach_thres", type=float, default=2.0)
  parser.add_argument("--ach_epsilon", type=float, default=0.05)
  parser.add_argument("--ach_reward_scale", type=float, default=1.0)
  parser.add_argument("--update_epochs", "--train_steps", type=int, default=1)
  parser.add_argument("--num_minibatches", type=int, default=1)
  parser.add_argument(
      "--train_batch_size",
      type=int,
      default=1024,
      help="Maximum samples materialized as tensors in one optimizer step.")
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument(
      "--device",
      default="cpu",
      help="Torch device, e.g. cpu or cuda. Defaults to CUDA if available.")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  print(f"Training ACH_poker-style ACH on {args.game_name}")
  train_ach_poker_style(
      game_name=args.game_name,
      num_iterations=args.num_iterations,
      cfr_batch_size=args.cfr_batch_size,
      eval_freq=args.eval_freq,
      save_freq=args.save_freq,
      checkpoint_prefix=args.checkpoint_prefix,
      checkpoint_dir=args.checkpoint_dir,
      resume=not args.no_resume,
      seed=args.seed,
      num_workers=args.num_workers,
      mp_start_method=args.mp_start_method,
      traversal_eta=args.traversal_eta,
      use_external_sampling=not args.self_play_collection,
      use_average_policy=not args.no_average_policy,
      compute_nash_conv=args.compute_nash_conv,
      parallel_average_policy=args.parallel_average_policy,
      stream_worker_updates=not args.no_stream_worker_updates,
      hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
      learning_rate=args.learning_rate,
      weight_decay=args.weight_decay,
      gamma=args.gamma,
      gae_lambda=args.gae_lambda,
      ach_eta=args.ach_eta,
      ach_alpha=args.ach_alpha,
      ach_beta=args.ach_beta,
      ach_thres=args.ach_thres,
      ach_epsilon=args.ach_epsilon,
      ach_reward_scale=args.ach_reward_scale,
      update_epochs=args.update_epochs,
      num_minibatches=args.num_minibatches,
      train_batch_size=args.train_batch_size,
      max_grad_norm=args.max_grad_norm,
      device=args.device)


if __name__ == "__main__":
  main()
