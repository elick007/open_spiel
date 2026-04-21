"""Practical ACH example for OpenSpiel.

This example follows the paper's practical Actor-Critic Hedge (ACH)
implementation more closely than ``ach_pytorch.py``:

- one shared actor-value network is used for all players;
- trajectories are sampled from the current self-play policy;
- only sampled state-action pairs are optimized;
- advantages are estimated with GAE(lambda);
- the policy loss follows the clipped ACH objective from Appendix E.

It is still a simplified trainer:

- no asynchronous actors or learner;
- one shared optimizer update per self-play batch;
- optional multiprocessing for self-play collection on Linux;
- only turn-based sequential games are supported.

Paper:
https://openreview.net/pdf?id=DTXZqTNV5nW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
import pyspiel
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


INVALID_ACTION_LOGIT = -1e9


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


def _latest_checkpoint_path(checkpoint_dir: Path,
                            checkpoint_prefix: str,
                            game_name: str) -> Optional[Path]:
  """Returns the newest matching checkpoint path in the checkpoint dir."""
  if not checkpoint_dir.exists():
    return None

  pattern = f"{checkpoint_prefix}_{game_name}_*.pth"
  candidates = [
      path for path in checkpoint_dir.glob(pattern)
      if path.is_file()
  ]
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_payload(agent: "PracticalACHAgent",
                        iteration: int) -> Dict[str, object]:
  """Builds a serializable checkpoint payload."""
  return {
      "iteration": iteration,
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
      state_size=agent_config["state_size"],
      action_size=agent_config["action_size"],
      hidden_sizes=agent_config["hidden_sizes"],
      gamma=agent_config["gamma"],
      gae_lambda=agent_config["gae_lambda"],
      device="cpu")
  worker_agent.network.load_state_dict(network_state)
  return worker_agent.collect_batch(game, min_samples=min_samples)


class SharedActorCritic(nn.Module):
  """Shared torso with separate policy and value heads."""

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


class PracticalACHAgent:
  """A simplified practical ACH learner based on Appendix E."""

  def __init__(
      self,
      state_size: int,
      action_size: int,
      hidden_sizes: Sequence[int] = (256, 256),
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
    self.state_size = state_size
    self.action_size = action_size
    self.hidden_sizes = tuple(hidden_sizes)
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
        input_size=state_size,
        num_actions=action_size,
        hidden_sizes=self.hidden_sizes).to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(),
                                lr=learning_rate,
                                eps=1e-5)

  def _to_tensor(self, array: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=self.device)

  def _policy_and_value(self,
                        states: torch.Tensor,
                        legal_masks: torch.Tensor
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
        info_state = get_state_tensor(state, player, game)
        legal_actions = state.legal_actions(player)
        mask = legal_actions_mask(legal_actions, self.action_size)
        action, old_prob, value = self.act(info_state, mask, training=True)

        if pending[player] is not None:
          pending[player].next_value = value
          pending[player].done = False
          player_steps[player].append(pending[player])

        pending[player] = PendingTransition(
            state=info_state.copy(),
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

    worker_counts = _split_sample_count(min_samples, resolved_workers)
    agent_config = {
        "state_size": self.state_size,
        "action_size": self.action_size,
        "hidden_sizes": self.hidden_sizes,
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

    states = self._to_tensor(np.asarray([t.state for t in transitions]))
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
        mb_masks = legal_masks[minibatch_indices]
        mb_actions = actions[minibatch_indices]
        mb_old_probs = old_probs[minibatch_indices].clamp_min(1e-8)
        mb_advantages = advantages[minibatch_indices]
        mb_returns = returns[minibatch_indices]

        if self.normalize_advantages and len(mb_advantages) > 1:
          mb_advantages = ((mb_advantages - mb_advantages.mean()) /
                           (mb_advantages.std() + 1e-8))

        logits, policy_probs, values = self._policy_and_value(mb_states, mb_masks)
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
    self.network.load_state_dict(checkpoint["network"])
    self.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", 0))


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


def train_practical_ach(
    game_name: str = "leduc_poker",
    num_iterations: int = 1000,
    batch_size: int = 512,
    eval_freq: int = 50,
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
  state_size = int(np.prod(state_shape))
  action_size = game.num_distinct_actions()

  agent = PracticalACHAgent(
      state_size=state_size,
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

  resolved_workers = _resolve_num_workers(num_workers)
  use_parallel = resolved_workers > 1
  pool = None

  if use_parallel:
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

      if eval_freq and iteration % eval_freq == 0:
        if game.num_players() == 2:
          win_rate, avg_return = evaluate_vs_random(agent, game_name, num_games=200)
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
              f"{checkpoint_prefix}_{game_name}_{iteration}.pth")
          agent.save(str(checkpoint_path), iteration=iteration)
          last_save_time = now
          print(f"Saved hourly checkpoint: {checkpoint_path}")
  finally:
    if pool is not None:
      pool.close()
      pool.join()

  return agent

def play_interactive_game(game_name='eren_yifang'):
    game = pyspiel.load_game(game_name)
    game_type = game.get_type()
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("This example only supports turn-based sequential games.")

    state_shape = get_state_shape(game)
    state_size = int(np.prod(state_shape))
    action_size = game.num_distinct_actions()

    agent = PracticalACHAgent(
        state_size=state_size,
        action_size=action_size)
    latest_checkpoint = _latest_checkpoint_path(
      checkpoint_dir=Path("checkpoints"),
      checkpoint_prefix="ach_practical",
      game_name=game_name)
    if latest_checkpoint is None:
      raise FileNotFoundError("No checkpoint found in checkpoints directory.")
    loaded_iteration = agent.load(str(latest_checkpoint))
    print(f"Loaded checkpoint iteration: {loaded_iteration}")
    """Play an interactive game against the trained agent."""
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()

    print(f"\n{'='*50}")
    print(f"Playing {game_name}")
    print(f"{'='*50}\n")

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
            info_state = get_state_tensor(state, current_player, game)
            mask = legal_actions_mask(legal_actions, game.num_distinct_actions())
            action, _, _ = agent.act(info_state, mask, training=False)
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
                    print("Please enter a number!")

        state.apply_action(action)

    print(f"\nGame over!")
    print(f"Returns: {state.returns()}")
    if state.returns()[1] > state.returns()[0]:
        print("You win! 🎉")
    elif state.returns()[0] > state.returns()[1]:
        print("Agent wins! 🤖")
    else:
        print("It's a draw!")


if __name__ == "__main__":
  print("Training practical ACH on Eren Yifang...")
  train_practical_ach(
      game_name="eren_yifang",
      num_iterations=1000000,
      batch_size=1024,
      eval_freq=1000,
      save_interval_seconds=3600.0,
      learning_rate=1e-4,
      gamma=0.99,
      gae_lambda=0.95,
      clip_ratio=0.05,
      logit_threshold=1.5,
      entropy_coef=0.02,
      value_coef=0.5,
      eta=1.0,
      max_grad_norm=0.5,
      update_epochs=2,
      num_minibatches=4,
      normalize_advantages=True,
      hidden_sizes=(256, 256),
      seed=0,
      num_workers=8,
  )
