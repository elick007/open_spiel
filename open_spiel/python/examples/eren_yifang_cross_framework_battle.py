"""Cross-framework battle script for Eren Yifang.

Runs head-to-head matches between:
- an OpenSpiel ACH agent trained by `ach_poker_pytorch.py`
- an RLCard DMC agent trained on `mahjong_two_by_one`
- or a random legal-action player

The script adapts OpenSpiel `eren_yifang` states into the RLCard observation and
action conventions used by the DMC model, and can also benchmark either model
against a random player.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib
from pathlib import Path
import sys
import types
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyspiel
import torch


THIS_DIR = Path(__file__).resolve().parent


def _resolve_local_path(path_str: str) -> Path:
  """Resolves Windows-style paths when the script runs under WSL/Linux."""
  candidates: List[Path] = [Path(path_str)]
  if len(path_str) >= 3 and path_str[1:3] == ":\\":
    drive = path_str[0].lower()
    remainder = path_str[3:].replace("\\", "/")
    candidates.append(Path("/mnt") / drive / remainder)
  elif "\\" in path_str:
    candidates.append(Path(path_str.replace("\\", "/")))

  seen = set()
  for candidate in candidates:
    normalized = str(candidate)
    if normalized in seen:
      continue
    seen.add(normalized)
    if candidate.exists():
      return candidate
  return candidates[0]


RLCard_ROOT = _resolve_local_path(r"D:\Projects\PYProjects\rlcard")
DMC_MODEL_DEFAULT = _resolve_local_path(
    r"D:\Projects\PYProjects\rlcard\auto_play_ai\train\1_2850553600.pth")

if str(THIS_DIR) not in sys.path:
  sys.path.append(str(THIS_DIR))
if str(RLCard_ROOT) not in sys.path:
  sys.path.append(str(RLCard_ROOT))

from ach_poker_pytorch import (  # pylint: disable=g-import-not-at-top
    ACHPokerAgent,
    _load_agents_checkpoint,
    _latest_checkpoint_path,
    get_state_shape,
    get_state_tensor,
    legal_actions_mask,
)


AGENT_TYPES = ("ach", "dmc", "random")

# OpenSpiel action ids from `open_spiel/games/eren_yifang/eren_yifang.h`.
OS_DRAW_ACTION = 0
OS_HU_ACTION = 1
OS_DISCARD_ACTION_BASE = 3
OS_DISCARD_ACTION_END = 11
OS_PONG_ACTION_BASE = 12
OS_PONG_ACTION_END = 20
OS_GONG_ACTION_BASE = 21
OS_GONG_ACTION_END = 29
OS_CONCEALED_GONG_ACTION_BASE = 30
OS_CONCEALED_GONG_ACTION_END = 38
OS_ADD_GONG_ACTION_BASE = 40
OS_ADD_GONG_ACTION_END = 48

# RLCard action ids from `rlcard/games/mahjong_two_by_one/utils.py`.
RL_DRAW_ACTION = 9
RL_PONG_ACTION = 10
RL_GONG_ACTION = 11
RL_STAND_ACTION = 12
RL_HU_ACTION = 13
RL_ZIMO_ACTION = 14

# Observation layout from `eren_yifang.h`.
OBS_SELF_HAND = 0
OBS_SELF_EXPOSED = 1
OBS_SELF_HIDDEN = 2
OBS_SELF_DISCARD_START = 3
OBS_DISCARD_PLANES = 13
OBS_OPP_EXPOSED = OBS_SELF_DISCARD_START + OBS_DISCARD_PLANES
OBS_OPP_HIDDEN = OBS_OPP_EXPOSED + 1
OBS_OPP_DISCARD_START = OBS_OPP_HIDDEN + 1


def _action_in_range(action: int, start: int, end: int) -> bool:
  return start <= action <= end


def _tile_counts_from_plane(plane: np.ndarray) -> np.ndarray:
  return plane.sum(axis=0).astype(np.int8)


def _tile_counts_from_discard_planes(obs: np.ndarray, start: int) -> np.ndarray:
  return obs[start:start + OBS_DISCARD_PLANES].sum(axis=(0, 1)).astype(np.int8)


def _counts_to_rlcard_plane(counts: np.ndarray) -> np.ndarray:
  plane = np.zeros((9, 4), dtype=np.int8)
  for tile, count in enumerate(np.asarray(counts, dtype=np.int16)):
    plane[tile, :max(0, min(int(count), 4))] = 1
  return plane


def _ensure_git_dependency_stub() -> None:
  """Avoids requiring GitPython just to unpickle an RLCard agent."""
  try:
    importlib.import_module("git")
    return
  except ModuleNotFoundError:
    pass

  git_module = types.ModuleType("git")

  class InvalidGitRepositoryError(Exception):
    pass

  class Repo:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
      del args, kwargs
      raise InvalidGitRepositoryError()

  git_module.InvalidGitRepositoryError = InvalidGitRepositoryError
  git_module.Repo = Repo
  sys.modules["git"] = git_module


def _load_rlcard_pickled_agent(model_path: str, device: torch.device):
  _ensure_git_dependency_stub()
  resolved_path = _resolve_local_path(model_path)
  try:
    agent = torch.load(
        str(resolved_path), map_location=device, weights_only=False)
  except TypeError:
    agent = torch.load(str(resolved_path), map_location=device)

  if not hasattr(agent, "eval_step"):
    raise TypeError(
        f"Checkpoint {resolved_path} did not load an RLCard agent object.")
  if hasattr(agent, "eval"):
    agent.eval()
  if hasattr(agent, "set_device"):
    agent.set_device(device)
  return agent


def _infer_dmc_model_paths(model_path: str, num_players: int) -> List[Path]:
  """Infers per-seat RLCard DMC checkpoints from one checkpoint path."""
  resolved_path = _resolve_local_path(model_path)
  model_paths = [resolved_path for _ in range(num_players)]

  stem_parts = resolved_path.stem.split("_", maxsplit=1)
  if len(stem_parts) != 2 or not stem_parts[0].isdigit():
    return model_paths

  seat = int(stem_parts[0])
  if seat < 0 or seat >= num_players:
    return model_paths

  suffix = resolved_path.suffix
  remainder = stem_parts[1]
  model_paths[seat] = resolved_path
  for other_seat in range(num_players):
    if other_seat == seat:
      continue
    sibling = resolved_path.with_name(f"{other_seat}_{remainder}{suffix}")
    if sibling.exists():
      model_paths[other_seat] = sibling
  return model_paths


def _resolve_dmc_model_paths(args: argparse.Namespace,
                             num_players: int) -> List[Path]:
  model_paths = _infer_dmc_model_paths(args.dmc_model, num_players)
  seat_overrides = [args.dmc_model_p0, args.dmc_model_p1]
  for seat, override in enumerate(seat_overrides[:num_players]):
    if override is not None:
      model_paths[seat] = _resolve_local_path(override)
  return model_paths


def _rlcard_raw_action(rl_action: int) -> int | str:
  if 0 <= rl_action <= 8:
    return rl_action + 1
  if rl_action == RL_DRAW_ACTION:
    return "draw"
  if rl_action == RL_PONG_ACTION:
    return "pong"
  if rl_action == RL_GONG_ACTION:
    return "gong"
  if rl_action == RL_STAND_ACTION:
    return "stand"
  if rl_action == RL_HU_ACTION:
    return "hu"
  if rl_action == RL_ZIMO_ACTION:
    return "zimo"
  raise ValueError(f"Unsupported RLCard action id: {rl_action}")


class RandomActionAgent:
  """Uniformly samples from OpenSpiel legal actions."""

  def __init__(self, seed: int | None = None):
    self.label = "Random"
    self.rng = np.random.default_rng(seed)

  def step(self,
           state: pyspiel.State,
           player: int,
           game: pyspiel.Game | None = None) -> int:
    del game
    legal_actions = state.legal_actions(player)
    return int(self.rng.choice(legal_actions))


class ACHModelWrapper:
  """Adapts ACH agents saved by `ach_poker_pytorch.py` to match play."""

  def __init__(self,
               agents: Sequence[ACHPokerAgent],
               model_paths: Sequence[str] | None = None):
    self.label = "ACH"
    self.agents = list(agents)
    self.model_paths = list(model_paths or [])

  def step(self,
           state: pyspiel.State,
           player: int,
           game: pyspiel.Game) -> int:
    legal_actions = state.legal_actions(player)
    info_state = get_state_tensor(state, player, game)
    mask = legal_actions_mask(legal_actions, game.num_distinct_actions())
    action, _, _ = self.agents[player].act(info_state, mask, training=False)
    if action not in legal_actions:
      action = int(legal_actions[np.argmax(mask[legal_actions])])
    return action


class RLCardDMCWrapper:
  """Adapts an RLCard DMC model to OpenSpiel Eren Yifang."""

  def __init__(self,
               model_paths: Sequence[str | Path],
               device: str | None = None):
    self.label = "DMC"
    self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.model_paths = [str(_resolve_local_path(str(path))) for path in model_paths]
    self.agents = [
        _load_rlcard_pickled_agent(model_path, self.device)
        for model_path in self.model_paths
    ]

  def _open_spiel_obs(self, state: pyspiel.State, player: int) -> np.ndarray:
    obs = np.asarray(state.observation_tensor(player), dtype=np.int8).reshape(
        state.get_game().observation_tensor_shape())
    if obs.shape != (31, 4, 9):
      raise ValueError(
          f"Expected OpenSpiel observation shape (31, 4, 9), got {obs.shape}.")
    return obs

  def _build_obs(self, state: pyspiel.State, player: int) -> np.ndarray:
    """Converts OpenSpiel's relative 31x4x9 tensor to RLCard's 4x9x4 tensor."""
    obs = self._open_spiel_obs(state, player)

    hand_counts = _tile_counts_from_plane(obs[OBS_SELF_HAND])
    self_exposed_counts = _tile_counts_from_plane(obs[OBS_SELF_EXPOSED])
    self_hidden_counts = _tile_counts_from_plane(obs[OBS_SELF_HIDDEN])
    opp_exposed_counts = _tile_counts_from_plane(obs[OBS_OPP_EXPOSED])
    opp_hidden_counts = _tile_counts_from_plane(obs[OBS_OPP_HIDDEN])

    self_discard_counts = _tile_counts_from_discard_planes(
        obs, OBS_SELF_DISCARD_START).astype(np.int16)
    opp_discard_counts = _tile_counts_from_discard_planes(
        obs, OBS_OPP_DISCARD_START).astype(np.int16)

    # RLCard `table` stores only discards still on the table. Claimed discards
    # disappear once converted into exposed melds, so we subtract one tile per
    # exposed meld.
    table_counts = self_discard_counts + opp_discard_counts
    table_counts -= (self_exposed_counts > 0).astype(np.int16)
    table_counts -= (opp_exposed_counts > 0).astype(np.int16)
    table_counts = np.clip(table_counts, 0, 4).astype(np.int8)

    self_pile_counts = (self_exposed_counts + self_hidden_counts).astype(np.int8)
    opp_pile_counts = (opp_exposed_counts + opp_hidden_counts).astype(np.int8)

    pile0_counts = self_pile_counts if player == 0 else opp_pile_counts
    pile1_counts = opp_pile_counts if player == 0 else self_pile_counts

    return np.stack([
        _counts_to_rlcard_plane(hand_counts),
        _counts_to_rlcard_plane(table_counts),
        _counts_to_rlcard_plane(pile0_counts),
        _counts_to_rlcard_plane(pile1_counts),
    ],
                    axis=0).astype(np.int8)

  def _build_legal_action_mapping(
      self, legal_actions: Sequence[int]) -> "OrderedDict[int, int]":
    """Maps RLCard's 15-way action ids onto the current OpenSpiel legal set."""
    mapping: "OrderedDict[int, int]" = OrderedDict()
    legal_set = {int(action) for action in legal_actions}

    discard_actions = [
        int(action) for action in legal_actions
        if _action_in_range(int(action), OS_DISCARD_ACTION_BASE,
                            OS_DISCARD_ACTION_END)
    ]
    pong_actions = [
        int(action) for action in legal_actions
        if _action_in_range(int(action), OS_PONG_ACTION_BASE, OS_PONG_ACTION_END)
    ]
    direct_gong_actions = [
        int(action) for action in legal_actions
        if _action_in_range(int(action), OS_GONG_ACTION_BASE, OS_GONG_ACTION_END)
    ]
    concealed_gong_actions = [
        int(action) for action in legal_actions
        if _action_in_range(
            int(action), OS_CONCEALED_GONG_ACTION_BASE,
            OS_CONCEALED_GONG_ACTION_END)
    ]
    add_gong_actions = [
        int(action) for action in legal_actions
        if _action_in_range(int(action), OS_ADD_GONG_ACTION_BASE,
                            OS_ADD_GONG_ACTION_END)
    ]

    # Actor turn in RLCard: `zimo`, `gong`, then discard tile ids.
    if discard_actions:
      if OS_HU_ACTION in legal_set:
        mapping[RL_ZIMO_ACTION] = OS_HU_ACTION
      preferred_gong = None
      if add_gong_actions:
        preferred_gong = add_gong_actions[0]
      elif concealed_gong_actions:
        preferred_gong = concealed_gong_actions[0]
      if preferred_gong is not None:
        mapping[RL_GONG_ACTION] = preferred_gong
      for action in discard_actions:
        mapping[action - OS_DISCARD_ACTION_BASE] = action
      return mapping

    # Discard response in RLCard: `hu`, `gong`, `pong`, `draw`.
    if (OS_DRAW_ACTION in legal_set or pong_actions or direct_gong_actions):
      if OS_HU_ACTION in legal_set:
        mapping[RL_HU_ACTION] = OS_HU_ACTION
      if direct_gong_actions:
        mapping[RL_GONG_ACTION] = direct_gong_actions[0]
      if pong_actions:
        mapping[RL_PONG_ACTION] = pong_actions[0]
      if OS_DRAW_ACTION in legal_set:
        mapping[RL_DRAW_ACTION] = OS_DRAW_ACTION
      return mapping

    # Rob-kong response in RLCard exposes only `hu`.
    if OS_HU_ACTION in legal_set:
      mapping[RL_HU_ACTION] = OS_HU_ACTION
      return mapping

    if OS_DRAW_ACTION in legal_set:
      mapping[RL_DRAW_ACTION] = OS_DRAW_ACTION
      return mapping

    raise ValueError(
        f"Could not convert OpenSpiel legal actions {list(legal_actions)} "
        "into RLCard legal actions.")

  def _build_rl_state(
      self, state: pyspiel.State, player: int
  ) -> Tuple[Dict[str, object], "OrderedDict[int, int]"]:
    legal_actions = state.legal_actions(player)
    action_mapping = self._build_legal_action_mapping(legal_actions)
    rl_state = {
        "obs": self._build_obs(state, player),
        "legal_actions": OrderedDict(
            (rl_action, None) for rl_action in action_mapping.keys()),
        "raw_legal_actions": [
            _rlcard_raw_action(rl_action) for rl_action in action_mapping.keys()
        ],
    }
    return rl_state, action_mapping

  def step(self,
           state: pyspiel.State,
           player: int,
           game: pyspiel.Game | None = None) -> int:
    del game
    rl_state, action_mapping = self._build_rl_state(state, player)
    action, _ = self.agents[player].eval_step(rl_state)
    rl_action = int(action)
    if rl_action in action_mapping:
      return action_mapping[rl_action]
    if len(action_mapping) == 1:
      return next(iter(action_mapping.values()))
    raise ValueError(
        f"RLCard agent returned illegal action {rl_action}; expected one of "
        f"{list(action_mapping.keys())}.")


def load_ach_agents(game_name: str,
                    checkpoint_path: str | None) -> Tuple[List[ACHPokerAgent],
                                                          List[str]]:
  game = pyspiel.load_game(game_name)
  state_size = int(np.prod(get_state_shape(game)))
  action_size = game.num_distinct_actions()
  agents = [
      ACHPokerAgent(state_size=state_size, action_size=action_size)
      for _ in range(game.num_players())
  ]

  path = _resolve_local_path(checkpoint_path) if checkpoint_path else _latest_checkpoint_path(
      checkpoint_dir=Path("checkpoints"),
      checkpoint_prefix="ach_poker_style",
      game_name=game_name)
  if path is None:
    raise FileNotFoundError("No ACH poker checkpoint found.")

  checkpoint = torch.load(str(path), map_location=agents[0].device)
  if "agents" not in checkpoint:
    raise ValueError(
        "ACH checkpoint must be an all-player bundle containing `agents`.")

  _load_agents_checkpoint(path, agents)
  model_paths = [str(path)] * game.num_players()
  return agents, model_paths


def _resolve_seat_agent_types(args: argparse.Namespace) -> List[str]:
  if args.player0 is not None or args.player1 is not None:
    player0 = args.player0 or "ach"
    player1 = args.player1 or "dmc"
    return [player0, player1]
  return ["ach", "dmc"] if args.ach_player == 0 else ["dmc", "ach"]


def _build_agent_pool(args: argparse.Namespace,
                      game_name: str,
                      num_players: int) -> Dict[str, object]:
  seat_agent_types = _resolve_seat_agent_types(args)
  required_types = set(seat_agent_types)
  agent_pool: Dict[str, object] = {}

  if "ach" in required_types:
    ach_agents, ach_model_paths = load_ach_agents(game_name,
                                                  args.ach_checkpoint)
    agent_pool["ach"] = ACHModelWrapper(ach_agents, ach_model_paths)
  if "dmc" in required_types:
    agent_pool["dmc"] = RLCardDMCWrapper(
        _resolve_dmc_model_paths(args, num_players), device=args.device)
  if "random" in required_types:
    agent_pool["random"] = RandomActionAgent(seed=args.seed)
  return agent_pool


def play_match(game: pyspiel.Game,
               seat_agents: Sequence[object]) -> Tuple[List[float], List[str]]:
  state = game.new_initial_state()
  trace: List[str] = []

  while not state.is_terminal():
    if state.is_chance_node():
      actions, probs = zip(*state.chance_outcomes())
      action = int(np.random.choice(actions, p=probs))
      trace.append(f"chance -> {state.action_to_string(state.current_player(), action)}")
      state.apply_action(action)
      continue

    player = state.current_player()
    agent = seat_agents[player]
    action = agent.step(state, player, game)
    actor = agent.label

    trace.append(f"P{player} {actor}: {state.action_to_string(player, action)}")
    state.apply_action(action)

  return state.returns(), trace


def run_battle(args: argparse.Namespace) -> None:
  if args.seed is not None:
    np.random.seed(args.seed)

  game = pyspiel.load_game(args.game)
  base_agent_types = _resolve_seat_agent_types(args)
  agent_pool = _build_agent_pool(args, args.game, game.num_players())

  if "dmc" in agent_pool:
    dmc_agent = agent_pool["dmc"]
    print(f"DMC checkpoints by seat: {dmc_agent.model_paths}")
    if len(set(dmc_agent.model_paths)) == 1 and (
        args.alternate_seats or "dmc" in base_agent_types):
      print(
          "Warning: DMC is using the same checkpoint for multiple seats. "
          "RLCard DMC models are usually position-specific.")
  if "ach" in agent_pool:
    ach_agent = agent_pool["ach"]
    print(f"ACH checkpoint bundle: {ach_agent.model_paths[0]}")

  totals = np.zeros(game.num_players(), dtype=np.float64)
  wins = np.zeros(game.num_players(), dtype=np.int32)
  draws = 0

  for game_index in range(args.num_games):
    seat_agent_types = list(base_agent_types)
    if args.alternate_seats and game_index % 2 == 1:
      seat_agent_types.reverse()
    seat_agents = [agent_pool[agent_type] for agent_type in seat_agent_types]

    returns, trace = play_match(game, seat_agents)
    totals += np.asarray(returns)

    if returns[0] == returns[1]:
      draws += 1
    else:
      wins[int(np.argmax(returns))] += 1

    if args.verbose:
      print(
          f"=== Game {game_index + 1} | "
          f"P0={seat_agents[0].label} P1={seat_agents[1].label} ===")
      for item in trace:
        print(item)
      print(f"Returns: {returns}\n")

  print("Battle finished")
  print(f"Games: {args.num_games}")
  print(f"Base matchup: P0={base_agent_types[0]} P1={base_agent_types[1]}")
  print(f"Seat mode: {'alternate' if args.alternate_seats else 'fixed'}")
  print(f"Wins P0/P1: {wins.tolist()}")
  print(f"Draws: {draws}")
  print(f"Average returns: {(totals / args.num_games).tolist()}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--game", default="eren_yifang")
  parser.add_argument(
      "--player0",
      choices=AGENT_TYPES,
      default=None,
      help="Agent type seated at P0. Defaults to the legacy ACH/DMC setup.")
  parser.add_argument(
      "--player1",
      choices=AGENT_TYPES,
      default=None,
      help="Agent type seated at P1. Defaults to the legacy ACH/DMC setup.")
  parser.add_argument("--ach_checkpoint", default=None)
  parser.add_argument(
      "--dmc_model",
      default=str(DMC_MODEL_DEFAULT))
  parser.add_argument(
      "--dmc_model_p0",
      default=None,
      help="Optional RLCard DMC checkpoint to use when DMC sits at P0.")
  parser.add_argument(
      "--dmc_model_p1",
      default=None,
      help="Optional RLCard DMC checkpoint to use when DMC sits at P1.")
  parser.add_argument("--num_games", type=int, default=100)
  parser.add_argument("--ach_player", type=int, default=0, choices=[0, 1])
  parser.add_argument("--alternate_seats", action="store_true")
  parser.add_argument("--device", default=None)
  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="Optional RNG seed used by chance nodes and the random player.")
  parser.add_argument("--verbose", action="store_true")
  return parser


if __name__ == "__main__":
  run_battle(build_parser().parse_args())
