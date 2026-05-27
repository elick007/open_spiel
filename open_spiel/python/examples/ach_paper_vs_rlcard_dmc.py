"""Battle an OpenSpiel ACH agent against an RLCard DMC agent on Eren Yifang.

The OpenSpiel side uses checkpoints produced by `ach_paper_pytorch.py` for
`eren_yifang`. The RLCard side adapts agents trained by
`~/RLCard-mahjong/examples/run_dmc.py` on `mahjong_two_by_one`.

`eren_yifang` deliberately uses the same 15 action ids as RLCard's
`mahjong_two_by_one`:

  0..8: discard tile 1..9, 9: draw/pass, 10: pong, 11: gong,
  12: stand, 13: hu, 14: zimo.

OpenSpiel observations are `[6, 9, 4]`: the first four channels match
RLCard's `[4, 9, 4]` DMC observation, followed by two seat-marker channels.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import importlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import types
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyspiel
import torch


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ACH_CHECKPOINT_DIR = THIS_DIR / "checkpoints"
DEFAULT_RLCARD_ROOT = Path("~/RLCard-mahjong").expanduser()
DEFAULT_DMC_MODEL_P0 = "~/RLCard-mahjong/auto_play_ai/train/0_2835094400.pth"
DEFAULT_DMC_MODEL_P1 = "~/RLCard-mahjong/auto_play_ai/train/1_2835094400.pth"

RLCARD_OBSERVATION_SHAPE = (4, 9, 4)
OPEN_SPIEL_OBSERVATION_SHAPE = (6, 9, 4)
NUM_RLCARD_ACTIONS = 15

ACTION_NAMES = {
    9: "draw",
    10: "pong",
    11: "gong",
    12: "stand",
    13: "hu",
    14: "zimo",
}


if str(THIS_DIR) not in sys.path:
  sys.path.append(str(THIS_DIR))

from ach_paper_pytorch import (  # pylint: disable=g-import-not-at-top
    PracticalACHAgent,
    _latest_checkpoint_path,
    get_auxiliary_features,
    get_state_tensor,
    legal_actions_mask,
)


def _expand_path(path: str | os.PathLike[str]) -> Path:
  return Path(path).expanduser().resolve()


def _safe_torch_load(path: Path, map_location):
  """Loads both tensor checkpoints and pickled RLCard agent objects."""
  try:
    return torch.load(str(path), map_location=map_location, weights_only=False)
  except TypeError:
    return torch.load(str(path), map_location=map_location)


def _ensure_git_dependency_stub() -> None:
  """Lets torch unpickle RLCard agents even when GitPython is not installed."""
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


def _action_to_rlcard_raw(action: int) -> int | str:
  if 0 <= action <= 8:
    return action + 1
  return ACTION_NAMES.get(action, action)


def _dmc_device_arg(device: torch.device) -> str:
  if device.type != "cuda":
    return "cpu"
  return str(0 if device.index is None else device.index)


def _set_rlcard_agent_device(agent, device: torch.device) -> None:
  if hasattr(agent, "net") and hasattr(agent.net, "to"):
    agent.net.to(device)
  if hasattr(agent, "set_device"):
    agent.set_device(device)
  if hasattr(agent, "eval"):
    agent.eval()


def _add_rlcard_to_path(rlcard_root: Path) -> None:
  root_str = str(rlcard_root)
  if root_str not in sys.path:
    sys.path.insert(0, root_str)


def _ensure_package_stub(name: str, package_path: Path) -> None:
  module = sys.modules.get(name)
  if module is not None:
    return
  module = types.ModuleType(name)
  module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
  sys.modules[name] = module


def _load_rlcard_dmc_model_module(rlcard_root: Path):
  """Loads RLCard's DMC model module without importing `rlcard.agents`."""
  module_name = "rlcard.agents.dmc_agent.model"
  if module_name in sys.modules:
    return sys.modules[module_name]

  rlcard_package = rlcard_root / "rlcard"
  dmc_package = rlcard_package / "agents" / "dmc_agent"
  model_py = dmc_package / "model.py"
  if not model_py.exists():
    raise FileNotFoundError(f"Could not find RLCard DMC model file: {model_py}")

  _add_rlcard_to_path(rlcard_root)
  _ensure_package_stub("rlcard", rlcard_package)
  _ensure_package_stub("rlcard.agents", rlcard_package / "agents")
  _ensure_package_stub("rlcard.agents.dmc_agent", dmc_package)

  spec = importlib.util.spec_from_file_location(module_name, model_py)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec for {model_py}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _frame_number(path: Path) -> int:
  match = re.match(r"^[01]_(\d+)\.pth$", path.name)
  return int(match.group(1)) if match else -1


def _latest_dmc_pair_anchor(directory: Path) -> Optional[Path]:
  candidates = [
      path for path in directory.glob("0_*.pth")
      if path.is_file() and (directory / path.name.replace("0_", "1_", 1)).exists()
  ]
  if not candidates:
    return None
  return max(candidates, key=lambda path: (_frame_number(path), path.stat().st_mtime))


def _default_dmc_checkpoint(rlcard_root: Path) -> Optional[Path]:
  """Finds a likely DMC checkpoint under a local RLCard checkout."""
  run_dmc_dir = rlcard_root / "experiments" / "dmc_result" / "mahjong_two_by_one"
  model_tar = run_dmc_dir / "model.tar"
  if model_tar.exists():
    return model_tar

  if run_dmc_dir.exists():
    anchor = _latest_dmc_pair_anchor(run_dmc_dir)
    if anchor is not None:
      return anchor
  return None


def _dmc_input_dim(agent) -> Optional[int]:
  net = getattr(agent, "net", None)
  if net is None:
    return None
  fc_layers = getattr(net, "fc_layers", None)
  if fc_layers is None:
    return None
  for module in fc_layers:
    if isinstance(module, torch.nn.Linear):
      return int(module.in_features)
  return None


def _validate_dmc_agent_shape(agent, source: Path) -> None:
  expected_input_dim = int(np.prod(RLCARD_OBSERVATION_SHAPE)) + NUM_RLCARD_ACTIONS
  input_dim = _dmc_input_dim(agent)
  if input_dim is None:
    return
  if input_dim != expected_input_dim:
    observed_dim = input_dim - NUM_RLCARD_ACTIONS
    raise ValueError(
        f"DMC checkpoint {source} expects {observed_dim} observation features "
        f"(network input {input_dim}), but current RLCard mahjong_two_by_one "
        f"uses {int(np.prod(RLCARD_OBSERVATION_SHAPE))} observation features "
        f"with shape {RLCARD_OBSERVATION_SHAPE}. Use a checkpoint trained by "
        "`~/RLCard-mahjong/examples/run_dmc.py` against the current "
        "`mahjong_two_by_one` environment.")


def _infer_dmc_agent_paths(anchor_path: Path, num_players: int) -> List[Path]:
  """Infers per-seat `0_*.pth`/`1_*.pth` paths from one anchor path."""
  paths = [anchor_path for _ in range(num_players)]
  match = re.match(r"^([01])_(.+)\.pth$", anchor_path.name)
  if not match:
    return paths

  seat = int(match.group(1))
  paths[seat] = anchor_path
  for other_seat in range(num_players):
    if other_seat == seat:
      continue
    sibling = anchor_path.with_name(f"{other_seat}_{match.group(2)}.pth")
    if sibling.exists():
      paths[other_seat] = sibling
  return paths


def _load_dmc_agents_from_model_tar(
    model_tar: Path,
    rlcard_root: Path,
    device: torch.device,
) -> List[object]:
  """Loads `run_dmc.py`'s bundled `model.tar` checkpoint."""
  dmc_model_module = _load_rlcard_dmc_model_module(rlcard_root)
  checkpoint = _safe_torch_load(model_tar, map_location=device)
  if "model_state_dict" not in checkpoint:
    raise ValueError(f"{model_tar} is not an RLCard DMC `model.tar` checkpoint.")

  num_players = len(checkpoint["model_state_dict"])
  state_shape = [list(RLCARD_OBSERVATION_SHAPE) for _ in range(num_players)]
  action_shape = [[NUM_RLCARD_ACTIONS] for _ in range(num_players)]
  model = dmc_model_module.DMCModel(
      state_shape,
      action_shape,
      exp_epsilon=0.0,
      device=_dmc_device_arg(device),
  )
  agents = []
  for player, state_dict in enumerate(checkpoint["model_state_dict"]):
    agent = model.get_agent(player)
    agent.load_state_dict(state_dict)
    _set_rlcard_agent_device(agent, device)
    _validate_dmc_agent_shape(agent, model_tar)
    agents.append(agent)
  return agents


def _load_dmc_agents_from_pickles(
    paths: Sequence[Path],
    rlcard_root: Path,
    device: torch.device,
) -> List[object]:
  """Loads per-seat pickled DMCAgent files such as `0_123456.pth`."""
  _load_rlcard_dmc_model_module(rlcard_root)
  _ensure_git_dependency_stub()
  agents = []
  for path in paths:
    agent = _safe_torch_load(path, map_location=device)
    if not hasattr(agent, "eval_step"):
      raise TypeError(f"{path} did not load an RLCard DMCAgent-like object.")
    _set_rlcard_agent_device(agent, device)
    _validate_dmc_agent_shape(agent, path)
    agents.append(agent)
  return agents


def load_dmc_agents(args: argparse.Namespace,
                    num_players: int,
                    device: torch.device) -> Tuple[List[object], List[Path]]:
  rlcard_root = _expand_path(args.rlcard_root)
  checkpoint = _expand_path(args.dmc_checkpoint) if args.dmc_checkpoint else None
  if checkpoint is None:
    checkpoint = _default_dmc_checkpoint(rlcard_root)
  if checkpoint is None and args.dmc_model_p0:
    checkpoint = _expand_path(args.dmc_model_p0)
  if checkpoint is None:
    raise FileNotFoundError(
        "No RLCard DMC checkpoint found. Pass --dmc_checkpoint pointing to "
        "`model.tar` or a per-seat file such as `0_123456.pth`.")

  if checkpoint.is_dir():
    model_tar = checkpoint / "model.tar"
    if model_tar.exists():
      checkpoint = model_tar
    else:
      anchor = _latest_dmc_pair_anchor(checkpoint)
      if anchor is None:
        raise FileNotFoundError(
            f"No model.tar or 0_*.pth/1_*.pth pair found in {checkpoint}.")
      checkpoint = anchor

  if checkpoint.name == "model.tar" or checkpoint.suffix == ".tar":
    agents = _load_dmc_agents_from_model_tar(checkpoint, rlcard_root, device)
    return agents[:num_players], [checkpoint] * min(len(agents), num_players)

  paths = _infer_dmc_agent_paths(checkpoint, num_players)
  overrides = [args.dmc_model_p0, args.dmc_model_p1]
  for seat, override in enumerate(overrides[:num_players]):
    if override:
      paths[seat] = _expand_path(override)
  agents = _load_dmc_agents_from_pickles(paths, rlcard_root, device)
  return agents, paths


class ACHPaperPlayer:
  """Greedy policy wrapper around `PracticalACHAgent`."""

  label = "ACH"

  def __init__(self, checkpoint: Path, game: pyspiel.Game, device: torch.device):
    self.checkpoint = checkpoint
    self.agent, self.iteration = PracticalACHAgent.from_checkpoint(
        str(checkpoint), device=str(device))
    self.agent.network.eval()
    self.game = game

  def step(self, state: pyspiel.State, player: int) -> int:
    legal_actions = state.legal_actions(player)
    obs = get_state_tensor(state, player, self.game)
    aux = get_auxiliary_features(
        state, player, self.game, self.agent.aux_feature_channels)
    mask = legal_actions_mask(legal_actions, self.game.num_distinct_actions())
    action, _, _ = self.agent.act(
        obs, mask, auxiliary_features=aux, training=False)
    if action not in legal_actions:
      return int(legal_actions[0])
    return int(action)


class RLCardDMCPlayer:
  """Adapts RLCard DMC agents to the OpenSpiel `eren_yifang` state API."""

  label = "DMC"

  def __init__(self, agents: Sequence[object], paths: Sequence[Path],
               game: pyspiel.Game):
    self.agents = list(agents)
    self.paths = list(paths)
    self.game = game

  def _rlcard_observation(self, state: pyspiel.State, player: int) -> np.ndarray:
    observation = np.asarray(
        state.observation_tensor(player), dtype=np.float32).reshape(
            self.game.observation_tensor_shape())
    if tuple(observation.shape) != OPEN_SPIEL_OBSERVATION_SHAPE:
      raise ValueError(
          "Expected OpenSpiel eren_yifang observation shape "
          f"{OPEN_SPIEL_OBSERVATION_SHAPE}, got {tuple(observation.shape)}.")
    return observation[:RLCARD_OBSERVATION_SHAPE[0]]

  def _rlcard_state(self, state: pyspiel.State, player: int) -> Dict[str, object]:
    legal_actions = [int(action) for action in state.legal_actions(player)]
    return {
        "obs": self._rlcard_observation(state, player),
        "legal_actions": OrderedDict((action, None) for action in legal_actions),
        "raw_legal_actions": [
            _action_to_rlcard_raw(action) for action in legal_actions
        ],
        "raw_obs": {},
    }

  def step(self, state: pyspiel.State, player: int) -> int:
    rl_state = self._rlcard_state(state, player)
    action, _ = self.agents[player].eval_step(rl_state)
    action = int(action)
    legal_actions = state.legal_actions(player)
    if action in legal_actions:
      return action
    if len(legal_actions) == 1:
      return int(legal_actions[0])
    raise ValueError(
        f"RLCard DMC returned illegal action {action}; legal actions are "
        f"{list(legal_actions)}.")


def _resolve_ach_checkpoint(args: argparse.Namespace) -> Path:
  if args.ach_checkpoint:
    return _expand_path(args.ach_checkpoint)
  checkpoint = _latest_checkpoint_path(
      checkpoint_dir=_expand_path(args.ach_checkpoint_dir),
      checkpoint_prefix=args.ach_checkpoint_prefix,
      game_name=args.game,
  )
  if checkpoint is None:
    raise FileNotFoundError(
        "No ACH checkpoint found. Pass --ach_checkpoint or adjust "
        "--ach_checkpoint_dir.")
  return checkpoint.resolve()


def _sample_chance_action(state: pyspiel.State) -> int:
  actions, probs = zip(*state.chance_outcomes())
  return int(np.random.choice(actions, p=np.asarray(probs, dtype=np.float64)))


def play_one_game(game: pyspiel.Game,
                  seat_players: Sequence[object],
                  verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
  state = game.new_initial_state()
  trace: List[str] = []

  while not state.is_terminal():
    if state.is_chance_node():
      action = _sample_chance_action(state)
      if verbose:
        trace.append(
            f"chance: {state.action_to_string(state.current_player(), action)}")
      state.apply_action(action)
      continue

    player = state.current_player()
    agent = seat_players[player]
    action = agent.step(state, player)
    if verbose:
      trace.append(
          f"P{player} {agent.label}: {state.action_to_string(player, action)}")
    state.apply_action(action)

  return np.asarray(state.returns(), dtype=np.float64), trace


def run_battle(args: argparse.Namespace) -> None:
  if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

  device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available()
                                       else "cpu"))
  game = pyspiel.load_game(args.game)
  if args.game != "eren_yifang":
    print(
        f"Warning: this adapter is intended for eren_yifang, got {args.game!r}.")
  if game.num_players() != 2 or game.num_distinct_actions() != NUM_RLCARD_ACTIONS:
    raise ValueError(
        "This script expects the two-player 15-action eren_yifang game.")

  ach_checkpoint = _resolve_ach_checkpoint(args)
  ach_player = ACHPaperPlayer(ach_checkpoint, game, device)
  dmc_agents, dmc_paths = load_dmc_agents(args, game.num_players(), device)
  dmc_player = RLCardDMCPlayer(dmc_agents, dmc_paths, game)

  print(f"Game: {args.game}")
  print(f"Device: {device}")
  print(f"ACH checkpoint: {ach_checkpoint} (iteration {ach_player.iteration})")
  print(f"DMC checkpoints by seat: {[str(path) for path in dmc_paths]}")

  base_seats = [None, None]
  base_seats[args.ach_seat] = ach_player
  base_seats[1 - args.ach_seat] = dmc_player

  seat_totals = np.zeros(game.num_players(), dtype=np.float64)
  seat_wins = np.zeros(game.num_players(), dtype=np.int64)
  agent_totals: Dict[str, float] = defaultdict(float)
  agent_games: Dict[str, int] = defaultdict(int)
  agent_wins: Dict[str, int] = defaultdict(int)
  draws = 0

  for game_index in range(args.num_games):
    seat_players = list(base_seats)
    if args.alternate_seats and game_index % 2 == 1:
      seat_players.reverse()

    returns, trace = play_one_game(
        game, seat_players, verbose=args.verbose)
    seat_totals += returns

    for seat, agent in enumerate(seat_players):
      agent_totals[agent.label] += float(returns[seat])
      agent_games[agent.label] += 1

    if returns[0] == returns[1]:
      draws += 1
    else:
      winning_seat = int(np.argmax(returns))
      seat_wins[winning_seat] += 1
      agent_wins[seat_players[winning_seat].label] += 1

    if args.verbose:
      print(
          f"\n=== Game {game_index + 1}: "
          f"P0={seat_players[0].label}, P1={seat_players[1].label} ===")
      for item in trace:
        print(item)
      print(f"Returns: {returns.tolist()}")

  print("\nBattle finished")
  print(f"Games: {args.num_games}")
  print(f"Seat mode: {'alternating' if args.alternate_seats else 'fixed'}")
  print(f"Seat wins P0/P1: {seat_wins.tolist()}, draws: {draws}")
  print(f"Average returns by seat: {(seat_totals / args.num_games).tolist()}")
  for label in sorted(agent_games):
    avg = agent_totals[label] / max(agent_games[label], 1)
    print(
        f"{label}: wins={agent_wins[label]}, "
        f"games={agent_games[label]}, average_return={avg:.6g}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--game", default="eren_yifang")
  parser.add_argument(
      "--ach_checkpoint",
      default=None,
      help="Exact checkpoint from ach_paper_pytorch.py. If omitted, the latest "
      "matching checkpoint is used.")
  parser.add_argument(
      "--ach_checkpoint_dir",
      default=str(DEFAULT_ACH_CHECKPOINT_DIR),
      help="Directory containing ach_practical_eren_yifang_*.pth files.")
  parser.add_argument("--ach_checkpoint_prefix", default="ach_practical")
  parser.add_argument(
      "--rlcard_root",
      default=str(DEFAULT_RLCARD_ROOT),
      help="Local checkout containing RLCard-mahjong.")
  parser.add_argument(
      "--dmc_checkpoint",
      default=None,
      help="RLCard DMC model.tar, a directory containing model.tar or "
      "0_*.pth/1_*.pth, or one per-seat .pth file.")
  parser.add_argument(
      "--dmc_model_p0",
      default=DEFAULT_DMC_MODEL_P0,
      help="Optional per-seat RLCard DMC .pth override for player 0.")
  parser.add_argument(
      "--dmc_model_p1",
      default=DEFAULT_DMC_MODEL_P1,
      help="Optional per-seat RLCard DMC .pth override for player 1.")
  parser.add_argument("--num_games", type=int, default=100)
  parser.add_argument(
      "--ach_seat",
      type=int,
      choices=[0, 1],
      default=0,
      help="Seat used by ACH when --alternate_seats is disabled.")
  parser.add_argument(
      "--alternate_seats",
      action="store_true",
      help="Swap ACH and DMC seats every other game.")
  parser.add_argument("--device", default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--verbose", action="store_true")
  return parser


if __name__ == "__main__":
  run_battle(build_parser().parse_args())
