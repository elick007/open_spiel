"""Cross-framework battle script for Eren Yifang.

Runs head-to-head matches between:
- an OpenSpiel ACH agent trained by `ach_paper_pytorch.py`
- an RLCard DMC agent trained on `mahjong_two_by_one`

The script adapts OpenSpiel `eren_yifang` states into the RLCard observation and
action conventions used by the DMC model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyspiel
import torch


THIS_DIR = Path(__file__).resolve().parent
OPEN_SPIEL_ROOT = THIS_DIR.parents[2]
RLCard_ROOT = Path(r"D:\Projects\PYProjects\rlcard")

if str(THIS_DIR) not in sys.path:
  sys.path.append(str(THIS_DIR))
if str(RLCard_ROOT) not in sys.path:
  sys.path.append(str(RLCard_ROOT))

from ach_paper_pytorch import (  # pylint: disable=g-import-not-at-top
    PracticalACHAgent,
    _latest_checkpoint_path,
    get_state_shape,
    get_state_tensor,
    legal_actions_mask,
)


RL_ACTION_TO_OPEN_SPIEL = {action: action for action in range(49)}


class RLCardDMCWrapper:
  """Adapts an RLCard DMC model to OpenSpiel Eren Yifang."""

  def __init__(self, model_path: str, device: str | None = None):
    self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.agent = torch.load(model_path, map_location=self.device)
    if hasattr(self.agent, "set_device"):
      self.agent.set_device(self.device)

  def _build_obs(self, state: pyspiel.State, player: int) -> Dict[str, object]:
    obs = np.asarray(state.observation_tensor(player), dtype=np.int8).reshape(
        state.get_game().observation_tensor_shape())
    return {"obs": obs}

  def _translate_action(self, rl_action: int, legal_actions: Sequence[int]) -> int:
    mapped = RL_ACTION_TO_OPEN_SPIEL.get(int(rl_action))
    if mapped in legal_actions:
      return mapped
    if len(legal_actions) == 1:
      return int(legal_actions[0])
    raise ValueError(
        f"Unable to map RLCard action {rl_action} to OpenSpiel legal actions {list(legal_actions)}")

  def step(self, state: pyspiel.State, player: int) -> int:
    legal_actions = state.legal_actions(player)
    rl_state = self._build_obs(state, player)
    rl_state["legal_actions"] = OrderedDict((self._to_rl_action(a, legal_actions), None)
                                             for a in legal_actions)
    rl_state["raw_legal_actions"] = list(rl_state["legal_actions"].keys())
    action, _ = self.agent.eval_step(rl_state)
    return self._translate_action(int(action), legal_actions)

  def _to_rl_action(self, open_spiel_action: int, legal_actions: Sequence[int]) -> int:
    del legal_actions
    if 0 <= open_spiel_action < 49:
      return open_spiel_action
    raise ValueError(f"Unsupported OpenSpiel action: {open_spiel_action}")


def load_ach_agent(game_name: str, checkpoint_path: str | None) -> PracticalACHAgent:
  game = pyspiel.load_game(game_name)
  state_size = int(np.prod(get_state_shape(game)))
  action_size = game.num_distinct_actions()
  agent = PracticalACHAgent(state_size=state_size, action_size=action_size)

  path = Path(checkpoint_path) if checkpoint_path else _latest_checkpoint_path(
      checkpoint_dir=Path("checkpoints"),
      checkpoint_prefix="ach_practical",
      game_name=game_name)
  if path is None:
    raise FileNotFoundError("No ACH checkpoint found.")
  agent.load(str(path))
  return agent


def play_match(game: pyspiel.Game,
               ach_agent: PracticalACHAgent,
               dmc_agent: RLCardDMCWrapper,
               ach_player: int) -> Tuple[List[float], List[str]]:
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
    legal_actions = state.legal_actions(player)
    if player == ach_player:
      info_state = get_state_tensor(state, player, game)
      mask = legal_actions_mask(legal_actions, game.num_distinct_actions())
      action, _, _ = ach_agent.act(info_state, mask, training=False)
      if action not in legal_actions:
        action = int(legal_actions[np.argmax(mask[legal_actions])])
      actor = "ACH"
    else:
      action = dmc_agent.step(state, player)
      actor = "DMC"

    trace.append(f"P{player} {actor}: {state.action_to_string(player, action)}")
    state.apply_action(action)

  return state.returns(), trace


def run_battle(args: argparse.Namespace) -> None:
  game = pyspiel.load_game(args.game)
  ach_agent = load_ach_agent(args.game, args.ach_checkpoint)
  dmc_agent = RLCardDMCWrapper(args.dmc_model, device=args.device)

  totals = np.zeros(game.num_players(), dtype=np.float64)
  wins = np.zeros(game.num_players(), dtype=np.int32)
  draws = 0

  for game_index in range(args.num_games):
    ach_player = game_index % 2 if args.alternate_seats else args.ach_player
    returns, trace = play_match(game, ach_agent, dmc_agent, ach_player)
    totals += np.asarray(returns)

    if returns[0] == returns[1]:
      draws += 1
    else:
      wins[int(np.argmax(returns))] += 1

    if args.verbose:
      print(f"=== Game {game_index + 1} | ACH seat={ach_player} ===")
      for item in trace:
        print(item)
      print(f"Returns: {returns}\n")

  print("Battle finished")
  print(f"Games: {args.num_games}")
  print(f"Seat mode: {'alternate' if args.alternate_seats else 'fixed'}")
  print(f"Wins P0/P1: {wins.tolist()}")
  print(f"Draws: {draws}")
  print(f"Average returns: {(totals / args.num_games).tolist()}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--game", default="eren_yifang")
  parser.add_argument("--ach-checkpoint", default=None)
  parser.add_argument(
      "--dmc-model",
      default=r"D:\Projects\PYProjects\rlcard\auto_play_ai\train\1_2850553600.pth")
  parser.add_argument("--num-games", type=int, default=100)
  parser.add_argument("--ach-player", type=int, default=0, choices=[0, 1])
  parser.add_argument("--alternate-seats", action="store_true")
  parser.add_argument("--device", default=None)
  parser.add_argument("--verbose", action="store_true")
  return parser


if __name__ == "__main__":
  run_battle(build_parser().parse_args())
