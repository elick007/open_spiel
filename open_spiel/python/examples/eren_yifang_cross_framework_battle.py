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


RL_ACTION_TO_OPEN_SPIEL = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: None,   # draw/passive transition, never chosen on OpenSpiel player nodes.
    10: 9,     # pong
    11: 11,    # gong -> tile-specific kong resolved from legal actions.
    12: 19,    # stand -> pass
    13: 18,    # hu
    14: 18,    # zimo
}


def _append_tile_count(row: np.ndarray, count: int) -> None:
  row[:] = 0
  if count > 0:
    row[:count] = 1


def _decode_open_spiel_discards(state_str: str, marker: str) -> List[int]:
  if marker not in state_str:
    return []
  part = state_str.split(marker, maxsplit=1)[1]
  part = part.split("|", maxsplit=1)[0].strip()
  if not part:
    return []
  tiles = []
  for token in part.split():
    if token.endswith("W"):
      tiles.append(int(token[:-1]) - 1)
  return tiles


def _parse_observation_string(obs_string: str) -> Dict[str, object]:
  result: Dict[str, object] = {
      "self_melds": [],
      "opp_exposed": [],
      "opp_concealed_kongs": 0,
      "self_discards": _decode_open_spiel_discards(obs_string, "Self discards:"),
      "opp_discards": _decode_open_spiel_discards(obs_string, "Opp discards:"),
      "last_discard": None,
  }

  if "| Self melds:" in obs_string:
    part = obs_string.split("| Self melds:", maxsplit=1)[1].split("|", maxsplit=1)[0]
    for chunk in part.split("["):
      if "]" not in chunk:
        continue
      inner = chunk.split("]", maxsplit=1)[0].strip()
      if not inner:
        continue
      label, tile = inner.split()
      result["self_melds"].append((label, int(tile[:-1]) - 1))

  if "| Opp exposed:" in obs_string:
    part = obs_string.split("| Opp exposed:", maxsplit=1)[1].split("|", maxsplit=1)[0]
    for chunk in part.split("["):
      if "]" not in chunk:
        continue
      inner = chunk.split("]", maxsplit=1)[0].strip()
      if not inner:
        continue
      label, tile = inner.split()
      result["opp_exposed"].append((label, int(tile[:-1]) - 1))

  if "| Opp an-gang count:" in obs_string:
    value = obs_string.split("| Opp an-gang count:", maxsplit=1)[1].split("|", maxsplit=1)[0].strip()
    result["opp_concealed_kongs"] = int(value)

  if "| Last discard:" in obs_string:
    token = obs_string.split("| Last discard:", maxsplit=1)[1].split()[0]
    result["last_discard"] = int(token[:-1]) - 1

  return result


class RLCardDMCWrapper:
  """Adapts an RLCard DMC model to OpenSpiel Eren Yifang."""

  def __init__(self, model_path: str, device: str | None = None):
    self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    self.agent = torch.load(model_path, map_location=self.device)
    if hasattr(self.agent, "set_device"):
      self.agent.set_device(self.device)

  def _build_obs(self, state: pyspiel.State, player: int) -> Dict[str, object]:
    obs = np.zeros((4, 9, 4), dtype=np.int8)
    obs_string = state.observation_string(player)
    parsed = _parse_observation_string(obs_string)

    hand_tensor = np.asarray(state.observation_tensor(player), dtype=np.float32)
    image_size = 4 * 9 * 69
    hand_map = hand_tensor[:image_size].reshape(69, 4, 9)[0].transpose(1, 0)
    obs[0] = hand_map

    for tile in parsed["self_discards"]:
      row = obs[1, tile]
      idx = int(row.sum())
      if idx < 4:
        row[idx] = 1
    for tile in parsed["opp_discards"]:
      row = obs[1, tile]
      idx = int(row.sum())
      if idx < 4:
        row[idx] = 1

    self_pile_index = 2
    opp_pile_index = 3
    for label, tile in parsed["self_melds"]:
      count = 4 if label in ("MingGang", "AnGang") else 3
      _append_tile_count(obs[self_pile_index, tile], count)
    for label, tile in parsed["opp_exposed"]:
      count = 4 if label == "MingGang" else 3
      _append_tile_count(obs[opp_pile_index, tile], count)

    return {"obs": obs}

  def _translate_action(self, rl_action: int, legal_actions: Sequence[int]) -> int:
    if rl_action in range(9):
      mapped = RL_ACTION_TO_OPEN_SPIEL[rl_action]
      if mapped in legal_actions:
        return mapped
    elif rl_action == 10 and 9 in legal_actions:
      return 9
    elif rl_action == 11:
      kong_actions = [action for action in legal_actions if 10 <= action <= 18]
      if len(kong_actions) == 1:
        return kong_actions[0]
      if kong_actions:
        return kong_actions[0]
    elif rl_action in (13, 14) and 18 in legal_actions:
      return 19
    elif rl_action == 9 and 20 in legal_actions:
      return 20

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
    if 0 <= open_spiel_action <= 8:
      return open_spiel_action
    if open_spiel_action == 9:
      return 10
    if 10 <= open_spiel_action <= 18:
      return 11
    if open_spiel_action == 18:
      if 13 in RL_ACTION_TO_OPEN_SPIEL and 18 in legal_actions:
        return 13
      return 14
    if open_spiel_action == 19:
      return 12
    if open_spiel_action == 20:
      return 9
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