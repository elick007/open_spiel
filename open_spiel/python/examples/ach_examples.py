"""Interactive play using the latest ACH_poker PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ach_poker_pytorch import (
    ACHPokerAgent,
    _ensure_pyspiel,
    _latest_checkpoint_path,
    _safe_game_name,
    get_state_shape,
    get_state_tensor,
    legal_actions_mask,
)

if TYPE_CHECKING:
    import pyspiel


GAME_ALIASES = {
    "kuhn": "kuhn_poker",
    "leduc": "leduc_poker",
    "liars_dice": "liars_dice",
    "xueliu_erma": "xueliu_erma",
}


def _load_checkpoint_payload(checkpoint_path: Path) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _infer_hidden_sizes(checkpoint: dict) -> Optional[Tuple[int, ...]]:
    """Infers ACHPokerAgent hidden sizes from a saved checkpoint."""
    saved_agents = checkpoint.get("agents")
    if not saved_agents:
        return None

    network_state = saved_agents[0].get("network", {})
    hidden_sizes = []
    layer_index = 0
    while True:
        weight = network_state.get(f"torso.{layer_index}.weight")
        if weight is None:
            break
        hidden_sizes.append(int(weight.shape[0]))
        layer_index += 2

    return tuple(hidden_sizes) if hidden_sizes else None


def _format_action(state: pyspiel.State, player: int, action: int) -> str:
    try:
        return f"{action}: {state.action_to_string(player, action)}"
    except (AttributeError, TypeError, ValueError):
        return str(action)


def _player_view(state: pyspiel.State, player: int, game: pyspiel.Game) -> str:
    game_type = game.get_type()
    if getattr(game_type, "provides_information_state_string", False):
        return state.information_state_string(player)
    if getattr(game_type, "provides_observation_string", False):
        return state.observation_string(player)
    return str(state)


def load_latest_trained_agents(
    game_name: str = "eren_yifang",
    checkpoint_dir: str = "checkpoints",
    checkpoint_prefix: str = "ach_poker_style",
    device: str = "cpu",
    hidden_sizes: Optional[Sequence[int]] = None,
) -> Tuple[pyspiel.Game, List[ACHPokerAgent], Path, int]:
    """Loads the latest all-player checkpoint written by ach_poker_pytorch.py."""
    pyspiel_module = _ensure_pyspiel()
    game = pyspiel_module.load_game(game_name)
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_path = _latest_checkpoint_path(
        checkpoint_dir=checkpoint_dir_path,
        checkpoint_prefix=checkpoint_prefix,
        game_name=game_name,
    )
    if checkpoint_path is None:
        pattern = f"{checkpoint_prefix}_{_safe_game_name(game_name)}_*.pth"
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir_path} matching {pattern}. "
            "Train with ach_poker_pytorch.py first, then run this script."
        )

    checkpoint = _load_checkpoint_payload(checkpoint_path)
    saved_agents = checkpoint.get("agents")
    if not saved_agents:
        raise ValueError(f"{checkpoint_path} is not an ach_poker_pytorch checkpoint.")
    if len(saved_agents) != game.num_players():
        raise ValueError(
            f"{checkpoint_path} has {len(saved_agents)} agents, but {game_name} "
            f"has {game.num_players()} players."
        )

    state_size = int(np.prod(get_state_shape(game)))
    action_size = game.num_distinct_actions()
    inferred_hidden_sizes = _infer_hidden_sizes(checkpoint)
    resolved_hidden_sizes = tuple(hidden_sizes or inferred_hidden_sizes or (256, 256))

    agents = [
        ACHPokerAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=resolved_hidden_sizes,
            device=device,
        )
        for _ in range(game.num_players())
    ]
    for agent, saved_agent in zip(agents, saved_agents):
        agent.network.load_state_dict(saved_agent["network"])
        agent.network.eval()
    loaded_iteration = int(checkpoint.get("iteration", 0))

    return game, agents, checkpoint_path, loaded_iteration


def _read_human_action(
    state: pyspiel.State,
    player: int,
    legal_actions: Sequence[int],
) -> int:
    action_help = ", ".join(
        _format_action(state, player, action) for action in legal_actions)
    print(f"Legal actions: {action_help}")

    while True:
        raw_action = input("Your action: ").strip()
        try:
            action = int(raw_action)
        except ValueError:
            print("Please enter the action number.")
            continue
        if action in legal_actions:
            return action
        print(f"Invalid action. Choose one of: {list(legal_actions)}")


def play_interactive_game(
    game_name: str = "eren_yifang",
    checkpoint_dir: str = "checkpoints",
    checkpoint_prefix: str = "ach_poker_style",
    human_player: int = 1,
    device: str = "cpu",
    hidden_sizes: Optional[Sequence[int]] = None,
) -> None:
    """Plays against the latest trained ACH_poker PyTorch agents."""
    game, agents, checkpoint_path, iteration = load_latest_trained_agents(
        game_name=game_name,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        device=device,
        hidden_sizes=hidden_sizes,
    )
    if not 0 <= human_player < game.num_players():
        raise ValueError(
            f"human_player must be in [0, {game.num_players() - 1}], got "
            f"{human_player}."
        )

    state = game.new_initial_state()
    num_actions = game.num_distinct_actions()

    print(f"\n{'=' * 50}")
    print(f"Playing {game_name}")
    print(f"Loaded checkpoint: {checkpoint_path} (iteration {iteration})")
    print(f"You are player {human_player}")
    print(f"{'=' * 50}\n")

    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(int(np.random.choice(actions, p=probs)))
            continue

        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        print(f"\nPlayer {current_player} to act")
        print(_player_view(state, current_player, game))

        if current_player == human_player:
            action = _read_human_action(state, current_player, legal_actions)
        else:
            info_state = get_state_tensor(state, current_player, game)
            mask = legal_actions_mask(legal_actions, num_actions)
            action, action_prob, value = agents[current_player].act(
                info_state, mask, training=False)
            print(
                "Agent chooses "
                f"{_format_action(state, current_player, action)} "
                f"(prob={action_prob:.3f}, value={value:.3f})"
            )

        state.apply_action(action)

    returns = state.returns()
    print("\nGame over!")
    print(f"Final state:\n{state}")
    print(f"Returns: {returns}")
    human_return = returns[human_player]
    best_return = max(returns)
    if human_return == best_return and returns.count(best_return) == 1:
        print("You win!")
    elif human_return == best_return:
        print("Draw.")
    else:
        print("Agent wins.")


def _parse_hidden_sizes(raw_hidden_sizes: str) -> Optional[Tuple[int, ...]]:
    if raw_hidden_sizes.lower() == "auto":
        return None
    return tuple(
        int(size.strip())
        for size in raw_hidden_sizes.split(",")
        if size.strip()
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play interactively against the latest checkpoint produced by "
            "ach_poker_pytorch.py."
        )
    )
    parser.add_argument(
        "game_positional",
        nargs="?",
        help="Optional game string or shortcut, e.g. kuhn or eren_yifang.",
    )
    parser.add_argument(
        "--game_name",
        "--game",
        default="eren_yifang",
        help="OpenSpiel game string used when training the checkpoint.",
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--checkpoint_prefix", default="ach_poker_style")
    parser.add_argument(
        "--human_player",
        type=int,
        default=1,
        help="Player id controlled by keyboard input.",
    )
    parser.add_argument(
        "--hidden_sizes",
        default="auto",
        help="Comma-separated model hidden sizes, or auto to infer them.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    game_name = args.game_positional or args.game_name
    game_name = GAME_ALIASES.get(game_name, game_name)
    play_interactive_game(
        game_name=game_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        human_player=args.human_player,
        device=args.device,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
    )


if __name__ == "__main__":
    main()
