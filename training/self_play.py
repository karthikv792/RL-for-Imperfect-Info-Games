# training/self_play.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from engine.game_state import GameState
from engine.actions import action_to_index
from models.board_encoder import BoardEncoder
from agents.ismcts.ismcts import ismcts_search


@dataclass
class SelfPlayConfig:
    num_simulations: int = 100
    num_determinizations: int = 20
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_moves: int = 500
    cards_per_hand: int = 5


def run_self_play_game(
    model: torch.nn.Module,
    encoder: BoardEncoder,
    config: SelfPlayConfig,
    seed: int | None = None,
    device: str = "cpu",
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one full game via self-play, returning training examples."""

    def policy_value_fn(state: GameState) -> tuple[np.ndarray, float]:
        tensor = encoder.encode_torch(state, player_perspective=state.current_player)
        tensor = tensor.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            policy, value = model(tensor)
        return policy.cpu().numpy()[0], value.cpu().item()

    game_state = GameState.new_game(seed=seed, cards_per_hand=config.cards_per_hand)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []  # (state_tensor, mcts_policy, player)

    for move_num in range(config.max_moves):
        if game_state.is_terminal():
            break

        current_player = game_state.current_player
        info_set = game_state.to_information_set(current_player)
        state_tensor = game_state.to_tensor(current_player)

        # Run ISMCTS to get action and visit counts
        from agents.ismcts.mcts import mcts_search
        from agents.ismcts.ismcts import determinize

        # Aggregate across determinizations
        aggregated_visits: dict[int, int] = {}
        for d in range(config.num_determinizations):
            det_state = determinize(info_set, current_player, seed=(seed or 0) * 1000 + move_num * 100 + d)
            _, visit_counts = mcts_search(
                state=det_state,
                policy_value_fn=policy_value_fn,
                num_simulations=config.num_simulations,
                cpuct=config.cpuct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
            )
            for action, visits in visit_counts.items():
                idx = action_to_index(action)
                aggregated_visits[idx] = aggregated_visits.get(idx, 0) + visits

        # Build policy target from visit counts
        total_visits = sum(aggregated_visits.values())
        policy_target = np.zeros(300, dtype=np.float32)
        for idx, visits in aggregated_visits.items():
            policy_target[idx] = visits / total_visits

        history.append((state_tensor, policy_target, current_player))

        # Select action (proportional to visits for exploration)
        probs = policy_target / policy_target.sum()
        chosen_idx = np.random.choice(300, p=probs)
        from engine.actions import index_to_action
        action = index_to_action(chosen_idx)

        # Ensure action is legal (fallback to most visited)
        legal = game_state.get_legal_actions()
        legal_indices = {action_to_index(a) for a in legal}
        if chosen_idx not in legal_indices:
            chosen_idx = max(aggregated_visits, key=aggregated_visits.get)
            action = index_to_action(chosen_idx)

        game_state = game_state.apply_action(action)

    # Assign values based on game outcome
    winner = game_state.get_winner()
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for state_tensor, policy_target, player in history:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        examples.append((state_tensor, policy_target, value))

    return examples
