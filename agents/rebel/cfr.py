from __future__ import annotations
import numpy as np
from typing import Callable
from engine.game_state import GameState
from engine.actions import action_to_index


class CFRNode:
    """A single CFR information set node."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, regrets: np.ndarray):
        self.regret_sum += regrets

    def update_strategy_sum(self, strategy: np.ndarray, weight: float = 1.0):
        self.strategy_sum += weight * strategy

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.num_actions) / self.num_actions


ValueFn = Callable[[GameState], float]


def run_cfr(
    state: GameState,
    value_fn: ValueFn,
    num_iterations: int = 100,
    max_depth: int = 5,
) -> np.ndarray:
    """Run CFR iterations on a subgame rooted at `state`.
    Returns a strategy vector of shape (300,) normalized to legal actions.
    """
    legal_actions = state.get_legal_actions()
    if not legal_actions:
        return np.zeros(300, dtype=np.float32)

    num_legal = len(legal_actions)
    node = CFRNode(num_actions=num_legal)

    for _ in range(num_iterations):
        strategy = node.get_strategy()
        action_values = np.zeros(num_legal, dtype=np.float64)

        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)
            if next_state.is_terminal():
                winner = next_state.get_winner()
                if winner is None:
                    action_values[i] = 0.0
                elif winner == state.current_player:
                    action_values[i] = 1.0
                else:
                    action_values[i] = -1.0
            elif max_depth <= 1:
                v = value_fn(next_state)
                if next_state.current_player != state.current_player:
                    v = -v
                action_values[i] = v
            else:
                v = value_fn(next_state)
                if next_state.current_player != state.current_player:
                    v = -v
                action_values[i] = v

        node_value = np.dot(strategy, action_values)
        regrets = action_values - node_value
        node.update_regrets(regrets)
        node.update_strategy_sum(strategy)

    avg_strategy = node.get_average_strategy()
    full_strategy = np.zeros(300, dtype=np.float32)
    for i, action in enumerate(legal_actions):
        idx = action_to_index(action)
        full_strategy[idx] = avg_strategy[i]

    total = full_strategy.sum()
    if total > 0:
        full_strategy /= total

    return full_strategy
