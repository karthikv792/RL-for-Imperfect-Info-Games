# agents/ismcts/mcts.py
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional
from engine.game_state import GameState
from engine.actions import Action, action_to_index, index_to_action


class MCTSNode:
    __slots__ = (
        "state", "parent", "action", "prior", "visit_count",
        "value_sum", "children",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional[MCTSNode] = None,
        action: Optional[Action] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}  # action_index -> child

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, legal_actions: list[Action], policy: np.ndarray):
        for action in legal_actions:
            idx = action_to_index(action)
            if idx not in self.children:
                child_state = self.state.apply_action(action)
                self.children[idx] = MCTSNode(
                    state=child_state,
                    parent=self,
                    action=action,
                    prior=float(policy[idx]),
                )

    def select_child(self, cpuct: float) -> MCTSNode:
        best_score = -math.inf
        best_child = None
        sqrt_parent = math.sqrt(self.visit_count)

        for child in self.children.values():
            ucb = child.q_value + cpuct * child.prior * sqrt_parent / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            # Flip value at each level since players alternate
            node.value_sum += value
            value = -value
            node = node.parent


PolicyValueFn = Callable[[GameState], tuple[np.ndarray, float]]


def mcts_search(
    state: GameState,
    policy_value_fn: PolicyValueFn,
    num_simulations: int,
    cpuct: float = 1.5,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
) -> tuple[Action, dict[Action, int]]:
    root = MCTSNode(state=state)

    # Expand root
    legal_actions = state.get_legal_actions()
    policy, _ = policy_value_fn(state)

    # Normalize policy to legal actions
    legal_indices = [action_to_index(a) for a in legal_actions]
    mask = np.zeros(300)
    mask[legal_indices] = 1.0
    policy = policy * mask
    policy_sum = policy.sum()
    if policy_sum > 0:
        policy = policy / policy_sum
    else:
        policy[legal_indices] = 1.0 / len(legal_indices)

    # Add Dirichlet noise at root for exploration
    if dirichlet_alpha > 0 and dirichlet_epsilon > 0:
        noise = np.zeros(300)
        noise[legal_indices] = np.random.dirichlet(
            [dirichlet_alpha] * len(legal_indices)
        )
        policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise

    root.expand(legal_actions, policy)

    for _ in range(num_simulations):
        node = root

        # SELECT: traverse tree using UCB
        while node.children and not node.state.is_terminal():
            node = node.select_child(cpuct)

        # EVALUATE
        if node.state.is_terminal():
            winner = node.state.get_winner()
            if winner is None:
                value = 0.0
            elif winner == state.current_player:
                value = 1.0
            else:
                value = -1.0
        else:
            # EXPAND and evaluate with neural network
            child_legal = node.state.get_legal_actions()
            if child_legal:
                child_policy, value = policy_value_fn(node.state)
                # Adjust value sign: NN returns value from current player's perspective
                if node.state.current_player != state.current_player:
                    value = -value
                node.expand(child_legal, child_policy)
            else:
                value = 0.0

        # BACKPROPAGATE
        node.backpropagate(value)

    # Extract visit counts
    visit_counts: dict[Action, int] = {}
    for idx, child in root.children.items():
        visit_counts[child.action] = child.visit_count

    # Select action with most visits
    best_action = max(visit_counts, key=visit_counts.get)
    return best_action, visit_counts
