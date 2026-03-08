# tests/test_agents/test_mcts.py
import numpy as np
from engine.game_state import GameState
from agents.ismcts.mcts import MCTSNode, mcts_search


class TestMCTSNode:
    def test_root_node(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        assert node.parent is None
        assert node.visit_count == 0
        assert len(node.children) == 0

    def test_expand(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        legal = gs.get_legal_actions()
        prior = np.ones(300) / 300  # uniform prior
        node.expand(legal, prior)
        assert len(node.children) == len(legal)

    def test_select_child_ucb(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        legal = gs.get_legal_actions()
        prior = np.ones(300) / 300
        node.expand(legal, prior)
        node.visit_count = 1
        child = node.select_child(cpuct=1.5)
        assert child is not None


class TestMCTSSearch:
    def test_returns_valid_action(self):
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            """Uniform policy, neutral value."""
            return np.ones(300) / 300, 0.0

        action, visit_counts = mcts_search(
            state=gs,
            policy_value_fn=dummy_policy_value,
            num_simulations=10,
            cpuct=1.5,
        )
        assert action in gs.get_legal_actions()

    def test_visit_counts_sum(self):
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        _, visit_counts = mcts_search(
            state=gs,
            policy_value_fn=dummy_policy_value,
            num_simulations=50,
            cpuct=1.5,
        )
        assert sum(visit_counts.values()) == 50

    def test_more_simulations_better(self):
        """With more simulations, the search should be more confident (higher max visits)."""
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        _, vc_low = mcts_search(gs, dummy_policy_value, num_simulations=10, cpuct=1.5)
        _, vc_high = mcts_search(gs, dummy_policy_value, num_simulations=100, cpuct=1.5)
        assert max(vc_high.values()) >= max(vc_low.values())
