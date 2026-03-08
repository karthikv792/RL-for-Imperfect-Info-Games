import numpy as np
from engine.game_state import GameState
from agents.rebel.cfr import CFRNode, run_cfr


class TestCFRNode:
    def test_initial_strategy_uniform(self):
        node = CFRNode(num_actions=5)
        strategy = node.get_strategy()
        assert strategy.shape == (5,)
        assert np.allclose(strategy, 0.2)

    def test_update_regrets(self):
        node = CFRNode(num_actions=3)
        node.update_regrets(np.array([1.0, -0.5, 0.0]))
        strategy = node.get_strategy()
        assert strategy[0] > strategy[1]


class TestRunCFR:
    def test_returns_strategy(self):
        gs = GameState.new_game(seed=42)

        def dummy_value_fn(state):
            return 0.0

        strategy = run_cfr(
            state=gs,
            value_fn=dummy_value_fn,
            num_iterations=5,
            max_depth=2,
        )
        assert isinstance(strategy, np.ndarray)
        assert strategy.shape == (300,)
        assert abs(strategy.sum() - 1.0) < 1e-5

    def test_strategy_is_valid_distribution(self):
        gs = GameState.new_game(seed=42)

        def dummy_value_fn(state):
            return 0.0

        strategy = run_cfr(
            state=gs,
            value_fn=dummy_value_fn,
            num_iterations=10,
            max_depth=3,
        )
        assert np.all(strategy >= 0)
        assert abs(strategy.sum() - 1.0) < 1e-5
