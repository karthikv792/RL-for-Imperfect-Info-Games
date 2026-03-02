# tests/test_training/test_arena.py
import pytest
from agents.random_agent import RandomAgent
from training.arena import Arena


class TestArena:
    def test_run_match(self):
        p1 = RandomAgent("r1", seed=1)
        p2 = RandomAgent("r2", seed=2)
        arena = Arena(num_games=10)
        result = arena.evaluate(p1, p2)
        assert "p1_wins" in result
        assert "p2_wins" in result
        assert "draws" in result
        assert result["p1_wins"] + result["p2_wins"] + result["draws"] == 10

    def test_win_rate(self):
        p1 = RandomAgent("r1", seed=1)
        p2 = RandomAgent("r2", seed=2)
        arena = Arena(num_games=10)
        result = arena.evaluate(p1, p2)
        assert 0.0 <= result["p1_win_rate"] <= 1.0
