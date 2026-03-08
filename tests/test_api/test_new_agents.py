# tests/test_api/test_new_agents.py
import pytest
from api.game_session import GameSessionManager


class TestNewAgents:
    def test_all_agents_listed(self):
        manager = GameSessionManager()
        agents = manager.list_agents()
        assert "random" in agents
        assert "heuristic" in agents
        assert "decision_tf" in agents
        assert "alphazero_belief" in agents
