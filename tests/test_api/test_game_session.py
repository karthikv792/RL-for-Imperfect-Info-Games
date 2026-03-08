from api.game_session import GameSession, GameSessionManager
from agents.random_agent import RandomAgent


class TestGameSession:
    def test_create_session(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        assert session.game_state is not None
        assert not session.is_over

    def test_human_move(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        legal = session.get_legal_moves()
        assert len(legal) > 0
        move = legal[0]
        result = session.apply_human_move(move["position"], move["type"])
        assert result["success"]

    def test_agent_responds_after_human(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        legal = session.get_legal_moves()
        session.apply_human_move(legal[0]["position"], legal[0]["type"])
        agent_result = session.get_agent_move()
        assert "action" in agent_result
        assert "thinking_time_ms" in agent_result

    def test_get_state_dict(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        state = session.to_dict()
        assert "board" in state
        assert "hand" in state
        assert "legal_moves" in state
        assert "turn" in state
        assert "sequences" in state


class TestGameSessionManager:
    def test_create_and_get(self):
        manager = GameSessionManager()
        sid = manager.create_session(agent_name="random", seed=42)
        session = manager.get_session(sid)
        assert session is not None

    def test_list_agents(self):
        manager = GameSessionManager()
        agents = manager.list_agents()
        assert "random" in agents
        assert "heuristic" in agents
