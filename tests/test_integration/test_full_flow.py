# tests/test_integration/test_full_flow.py
from fastapi.testclient import TestClient
from api.main import app


class TestFullFlow:
    def test_create_game_play_move(self):
        """End-to-end: create game, verify state, play a move."""
        client = TestClient(app)

        # List agents
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert "random" in agents

        # Create game
        resp = client.post("/api/game", json={"agent": "random", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["session_id"]
        assert "state" in data

        # Get game state
        resp = client.get(f"/api/game/{session_id}")
        assert resp.status_code == 200
        state = resp.json()
        assert state["turn"] == "human"
        assert len(state["hand"]) > 0
        assert len(state["legal_moves"]) > 0

        # Make a move
        move = state["legal_moves"][0]
        resp = client.post(f"/api/game/{session_id}/move", json={
            "position": move["position"],
            "type": move["type"],
        })
        assert resp.status_code == 200

    def test_spectator_flow(self):
        """End-to-end: start spectator match, advance 5 moves."""
        client = TestClient(app)

        # Start match
        resp = client.post("/api/spectate?agent1=random&agent2=heuristic&seed=42")
        assert resp.status_code == 200
        match_id = resp.json()["match_id"]

        # Advance 5 times
        for _ in range(5):
            resp = client.post(f"/api/spectate/{match_id}/advance")
            assert resp.status_code == 200

    def test_models_endpoint(self):
        """Models endpoint returns a list."""
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_leaderboard_endpoint(self):
        """Leaderboard endpoint returns data."""
        client = TestClient(app)
        resp = client.get("/api/leaderboard")
        assert resp.status_code == 200
        assert "leaderboard" in resp.json()
