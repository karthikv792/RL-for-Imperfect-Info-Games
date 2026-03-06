import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestRESTEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_list_agents(self, client):
        response = client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert "random" in data["agents"]
        assert "heuristic" in data["agents"]

    def test_create_game(self, client):
        response = client.post("/api/game", json={
            "agent": "random",
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "state" in data

    def test_get_game_state(self, client):
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        sid = create.json()["session_id"]
        response = client.get(f"/api/game/{sid}")
        assert response.status_code == 200
        data = response.json()
        assert "board" in data
        assert "hand" in data

    def test_make_move(self, client):
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        sid = create.json()["session_id"]
        state = create.json()["state"]
        legal = state["legal_moves"]
        assert len(legal) > 0
        move = legal[0]
        response = client.post(f"/api/game/{sid}/move", json={
            "position": move["position"],
            "type": move["type"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["move_result"]["success"]
        assert "agent_move" in data

    def test_nonexistent_game(self, client):
        response = client.get("/api/game/nonexistent")
        assert response.status_code == 404
