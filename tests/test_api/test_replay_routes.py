# tests/test_api/test_replay_routes.py
from fastapi.testclient import TestClient
from api.main import app


class TestReplayRoutes:
    def test_list_replays_empty(self):
        client = TestClient(app)
        resp = client.get("/api/replays")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_replay_not_found(self):
        client = TestClient(app)
        resp = client.get("/api/replays/nonexistent")
        assert resp.status_code == 404
