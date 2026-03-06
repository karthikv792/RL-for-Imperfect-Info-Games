import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestWebSocket:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_websocket_new_game(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "new_game", "agent": "random", "seed": 42})
            data = ws.receive_json()
            assert data["type"] == "game_state"
            assert "board" in data
            assert "hand" in data

    def test_websocket_make_move(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "new_game", "agent": "random", "seed": 42})
            state = ws.receive_json()
            legal = state["legal_moves"]
            assert len(legal) > 0

            ws.send_json({
                "type": "make_move",
                "position": legal[0]["position"],
                "move_type": legal[0]["type"],
            })
            response = ws.receive_json()
            assert response["type"] in ("game_state", "agent_move", "error")

    def test_websocket_invalid_message(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "unknown_command"})
            data = ws.receive_json()
            assert data["type"] == "error"
