import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestFullGameViaREST:
    def test_play_complete_game(self):
        client = TestClient(app)

        # Create game
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        assert create.status_code == 200
        sid = create.json()["session_id"]

        # Play up to 200 moves (should finish)
        for _ in range(200):
            state = client.get(f"/api/game/{sid}").json()
            if state["is_over"]:
                break

            if state["turn"] != "human":
                break

            legal = state["legal_moves"]
            if not legal:
                break

            move = legal[0]
            response = client.post(f"/api/game/{sid}/move", json={
                "position": move["position"],
                "type": move["type"],
            })
            assert response.status_code == 200

        # Game should have completed
        final = client.get(f"/api/game/{sid}").json()
        assert final["is_over"]
