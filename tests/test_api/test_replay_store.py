import pytest
from api.replay_store import ReplayStore


class TestReplayStore:
    def test_save_and_load(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        replay_id = store.save_replay(
            agent1="human",
            agent2="heuristic",
            winner="heuristic",
            moves=[
                {"player": 1, "position": [1, 2], "type": "place"},
                {"player": 2, "position": [3, 4], "type": "place"},
            ],
            seed=42,
        )
        replay = store.get_replay(replay_id)
        assert replay is not None
        assert replay["winner"] == "heuristic"
        assert len(replay["moves"]) == 2

    def test_list_replays(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        store.save_replay("human", "random", "human", [], seed=1)
        store.save_replay("human", "heuristic", "heuristic", [], seed=2)
        replays = store.list_replays(limit=10)
        assert len(replays) == 2

    def test_list_by_agent(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        store.save_replay("human", "random", "human", [], seed=1)
        store.save_replay("human", "heuristic", "heuristic", [], seed=2)
        replays = store.list_replays(agent="random")
        assert len(replays) == 1
