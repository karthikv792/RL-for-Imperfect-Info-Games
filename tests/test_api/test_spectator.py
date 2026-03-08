# tests/test_api/test_spectator.py
from api.spectator import SpectatorManager


class TestSpectatorManager:
    def test_start_match(self):
        manager = SpectatorManager()
        match_id = manager.start_match(
            agent1_name="random",
            agent2_name="heuristic",
            seed=42,
        )
        assert match_id is not None
        assert manager.get_match(match_id) is not None

    def test_advance_match(self):
        manager = SpectatorManager()
        match_id = manager.start_match("random", "heuristic", seed=42)
        result = manager.advance_match(match_id)
        assert "board" in result or "error" in result

    def test_list_matches(self):
        manager = SpectatorManager()
        manager.start_match("random", "heuristic", seed=42)
        matches = manager.list_matches()
        assert len(matches) >= 1
