import pytest
from api.leaderboard import LeaderboardManager


class TestLeaderboardManager:
    def test_record_result(self):
        lb = LeaderboardManager()
        lb.record_result(agent1="random", agent2="heuristic", winner="heuristic")
        stats = lb.get_stats()
        assert "heuristic" in stats
        assert stats["heuristic"]["wins"] == 1

    def test_head_to_head(self):
        lb = LeaderboardManager()
        lb.record_result("random", "heuristic", "heuristic")
        lb.record_result("random", "heuristic", "random")
        h2h = lb.get_head_to_head("random", "heuristic")
        assert h2h["random"] == 1
        assert h2h["heuristic"] == 1

    def test_elo_ratings(self):
        lb = LeaderboardManager()
        for _ in range(5):
            lb.record_result("random", "heuristic", "heuristic")
        stats = lb.get_stats()
        assert stats["heuristic"]["elo"] > stats["random"]["elo"]
