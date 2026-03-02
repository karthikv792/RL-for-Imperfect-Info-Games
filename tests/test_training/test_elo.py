# tests/test_training/test_elo.py
import pytest
from training.elo import EloRating


class TestEloRating:
    def test_initial_rating(self):
        elo = EloRating()
        elo.add_player("agent_a")
        assert elo.get_rating("agent_a") == 1500

    def test_winner_gains_rating(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner="a")
        assert elo.get_rating("a") > 1500
        assert elo.get_rating("b") < 1500

    def test_draw_equal_players(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner=None)
        # Draw between equal players shouldn't change ratings much
        assert abs(elo.get_rating("a") - 1500) < 1

    def test_leaderboard(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner="a")
        board = elo.leaderboard()
        assert board[0][0] == "a"
        assert board[1][0] == "b"
