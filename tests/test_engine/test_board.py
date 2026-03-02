import numpy as np
import pytest
from engine.board import BOARD_LAYOUT, CORNERS, Occupant


class TestBoardLayout:
    def test_board_is_10x10(self):
        assert BOARD_LAYOUT.shape == (10, 10)

    def test_corners_are_empty_string(self):
        for r, c in CORNERS:
            assert BOARD_LAYOUT[r][c] == ""

    def test_non_corner_cells_have_cards(self):
        for r in range(10):
            for c in range(10):
                if (r, c) not in CORNERS:
                    assert BOARD_LAYOUT[r][c] != "", f"Cell ({r},{c}) should have a card"

    def test_known_cell_values(self):
        assert BOARD_LAYOUT[0][1] == "2S"
        assert BOARD_LAYOUT[1][0] == "6C"
        assert BOARD_LAYOUT[9][1] == "AD"


class TestOccupant:
    def test_empty(self):
        assert Occupant.EMPTY == 0

    def test_player1(self):
        assert Occupant.PLAYER1 == 1

    def test_player2(self):
        assert Occupant.PLAYER2 == 2

    def test_corner(self):
        assert Occupant.CORNER == 3
