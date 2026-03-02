import numpy as np
import pytest
from engine.board import Occupant, CORNERS, BOARD_LAYOUT
from engine.actions import Action, ActionType
from engine.deck import Card
from engine.rules import (
    find_sequences,
    check_win,
    get_legal_actions,
    initial_occupancy,
)


class TestInitialOccupancy:
    def test_shape(self):
        occ = initial_occupancy()
        assert occ.shape == (10, 10)

    def test_corners_marked(self):
        occ = initial_occupancy()
        for r, c in CORNERS:
            assert occ[r][c] == Occupant.CORNER

    def test_non_corners_empty(self):
        occ = initial_occupancy()
        for r in range(10):
            for c in range(10):
                if (r, c) not in CORNERS:
                    assert occ[r][c] == Occupant.EMPTY


class TestFindSequences:
    def test_no_sequences_empty_board(self):
        occ = initial_occupancy()
        assert find_sequences(occ, Occupant.PLAYER1) == []

    def test_horizontal_sequence(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_vertical_sequence(self):
        occ = initial_occupancy()
        for r in range(1, 6):
            occ[r][3] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_diagonal_sequence(self):
        occ = initial_occupancy()
        for i in range(5):
            occ[1 + i][1 + i] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_corner_counts_for_both_players(self):
        occ = initial_occupancy()
        for c in range(1, 5):
            occ[0][c] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1


class TestCheckWin:
    def test_no_win_empty_board(self):
        occ = initial_occupancy()
        assert check_win(occ, Occupant.PLAYER1) is False

    def test_one_sequence_not_win(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        assert check_win(occ, Occupant.PLAYER1) is False

    def test_two_sequences_is_win(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        for c in range(5):
            occ[3][c] = Occupant.PLAYER1
        assert check_win(occ, Occupant.PLAYER1) is True


class TestGetLegalActions:
    def test_empty_board_with_matching_card(self):
        occ = initial_occupancy()
        hand = [Card.from_str("2S")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        assert any(a.row == 0 and a.col == 1 for a in place_actions)

    def test_two_eyed_jack_can_go_anywhere_empty(self):
        occ = initial_occupancy()
        hand = [Card.from_str("J2C")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        wild_actions = [a for a in actions if a.action_type == ActionType.WILD]
        empty_count = sum(
            1 for r in range(10) for c in range(10)
            if occ[r][c] == Occupant.EMPTY
        )
        assert len(wild_actions) == empty_count

    def test_one_eyed_jack_removes_opponent(self):
        occ = initial_occupancy()
        occ[1][0] = Occupant.PLAYER2
        hand = [Card.from_str("J1S")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        remove_actions = [a for a in actions if a.action_type == ActionType.REMOVE]
        assert any(a.row == 1 and a.col == 0 for a in remove_actions)

    def test_cannot_place_on_occupied(self):
        occ = initial_occupancy()
        occ[0][1] = Occupant.PLAYER2
        hand = [Card.from_str("2S")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        assert not any(a.row == 0 and a.col == 1 for a in place_actions)

    def test_no_legal_actions_returns_empty(self):
        occ = initial_occupancy()
        hand = []
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        assert actions == []
