from __future__ import annotations
import numpy as np
from engine.board import BOARD_LAYOUT, CORNERS, Occupant
from engine.actions import Action, ActionType
from engine.deck import Card

# Precompute card -> board positions mapping
_CARD_POSITIONS: dict[str, list[tuple[int, int]]] = {}
for _r in range(10):
    for _c in range(10):
        _card_str = BOARD_LAYOUT[_r][_c]
        if _card_str:
            _CARD_POSITIONS.setdefault(_card_str, []).append((_r, _c))

_CORNERS_SET = set(CORNERS)


def initial_occupancy() -> np.ndarray:
    occ = np.full((10, 10), Occupant.EMPTY, dtype=np.int8)
    for r, c in CORNERS:
        occ[r][c] = Occupant.CORNER
    return occ


def find_sequences(
    occupancy: np.ndarray,
    player: Occupant,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    sequences: list[tuple[tuple[int, int], tuple[int, int]]] = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    def matches(r: int, c: int) -> bool:
        if not (0 <= r < 10 and 0 <= c < 10):
            return False
        return occupancy[r][c] == player or occupancy[r][c] == Occupant.CORNER

    for dr, dc in directions:
        for r in range(10):
            for c in range(10):
                cells = [(r + i * dr, c + i * dc) for i in range(5)]
                if all(matches(cr, cc) for cr, cc in cells):
                    start, end = cells[0], cells[-1]
                    sequences.append((start, end))
    return sequences


def check_win(occupancy: np.ndarray, player: Occupant) -> bool:
    seqs = find_sequences(occupancy, player)
    if len(seqs) < 2:
        return False
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            cells_i = _sequence_cells(seqs[i])
            cells_j = _sequence_cells(seqs[j])
            non_corner_i = {c for c in cells_i if c not in _CORNERS_SET}
            non_corner_j = {c for c in cells_j if c not in _CORNERS_SET}
            if non_corner_i != non_corner_j:
                return True
    return False


def _sequence_cells(
    seq: tuple[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = seq
    dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
    dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
    return [(r1 + i * dr, c1 + i * dc) for i in range(5)]


def get_legal_actions(
    occupancy: np.ndarray,
    hand: list[Card],
    player: Occupant,
) -> list[Action]:
    actions: list[Action] = []
    has_one_eyed_jack = any(c.is_one_eyed_jack for c in hand)
    has_two_eyed_jack = any(c.is_two_eyed_jack for c in hand)
    opponent = Occupant.PLAYER2 if player == Occupant.PLAYER1 else Occupant.PLAYER1

    hand_card_strs = {str(c) for c in hand if not c.is_one_eyed_jack and not c.is_two_eyed_jack}

    for r in range(10):
        for c in range(10):
            cell_card = BOARD_LAYOUT[r][c]
            if not cell_card:
                continue

            if occupancy[r][c] == Occupant.EMPTY:
                if cell_card in hand_card_strs:
                    actions.append(Action(row=r, col=c, action_type=ActionType.PLACE))
                if has_two_eyed_jack:
                    actions.append(Action(row=r, col=c, action_type=ActionType.WILD))
            elif occupancy[r][c] == opponent:
                if has_one_eyed_jack:
                    actions.append(Action(row=r, col=c, action_type=ActionType.REMOVE))

    return actions
