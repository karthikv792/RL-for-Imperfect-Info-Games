from __future__ import annotations
from enum import IntEnum
import numpy as np


class Occupant(IntEnum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    CORNER = 3


CORNERS: tuple[tuple[int, int], ...] = ((0, 0), (0, 9), (9, 0), (9, 9))

_LAYOUT_ROWS = [
    ["", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", ""],
    ["6C", "5C", "4C", "3C", "2C", "AH", "KH", "QH", "10H", "10S"],
    ["7C", "AS", "2D", "3D", "4D", "5D", "6D", "7D", "9H", "QS"],
    ["8C", "KS", "6C", "5C", "4C", "3C", "2C", "8D", "8H", "KS"],
    ["9C", "QS", "7C", "6H", "5H", "4H", "AH", "9D", "7H", "AS"],
    ["10C", "10S", "8C", "7H", "2H", "3H", "KH", "10D", "6H", "2D"],
    ["QC", "9S", "9C", "8H", "9H", "10H", "QH", "QD", "5H", "3D"],
    ["KC", "8S", "10C", "QC", "KC", "AC", "AD", "KD", "4H", "4D"],
    ["AC", "7S", "6S", "5S", "4S", "3S", "2S", "2H", "3H", "5D"],
    ["", "AD", "KD", "QD", "10D", "9D", "8D", "7D", "6D", ""],
]

BOARD_LAYOUT: np.ndarray = np.array(_LAYOUT_ROWS, dtype=object)
