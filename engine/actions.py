from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

NUM_POSITIONS = 100
ACTION_SPACE_SIZE = 300


class ActionType(IntEnum):
    PLACE = 0
    REMOVE = 1
    WILD = 2


@dataclass(frozen=True)
class Action:
    row: int
    col: int
    action_type: ActionType


def action_to_index(action: Action) -> int:
    return int(action.action_type) * NUM_POSITIONS + action.row * 10 + action.col


def index_to_action(index: int) -> Action:
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"Action index must be 0-299, got {index}")
    action_type = ActionType(index // NUM_POSITIONS)
    position = index % NUM_POSITIONS
    row, col = divmod(position, 10)
    return Action(row=row, col=col, action_type=action_type)
