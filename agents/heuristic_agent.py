# agents/heuristic_agent.py
from __future__ import annotations
import math
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, ActionType
from engine.board import Occupant


class HeuristicAgent(Agent):
    """Manhattan-distance heuristic: place tokens near existing tokens."""

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        legal = state.get_legal_actions()
        me = Occupant.PLAYER1 if state.current_player == 1 else Occupant.PLAYER2

        my_positions = [
            (r, c) for r in range(10) for c in range(10)
            if state.occupancy[r][c] == me
        ]

        if not my_positions:
            # No tokens yet — pick first place action, or any action
            place_actions = [a for a in legal if a.action_type != ActionType.REMOVE]
            return place_actions[0] if place_actions else legal[0]

        best_action = legal[0]
        best_score = math.inf

        for action in legal:
            min_dist = min(
                abs(action.row - r) + abs(action.col - c) for r, c in my_positions
            )
            if action.action_type == ActionType.REMOVE:
                min_dist += 0.5  # slight penalty for remove vs place
            if min_dist < best_score:
                best_score = min_dist
                best_action = action

        return best_action
