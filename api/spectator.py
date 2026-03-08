# api/spectator.py
from __future__ import annotations
import uuid
from engine.game_state import GameState
from engine.board import BOARD_LAYOUT
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

SPECTATOR_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
}


class SpectatorMatch:
    def __init__(self, agent1: Agent, agent2: Agent, seed: int | None = None):
        self.match_id = str(uuid.uuid4())
        self.agents = {1: agent1, 2: agent2}
        self.game_state = GameState.new_game(seed=seed)
        self.move_count = 0

    @property
    def is_over(self) -> bool:
        return self.game_state.is_terminal()

    def advance(self) -> dict:
        if self.is_over:
            return {"done": True, "winner": self.game_state.get_winner()}
        current = self.game_state.current_player
        agent = self.agents[current]
        info_set = self.game_state.to_information_set(current)
        action = agent.select_action(self.game_state, info_set)
        self.game_state = self.game_state.apply_action(action)
        self.move_count += 1
        return {
            "move": {"player": current, "position": [action.row, action.col], "type": action.action_type.name.lower()},
            "board": [[{"card": BOARD_LAYOUT[r][c], "occupant": int(self.game_state.occupancy[r][c])} for c in range(10)] for r in range(10)],
            "move_count": self.move_count,
            "is_over": self.is_over,
            "winner": self.game_state.get_winner(),
        }


class SpectatorManager:
    def __init__(self):
        self.matches: dict[str, SpectatorMatch] = {}

    def start_match(self, agent1_name: str, agent2_name: str, seed: int | None = None) -> str:
        a1_cls = SPECTATOR_AGENTS.get(agent1_name)
        a2_cls = SPECTATOR_AGENTS.get(agent2_name)
        if not a1_cls or not a2_cls:
            raise ValueError(f"Unknown agent")
        a1 = a1_cls(name=agent1_name, seed=seed) if agent1_name == "random" else a1_cls(name=agent1_name)
        a2 = a2_cls(name=agent2_name, seed=(seed or 0) + 1) if agent2_name == "random" else a2_cls(name=agent2_name)
        match = SpectatorMatch(a1, a2, seed=seed)
        self.matches[match.match_id] = match
        return match.match_id

    def get_match(self, match_id: str) -> SpectatorMatch | None:
        return self.matches.get(match_id)

    def advance_match(self, match_id: str) -> dict:
        match = self.matches.get(match_id)
        if not match:
            return {"error": "Match not found"}
        return match.advance()

    def list_matches(self) -> list[dict]:
        return [
            {"match_id": m.match_id, "move_count": m.move_count, "is_over": m.is_over}
            for m in self.matches.values()
        ]
