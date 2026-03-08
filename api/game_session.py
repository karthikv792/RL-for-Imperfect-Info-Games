from __future__ import annotations
import time
import uuid
from engine.game_state import GameState
from engine.actions import Action, ActionType
from engine.board import BOARD_LAYOUT, Occupant
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.decision_tf.agent import DecisionTFAgent
from agents.alphazero_belief.agent import AlphaZeroBeliefAgent


AVAILABLE_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "decision_tf": DecisionTFAgent,
    "alphazero_belief": AlphaZeroBeliefAgent,
}


class GameSession:
    """Manages a single human-vs-AI game."""

    def __init__(
        self,
        agent: Agent,
        seed: int | None = None,
        human_player: int = 1,
    ):
        self.agent = agent
        self.human_player = human_player
        self.ai_player = 2 if human_player == 1 else 1
        self.game_state = GameState.new_game(seed=seed)
        self.move_history: list[dict] = []
        self.session_id = str(uuid.uuid4())

    @property
    def is_over(self) -> bool:
        return self.game_state.is_terminal()

    @property
    def is_human_turn(self) -> bool:
        return self.game_state.current_player == self.human_player

    def get_legal_moves(self) -> list[dict]:
        if not self.is_human_turn or self.is_over:
            return []
        actions = self.game_state.get_legal_actions()
        moves = []
        for a in actions:
            moves.append({
                "position": [a.row, a.col],
                "type": a.action_type.name.lower(),
            })
        return moves

    def apply_human_move(self, position: list[int], move_type: str) -> dict:
        if not self.is_human_turn or self.is_over:
            return {"success": False, "error": "Not your turn"}

        action_type = ActionType[move_type.upper()]
        action = Action(row=position[0], col=position[1], action_type=action_type)

        legal = self.game_state.get_legal_actions()
        if action not in legal:
            return {"success": False, "error": "Illegal move"}

        self.game_state = self.game_state.apply_action(action)
        self.move_history.append({
            "player": self.human_player,
            "position": position,
            "type": move_type,
        })
        return {"success": True}

    def get_agent_move(self) -> dict:
        if self.is_human_turn or self.is_over:
            return {"error": "Not AI's turn"}

        start = time.time()
        info_set = self.game_state.to_information_set(self.ai_player)
        action = self.agent.select_action(self.game_state, info_set)
        elapsed_ms = int((time.time() - start) * 1000)

        self.game_state = self.game_state.apply_action(action)
        move_data = {
            "player": self.ai_player,
            "position": [action.row, action.col],
            "type": action.action_type.name.lower(),
        }
        self.move_history.append(move_data)
        return {
            "action": move_data,
            "thinking_time_ms": elapsed_ms,
        }

    def to_dict(self) -> dict:
        board = []
        for r in range(10):
            row = []
            for c in range(10):
                occ = int(self.game_state.occupancy[r][c])
                card = BOARD_LAYOUT[r][c]
                row.append({"card": card, "occupant": occ})
            board.append(row)

        return {
            "board": board,
            "hand": list(self.game_state.hands[self.human_player]),
            "legal_moves": self.get_legal_moves(),
            "turn": "human" if self.is_human_turn else "ai",
            "current_player": self.game_state.current_player,
            "sequences": {
                str(k): list(v) for k, v in self.game_state.sequences.items()
            },
            "deck_size": self.game_state.deck_size,
            "is_over": self.is_over,
            "winner": self.game_state.get_winner(),
            "move_history": self.move_history,
        }


class GameSessionManager:
    """Manages multiple concurrent game sessions."""

    def __init__(self):
        self.sessions: dict[str, GameSession] = {}

    def create_session(
        self,
        agent_name: str = "random",
        seed: int | None = None,
        human_player: int = 1,
    ) -> str:
        agent_cls = AVAILABLE_AGENTS.get(agent_name)
        if agent_cls is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        if agent_name == "random":
            agent = agent_cls(name=agent_name, seed=seed)
        else:
            agent = agent_cls(name=agent_name)

        session = GameSession(agent=agent, seed=seed, human_player=human_player)
        self.sessions[session.session_id] = session
        return session.session_id

    def get_session(self, session_id: str) -> GameSession | None:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def list_agents(self) -> list[str]:
        return list(AVAILABLE_AGENTS.keys())
