# agents/random_agent.py
from __future__ import annotations
import random
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action


class RandomAgent(Agent):
    def __init__(self, name: str = "random", seed: int | None = None):
        super().__init__(name)
        self.rng = random.Random(seed)

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        legal = state.get_legal_actions()
        return self.rng.choice(legal)
