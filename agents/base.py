# agents/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from engine.game_state import GameState, InformationSet
from engine.actions import Action


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        ...

    def train(self, training_data) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
