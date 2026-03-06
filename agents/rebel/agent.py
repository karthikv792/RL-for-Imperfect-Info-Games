from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_tokenizer import BoardTokenizer
from agents.rebel.cfr import run_cfr


class RebelAgent(Agent):
    """ReBeL-style agent using CFR + DeBERTa value network."""

    def __init__(
        self,
        name: str = "rebel_deberta",
        model: torch.nn.Module | None = None,
        tokenizer: BoardTokenizer | None = None,
        num_cfr_iterations: int = 50,
        max_depth: int = 3,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer or BoardTokenizer()
        self.num_cfr_iterations = num_cfr_iterations
        self.max_depth = max_depth

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.model is not None:
            self.model.to(self.device)

    def _value_fn(self, state: GameState) -> float:
        tokens, mask = self.tokenizer.tokenize_with_mask(
            state, player_perspective=state.current_player
        )
        t = torch.from_numpy(tokens).unsqueeze(0).long().to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).long().to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, value = self.model(t, m)
        return value.cpu().item()

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        strategy = run_cfr(
            state=state,
            value_fn=self._value_fn,
            num_iterations=self.num_cfr_iterations,
            max_depth=self.max_depth,
        )
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]
        best_idx = max(legal_indices, key=lambda i: strategy[i])
        return index_to_action(best_idx)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load(self, path: str) -> None:
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
