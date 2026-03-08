from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_tokenizer import BoardTokenizer


class DecisionTFAgent(Agent):
    """Decision Transformer agent — treats RL as sequence modeling."""

    def __init__(
        self,
        name: str = "decision_tf",
        model: torch.nn.Module | None = None,
        tokenizer: BoardTokenizer | None = None,
        target_return: float = 1.0,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer or BoardTokenizer()
        self.target_return = target_return

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

        # Episode history
        self._state_history: list[torch.Tensor] = []
        self._action_history: list[int] = []
        self._return_history: list[float] = []
        self._timestep = 0

    def reset(self):
        """Reset episode history for a new game."""
        self._state_history.clear()
        self._action_history.clear()
        self._return_history.clear()
        self._timestep = 0

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        tokens = self.tokenizer.tokenize(state, player_perspective=state.current_player)
        state_tensor = torch.from_numpy(tokens).long()
        self._state_history.append(state_tensor)

        # Compute return-to-go (decreasing from target)
        rtg = self.target_return - self._timestep * 0.002
        self._return_history.append(rtg)

        # Get action distribution from model
        action_probs = self.model.get_action(
            states=self._state_history,
            actions=self._action_history,
            returns_to_go=self._return_history,
            timesteps=list(range(len(self._state_history))),
        )

        # Mask to legal actions
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]
        masked = torch.zeros(300)
        for idx in legal_indices:
            masked[idx] = action_probs[idx]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            for idx in legal_indices:
                masked[idx] = 1.0 / len(legal_indices)

        best_idx = masked.argmax().item()
        self._action_history.append(best_idx)
        self._timestep += 1
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
