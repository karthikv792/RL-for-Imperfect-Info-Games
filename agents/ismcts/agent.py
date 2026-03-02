# agents/ismcts/agent.py
from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action
from models.board_encoder import BoardEncoder
from agents.ismcts.ismcts import ismcts_search


class ISMCTSAgent(Agent):
    """ISMCTS agent using a ViT policy/value network."""

    def __init__(
        self,
        name: str = "ismcts_vit",
        model: torch.nn.Module | None = None,
        encoder: BoardEncoder | None = None,
        num_determinizations: int = 20,
        num_simulations: int = 100,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.encoder = encoder or BoardEncoder(target_size=224)
        self.num_determinizations = num_determinizations
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

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

    def _policy_value_fn(self, state: GameState) -> tuple[np.ndarray, float]:
        tensor = self.encoder.encode_torch(state, player_perspective=state.current_player)
        tensor = tensor.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(tensor)
        return policy.cpu().numpy()[0], value.cpu().item()

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        return ismcts_search(
            info_set=info_set,
            player=state.current_player,
            policy_value_fn=self._policy_value_fn,
            num_determinizations=self.num_determinizations,
            num_simulations=self.num_simulations,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )

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
