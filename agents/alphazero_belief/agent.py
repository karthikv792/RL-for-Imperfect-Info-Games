from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_encoder import BoardEncoder
from agents.ismcts.mcts import mcts_search
from agents.ismcts.ismcts import determinize


class AlphaZeroBeliefAgent(Agent):
    """AlphaZero-style agent with belief sampling for imperfect information."""

    def __init__(
        self,
        name: str = "alphazero_belief",
        model: torch.nn.Module | None = None,
        encoder: BoardEncoder | None = None,
        num_simulations: int = 200,
        num_determinizations: int = 20,
        cpuct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.encoder = encoder or BoardEncoder()
        self.num_simulations = num_simulations
        self.num_determinizations = num_determinizations
        self.cpuct = cpuct
        self.temperature = temperature
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
        current_player = state.current_player
        aggregated_visits: dict[int, int] = {}

        for d in range(self.num_determinizations):
            det_state = determinize(info_set, current_player, seed=d * 1000)
            _, visit_counts = mcts_search(
                state=det_state,
                policy_value_fn=self._policy_value_fn,
                num_simulations=self.num_simulations,
                cpuct=self.cpuct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
            )
            for action, visits in visit_counts.items():
                idx = action_to_index(action)
                aggregated_visits[idx] = aggregated_visits.get(idx, 0) + visits

        # Build policy from visit counts
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]

        if self.temperature == 0.0:
            # Greedy
            best_idx = max(legal_indices, key=lambda i: aggregated_visits.get(i, 0))
        else:
            # Temperature-scaled
            visits = np.array([aggregated_visits.get(i, 0) for i in legal_indices], dtype=np.float64)
            if self.temperature != 1.0:
                visits = visits ** (1.0 / self.temperature)
            total = visits.sum()
            if total > 0:
                probs = visits / total
            else:
                probs = np.ones(len(legal_indices)) / len(legal_indices)
            chosen = np.random.choice(len(legal_indices), p=probs)
            best_idx = legal_indices[chosen]

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
