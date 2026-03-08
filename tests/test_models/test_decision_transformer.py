# tests/test_models/test_decision_transformer.py
import torch
import pytest
from models.decision_transformer import DecisionTransformer


class TestDecisionTransformer:
    @pytest.fixture
    def model(self):
        return DecisionTransformer(
            state_dim=317,
            action_dim=300,
            max_timestep=500,
            context_length=20,
            hidden_size=256,
            n_layer=4,
            n_head=4,
        )

    def test_forward_shapes(self, model):
        B, T = 2, 10
        states = torch.randint(0, 10, (B, T, 317))
        actions = torch.randint(0, 300, (B, T))
        returns_to_go = torch.randn(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)

        action_preds, value_preds = model(states, actions, returns_to_go, timesteps)
        assert action_preds.shape == (B, T, 300)
        assert value_preds.shape == (B, T, 1)

    def test_action_prediction_is_distribution(self, model):
        B, T = 1, 5
        states = torch.randint(0, 10, (B, T, 317))
        actions = torch.randint(0, 300, (B, T))
        returns_to_go = torch.ones(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0)

        action_preds, _ = model(states, actions, returns_to_go, timesteps)
        probs = action_preds[0, -1]  # last timestep
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_get_action(self, model):
        import numpy as np
        states = [torch.randint(0, 10, (317,)) for _ in range(3)]
        actions = [0, 5]
        returns_to_go = [1.0, 0.8, 0.6]
        timesteps = [0, 1, 2]

        action_probs = model.get_action(states, actions, returns_to_go, timesteps)
        assert action_probs.shape == (300,)
        assert abs(action_probs.sum().item() - 1.0) < 1e-5
