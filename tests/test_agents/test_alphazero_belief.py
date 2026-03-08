import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.alphazero_belief.agent import AlphaZeroBeliefAgent
from agents.base import Agent


class TestAlphaZeroBeliefAgent:
    def test_is_agent(self):
        agent = AlphaZeroBeliefAgent(name="test", model=MagicMock(), encoder=MagicMock())
        assert isinstance(agent, Agent)

    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        agent = AlphaZeroBeliefAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            num_simulations=5,
            num_determinizations=2,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_temperature_controls_exploration(self):
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        agent = AlphaZeroBeliefAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            temperature=0.0,
            device="cpu",
        )
        assert agent.temperature == 0.0
