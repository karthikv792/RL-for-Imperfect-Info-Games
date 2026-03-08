import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.decision_tf.agent import DecisionTFAgent
from agents.base import Agent


class TestDecisionTFAgent:
    def test_is_agent(self):
        agent = DecisionTFAgent(name="test", model=MagicMock(), tokenizer=MagicMock())
        assert isinstance(agent, Agent)

    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        mock_model = MagicMock()
        mock_model.get_action.return_value = torch.ones(300) / 300
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = np.zeros(317, dtype=np.int32)

        agent = DecisionTFAgent(
            name="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            target_return=1.0,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_reset_clears_history(self):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        agent = DecisionTFAgent(name="test", model=mock_model, device="cpu")
        agent._state_history.append(torch.zeros(317))
        agent.reset()
        assert len(agent._state_history) == 0
