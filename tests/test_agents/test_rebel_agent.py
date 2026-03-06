import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.rebel.agent import RebelAgent
from agents.base import Agent


class TestRebelAgent:
    def test_is_agent(self):
        agent = RebelAgent(name="test", model=MagicMock(), tokenizer=MagicMock())
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

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize_with_mask.return_value = (
            np.zeros(317, dtype=np.int32),
            np.ones(317, dtype=np.int32),
        )

        agent = RebelAgent(
            name="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            num_cfr_iterations=3,
            max_depth=1,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_save_and_load(self, tmp_path):
        from models.deberta_policy_value import DeBERTaPolicyValue
        from models.board_tokenizer import BoardTokenizer

        model = DeBERTaPolicyValue(num_actions=300, seq_len=317)
        tokenizer = BoardTokenizer()
        agent = RebelAgent(name="test", model=model, tokenizer=tokenizer, device="cpu")

        save_path = str(tmp_path / "rebel_agent")
        agent.save(save_path)
        agent.load(save_path)
