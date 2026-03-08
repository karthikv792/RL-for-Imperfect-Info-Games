# tests/test_agents/test_ismcts_agent.py
from unittest.mock import MagicMock
import torch
from engine.game_state import GameState
from agents.ismcts.agent import ISMCTSAgent
from agents.base import Agent


class TestISMCTSAgent:
    def test_is_agent(self):
        agent = ISMCTSAgent(name="test", model=MagicMock(), encoder=MagicMock())
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

        agent = ISMCTSAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            num_determinizations=2,
            num_simulations=5,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_save_and_load(self, tmp_path):
        from models.vit_policy_value import ViTPolicyValue
        from models.board_encoder import BoardEncoder

        model = ViTPolicyValue(num_actions=300, num_input_channels=22)
        encoder = BoardEncoder(target_size=224)
        agent = ISMCTSAgent(name="test", model=model, encoder=encoder, device="cpu")

        save_path = str(tmp_path / "agent")
        agent.save(save_path)
        agent.load(save_path)
