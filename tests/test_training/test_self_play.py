# tests/test_training/test_self_play.py
from unittest.mock import MagicMock
import torch
from training.self_play import run_self_play_game, SelfPlayConfig


class TestSelfPlayGame:
    def test_generates_training_examples(self):
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        config = SelfPlayConfig(
            num_simulations=5,
            num_determinizations=2,
            cpuct=1.5,
        )

        examples = run_self_play_game(
            model=mock_model,
            encoder=mock_encoder,
            config=config,
            seed=42,
            device="cpu",
        )
        assert len(examples) > 0
        state, policy, value = examples[0]
        assert state.shape == (22, 10, 10)
        assert policy.shape == (300,)
        assert isinstance(value, float)
