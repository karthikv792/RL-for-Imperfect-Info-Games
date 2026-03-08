# tests/test_training/test_hub_publisher.py
from training.hub_publisher import create_model_card


class TestModelCard:
    def test_create_card(self):
        card = create_model_card(
            model_name="sequence-ai-vit",
            architecture="ViT + MCTS",
            description="Vision Transformer policy-value network for Sequence",
            metrics={"elo": 1200, "win_rate_vs_random": 0.85},
        )
        assert "sequence-ai-vit" in card
        assert "ViT + MCTS" in card
        assert "1200" in card

    def test_card_has_required_sections(self):
        card = create_model_card(
            model_name="test",
            architecture="test",
            description="test",
            metrics={},
        )
        assert "## Model Description" in card
        assert "## Training" in card
        assert "## Usage" in card
