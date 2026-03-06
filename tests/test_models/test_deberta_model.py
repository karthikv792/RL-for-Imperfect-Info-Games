import torch
import pytest
from models.deberta_policy_value import DeBERTaPolicyValue


class TestDeBERTaPolicyValue:
    @pytest.fixture
    def model(self):
        return DeBERTaPolicyValue(
            backbone_name="microsoft/deberta-v3-small",
            num_actions=300,
            seq_len=317,
        )

    def test_forward_shapes(self, model):
        tokens = torch.randint(0, 10, (2, 317))
        mask = torch.ones(2, 317, dtype=torch.long)
        policy, value = model(tokens, mask)
        assert policy.shape == (2, 300)
        assert value.shape == (2, 1)

    def test_policy_sums_to_one(self, model):
        tokens = torch.randint(0, 10, (1, 317))
        mask = torch.ones(1, 317, dtype=torch.long)
        policy, _ = model(tokens, mask)
        total = policy.sum(dim=1)
        assert torch.allclose(total, torch.ones(1), atol=1e-5)

    def test_value_in_range(self, model):
        tokens = torch.randint(0, 10, (1, 317))
        mask = torch.ones(1, 317, dtype=torch.long)
        _, value = model(tokens, mask)
        assert -1.0 <= value.item() <= 1.0

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.policy_head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad
