# tests/test_models/test_vit_model.py
import torch
import pytest
from models.vit_policy_value import ViTPolicyValue


class TestViTPolicyValue:
    @pytest.fixture
    def model(self):
        return ViTPolicyValue(
            backbone_name="google/vit-base-patch16-224",
            num_actions=300,
            policy_hidden=512,
            value_hidden=256,
            num_input_channels=22,
        )

    def test_forward_shapes(self, model):
        x = torch.randn(2, 22, 224, 224)
        policy, value = model(x)
        assert policy.shape == (2, 300)
        assert value.shape == (2, 1)

    def test_policy_sums_to_one(self, model):
        x = torch.randn(1, 22, 224, 224)
        policy, _ = model(x)
        total = policy.sum(dim=1)
        assert torch.allclose(total, torch.ones(1), atol=1e-5)

    def test_value_in_range(self, model):
        x = torch.randn(1, 22, 224, 224)
        _, value = model(x)
        assert -1.0 <= value.item() <= 1.0

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.policy_head.parameters():
            assert param.requires_grad
        for param in model.value_head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad
