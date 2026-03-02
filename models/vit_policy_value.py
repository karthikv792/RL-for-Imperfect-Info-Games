# models/vit_policy_value.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ViTPolicyValue(nn.Module):
    """ViT backbone with custom policy and value heads for AlphaZero-style training."""

    def __init__(
        self,
        backbone_name: str = "google/vit-base-patch16-224",
        num_actions: int = 300,
        policy_hidden: int = 512,
        value_hidden: int = 256,
        num_input_channels: int = 22,
    ):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for base

        # Replace the patch embedding to accept 22 channels instead of 3
        old_embed = self.backbone.embeddings.patch_embeddings.projection
        self.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=old_embed.out_channels,
            kernel_size=old_embed.kernel_size,
            stride=old_embed.stride,
            padding=old_embed.padding,
        )
        # Update the stored num_channels so the forward-pass shape check passes
        self.backbone.embeddings.patch_embeddings.num_channels = num_input_channels
        self.backbone.config.num_channels = num_input_channels

        # Policy head: [CLS] -> action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, num_actions),
            nn.Softmax(dim=-1),
        )

        # Value head: [CLS] -> scalar value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        policy = self.policy_head(cls_token)
        value = self.value_head(cls_token)
        return policy, value

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
