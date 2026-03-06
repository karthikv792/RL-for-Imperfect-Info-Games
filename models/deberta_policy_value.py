from __future__ import annotations
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config
from models.board_tokenizer import SUIT_VOCAB, RANK_VOCAB, OCCUPANT_VOCAB, METADATA_VOCAB


class DeBERTaPolicyValue(nn.Module):
    """DeBERTa backbone with custom policy and value heads."""

    def __init__(
        self,
        backbone_name: str = "microsoft/deberta-v3-small",
        num_actions: int = 300,
        seq_len: int = 317,
        policy_hidden: int = 512,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.backbone = DebertaV2Model.from_pretrained(backbone_name)
        self.backbone = self.backbone.float()  # ensure float32 for custom embeddings
        hidden_size = self.backbone.config.hidden_size

        max_vocab = max(SUIT_VOCAB, RANK_VOCAB, OCCUPANT_VOCAB, METADATA_VOCAB)
        self.token_embedding = nn.Embedding(max_vocab + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(seq_len, hidden_size)

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, num_actions),
            nn.Softmax(dim=-1),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(pos_ids)
        embeddings = tok_emb + pos_emb

        outputs = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state

        cls_token = hidden[:, 0]
        policy = self.policy_head(cls_token)
        value = self.value_head(cls_token)
        return policy, value

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
