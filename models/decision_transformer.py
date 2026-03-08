# models/decision_transformer.py
from __future__ import annotations
import torch
import torch.nn as nn
import math


class DecisionTransformer(nn.Module):
    """Decision Transformer for Sequence game using causal GPT-like architecture."""

    def __init__(
        self,
        state_dim: int = 317,
        action_dim: int = 300,
        max_timestep: int = 500,
        context_length: int = 20,
        hidden_size: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.context_length = context_length

        # Embeddings for each modality
        self.state_embedding = nn.Sequential(
            nn.Embedding(120, 32),  # token embedding
            nn.Flatten(start_dim=-2),  # (B, T, 317*32)
            nn.Linear(state_dim * 32, hidden_size),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(action_dim + 1, hidden_size)  # +1 for padding
        self.return_embedding = nn.Linear(1, hidden_size)
        self.timestep_embedding = nn.Embedding(max_timestep + 1, hidden_size)

        # Transformer backbone (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Layer norm
        self.ln = nn.LayerNorm(hidden_size)

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,      # (B, T, 317) int
        actions: torch.Tensor,      # (B, T) int
        returns_to_go: torch.Tensor,  # (B, T, 1) float
        timesteps: torch.Tensor,    # (B, T) int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = states.shape
        device = states.device

        # Embed each modality
        state_emb = self.state_embedding(states)        # (B, T, H)
        action_emb = self.action_embedding(actions)     # (B, T, H)
        return_emb = self.return_embedding(returns_to_go)  # (B, T, H)
        time_emb = self.timestep_embedding(timesteps)   # (B, T, H)

        # Interleave: [R1, S1, A1, R2, S2, A2, ...]
        # Each timestep contributes 3 tokens
        seq_len = T * 3
        tokens = torch.zeros(B, seq_len, self.hidden_size, device=device)
        tokens[:, 0::3] = return_emb + time_emb
        tokens[:, 1::3] = state_emb + time_emb
        tokens[:, 2::3] = action_emb + time_emb

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

        # Transformer forward
        hidden = self.transformer(tokens, mask=causal_mask)
        hidden = self.ln(hidden)

        # Extract state position outputs (positions 1, 4, 7, ...)
        state_hidden = hidden[:, 1::3]  # (B, T, H)

        action_preds = self.action_head(state_hidden)  # (B, T, 300)
        value_preds = self.value_head(state_hidden)    # (B, T, 1)

        return action_preds, value_preds

    @torch.no_grad()
    def get_action(
        self,
        states: list[torch.Tensor],
        actions: list[int],
        returns_to_go: list[float],
        timesteps: list[int],
    ) -> torch.Tensor:
        """Get action distribution for the current (last) timestep."""
        self.eval()
        device = next(self.parameters()).device

        T = len(states)
        s = torch.stack(states).unsqueeze(0).to(device)  # (1, T, 317)
        # Pad actions to length T (last action is dummy)
        a_list = actions + [0] * (T - len(actions))
        a = torch.tensor(a_list[:T], dtype=torch.long).unsqueeze(0).to(device)
        r = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        t = torch.tensor(timesteps, dtype=torch.long).unsqueeze(0).to(device)

        action_preds, _ = self(s, a, r, t)
        return action_preds[0, -1].cpu()  # (300,)
