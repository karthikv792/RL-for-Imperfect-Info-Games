# models/board_encoder.py
from __future__ import annotations
import numpy as np
import torch
from engine.game_state import GameState


class BoardEncoder:
    """Encode GameState as a multi-channel tensor suitable for ViT input."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def encode(self, state: GameState, player_perspective: int) -> np.ndarray:
        raw = state.to_tensor(player_perspective)  # (22, 10, 10)
        # Upscale to target_size x target_size using nearest neighbor
        scale = self.target_size // 10
        upscaled = np.repeat(np.repeat(raw, scale, axis=1), scale, axis=2)
        # Pad if needed to reach exact target size
        pad_h = self.target_size - upscaled.shape[1]
        pad_w = self.target_size - upscaled.shape[2]
        if pad_h > 0 or pad_w > 0:
            upscaled = np.pad(upscaled, ((0, 0), (0, pad_h), (0, pad_w)))
        return upscaled[:, :self.target_size, :self.target_size].astype(np.float32)

    def encode_torch(self, state: GameState, player_perspective: int) -> torch.Tensor:
        return torch.from_numpy(self.encode(state, player_perspective))

    def batch_encode(
        self,
        states: list[GameState],
        player_perspectives: list[int],
    ) -> torch.Tensor:
        tensors = [
            self.encode(s, p) for s, p in zip(states, player_perspectives)
        ]
        return torch.from_numpy(np.stack(tensors))
