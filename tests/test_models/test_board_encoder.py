# tests/test_models/test_board_encoder.py
import numpy as np
import torch
import pytest
from engine.game_state import GameState
from models.board_encoder import BoardEncoder


class TestBoardEncoder:
    def test_encode_shape(self):
        gs = GameState.new_game(seed=42)
        encoder = BoardEncoder(target_size=224)
        tensor = encoder.encode(gs, player_perspective=1)
        assert tensor.shape == (22, 224, 224)

    def test_encode_dtype(self):
        gs = GameState.new_game(seed=42)
        encoder = BoardEncoder(target_size=224)
        tensor = encoder.encode(gs, player_perspective=1)
        assert tensor.dtype == np.float32

    def test_encode_to_torch(self):
        gs = GameState.new_game(seed=42)
        encoder = BoardEncoder(target_size=224)
        t = encoder.encode_torch(gs, player_perspective=1)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (22, 224, 224)

    def test_batch_encode(self):
        states = [GameState.new_game(seed=i) for i in range(4)]
        encoder = BoardEncoder(target_size=224)
        batch = encoder.batch_encode(states, player_perspectives=[1, 2, 1, 2])
        assert batch.shape == (4, 22, 224, 224)
