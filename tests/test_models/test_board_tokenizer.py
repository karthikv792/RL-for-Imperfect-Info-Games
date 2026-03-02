import numpy as np
import pytest
from engine.game_state import GameState
from models.board_tokenizer import BoardTokenizer


class TestBoardTokenizer:
    def test_tokenize_shape(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens = tokenizer.tokenize(gs, player_perspective=1)
        assert tokens.shape == (317,)
        assert tokens.dtype == np.int32

    def test_tokenize_attention_mask(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens, mask = tokenizer.tokenize_with_mask(gs, player_perspective=1)
        assert mask.shape == tokens.shape
        assert mask.dtype == np.int32
        assert mask.sum() > 0

    def test_batch_tokenize(self):
        states = [GameState.new_game(seed=i) for i in range(4)]
        tokenizer = BoardTokenizer()
        batch_tokens, batch_masks = tokenizer.batch_tokenize(
            states, player_perspectives=[1, 2, 1, 2]
        )
        assert batch_tokens.shape == (4, 317)
        assert batch_masks.shape == (4, 317)

    def test_corner_cells_encoded_correctly(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens = tokenizer.tokenize(gs, player_perspective=1)
        assert tokens[0] == 0  # no suit
        assert tokens[1] == 0  # no rank
        assert tokens[2] == 3  # CORNER occupant

    def test_hand_cards_appended(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens = tokenizer.tokenize(gs, player_perspective=1)
        hand_section = tokens[300:315]
        assert hand_section.sum() > 0
