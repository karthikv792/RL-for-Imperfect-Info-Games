from __future__ import annotations
import numpy as np
from engine.game_state import GameState
from engine.board import BOARD_LAYOUT, Occupant

SUIT_ID = {"C": 1, "S": 2, "H": 3, "D": 4}
RANK_ID = {
    "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7,
    "9": 8, "10": 9, "Q": 10, "K": 11, "A": 12, "J1": 13, "J2": 14,
}
OCCUPANT_ID = {
    Occupant.EMPTY: 0, Occupant.PLAYER1: 1,
    Occupant.PLAYER2: 2, Occupant.CORNER: 3,
}

SUIT_VOCAB = 5
RANK_VOCAB = 15
OCCUPANT_VOCAB = 4
METADATA_VOCAB = 105
SEQ_LEN = 317


class BoardTokenizer:
    """Convert GameState into integer token sequences for DeBERTa input."""

    def _encode_card_str(self, card_str: str) -> tuple[int, int]:
        if not card_str:
            return 0, 0
        suit_char = card_str[-1]
        rank_str = card_str[:-1]
        return SUIT_ID.get(suit_char, 0), RANK_ID.get(rank_str, 0)

    def tokenize(self, state: GameState, player_perspective: int) -> np.ndarray:
        tokens = np.zeros(SEQ_LEN, dtype=np.int32)
        me = Occupant.PLAYER1 if player_perspective == 1 else Occupant.PLAYER2
        opp = Occupant.PLAYER2 if player_perspective == 1 else Occupant.PLAYER1

        for r in range(10):
            for c in range(10):
                idx = (r * 10 + c) * 3
                card_str = BOARD_LAYOUT[r][c]
                s_id, r_id = self._encode_card_str(card_str)
                tokens[idx] = s_id
                tokens[idx + 1] = r_id
                occ = state.occupancy[r][c]
                if occ == me:
                    tokens[idx + 2] = 1
                elif occ == opp:
                    tokens[idx + 2] = 2
                else:
                    tokens[idx + 2] = OCCUPANT_ID.get(Occupant(occ), 0)

        hand = state.hands[player_perspective]
        for i, card_str in enumerate(hand):
            idx = 300 + i * 3
            s_id, r_id = self._encode_card_str(card_str)
            tokens[idx] = s_id
            tokens[idx + 1] = r_id
            tokens[idx + 2] = 1

        tokens[315] = min(state.deck_size, 104)
        tokens[316] = len(state.discard_pile) % 105

        return tokens

    def tokenize_with_mask(
        self, state: GameState, player_perspective: int
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenize(state, player_perspective)
        mask = np.ones(SEQ_LEN, dtype=np.int32)
        return tokens, mask

    def batch_tokenize(
        self,
        states: list[GameState],
        player_perspectives: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens_list = []
        masks_list = []
        for s, p in zip(states, player_perspectives):
            t, m = self.tokenize_with_mask(s, p)
            tokens_list.append(t)
            masks_list.append(m)
        return np.stack(tokens_list), np.stack(masks_list)
