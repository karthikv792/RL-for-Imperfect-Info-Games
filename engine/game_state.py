from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from engine.board import BOARD_LAYOUT, CORNERS, Occupant
from engine.deck import Card, Deck
from engine.actions import Action, ActionType
from engine.rules import (
    initial_occupancy,
    get_legal_actions,
    check_win,
    find_sequences,
    _sequence_cells,
)


@dataclass(frozen=True)
class InformationSet:
    occupancy: np.ndarray
    current_player: int
    own_hand: tuple[str, ...]
    deck_size: int
    discard_pile: tuple[str, ...]
    sequences: dict[int, tuple]

    def __eq__(self, other):
        if not isinstance(other, InformationSet):
            return False
        return (
            np.array_equal(self.occupancy, other.occupancy)
            and self.current_player == other.current_player
            and self.own_hand == other.own_hand
            and self.deck_size == other.deck_size
            and self.discard_pile == other.discard_pile
        )

    def __hash__(self):
        return hash((
            self.occupancy.tobytes(),
            self.current_player,
            self.own_hand,
            self.deck_size,
            self.discard_pile,
        ))


class GameState:
    __slots__ = (
        "occupancy", "current_player", "hands", "deck_cards",
        "discard_pile", "sequences",
    )

    def __init__(
        self,
        occupancy: np.ndarray,
        current_player: int,
        hands: dict[int, tuple[str, ...]],
        deck_cards: tuple[str, ...],
        discard_pile: tuple[str, ...],
        sequences: dict[int, tuple],
    ):
        self.occupancy = occupancy
        self.occupancy.flags.writeable = False
        self.current_player = current_player
        self.hands = hands
        self.deck_cards = deck_cards
        self.discard_pile = discard_pile
        self.sequences = sequences

    @classmethod
    def new_game(cls, seed: int | None = None, cards_per_hand: int = 5) -> GameState:
        deck = Deck(seed=seed)
        hands: dict[int, list[str]] = {1: [], 2: []}
        for _ in range(cards_per_hand):
            for p in (1, 2):
                hands[p].append(str(deck.draw()))
        return cls(
            occupancy=initial_occupancy(),
            current_player=1,
            hands={p: tuple(h) for p, h in hands.items()},
            deck_cards=tuple(str(c) for c in deck.cards),
            discard_pile=(),
            sequences={1: (), 2: ()},
        )

    @property
    def deck_size(self) -> int:
        return len(self.deck_cards)

    def get_legal_actions(self) -> list[Action]:
        hand_cards = [Card.from_str(s) for s in self.hands[self.current_player]]
        player_occ = Occupant.PLAYER1 if self.current_player == 1 else Occupant.PLAYER2
        return get_legal_actions(self.occupancy, hand_cards, player_occ)

    def apply_action(self, action: Action) -> GameState:
        new_occ = self.occupancy.copy()
        player_occ = Occupant.PLAYER1 if self.current_player == 1 else Occupant.PLAYER2
        hand = list(self.hands[self.current_player])
        discarded_card: str

        if action.action_type == ActionType.PLACE:
            new_occ[action.row][action.col] = player_occ
            card_str = BOARD_LAYOUT[action.row][action.col]
            hand.remove(card_str)
            discarded_card = card_str
        elif action.action_type == ActionType.REMOVE:
            new_occ[action.row][action.col] = Occupant.EMPTY
            for i, c in enumerate(hand):
                if Card.from_str(c).is_one_eyed_jack:
                    discarded_card = hand.pop(i)
                    break
        elif action.action_type == ActionType.WILD:
            new_occ[action.row][action.col] = player_occ
            for i, c in enumerate(hand):
                if Card.from_str(c).is_two_eyed_jack:
                    discarded_card = hand.pop(i)
                    break

        # Draw from deck
        deck_list = list(self.deck_cards)
        new_discard = self.discard_pile + (discarded_card,)
        if deck_list:
            drawn = deck_list.pop()
            hand.append(drawn)
        else:
            import random
            deck_list = list(new_discard[:-1])
            random.shuffle(deck_list)
            new_discard = (discarded_card,)
            if deck_list:
                drawn = deck_list.pop()
                hand.append(drawn)

        new_hands = dict(self.hands)
        new_hands[self.current_player] = tuple(hand)

        new_seqs = dict(self.sequences)
        found = find_sequences(new_occ, player_occ)
        new_seqs[self.current_player] = tuple(found)

        next_player = 2 if self.current_player == 1 else 1

        return GameState(
            occupancy=new_occ,
            current_player=next_player,
            hands=new_hands,
            deck_cards=tuple(deck_list),
            discard_pile=new_discard,
            sequences=new_seqs,
        )

    def is_terminal(self) -> bool:
        for p in (1, 2):
            occ = Occupant.PLAYER1 if p == 1 else Occupant.PLAYER2
            if check_win(self.occupancy, occ):
                return True
        if not self.get_legal_actions():
            return True
        return False

    def get_winner(self) -> Optional[int]:
        for p in (1, 2):
            occ = Occupant.PLAYER1 if p == 1 else Occupant.PLAYER2
            if check_win(self.occupancy, occ):
                return p
        return None

    def to_information_set(self, player: int) -> InformationSet:
        return InformationSet(
            occupancy=self.occupancy,
            current_player=self.current_player,
            own_hand=self.hands[player],
            deck_size=self.deck_size,
            discard_pile=self.discard_pile,
            sequences=self.sequences,
        )

    def to_tensor(self, player_perspective: int) -> np.ndarray:
        tensor = np.zeros((22, 10, 10), dtype=np.float32)

        me = Occupant.PLAYER1 if player_perspective == 1 else Occupant.PLAYER2
        opp = Occupant.PLAYER2 if player_perspective == 1 else Occupant.PLAYER1

        tensor[0] = (self.occupancy == me).astype(np.float32)
        tensor[1] = (self.occupancy == opp).astype(np.float32)
        tensor[2] = (self.occupancy == Occupant.CORNER).astype(np.float32)

        suit_map = {"C": 3, "S": 4, "H": 5, "D": 6}
        rank_map = {
            "2": 7, "3": 8, "4": 9, "5": 10, "6": 11, "7": 12,
            "8": 13, "9": 14, "10": 15, "Q": 16, "K": 17, "A": 18,
        }

        for r in range(10):
            for c in range(10):
                card_str = BOARD_LAYOUT[r][c]
                if not card_str:
                    continue
                suit_char = card_str[-1]
                rank_str = card_str[:-1]
                if suit_char in suit_map:
                    tensor[suit_map[suit_char]][r][c] = 1.0
                if rank_str in rank_map:
                    tensor[rank_map[rank_str]][r][c] = 1.0

        hand_cards = [Card.from_str(s) for s in self.hands[player_perspective]]
        legal = get_legal_actions(self.occupancy, hand_cards, me)
        for a in legal:
            if a.action_type in (ActionType.PLACE, ActionType.WILD):
                tensor[20][a.row][a.col] = 1.0

        my_seqs = find_sequences(self.occupancy, me)
        for seq in my_seqs:
            for r, c in _sequence_cells(seq):
                tensor[21][r][c] = 1.0

        return tensor
