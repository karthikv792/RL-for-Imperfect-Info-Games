from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import random


class Suit(IntEnum):
    CLUBS = 0
    SPADES = 1
    HEARTS = 2
    DIAMONDS = 3


SUIT_CHARS = {Suit.CLUBS: "C", Suit.SPADES: "S", Suit.HEARTS: "H", Suit.DIAMONDS: "D"}
CHAR_TO_SUIT = {v: k for k, v in SUIT_CHARS.items()}


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    QUEEN = 12
    KING = 13
    ACE = 14
    ONE_EYED_JACK = 11
    TWO_EYED_JACK = 15


RANK_STRS = {
    Rank.TWO: "2", Rank.THREE: "3", Rank.FOUR: "4", Rank.FIVE: "5",
    Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9",
    Rank.TEN: "10", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
    Rank.ONE_EYED_JACK: "J1", Rank.TWO_EYED_JACK: "J2",
}
STR_TO_RANK = {v: k for k, v in RANK_STRS.items()}


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{RANK_STRS[self.rank]}{SUIT_CHARS[self.suit]}"

    @classmethod
    def from_str(cls, s: str) -> Card:
        if s.startswith("J1"):
            return cls(suit=CHAR_TO_SUIT[s[2]], rank=Rank.ONE_EYED_JACK)
        if s.startswith("J2"):
            return cls(suit=CHAR_TO_SUIT[s[2]], rank=Rank.TWO_EYED_JACK)
        if s.startswith("10"):
            return cls(suit=CHAR_TO_SUIT[s[2]], rank=Rank.TEN)
        return cls(suit=CHAR_TO_SUIT[s[1]], rank=STR_TO_RANK[s[0]])

    @property
    def is_one_eyed_jack(self) -> bool:
        return self.rank == Rank.ONE_EYED_JACK

    @property
    def is_two_eyed_jack(self) -> bool:
        return self.rank == Rank.TWO_EYED_JACK


def _build_sequence_deck() -> tuple[Card, ...]:
    cards: list[Card] = []
    for _ in range(2):
        for suit in Suit:
            for rank in Rank:
                if rank == Rank.ONE_EYED_JACK and suit not in (Suit.SPADES, Suit.HEARTS):
                    continue
                if rank == Rank.TWO_EYED_JACK and suit not in (Suit.CLUBS, Suit.DIAMONDS):
                    continue
                cards.append(Card(suit=suit, rank=rank))
    return tuple(cards)


SEQUENCE_DECK: tuple[Card, ...] = _build_sequence_deck()


class Deck:
    def __init__(self, seed: int | None = None):
        self.cards: list[Card] = list(SEQUENCE_DECK)
        rng = random.Random(seed)
        rng.shuffle(self.cards)

    def draw(self) -> Card:
        return self.cards.pop()

    @property
    def remaining(self) -> int:
        return len(self.cards)
