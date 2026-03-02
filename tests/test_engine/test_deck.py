import pytest
from engine.deck import Card, Suit, Rank, Deck, SEQUENCE_DECK


class TestCard:
    def test_card_creation(self):
        c = Card(suit=Suit.HEARTS, rank=Rank.ACE)
        assert c.suit == Suit.HEARTS
        assert c.rank == Rank.ACE

    def test_card_str(self):
        c = Card(suit=Suit.SPADES, rank=Rank.TEN)
        assert str(c) == "10S"

    def test_card_from_str(self):
        c = Card.from_str("10S")
        assert c.suit == Suit.SPADES
        assert c.rank == Rank.TEN

    def test_card_from_str_ace(self):
        c = Card.from_str("AH")
        assert c.rank == Rank.ACE
        assert c.suit == Suit.HEARTS

    def test_one_eyed_jack(self):
        c = Card.from_str("J1S")
        assert c.rank == Rank.ONE_EYED_JACK
        assert c.is_one_eyed_jack

    def test_two_eyed_jack(self):
        c = Card.from_str("J2C")
        assert c.rank == Rank.TWO_EYED_JACK
        assert c.is_two_eyed_jack

    def test_card_frozen(self):
        c = Card(suit=Suit.CLUBS, rank=Rank.KING)
        with pytest.raises(AttributeError):
            c.suit = Suit.HEARTS


class TestDeck:
    def test_full_deck_size(self):
        d = Deck()
        assert len(d.cards) == 104

    def test_deck_shuffle_changes_order(self):
        d1 = Deck(seed=42)
        d2 = Deck(seed=99)
        assert d1.cards != d2.cards

    def test_draw_removes_card(self):
        d = Deck(seed=42)
        initial_size = len(d.cards)
        card = d.draw()
        assert len(d.cards) == initial_size - 1
        assert isinstance(card, Card)

    def test_draw_empty_raises(self):
        d = Deck()
        for _ in range(104):
            d.draw()
        with pytest.raises(IndexError):
            d.draw()

    def test_remaining(self):
        d = Deck()
        assert d.remaining == 104
        d.draw()
        assert d.remaining == 103


class TestSequenceDeck:
    def test_deck_constant_size(self):
        assert len(SEQUENCE_DECK) == 104

    def test_deck_has_one_eyed_jacks(self):
        oej = [c for c in SEQUENCE_DECK if c.is_one_eyed_jack]
        assert len(oej) == 4

    def test_deck_has_two_eyed_jacks(self):
        tej = [c for c in SEQUENCE_DECK if c.is_two_eyed_jack]
        assert len(tej) == 4
