# Sequence AI Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Sequence board game engine and build an ISMCTS+ViT AI agent from scratch, with a training pipeline and web interface for human play.

**Architecture:** Monorepo with clean separation — `engine/` for pure game logic, `agents/` for AI, `models/` for HuggingFace Transformers, `training/` for self-play pipeline, `api/` for FastAPI backend, `webapp/` for React frontend. Everything is TDD with immutable game state.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers/Trainer/Accelerate, FastAPI, WebSockets, React/Next.js, pytest, Weights & Biases.

**Existing Code Reference:** The old code lives in `Sequence/` and `agents/` directories. The board layout at `Sequence/board.py:16-25` is the canonical 10x10 card grid. The action encoding at `agents/NeuralNetworkAgent/__init__.py:173-196` (0-99 place, 100-199 remove, 200-299 wild) is reused.

---

## Phase 1A: Game Engine (Track A)

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `engine/__init__.py`
- Create: `engine/actions.py` (empty placeholder)
- Create: `agents/__init__.py` (new, clean version)
- Create: `models/__init__.py`
- Create: `training/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_engine/__init__.py`
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_models/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "sequence-ai"
version = "0.1.0"
description = "Professional Sequence board game with state-of-the-art AI agents"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "torch>=2.1",
    "transformers>=4.36",
    "accelerate>=0.25",
    "safetensors>=0.4",
    "wandb>=0.16",
    "pyyaml>=6.0",
    "fastapi>=0.108",
    "uvicorn>=0.25",
    "websockets>=12.0",
    "pydantic>=2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-benchmark>=4.0",
    "ruff>=0.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
target-version = "py311"
line-length = 100
```

**Step 2: Create directory structure**

```bash
mkdir -p engine agents/ismcts agents/rebel models training/configs tests/test_engine tests/test_agents tests/test_models tests/test_training api webapp
touch engine/__init__.py agents/__init__.py models/__init__.py training/__init__.py
touch tests/__init__.py tests/test_engine/__init__.py tests/test_agents/__init__.py tests/test_models/__init__.py tests/test_training/__init__.py
touch agents/ismcts/__init__.py agents/rebel/__init__.py
```

**Step 3: Install in editable mode and verify pytest runs**

```bash
pip install -e ".[dev]"
pytest --co  # should collect 0 tests, no errors
```
Expected: `no tests ran`

**Step 4: Commit**

```bash
git add pyproject.toml engine/ agents/ models/ training/ tests/ api/ webapp/
git commit -m "feat: scaffold project structure and dependencies"
```

---

### Task 2: Action types and encoding

**Files:**
- Create: `engine/actions.py`
- Create: `tests/test_engine/test_actions.py`

**Step 1: Write the failing tests**

```python
# tests/test_engine/test_actions.py
import pytest
from engine.actions import Action, ActionType, action_to_index, index_to_action


class TestAction:
    def test_place_action_creation(self):
        a = Action(row=3, col=5, action_type=ActionType.PLACE)
        assert a.row == 3
        assert a.col == 5
        assert a.action_type == ActionType.PLACE

    def test_remove_action_creation(self):
        a = Action(row=7, col=2, action_type=ActionType.REMOVE)
        assert a.action_type == ActionType.REMOVE

    def test_wild_action_creation(self):
        a = Action(row=0, col=1, action_type=ActionType.WILD)
        assert a.action_type == ActionType.WILD


class TestActionEncoding:
    def test_place_to_index(self):
        a = Action(row=3, col=5, action_type=ActionType.PLACE)
        assert action_to_index(a) == 35  # 3*10+5

    def test_remove_to_index(self):
        a = Action(row=7, col=2, action_type=ActionType.REMOVE)
        assert action_to_index(a) == 172  # 100 + 7*10+2

    def test_wild_to_index(self):
        a = Action(row=0, col=1, action_type=ActionType.WILD)
        assert action_to_index(a) == 201  # 200 + 0*10+1

    def test_index_to_place(self):
        a = index_to_action(35)
        assert a.row == 3
        assert a.col == 5
        assert a.action_type == ActionType.PLACE

    def test_index_to_remove(self):
        a = index_to_action(172)
        assert a.row == 7
        assert a.col == 2
        assert a.action_type == ActionType.REMOVE

    def test_index_to_wild(self):
        a = index_to_action(201)
        assert a.row == 0
        assert a.col == 1
        assert a.action_type == ActionType.WILD

    def test_roundtrip_all_actions(self):
        for i in range(300):
            a = index_to_action(i)
            assert action_to_index(a) == i

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            index_to_action(300)
        with pytest.raises(ValueError):
            index_to_action(-1)

    def test_action_is_frozen(self):
        a = Action(row=0, col=0, action_type=ActionType.PLACE)
        with pytest.raises(AttributeError):
            a.row = 1
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_engine/test_actions.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.actions'`

**Step 3: Implement engine/actions.py**

```python
# engine/actions.py
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

NUM_POSITIONS = 100  # 10x10 board
ACTION_SPACE_SIZE = 300  # 100 place + 100 remove + 100 wild


class ActionType(IntEnum):
    PLACE = 0    # Place token using matching card
    REMOVE = 1   # Remove opponent token using one-eyed Jack
    WILD = 2     # Place token using two-eyed Jack (wild)


@dataclass(frozen=True)
class Action:
    row: int
    col: int
    action_type: ActionType


def action_to_index(action: Action) -> int:
    return int(action.action_type) * NUM_POSITIONS + action.row * 10 + action.col


def index_to_action(index: int) -> Action:
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"Action index must be 0-299, got {index}")
    action_type = ActionType(index // NUM_POSITIONS)
    position = index % NUM_POSITIONS
    row, col = divmod(position, 10)
    return Action(row=row, col=col, action_type=action_type)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_engine/test_actions.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/actions.py tests/test_engine/test_actions.py
git commit -m "feat: add Action types and index encoding"
```

---

### Task 3: Card deck module

**Files:**
- Create: `engine/deck.py`
- Create: `tests/test_engine/test_deck.py`

**Step 1: Write the failing tests**

```python
# tests/test_engine/test_deck.py
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
        """Sequence uses 2 standard decks = 104 cards, minus 8 jacks + 8 special jacks"""
        d = Deck()
        assert len(d.cards) == 104

    def test_deck_shuffle_changes_order(self):
        d1 = Deck(seed=42)
        d2 = Deck(seed=99)
        assert d1.cards != d2.cards  # different seeds -> different order

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
        assert len(oej) == 4  # JS and JH from each of 2 decks

    def test_deck_has_two_eyed_jacks(self):
        tej = [c for c in SEQUENCE_DECK if c.is_two_eyed_jack]
        assert len(tej) == 4  # JC and JD from each of 2 decks
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_engine/test_deck.py -v
```
Expected: FAIL

**Step 3: Implement engine/deck.py**

```python
# engine/deck.py
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
    ONE_EYED_JACK = 11   # J1 — removes opponent token
    TWO_EYED_JACK = 15   # J2 — wild placement


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
        # Handle J1X, J2X, 10X, and single-char ranks
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
    """Build the standard Sequence deck: 2 copies of 52 cards with J1/J2 split."""
    cards: list[Card] = []
    for _ in range(2):  # two copies
        for suit in Suit:
            for rank in Rank:
                # One-eyed jacks are only Spades and Hearts
                if rank == Rank.ONE_EYED_JACK and suit not in (Suit.SPADES, Suit.HEARTS):
                    continue
                # Two-eyed jacks are only Clubs and Diamonds
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
```

**Step 4: Run tests**

```bash
pytest tests/test_engine/test_deck.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/deck.py tests/test_engine/test_deck.py
git commit -m "feat: add Card, Suit, Rank, and Deck with full Sequence deck"
```

---

### Task 4: Board layout and representation

**Files:**
- Create: `engine/board.py`
- Create: `tests/test_engine/test_board.py`

**Step 1: Write failing tests**

```python
# tests/test_engine/test_board.py
import numpy as np
import pytest
from engine.board import BOARD_LAYOUT, CORNERS, Occupant


class TestBoardLayout:
    def test_board_is_10x10(self):
        assert BOARD_LAYOUT.shape == (10, 10)

    def test_corners_are_empty_string(self):
        for r, c in CORNERS:
            assert BOARD_LAYOUT[r][c] == ""

    def test_non_corner_cells_have_cards(self):
        for r in range(10):
            for c in range(10):
                if (r, c) not in CORNERS:
                    assert BOARD_LAYOUT[r][c] != "", f"Cell ({r},{c}) should have a card"

    def test_known_cell_values(self):
        """Verify a few known positions from the original board."""
        assert BOARD_LAYOUT[0][1] == "2S"
        assert BOARD_LAYOUT[1][0] == "6C"
        assert BOARD_LAYOUT[9][1] == "AD"


class TestOccupant:
    def test_empty(self):
        assert Occupant.EMPTY == 0

    def test_player1(self):
        assert Occupant.PLAYER1 == 1

    def test_player2(self):
        assert Occupant.PLAYER2 == 2

    def test_corner(self):
        assert Occupant.CORNER == 3
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_engine/test_board.py -v
```

**Step 3: Implement engine/board.py**

```python
# engine/board.py
from __future__ import annotations
from enum import IntEnum
import numpy as np


class Occupant(IntEnum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2
    CORNER = 3


CORNERS: tuple[tuple[int, int], ...] = ((0, 0), (0, 9), (9, 0), (9, 9))

# The canonical Sequence board layout (10x10).
# Corners are "" (free/wild). All other cells have a card string.
# From original code at Sequence/board.py:16-25
_LAYOUT_ROWS = [
    ["", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", ""],
    ["6C", "5C", "4C", "3C", "2C", "AH", "KH", "QH", "10H", "10S"],
    ["7C", "AS", "2D", "3D", "4D", "5D", "6D", "7D", "9H", "QS"],
    ["8C", "KS", "6C", "5C", "4C", "3C", "2C", "8D", "8H", "KS"],
    ["9C", "QS", "7C", "6H", "5H", "4H", "AH", "9D", "7H", "AS"],
    ["10C", "10S", "8C", "7H", "2H", "3H", "KH", "10D", "6H", "2D"],
    ["QC", "9S", "9C", "8H", "9H", "10H", "QH", "QD", "5H", "3D"],
    ["KC", "8S", "10C", "QC", "KC", "AC", "AD", "KD", "4H", "4D"],
    ["AC", "7S", "6S", "5S", "4S", "3S", "2S", "2H", "3H", "5D"],
    ["", "AD", "KD", "QD", "10D", "9D", "8D", "7D", "6D", ""],
]

BOARD_LAYOUT: np.ndarray = np.array(_LAYOUT_ROWS, dtype=object)
```

**Step 4: Run tests**

```bash
pytest tests/test_engine/test_board.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/board.py tests/test_engine/test_board.py
git commit -m "feat: add board layout constants and Occupant enum"
```

---

### Task 5: Win detection and legal move generation (rules module)

**Files:**
- Create: `engine/rules.py`
- Create: `tests/test_engine/test_rules.py`

**Step 1: Write failing tests**

```python
# tests/test_engine/test_rules.py
import numpy as np
import pytest
from engine.board import Occupant, CORNERS, BOARD_LAYOUT
from engine.actions import Action, ActionType
from engine.deck import Card
from engine.rules import (
    find_sequences,
    check_win,
    get_legal_actions,
    initial_occupancy,
)


class TestInitialOccupancy:
    def test_shape(self):
        occ = initial_occupancy()
        assert occ.shape == (10, 10)

    def test_corners_marked(self):
        occ = initial_occupancy()
        for r, c in CORNERS:
            assert occ[r][c] == Occupant.CORNER

    def test_non_corners_empty(self):
        occ = initial_occupancy()
        for r in range(10):
            for c in range(10):
                if (r, c) not in CORNERS:
                    assert occ[r][c] == Occupant.EMPTY


class TestFindSequences:
    def test_no_sequences_empty_board(self):
        occ = initial_occupancy()
        assert find_sequences(occ, Occupant.PLAYER1) == []

    def test_horizontal_sequence(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_vertical_sequence(self):
        occ = initial_occupancy()
        for r in range(1, 6):
            occ[r][3] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_diagonal_sequence(self):
        occ = initial_occupancy()
        for i in range(5):
            occ[1 + i][1 + i] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1

    def test_corner_counts_for_both_players(self):
        """Corners act as wild for any player's sequence."""
        occ = initial_occupancy()
        # Place 4 tokens in row 0 cols 1-4, corner at (0,0) completes it
        for c in range(1, 5):
            occ[0][c] = Occupant.PLAYER1
        seqs = find_sequences(occ, Occupant.PLAYER1)
        assert len(seqs) >= 1


class TestCheckWin:
    def test_no_win_empty_board(self):
        occ = initial_occupancy()
        assert check_win(occ, Occupant.PLAYER1) is False

    def test_one_sequence_not_win(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        assert check_win(occ, Occupant.PLAYER1) is False  # need 2 sequences

    def test_two_sequences_is_win(self):
        occ = initial_occupancy()
        for c in range(5):
            occ[1][c] = Occupant.PLAYER1
        for c in range(5):
            occ[3][c] = Occupant.PLAYER1
        assert check_win(occ, Occupant.PLAYER1) is True


class TestGetLegalActions:
    def test_empty_board_with_matching_card(self):
        occ = initial_occupancy()
        hand = [Card.from_str("2S")]  # 2S is at (0,1)
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        assert any(a.row == 0 and a.col == 1 for a in place_actions)

    def test_two_eyed_jack_can_go_anywhere_empty(self):
        occ = initial_occupancy()
        hand = [Card.from_str("J2C")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        wild_actions = [a for a in actions if a.action_type == ActionType.WILD]
        # Should be all empty non-corner cells
        empty_count = sum(
            1 for r in range(10) for c in range(10)
            if occ[r][c] == Occupant.EMPTY
        )
        assert len(wild_actions) == empty_count

    def test_one_eyed_jack_removes_opponent(self):
        occ = initial_occupancy()
        occ[1][0] = Occupant.PLAYER2  # opponent token at (1,0)
        hand = [Card.from_str("J1S")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        remove_actions = [a for a in actions if a.action_type == ActionType.REMOVE]
        assert any(a.row == 1 and a.col == 0 for a in remove_actions)

    def test_cannot_place_on_occupied(self):
        occ = initial_occupancy()
        occ[0][1] = Occupant.PLAYER2  # (0,1) has a 2S, but occupied
        hand = [Card.from_str("2S")]
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        assert not any(a.row == 0 and a.col == 1 for a in place_actions)

    def test_no_legal_actions_returns_empty(self):
        occ = initial_occupancy()
        hand = []  # no cards
        actions = get_legal_actions(occ, hand, Occupant.PLAYER1)
        assert actions == []
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_engine/test_rules.py -v
```

**Step 3: Implement engine/rules.py**

```python
# engine/rules.py
from __future__ import annotations
import numpy as np
from engine.board import BOARD_LAYOUT, CORNERS, Occupant
from engine.actions import Action, ActionType
from engine.deck import Card

# Precompute card -> board positions mapping
_CARD_POSITIONS: dict[str, list[tuple[int, int]]] = {}
for _r in range(10):
    for _c in range(10):
        _card_str = BOARD_LAYOUT[_r][_c]
        if _card_str:
            _CARD_POSITIONS.setdefault(_card_str, []).append((_r, _c))


def initial_occupancy() -> np.ndarray:
    occ = np.full((10, 10), Occupant.EMPTY, dtype=np.int8)
    for r, c in CORNERS:
        occ[r][c] = Occupant.CORNER
    return occ


def find_sequences(
    occupancy: np.ndarray,
    player: Occupant,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Find all 5-in-a-row sequences for a player. Corners count as matching."""
    sequences: list[tuple[tuple[int, int], tuple[int, int]]] = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag-right, diag-left

    def matches(r: int, c: int) -> bool:
        if not (0 <= r < 10 and 0 <= c < 10):
            return False
        return occupancy[r][c] == player or occupancy[r][c] == Occupant.CORNER

    for dr, dc in directions:
        for r in range(10):
            for c in range(10):
                cells = [(r + i * dr, c + i * dc) for i in range(5)]
                if all(matches(cr, cc) for cr, cc in cells):
                    start, end = cells[0], cells[-1]
                    sequences.append((start, end))
    return sequences


def check_win(occupancy: np.ndarray, player: Occupant) -> bool:
    """A player wins with 2 or more distinct sequences."""
    seqs = find_sequences(occupancy, player)
    if len(seqs) < 2:
        return False
    # Check that at least 2 sequences don't fully overlap
    # Two sequences are distinct if they differ in at least one non-corner cell
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            cells_i = _sequence_cells(seqs[i])
            cells_j = _sequence_cells(seqs[j])
            non_corner_i = {c for c in cells_i if c not in CORNERS}
            non_corner_j = {c for c in cells_j if c not in CORNERS}
            if non_corner_i != non_corner_j:
                return True
    return False


def _sequence_cells(
    seq: tuple[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = seq
    dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
    dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
    return [(r1 + i * dr, c1 + i * dc) for i in range(5)]


def get_legal_actions(
    occupancy: np.ndarray,
    hand: list[Card],
    player: Occupant,
) -> list[Action]:
    """Get all legal actions for a player given board state and hand."""
    actions: list[Action] = []
    has_one_eyed_jack = any(c.is_one_eyed_jack for c in hand)
    has_two_eyed_jack = any(c.is_two_eyed_jack for c in hand)
    opponent = Occupant.PLAYER2 if player == Occupant.PLAYER1 else Occupant.PLAYER1

    # Collect card strings in hand (excluding jacks)
    hand_card_strs = {str(c) for c in hand if not c.is_one_eyed_jack and not c.is_two_eyed_jack}

    for r in range(10):
        for c in range(10):
            cell_card = BOARD_LAYOUT[r][c]
            if not cell_card:  # corner
                continue

            if occupancy[r][c] == Occupant.EMPTY:
                # PLACE: matching card in hand
                if cell_card in hand_card_strs:
                    actions.append(Action(row=r, col=c, action_type=ActionType.PLACE))
                # WILD: two-eyed jack
                if has_two_eyed_jack:
                    actions.append(Action(row=r, col=c, action_type=ActionType.WILD))
            elif occupancy[r][c] == opponent:
                # REMOVE: one-eyed jack
                if has_one_eyed_jack:
                    actions.append(Action(row=r, col=c, action_type=ActionType.REMOVE))

    return actions
```

**Step 4: Run tests**

```bash
pytest tests/test_engine/test_rules.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/rules.py tests/test_engine/test_rules.py
git commit -m "feat: add win detection, sequence finding, and legal move generation"
```

---

### Task 6: Immutable GameState with apply_action

**Files:**
- Create: `engine/game_state.py`
- Create: `tests/test_engine/test_game_state.py`

**Step 1: Write failing tests**

```python
# tests/test_engine/test_game_state.py
import numpy as np
import pytest
from engine.game_state import GameState
from engine.actions import Action, ActionType
from engine.deck import Card
from engine.board import Occupant


class TestGameStateCreation:
    def test_new_game(self):
        gs = GameState.new_game(seed=42)
        assert gs.current_player == 1
        assert len(gs.hands[1]) == 5
        assert len(gs.hands[2]) == 5
        assert gs.deck_size > 0
        assert gs.discard_pile == ()
        assert not gs.is_terminal()

    def test_new_game_deterministic(self):
        gs1 = GameState.new_game(seed=42)
        gs2 = GameState.new_game(seed=42)
        assert gs1.hands[1] == gs2.hands[1]
        assert gs1.hands[2] == gs2.hands[2]


class TestApplyAction:
    def test_place_adds_token(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        place_actions = [a for a in legal if a.action_type == ActionType.PLACE]
        if place_actions:
            action = place_actions[0]
            new_gs = gs.apply_action(action)
            assert new_gs.occupancy[action.row][action.col] == Occupant.PLAYER1
            # Original unchanged (immutability)
            assert gs.occupancy[action.row][action.col] == Occupant.EMPTY

    def test_apply_action_switches_player(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        if legal:
            new_gs = gs.apply_action(legal[0])
            assert new_gs.current_player == 2

    def test_apply_action_draws_card(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        place_actions = [a for a in legal if a.action_type == ActionType.PLACE]
        if place_actions:
            old_hand_size = len(gs.hands[gs.current_player])
            new_gs = gs.apply_action(place_actions[0])
            # Hand size stays the same (play one, draw one)
            assert len(new_gs.hands[1]) == old_hand_size

    def test_apply_action_adds_to_discard(self):
        gs = GameState.new_game(seed=42)
        legal = gs.get_legal_actions()
        if legal:
            new_gs = gs.apply_action(legal[0])
            assert len(new_gs.discard_pile) == 1


class TestTerminalState:
    def test_new_game_not_terminal(self):
        gs = GameState.new_game(seed=42)
        assert not gs.is_terminal()
        assert gs.get_winner() is None


class TestInformationSet:
    def test_info_set_hides_opponent_hand(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        assert info.own_hand == gs.hands[1]
        assert not hasattr(info, "opponent_hand")

    def test_info_set_preserves_public_info(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        assert info.current_player == gs.current_player
        assert info.deck_size == gs.deck_size
        np.testing.assert_array_equal(info.occupancy, gs.occupancy)


class TestTensorEncoding:
    def test_tensor_shape(self):
        gs = GameState.new_game(seed=42)
        tensor = gs.to_tensor(player_perspective=1)
        assert tensor.shape == (22, 10, 10)

    def test_tensor_dtype(self):
        gs = GameState.new_game(seed=42)
        tensor = gs.to_tensor(player_perspective=1)
        assert tensor.dtype == np.float32
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_engine/test_game_state.py -v
```

**Step 3: Implement engine/game_state.py**

```python
# engine/game_state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from copy import deepcopy

from engine.board import BOARD_LAYOUT, CORNERS, Occupant
from engine.deck import Card, Deck, Suit, Rank, SUIT_CHARS, RANK_STRS
from engine.actions import Action, ActionType
from engine.rules import (
    initial_occupancy,
    get_legal_actions,
    check_win,
    find_sequences,
    _CARD_POSITIONS,
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
    """Immutable game state. All mutation returns a new GameState."""

    __slots__ = (
        "occupancy", "current_player", "hands", "deck_cards",
        "discard_pile", "sequences", "_board_cards",
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
        self._board_cards = BOARD_LAYOUT

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
            # Remove one one-eyed jack from hand
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
            # Reshuffle discard pile into deck (minus the just-discarded card)
            import random
            deck_list = list(new_discard[:-1])
            random.shuffle(deck_list)
            new_discard = (discarded_card,)
            if deck_list:
                drawn = deck_list.pop()
                hand.append(drawn)

        new_hands = dict(self.hands)
        new_hands[self.current_player] = tuple(hand)

        # Check for new sequences
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
        # Also terminal if current player has no legal actions
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
        """Encode state as 22-channel 10x10 tensor for neural network input."""
        tensor = np.zeros((22, 10, 10), dtype=np.float32)

        me = Occupant.PLAYER1 if player_perspective == 1 else Occupant.PLAYER2
        opp = Occupant.PLAYER2 if player_perspective == 1 else Occupant.PLAYER1

        # Channel 0: My tokens
        tensor[0] = (self.occupancy == me).astype(np.float32)
        # Channel 1: Opponent tokens
        tensor[1] = (self.occupancy == opp).astype(np.float32)
        # Channel 2: Corners
        tensor[2] = (self.occupancy == Occupant.CORNER).astype(np.float32)

        # Channels 3-6: Card suits (C, S, H, D)
        suit_map = {"C": 3, "S": 4, "H": 5, "D": 6}
        # Channels 7-19: Card ranks (2-10, Q, K, A, J — 13 channels)
        rank_map = {
            "2": 7, "3": 8, "4": 9, "5": 10, "6": 11, "7": 12,
            "8": 13, "9": 14, "10": 15, "Q": 16, "K": 17, "A": 18,
        }
        # Channel 19 unused (could be J but Jacks aren't on board)

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

        # Channel 20: Current player's playable positions
        hand_cards = [Card.from_str(s) for s in self.hands[player_perspective]]
        legal = get_legal_actions(self.occupancy, hand_cards, me)
        for a in legal:
            if a.action_type in (ActionType.PLACE, ActionType.WILD):
                tensor[20][a.row][a.col] = 1.0

        # Channel 21: Sequences formed by current player
        my_seqs = find_sequences(self.occupancy, me)
        for seq in my_seqs:
            from engine.rules import _sequence_cells
            for r, c in _sequence_cells(seq):
                tensor[21][r][c] = 1.0

        return tensor
```

**Step 4: Run tests**

```bash
pytest tests/test_engine/test_game_state.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add engine/game_state.py tests/test_engine/test_game_state.py
git commit -m "feat: add immutable GameState with apply_action, info set, and tensor encoding"
```

---

### Task 7: Full game simulation end-to-end test

**Files:**
- Create: `tests/test_engine/test_game_simulation.py`

**Step 1: Write the test**

```python
# tests/test_engine/test_game_simulation.py
import pytest
from engine.game_state import GameState
from engine.actions import ActionType
import random


class TestFullGameSimulation:
    def test_random_game_completes(self):
        """Play a full game with random moves. Must terminate without error."""
        gs = GameState.new_game(seed=123)
        max_moves = 500
        for _ in range(max_moves):
            if gs.is_terminal():
                break
            legal = gs.get_legal_actions()
            assert len(legal) > 0, "Non-terminal state has no legal actions"
            action = random.choice(legal)
            gs = gs.apply_action(action)

    def test_ten_random_games(self):
        """10 random games all complete."""
        for seed in range(10):
            gs = GameState.new_game(seed=seed)
            for _ in range(500):
                if gs.is_terminal():
                    break
                action = random.choice(gs.get_legal_actions())
                gs = gs.apply_action(action)

    def test_winner_is_valid(self):
        gs = GameState.new_game(seed=42)
        for _ in range(500):
            if gs.is_terminal():
                winner = gs.get_winner()
                assert winner in (None, 1, 2)
                break
            action = random.choice(gs.get_legal_actions())
            gs = gs.apply_action(action)
```

**Step 2: Run test**

```bash
pytest tests/test_engine/test_game_simulation.py -v
```
Expected: ALL PASS (uses already-implemented modules)

**Step 3: Commit**

```bash
git add tests/test_engine/test_game_simulation.py
git commit -m "test: add full game simulation end-to-end tests"
```

---

## Phase 1B: ISMCTS + ViT Agent (Track B)

### Task 8: Agent protocol and baseline agents

**Files:**
- Create: `agents/base.py`
- Create: `agents/random_agent.py`
- Create: `agents/heuristic_agent.py`
- Create: `tests/test_agents/test_baselines.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_baselines.py
import pytest
from engine.game_state import GameState
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


class TestAgentProtocol:
    def test_random_agent_is_agent(self):
        agent = RandomAgent(name="random")
        assert isinstance(agent, Agent)

    def test_heuristic_agent_is_agent(self):
        agent = HeuristicAgent(name="heuristic")
        assert isinstance(agent, Agent)


class TestRandomAgent:
    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        agent = RandomAgent(name="random", seed=42)
        info = gs.to_information_set(player=gs.current_player)
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_plays_full_game(self):
        gs = GameState.new_game(seed=42)
        agents = {1: RandomAgent("r1", seed=1), 2: RandomAgent("r2", seed=2)}
        for _ in range(500):
            if gs.is_terminal():
                break
            info = gs.to_information_set(gs.current_player)
            action = agents[gs.current_player].select_action(gs, info)
            gs = gs.apply_action(action)
        assert gs.is_terminal()


class TestHeuristicAgent:
    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        agent = HeuristicAgent(name="heuristic")
        info = gs.to_information_set(player=gs.current_player)
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_plays_full_game(self):
        gs = GameState.new_game(seed=42)
        agents = {1: HeuristicAgent("h1"), 2: HeuristicAgent("h2")}
        for _ in range(500):
            if gs.is_terminal():
                break
            info = gs.to_information_set(gs.current_player)
            action = agents[gs.current_player].select_action(gs, info)
            gs = gs.apply_action(action)
        assert gs.is_terminal()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_agents/test_baselines.py -v
```

**Step 3: Implement agents**

```python
# agents/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from engine.game_state import GameState, InformationSet
from engine.actions import Action


class Agent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        ...

    def train(self, training_data) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
```

```python
# agents/random_agent.py
from __future__ import annotations
import random
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action


class RandomAgent(Agent):
    def __init__(self, name: str = "random", seed: int | None = None):
        super().__init__(name)
        self.rng = random.Random(seed)

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        legal = state.get_legal_actions()
        return self.rng.choice(legal)
```

```python
# agents/heuristic_agent.py
from __future__ import annotations
import math
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, ActionType
from engine.board import Occupant


class HeuristicAgent(Agent):
    """Manhattan-distance heuristic: place tokens near existing tokens."""

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        legal = state.get_legal_actions()
        me = Occupant.PLAYER1 if state.current_player == 1 else Occupant.PLAYER2

        my_positions = [
            (r, c) for r in range(10) for c in range(10)
            if state.occupancy[r][c] == me
        ]

        if not my_positions:
            # No tokens yet — pick first place action, or any action
            place_actions = [a for a in legal if a.action_type != ActionType.REMOVE]
            return place_actions[0] if place_actions else legal[0]

        best_action = legal[0]
        best_score = math.inf

        for action in legal:
            min_dist = min(
                abs(action.row - r) + abs(action.col - c) for r, c in my_positions
            )
            if action.action_type == ActionType.REMOVE:
                min_dist += 0.5  # slight penalty for remove vs place
            if min_dist < best_score:
                best_score = min_dist
                best_action = action

        return best_action
```

**Step 4: Run tests**

```bash
pytest tests/test_agents/test_baselines.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/base.py agents/random_agent.py agents/heuristic_agent.py tests/test_agents/test_baselines.py
git commit -m "feat: add Agent base class, RandomAgent, and HeuristicAgent"
```

---

### Task 9: Board tensor encoder for ViT

**Files:**
- Create: `models/board_encoder.py`
- Create: `tests/test_models/test_board_encoder.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_models/test_board_encoder.py -v
```

**Step 3: Implement models/board_encoder.py**

```python
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
```

**Step 4: Run tests**

```bash
pytest tests/test_models/test_board_encoder.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add models/board_encoder.py tests/test_models/test_board_encoder.py
git commit -m "feat: add BoardEncoder for ViT-compatible tensor encoding"
```

---

### Task 10: ViT policy/value model with HuggingFace

**Files:**
- Create: `models/vit_policy_value.py`
- Create: `tests/test_models/test_vit_model.py`

**Step 1: Write failing tests**

```python
# tests/test_models/test_vit_model.py
import torch
import pytest
from models.vit_policy_value import ViTPolicyValue


class TestViTPolicyValue:
    @pytest.fixture
    def model(self):
        return ViTPolicyValue(
            backbone_name="google/vit-base-patch16-224",
            num_actions=300,
            policy_hidden=512,
            value_hidden=256,
            num_input_channels=22,
        )

    def test_forward_shapes(self, model):
        x = torch.randn(2, 22, 224, 224)
        policy, value = model(x)
        assert policy.shape == (2, 300)
        assert value.shape == (2, 1)

    def test_policy_sums_to_one(self, model):
        x = torch.randn(1, 22, 224, 224)
        policy, _ = model(x)
        total = policy.sum(dim=1)
        assert torch.allclose(total, torch.ones(1), atol=1e-5)

    def test_value_in_range(self, model):
        x = torch.randn(1, 22, 224, 224)
        _, value = model(x)
        assert -1.0 <= value.item() <= 1.0

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.policy_head.parameters():
            assert param.requires_grad
        for param in model.value_head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_models/test_vit_model.py -v
```

**Step 3: Implement models/vit_policy_value.py**

```python
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
```

**Step 4: Run tests**

```bash
pytest tests/test_models/test_vit_model.py -v
```
Expected: ALL PASS (may take a moment to download ViT weights first time)

**Step 5: Commit**

```bash
git add models/vit_policy_value.py tests/test_models/test_vit_model.py
git commit -m "feat: add ViTPolicyValue model with pretrained backbone and custom heads"
```

---

### Task 11: MCTS core implementation

**Files:**
- Create: `agents/ismcts/mcts.py`
- Create: `tests/test_agents/test_mcts.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_mcts.py
import numpy as np
import pytest
from engine.game_state import GameState
from agents.ismcts.mcts import MCTSNode, mcts_search


class TestMCTSNode:
    def test_root_node(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        assert node.parent is None
        assert node.visit_count == 0
        assert len(node.children) == 0

    def test_expand(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        legal = gs.get_legal_actions()
        prior = np.ones(300) / 300  # uniform prior
        node.expand(legal, prior)
        assert len(node.children) == len(legal)

    def test_select_child_ucb(self):
        gs = GameState.new_game(seed=42)
        node = MCTSNode(state=gs)
        legal = gs.get_legal_actions()
        prior = np.ones(300) / 300
        node.expand(legal, prior)
        node.visit_count = 1
        child = node.select_child(cpuct=1.5)
        assert child is not None


class TestMCTSSearch:
    def test_returns_valid_action(self):
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            """Uniform policy, neutral value."""
            return np.ones(300) / 300, 0.0

        action, visit_counts = mcts_search(
            state=gs,
            policy_value_fn=dummy_policy_value,
            num_simulations=10,
            cpuct=1.5,
        )
        assert action in gs.get_legal_actions()

    def test_visit_counts_sum(self):
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        _, visit_counts = mcts_search(
            state=gs,
            policy_value_fn=dummy_policy_value,
            num_simulations=50,
            cpuct=1.5,
        )
        assert sum(visit_counts.values()) == 50

    def test_more_simulations_better(self):
        """With more simulations, the search should be more confident (higher max visits)."""
        gs = GameState.new_game(seed=42)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        _, vc_low = mcts_search(gs, dummy_policy_value, num_simulations=10, cpuct=1.5)
        _, vc_high = mcts_search(gs, dummy_policy_value, num_simulations=100, cpuct=1.5)
        assert max(vc_high.values()) >= max(vc_low.values())
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_agents/test_mcts.py -v
```

**Step 3: Implement agents/ismcts/mcts.py**

```python
# agents/ismcts/mcts.py
from __future__ import annotations
import math
import numpy as np
from typing import Callable, Optional
from engine.game_state import GameState
from engine.actions import Action, action_to_index, index_to_action


class MCTSNode:
    __slots__ = (
        "state", "parent", "action", "prior", "visit_count",
        "value_sum", "children",
    )

    def __init__(
        self,
        state: GameState,
        parent: Optional[MCTSNode] = None,
        action: Optional[Action] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}  # action_index -> child

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, legal_actions: list[Action], policy: np.ndarray):
        for action in legal_actions:
            idx = action_to_index(action)
            if idx not in self.children:
                child_state = self.state.apply_action(action)
                self.children[idx] = MCTSNode(
                    state=child_state,
                    parent=self,
                    action=action,
                    prior=float(policy[idx]),
                )

    def select_child(self, cpuct: float) -> MCTSNode:
        best_score = -math.inf
        best_child = None
        sqrt_parent = math.sqrt(self.visit_count)

        for child in self.children.values():
            ucb = child.q_value + cpuct * child.prior * sqrt_parent / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.visit_count += 1
            # Flip value at each level since players alternate
            node.value_sum += value
            value = -value
            node = node.parent


PolicyValueFn = Callable[[GameState], tuple[np.ndarray, float]]


def mcts_search(
    state: GameState,
    policy_value_fn: PolicyValueFn,
    num_simulations: int,
    cpuct: float = 1.5,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
) -> tuple[Action, dict[Action, int]]:
    root = MCTSNode(state=state)

    # Expand root
    legal_actions = state.get_legal_actions()
    policy, _ = policy_value_fn(state)

    # Normalize policy to legal actions
    legal_indices = [action_to_index(a) for a in legal_actions]
    mask = np.zeros(300)
    mask[legal_indices] = 1.0
    policy = policy * mask
    policy_sum = policy.sum()
    if policy_sum > 0:
        policy = policy / policy_sum
    else:
        policy[legal_indices] = 1.0 / len(legal_indices)

    # Add Dirichlet noise at root for exploration
    if dirichlet_alpha > 0 and dirichlet_epsilon > 0:
        noise = np.zeros(300)
        noise[legal_indices] = np.random.dirichlet(
            [dirichlet_alpha] * len(legal_indices)
        )
        policy = (1 - dirichlet_epsilon) * policy + dirichlet_epsilon * noise

    root.expand(legal_actions, policy)

    for _ in range(num_simulations):
        node = root

        # SELECT: traverse tree using UCB
        while node.children and not node.state.is_terminal():
            node = node.select_child(cpuct)

        # EVALUATE
        if node.state.is_terminal():
            winner = node.state.get_winner()
            if winner is None:
                value = 0.0
            elif winner == state.current_player:
                value = 1.0
            else:
                value = -1.0
        else:
            # EXPAND and evaluate with neural network
            child_legal = node.state.get_legal_actions()
            if child_legal:
                child_policy, value = policy_value_fn(node.state)
                # Adjust value sign: NN returns value from current player's perspective
                if node.state.current_player != state.current_player:
                    value = -value
                node.expand(child_legal, child_policy)
            else:
                value = 0.0

        # BACKPROPAGATE
        node.backpropagate(value)

    # Extract visit counts
    visit_counts: dict[Action, int] = {}
    for idx, child in root.children.items():
        visit_counts[child.action] = child.visit_count

    # Select action with most visits
    best_action = max(visit_counts, key=visit_counts.get)
    return best_action, visit_counts
```

**Step 4: Run tests**

```bash
pytest tests/test_agents/test_mcts.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/ismcts/mcts.py tests/test_agents/test_mcts.py
git commit -m "feat: add MCTS core with UCB selection, expansion, and backpropagation"
```

---

### Task 12: Information Set MCTS (determinization layer)

**Files:**
- Create: `agents/ismcts/ismcts.py`
- Create: `tests/test_agents/test_ismcts.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_ismcts.py
import numpy as np
import pytest
from engine.game_state import GameState, InformationSet
from agents.ismcts.ismcts import ismcts_search, determinize


class TestDeterminize:
    def test_returns_valid_game_state(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        assert isinstance(det, GameState)
        assert det.current_player == info.current_player

    def test_preserves_own_hand(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        assert det.hands[1] == info.own_hand

    def test_opponent_hand_is_valid(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det = determinize(info, player=1, seed=99)
        # Opponent hand should not contain cards in own hand or on discard
        for card in det.hands[2]:
            assert card not in info.own_hand

    def test_determinize_different_seeds_give_different_hands(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=1)
        det1 = determinize(info, player=1, seed=1)
        det2 = determinize(info, player=1, seed=2)
        # Different seeds should usually give different opponent hands
        # (extremely unlikely to be equal)
        assert det1.hands[2] != det2.hands[2]


class TestISMCTSSearch:
    def test_returns_valid_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        def dummy_policy_value(state):
            return np.ones(300) / 300, 0.0

        action = ismcts_search(
            info_set=info,
            player=gs.current_player,
            policy_value_fn=dummy_policy_value,
            num_determinizations=3,
            num_simulations=5,
            cpuct=1.5,
        )
        assert action in gs.get_legal_actions()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_agents/test_ismcts.py -v
```

**Step 3: Implement agents/ismcts/ismcts.py**

```python
# agents/ismcts/ismcts.py
from __future__ import annotations
import random
from typing import Callable
import numpy as np
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index
from engine.board import BOARD_LAYOUT, Occupant
from engine.deck import SEQUENCE_DECK
from agents.ismcts.mcts import mcts_search, PolicyValueFn


def determinize(
    info_set: InformationSet,
    player: int,
    seed: int | None = None,
) -> GameState:
    """Create a concrete GameState from an InformationSet by sampling opponent's hand."""
    rng = random.Random(seed)
    opponent = 2 if player == 1 else 1

    # All cards in the deck (as strings)
    all_cards = [str(c) for c in SEQUENCE_DECK]

    # Remove: own hand, discard pile, cards on the board
    used_cards = set(info_set.own_hand) | set(info_set.discard_pile)

    # Cards placed on the board — figure out which cards are "used" by board placement
    for r in range(10):
        for c in range(10):
            if info_set.occupancy[r][c] in (Occupant.PLAYER1, Occupant.PLAYER2):
                card_str = BOARD_LAYOUT[r][c]
                if card_str and card_str in all_cards:
                    # Remove one copy of this card from the available pool
                    idx = all_cards.index(card_str)
                    all_cards.pop(idx)

    # Remove own hand and discard from available pool
    available = list(all_cards)
    for card in info_set.own_hand:
        if card in available:
            available.remove(card)
    for card in info_set.discard_pile:
        if card in available:
            available.remove(card)

    rng.shuffle(available)

    # Deal opponent hand
    hand_size = len(info_set.own_hand)
    opponent_hand = tuple(available[:hand_size])
    remaining_deck = tuple(available[hand_size:])

    return GameState(
        occupancy=info_set.occupancy.copy(),
        current_player=info_set.current_player,
        hands={
            player: info_set.own_hand,
            opponent: opponent_hand,
        },
        deck_cards=remaining_deck,
        discard_pile=info_set.discard_pile,
        sequences=info_set.sequences,
    )


def ismcts_search(
    info_set: InformationSet,
    player: int,
    policy_value_fn: PolicyValueFn,
    num_determinizations: int = 20,
    num_simulations: int = 100,
    cpuct: float = 1.5,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
) -> Action:
    """Run ISMCTS: determinize, run MCTS on each, aggregate visit counts."""
    aggregated: dict[int, int] = {}  # action_index -> total visits

    for d in range(num_determinizations):
        det_state = determinize(info_set, player, seed=d)
        _, visit_counts = mcts_search(
            state=det_state,
            policy_value_fn=policy_value_fn,
            num_simulations=num_simulations,
            cpuct=cpuct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
        for action, visits in visit_counts.items():
            idx = action_to_index(action)
            aggregated[idx] = aggregated.get(idx, 0) + visits

    best_idx = max(aggregated, key=aggregated.get)
    from engine.actions import index_to_action
    return index_to_action(best_idx)
```

**Step 4: Run tests**

```bash
pytest tests/test_agents/test_ismcts.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/ismcts/ismcts.py tests/test_agents/test_ismcts.py
git commit -m "feat: add ISMCTS with determinization and aggregated search"
```

---

### Task 13: ISMCTS + ViT agent wrapper

**Files:**
- Create: `agents/ismcts/agent.py`
- Create: `tests/test_agents/test_ismcts_agent.py`

**Step 1: Write failing tests**

```python
# tests/test_agents/test_ismcts_agent.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from engine.game_state import GameState
from agents.ismcts.agent import ISMCTSAgent
from agents.base import Agent


class TestISMCTSAgent:
    def test_is_agent(self):
        agent = ISMCTSAgent(name="test", model=MagicMock(), encoder=MagicMock())
        assert isinstance(agent, Agent)

    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        agent = ISMCTSAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            num_determinizations=2,
            num_simulations=5,
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_save_and_load(self, tmp_path):
        from models.vit_policy_value import ViTPolicyValue
        from models.board_encoder import BoardEncoder

        model = ViTPolicyValue(num_actions=300, num_input_channels=22)
        encoder = BoardEncoder(target_size=224)
        agent = ISMCTSAgent(name="test", model=model, encoder=encoder)

        save_path = str(tmp_path / "agent")
        agent.save(save_path)
        agent.load(save_path)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_agents/test_ismcts_agent.py -v
```

**Step 3: Implement agents/ismcts/agent.py**

```python
# agents/ismcts/agent.py
from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action
from models.board_encoder import BoardEncoder
from agents.ismcts.ismcts import ismcts_search


class ISMCTSAgent(Agent):
    """ISMCTS agent using a ViT policy/value network."""

    def __init__(
        self,
        name: str = "ismcts_vit",
        model: torch.nn.Module | None = None,
        encoder: BoardEncoder | None = None,
        num_determinizations: int = 20,
        num_simulations: int = 100,
        cpuct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.encoder = encoder or BoardEncoder(target_size=224)
        self.num_determinizations = num_determinizations
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.model is not None:
            self.model.to(self.device)

    def _policy_value_fn(self, state: GameState) -> tuple[np.ndarray, float]:
        tensor = self.encoder.encode_torch(state, player_perspective=state.current_player)
        tensor = tensor.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(tensor)
        return policy.cpu().numpy()[0], value.cpu().item()

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        return ismcts_search(
            info_set=info_set,
            player=state.current_player,
            policy_value_fn=self._policy_value_fn,
            num_determinizations=self.num_determinizations,
            num_simulations=self.num_simulations,
            cpuct=self.cpuct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def load(self, path: str) -> None:
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
```

**Step 4: Run tests**

```bash
pytest tests/test_agents/test_ismcts_agent.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/ismcts/agent.py tests/test_agents/test_ismcts_agent.py
git commit -m "feat: add ISMCTSAgent wrapping ViT model with ISMCTS search"
```

---

## Phase 1C: Training Pipeline

### Task 14: Experience buffer

**Files:**
- Create: `training/experience_buffer.py`
- Create: `tests/test_training/test_buffer.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_buffer.py
import numpy as np
import pytest
from training.experience_buffer import ExperienceBuffer


class TestExperienceBuffer:
    def test_add_and_size(self):
        buf = ExperienceBuffer(max_size=100)
        state = np.zeros((22, 10, 10))
        policy = np.ones(300) / 300
        buf.add(state, policy, 1.0)
        assert len(buf) == 1

    def test_max_size(self):
        buf = ExperienceBuffer(max_size=5)
        for i in range(10):
            buf.add(np.zeros((22, 10, 10)), np.ones(300) / 300, float(i))
        assert len(buf) == 5

    def test_sample(self):
        buf = ExperienceBuffer(max_size=100)
        for i in range(20):
            buf.add(np.random.randn(22, 10, 10).astype(np.float32), np.ones(300) / 300, 1.0)
        states, policies, values = buf.sample(batch_size=8)
        assert states.shape == (8, 22, 10, 10)
        assert policies.shape == (8, 300)
        assert values.shape == (8,)

    def test_save_and_load(self, tmp_path):
        buf = ExperienceBuffer(max_size=100)
        for i in range(10):
            buf.add(np.random.randn(22, 10, 10).astype(np.float32), np.ones(300) / 300, 1.0)
        path = str(tmp_path / "buffer.npz")
        buf.save(path)
        buf2 = ExperienceBuffer.load(path, max_size=100)
        assert len(buf2) == len(buf)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_training/test_buffer.py -v
```

**Step 3: Implement training/experience_buffer.py**

```python
# training/experience_buffer.py
from __future__ import annotations
from collections import deque
import numpy as np


class ExperienceBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.states: deque[np.ndarray] = deque(maxlen=max_size)
        self.policies: deque[np.ndarray] = deque(maxlen=max_size)
        self.values: deque[float] = deque(maxlen=max_size)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = np.stack([self.states[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        return states, policies, values

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=np.stack(list(self.states)),
            policies=np.stack(list(self.policies)),
            values=np.array(list(self.values), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str, max_size: int = 500_000) -> ExperienceBuffer:
        data = np.load(path)
        buf = cls(max_size=max_size)
        for s, p, v in zip(data["states"], data["policies"], data["values"]):
            buf.add(s, p, float(v))
        return buf
```

**Step 4: Run tests**

```bash
pytest tests/test_training/test_buffer.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/experience_buffer.py tests/test_training/test_buffer.py
git commit -m "feat: add ExperienceBuffer with save/load"
```

---

### Task 15: Self-play engine

**Files:**
- Create: `training/self_play.py`
- Create: `tests/test_training/test_self_play.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_self_play.py
import numpy as np
import pytest
from unittest.mock import MagicMock
import torch
from training.self_play import run_self_play_game, SelfPlayConfig


class TestSelfPlayGame:
    def test_generates_training_examples(self):
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()

        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        config = SelfPlayConfig(
            num_simulations=5,
            num_determinizations=2,
            cpuct=1.5,
        )

        examples = run_self_play_game(
            model=mock_model,
            encoder=mock_encoder,
            config=config,
            seed=42,
            device="cpu",
        )
        assert len(examples) > 0
        state, policy, value = examples[0]
        assert state.shape == (22, 10, 10)
        assert policy.shape == (300,)
        assert isinstance(value, float)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_training/test_self_play.py -v
```

**Step 3: Implement training/self_play.py**

```python
# training/self_play.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from engine.game_state import GameState
from engine.actions import action_to_index
from models.board_encoder import BoardEncoder
from agents.ismcts.ismcts import ismcts_search


@dataclass
class SelfPlayConfig:
    num_simulations: int = 100
    num_determinizations: int = 20
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_moves: int = 500
    cards_per_hand: int = 5


def run_self_play_game(
    model: torch.nn.Module,
    encoder: BoardEncoder,
    config: SelfPlayConfig,
    seed: int | None = None,
    device: str = "cpu",
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one full game via self-play, returning training examples."""

    def policy_value_fn(state: GameState) -> tuple[np.ndarray, float]:
        tensor = encoder.encode_torch(state, player_perspective=state.current_player)
        tensor = tensor.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            policy, value = model(tensor)
        return policy.cpu().numpy()[0], value.cpu().item()

    game_state = GameState.new_game(seed=seed, cards_per_hand=config.cards_per_hand)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []  # (state_tensor, mcts_policy, player)

    for move_num in range(config.max_moves):
        if game_state.is_terminal():
            break

        current_player = game_state.current_player
        info_set = game_state.to_information_set(current_player)
        state_tensor = game_state.to_tensor(current_player)

        # Run ISMCTS to get action and visit counts
        from agents.ismcts.mcts import mcts_search
        from agents.ismcts.ismcts import determinize

        # Aggregate across determinizations
        aggregated_visits: dict[int, int] = {}
        for d in range(config.num_determinizations):
            det_state = determinize(info_set, current_player, seed=(seed or 0) * 1000 + move_num * 100 + d)
            _, visit_counts = mcts_search(
                state=det_state,
                policy_value_fn=policy_value_fn,
                num_simulations=config.num_simulations,
                cpuct=config.cpuct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
            )
            for action, visits in visit_counts.items():
                idx = action_to_index(action)
                aggregated_visits[idx] = aggregated_visits.get(idx, 0) + visits

        # Build policy target from visit counts
        total_visits = sum(aggregated_visits.values())
        policy_target = np.zeros(300, dtype=np.float32)
        for idx, visits in aggregated_visits.items():
            policy_target[idx] = visits / total_visits

        history.append((state_tensor, policy_target, current_player))

        # Select action (proportional to visits for exploration)
        probs = policy_target / policy_target.sum()
        chosen_idx = np.random.choice(300, p=probs)
        from engine.actions import index_to_action
        action = index_to_action(chosen_idx)

        # Ensure action is legal (fallback to most visited)
        legal = game_state.get_legal_actions()
        legal_indices = {action_to_index(a) for a in legal}
        if chosen_idx not in legal_indices:
            chosen_idx = max(aggregated_visits, key=aggregated_visits.get)
            action = index_to_action(chosen_idx)

        game_state = game_state.apply_action(action)

    # Assign values based on game outcome
    winner = game_state.get_winner()
    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for state_tensor, policy_target, player in history:
        if winner is None:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        examples.append((state_tensor, policy_target, value))

    return examples
```

**Step 4: Run tests**

```bash
pytest tests/test_training/test_self_play.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/self_play.py tests/test_training/test_self_play.py
git commit -m "feat: add self-play engine generating training examples"
```

---

### Task 16: Training config and HF Trainer wrapper

**Files:**
- Create: `training/configs/ismcts_vit.yaml`
- Create: `training/trainer.py`
- Create: `training/train.py`
- Create: `tests/test_training/test_trainer.py`

**Step 1: Write the config file**

```yaml
# training/configs/ismcts_vit.yaml
model:
  backbone: "google/vit-base-patch16-224"
  freeze_backbone_epochs: 5
  policy_hidden: 512
  value_hidden: 256
  num_input_channels: 22
  target_size: 224

mcts:
  simulations: 400
  cpuct: 1.5
  determinizations: 30
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs_per_iteration: 10
  buffer_size: 500000

self_play:
  games_per_iteration: 100
  parallel_workers: 8
  max_moves: 500

arena:
  games: 100
  promotion_threshold: 0.55

monitoring:
  wandb_project: "sequence-ai"
  log_every_n_steps: 10
```

**Step 2: Write failing tests**

```python
# tests/test_training/test_trainer.py
import numpy as np
import pytest
import torch
from training.trainer import SequenceTrainer, TrainingConfig


class TestTrainingConfig:
    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs_per_iteration: 2
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)
        config = TrainingConfig.from_yaml(str(config_path))
        assert config.batch_size == 32
        assert config.learning_rate == 0.001


class TestSequenceTrainer:
    def test_train_step(self):
        from models.vit_policy_value import ViTPolicyValue
        model = ViTPolicyValue(
            num_actions=300,
            num_input_channels=22,
            policy_hidden=64,
            value_hidden=32,
        )
        config = TrainingConfig(batch_size=4, learning_rate=0.001, epochs_per_iteration=1)
        trainer = SequenceTrainer(model=model, config=config, device="cpu")

        # Synthetic training data
        states = np.random.randn(8, 22, 10, 10).astype(np.float32)
        policies = np.random.dirichlet(np.ones(300), size=8).astype(np.float32)
        values = np.random.uniform(-1, 1, size=8).astype(np.float32)

        # Note: upscaling from 10x10 to 224x224 happens inside trainer
        metrics = trainer.train_epoch(states, policies, values)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics
```

**Step 3: Run to verify failure**

```bash
pytest tests/test_training/test_trainer.py -v
```

**Step 4: Implement training/trainer.py**

```python
# training/trainer.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import yaml


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs_per_iteration: int = 10
    target_size: int = 224

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        training = data.get("training", {})
        return cls(
            batch_size=training.get("batch_size", 256),
            learning_rate=training.get("learning_rate", 0.001),
            weight_decay=training.get("weight_decay", 0.0001),
            epochs_per_iteration=training.get("epochs_per_iteration", 10),
        )


class SequenceTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _upscale_states(self, states: np.ndarray) -> np.ndarray:
        """Upscale (B, 22, 10, 10) to (B, 22, 224, 224) via nearest neighbor."""
        scale = self.config.target_size // 10
        # np.repeat along spatial dims
        up = np.repeat(np.repeat(states, scale, axis=2), scale, axis=3)
        return up[:, :, :self.config.target_size, :self.config.target_size]

    def train_epoch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, float]:
        self.model.train()
        n = len(states)
        indices = np.arange(n)
        np.random.shuffle(indices)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        num_batches = 0

        for start in range(0, n, self.config.batch_size):
            end = min(start + self.config.batch_size, n)
            batch_idx = indices[start:end]

            batch_states = self._upscale_states(states[batch_idx])
            s = torch.FloatTensor(batch_states).to(self.device)
            p = torch.FloatTensor(policies[batch_idx]).to(self.device)
            v = torch.FloatTensor(values[batch_idx]).to(self.device)

            pred_p, pred_v = self.model(s)

            # Policy loss: cross-entropy (since pred_p is softmax output)
            pi_loss = -torch.sum(p * torch.log(pred_p + 1e-8)) / p.size(0)
            # Value loss: MSE
            v_loss = torch.mean((v - pred_v.squeeze()) ** 2)
            loss = pi_loss + v_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            num_batches += 1

        return {
            "policy_loss": total_pi_loss / max(num_batches, 1),
            "value_loss": total_v_loss / max(num_batches, 1),
            "total_loss": (total_pi_loss + total_v_loss) / max(num_batches, 1),
        }
```

**Step 5: Run tests**

```bash
pytest tests/test_training/test_trainer.py -v
```
Expected: ALL PASS

**Step 6: Create training/train.py entry point**

```python
# training/train.py
"""Main training entry point for ISMCTS + ViT agent."""
from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import torch

from models.vit_policy_value import ViTPolicyValue
from models.board_encoder import BoardEncoder
from training.trainer import SequenceTrainer, TrainingConfig
from training.self_play import run_self_play_game, SelfPlayConfig
from training.experience_buffer import ExperienceBuffer


def main():
    parser = argparse.ArgumentParser(description="Train Sequence AI agent")
    parser.add_argument("--config", type=str, default="training/configs/ismcts_vit.yaml")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize model
    model_cfg = config["model"]
    model = ViTPolicyValue(
        backbone_name=model_cfg["backbone"],
        num_actions=300,
        policy_hidden=model_cfg["policy_hidden"],
        value_hidden=model_cfg["value_hidden"],
        num_input_channels=model_cfg["num_input_channels"],
    ).to(device)

    encoder = BoardEncoder(target_size=model_cfg["target_size"])

    # Training config
    training_config = TrainingConfig(
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        epochs_per_iteration=config["training"]["epochs_per_iteration"],
        target_size=model_cfg["target_size"],
    )
    trainer = SequenceTrainer(model=model, config=training_config, device=device)

    # Self-play config
    sp_cfg = config["self_play"]
    mcts_cfg = config["mcts"]
    self_play_config = SelfPlayConfig(
        num_simulations=mcts_cfg["simulations"],
        num_determinizations=mcts_cfg["determinizations"],
        cpuct=mcts_cfg["cpuct"],
        dirichlet_alpha=mcts_cfg["dirichlet_alpha"],
        dirichlet_epsilon=mcts_cfg["dirichlet_epsilon"],
        max_moves=sp_cfg["max_moves"],
    )

    buffer = ExperienceBuffer(max_size=config["training"]["buffer_size"])

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")

        # Self-play phase
        print(f"Running {sp_cfg['games_per_iteration']} self-play games...")
        for game_idx in range(sp_cfg["games_per_iteration"]):
            examples = run_self_play_game(
                model=model,
                encoder=encoder,
                config=self_play_config,
                seed=iteration * 10000 + game_idx,
                device=device,
            )
            for state, policy, value in examples:
                buffer.add(state, policy, value)

        print(f"Buffer size: {len(buffer)}")

        # Training phase
        if len(buffer) >= training_config.batch_size:
            print("Training...")
            for epoch in range(training_config.epochs_per_iteration):
                states, policies, values = buffer.sample(
                    min(len(buffer), training_config.batch_size * 10)
                )
                metrics = trainer.train_epoch(states, policies, values)
                print(f"  Epoch {epoch + 1}: {metrics}")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"iteration_{iteration + 1}")
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
```

**Step 7: Commit**

```bash
git add training/trainer.py training/train.py training/configs/ismcts_vit.yaml tests/test_training/test_trainer.py
git commit -m "feat: add training pipeline with HF-style trainer and training entry point"
```

---

### Task 17: Arena evaluation

**Files:**
- Create: `training/arena.py`
- Create: `tests/test_training/test_arena.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_arena.py
import pytest
from agents.random_agent import RandomAgent
from training.arena import Arena


class TestArena:
    def test_run_match(self):
        p1 = RandomAgent("r1", seed=1)
        p2 = RandomAgent("r2", seed=2)
        arena = Arena(num_games=10)
        result = arena.evaluate(p1, p2)
        assert "p1_wins" in result
        assert "p2_wins" in result
        assert "draws" in result
        assert result["p1_wins"] + result["p2_wins"] + result["draws"] == 10

    def test_win_rate(self):
        p1 = RandomAgent("r1", seed=1)
        p2 = RandomAgent("r2", seed=2)
        arena = Arena(num_games=10)
        result = arena.evaluate(p1, p2)
        assert 0.0 <= result["p1_win_rate"] <= 1.0
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_training/test_arena.py -v
```

**Step 3: Implement training/arena.py**

```python
# training/arena.py
from __future__ import annotations
from engine.game_state import GameState
from agents.base import Agent


class Arena:
    def __init__(self, num_games: int = 100, max_moves: int = 500):
        self.num_games = num_games
        self.max_moves = max_moves

    def evaluate(
        self,
        player1: Agent,
        player2: Agent,
        base_seed: int = 0,
    ) -> dict[str, float | int]:
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for game_idx in range(self.num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                agents = {1: player1, 2: player2}
            else:
                agents = {1: player2, 2: player1}

            gs = GameState.new_game(seed=base_seed + game_idx)
            for _ in range(self.max_moves):
                if gs.is_terminal():
                    break
                current = gs.current_player
                info = gs.to_information_set(current)
                action = agents[current].select_action(gs, info)
                gs = gs.apply_action(action)

            winner = gs.get_winner()
            if winner is None:
                draws += 1
            elif agents[winner] is player1:
                p1_wins += 1
            else:
                p2_wins += 1

        return {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "p1_win_rate": p1_wins / self.num_games,
            "p2_win_rate": p2_wins / self.num_games,
        }
```

**Step 4: Run tests**

```bash
pytest tests/test_training/test_arena.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/arena.py tests/test_training/test_arena.py
git commit -m "feat: add Arena for evaluating agents head-to-head"
```

---

### Task 18: Elo rating system

**Files:**
- Create: `training/elo.py`
- Create: `tests/test_training/test_elo.py`

**Step 1: Write failing tests**

```python
# tests/test_training/test_elo.py
import pytest
from training.elo import EloRating


class TestEloRating:
    def test_initial_rating(self):
        elo = EloRating()
        elo.add_player("agent_a")
        assert elo.get_rating("agent_a") == 1500

    def test_winner_gains_rating(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner="a")
        assert elo.get_rating("a") > 1500
        assert elo.get_rating("b") < 1500

    def test_draw_equal_players(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner=None)
        # Draw between equal players shouldn't change ratings much
        assert abs(elo.get_rating("a") - 1500) < 1

    def test_leaderboard(self):
        elo = EloRating()
        elo.add_player("a")
        elo.add_player("b")
        elo.record_result("a", "b", winner="a")
        board = elo.leaderboard()
        assert board[0][0] == "a"
        assert board[1][0] == "b"
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_training/test_elo.py -v
```

**Step 3: Implement training/elo.py**

```python
# training/elo.py
from __future__ import annotations
import json


class EloRating:
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k = k_factor
        self.initial = initial_rating
        self.ratings: dict[str, float] = {}

    def add_player(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.initial

    def get_rating(self, name: str) -> float:
        return self.ratings[name]

    def record_result(self, player_a: str, player_b: str, winner: str | None):
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        eb = 1.0 - ea

        if winner == player_a:
            sa, sb = 1.0, 0.0
        elif winner == player_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] = ra + self.k * (sa - ea)
        self.ratings[player_b] = rb + self.k * (sb - eb)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.ratings, f, indent=2)

    @classmethod
    def load(cls, path: str) -> EloRating:
        elo = cls()
        with open(path) as f:
            elo.ratings = json.load(f)
        return elo
```

**Step 4: Run tests**

```bash
pytest tests/test_training/test_elo.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/elo.py tests/test_training/test_elo.py
git commit -m "feat: add Elo rating system for agent evaluation"
```

---

### Task 19: Run all tests and verify full pipeline

**Step 1: Run entire test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS

**Step 2: Quick integration smoke test**

```bash
python -c "
from engine.game_state import GameState
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from training.arena import Arena

arena = Arena(num_games=5)
result = arena.evaluate(RandomAgent('r', seed=1), HeuristicAgent('h'))
print('Random vs Heuristic:', result)
"
```
Expected: Prints win/loss/draw stats

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: verify full Phase 1 pipeline passes all tests"
```

---

## Summary

**Phase 1 delivers:**
- Clean, immutable game engine (`engine/`) with full Sequence rules
- Agent framework with `Agent` protocol + Random and Heuristic baselines
- ViT policy/value model with pretrained backbone + custom heads
- MCTS core + ISMCTS (determinization for imperfect info)
- Full ISMCTS+ViT agent ready to train
- Training pipeline: self-play, experience buffer, trainer, arena, Elo
- Training config (YAML) + entry point script
- Comprehensive test suite for every module

**Next phases (separate plans):**
- Phase 2: ReBeL+DeBERTa agent, Web app, Accelerate integration
- Phase 3: Decision Transformer, AlphaZero+belief, K8s, Spectator mode
- Phase 4: Polish, HF Hub publishing, deployment
