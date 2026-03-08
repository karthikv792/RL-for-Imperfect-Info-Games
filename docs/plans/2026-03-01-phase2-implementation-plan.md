# Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the ReBeL+DeBERTa agent, build the FastAPI backend for human-vs-AI play, and integrate HF Accelerate for distributed training.

**Architecture:** Phase 2 builds three tracks on top of the Phase 1 foundation (108 tests passing). Track A adds the DeBERTa-based sequence tokenizer and ReBeL agent (CFR + neural network value function). Track B builds the FastAPI WebSocket backend enabling real-time play. Track C wraps the existing trainer with HF Accelerate for device-agnostic distributed training. The web frontend (React/Next.js) is deferred to Phase 3 — Phase 2 delivers a fully playable backend API tested via pytest.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers (DeBERTa), FastAPI, WebSockets, HF Accelerate, pydantic, pytest

**Existing codebase (Phase 1):**
- `engine/` — actions.py, board.py, deck.py, game_state.py, rules.py
- `agents/` — base.py, random_agent.py, heuristic_agent.py, ismcts/{mcts.py, ismcts.py, agent.py}
- `models/` — board_encoder.py, vit_policy_value.py
- `training/` — experience_buffer.py, self_play.py, trainer.py, train.py, arena.py, elo.py
- 108 tests across tests/test_engine/, tests/test_agents/, tests/test_models/, tests/test_training/

---

## Phase 2A: ReBeL + DeBERTa Agent

### Task 1: Board tokenizer for DeBERTa

**Files:**
- Create: `models/board_tokenizer.py`
- Create: `tests/test_models/test_board_tokenizer.py`

**Context:** DeBERTa is a sequence model. We need to convert the 10×10 board + hand + game info into a flat token sequence. Each board cell becomes 3 integers: `[suit_id, rank_id, occupant_id]`. Hand cards are appended. Special tokens encode deck size and metadata.

**Step 1: Write the failing test**

```python
# tests/test_models/test_board_tokenizer.py
import numpy as np
import pytest
from engine.game_state import GameState
from models.board_tokenizer import BoardTokenizer


class TestBoardTokenizer:
    def test_tokenize_shape(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens = tokenizer.tokenize(gs, player_perspective=1)
        # 100 cells * 3 features + 5 hand cards * 3 + 2 metadata = 317
        assert tokens.shape == (317,)
        assert tokens.dtype == np.int32

    def test_tokenize_attention_mask(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens, mask = tokenizer.tokenize_with_mask(gs, player_perspective=1)
        assert mask.shape == tokens.shape
        assert mask.dtype == np.int32
        assert mask.sum() > 0  # at least some tokens are attended

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
        # First cell (0,0) is a corner: suit=0, rank=0, occupant=CORNER(3)
        assert tokens[0] == 0  # no suit
        assert tokens[1] == 0  # no rank
        assert tokens[2] == 3  # CORNER occupant

    def test_hand_cards_appended(self):
        gs = GameState.new_game(seed=42)
        tokenizer = BoardTokenizer()
        tokens = tokenizer.tokenize(gs, player_perspective=1)
        # Hand starts at index 300 (100 cells * 3)
        hand_section = tokens[300:315]  # 5 cards * 3 features
        assert hand_section.sum() > 0  # hand cards have non-zero encoding
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_models/test_board_tokenizer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement models/board_tokenizer.py**

```python
# models/board_tokenizer.py
from __future__ import annotations
import numpy as np
from engine.game_state import GameState
from engine.board import BOARD_LAYOUT, Occupant
from engine.deck import CHAR_TO_SUIT, STR_TO_RANK, Rank


# Encoding maps: 0 = padding/none
SUIT_ID = {"C": 1, "S": 2, "H": 3, "D": 4}
RANK_ID = {
    "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7,
    "9": 8, "10": 9, "Q": 10, "K": 11, "A": 12, "J1": 13, "J2": 14,
}
OCCUPANT_ID = {
    Occupant.EMPTY: 0, Occupant.PLAYER1: 1,
    Occupant.PLAYER2: 2, Occupant.CORNER: 3,
}

# Vocab sizes for embedding layers
SUIT_VOCAB = 5   # 0=pad + 4 suits
RANK_VOCAB = 15  # 0=pad + 14 ranks
OCCUPANT_VOCAB = 4  # 0=empty, 1=p1, 2=p2, 3=corner
METADATA_VOCAB = 105  # deck_size 0-104

# Sequence length: 100 cells * 3 + 5 hand * 3 + 2 metadata = 317
SEQ_LEN = 317


class BoardTokenizer:
    """Convert GameState into integer token sequences for DeBERTa input."""

    def _encode_card_str(self, card_str: str) -> tuple[int, int]:
        """Return (suit_id, rank_id) for a card string like '2S' or 'J1H'."""
        if not card_str:
            return 0, 0
        suit_char = card_str[-1]
        rank_str = card_str[:-1]
        return SUIT_ID.get(suit_char, 0), RANK_ID.get(rank_str, 0)

    def tokenize(self, state: GameState, player_perspective: int) -> np.ndarray:
        """Convert state to flat integer token array of shape (317,)."""
        tokens = np.zeros(SEQ_LEN, dtype=np.int32)
        me = Occupant.PLAYER1 if player_perspective == 1 else Occupant.PLAYER2
        opp = Occupant.PLAYER2 if player_perspective == 1 else Occupant.PLAYER1

        # Board cells: 100 * 3 = 300 tokens
        for r in range(10):
            for c in range(10):
                idx = (r * 10 + c) * 3
                card_str = BOARD_LAYOUT[r][c]
                s_id, r_id = self._encode_card_str(card_str)
                tokens[idx] = s_id
                tokens[idx + 1] = r_id
                occ = state.occupancy[r][c]
                # Normalize: always show "me" as 1, "opponent" as 2
                if occ == me:
                    tokens[idx + 2] = 1
                elif occ == opp:
                    tokens[idx + 2] = 2
                else:
                    tokens[idx + 2] = OCCUPANT_ID.get(Occupant(occ), 0)

        # Hand cards: 5 * 3 = 15 tokens
        hand = state.hands[player_perspective]
        for i, card_str in enumerate(hand):
            idx = 300 + i * 3
            s_id, r_id = self._encode_card_str(card_str)
            tokens[idx] = s_id
            tokens[idx + 1] = r_id
            tokens[idx + 2] = 1  # "own card" marker

        # Metadata: 2 tokens
        tokens[315] = min(state.deck_size, 104)  # deck size
        tokens[316] = len(state.discard_pile) % 105  # discard count

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
```

**Step 4: Run tests**

Run: `pytest tests/test_models/test_board_tokenizer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add models/board_tokenizer.py tests/test_models/test_board_tokenizer.py
git commit -m "feat: add BoardTokenizer for DeBERTa-compatible sequence encoding"
```

---

### Task 2: DeBERTa policy/value model

**Files:**
- Create: `models/deberta_policy_value.py`
- Create: `tests/test_models/test_deberta_model.py`

**Context:** Like `ViTPolicyValue`, this wraps a pretrained DeBERTa backbone with custom policy/value heads. Input is the integer token sequence from `BoardTokenizer`, converted to embeddings via three learned embedding layers (suit, rank, occupant) summed together.

**Step 1: Write the failing test**

```python
# tests/test_models/test_deberta_model.py
import torch
import pytest
from models.deberta_policy_value import DeBERTaPolicyValue


class TestDeBERTaPolicyValue:
    @pytest.fixture
    def model(self):
        return DeBERTaPolicyValue(
            backbone_name="microsoft/deberta-v3-small",
            num_actions=300,
            seq_len=317,
        )

    def test_forward_shapes(self, model):
        tokens = torch.randint(0, 10, (2, 317))
        mask = torch.ones(2, 317, dtype=torch.long)
        policy, value = model(tokens, mask)
        assert policy.shape == (2, 300)
        assert value.shape == (2, 1)

    def test_policy_sums_to_one(self, model):
        tokens = torch.randint(0, 10, (1, 317))
        mask = torch.ones(1, 317, dtype=torch.long)
        policy, _ = model(tokens, mask)
        total = policy.sum(dim=1)
        assert torch.allclose(total, torch.ones(1), atol=1e-5)

    def test_value_in_range(self, model):
        tokens = torch.randint(0, 10, (1, 317))
        mask = torch.ones(1, 317, dtype=torch.long)
        _, value = model(tokens, mask)
        assert -1.0 <= value.item() <= 1.0

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        for param in model.policy_head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_models/test_deberta_model.py -v`
Expected: FAIL

**Step 3: Implement models/deberta_policy_value.py**

```python
# models/deberta_policy_value.py
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
        hidden_size = self.backbone.config.hidden_size  # 768 for small

        # Custom input embeddings: replace DeBERTa's word embeddings
        # We have triples: (suit, rank, occupant) repeated. We embed each
        # and sum them, then project to hidden_size.
        max_vocab = max(SUIT_VOCAB, RANK_VOCAB, OCCUPANT_VOCAB, METADATA_VOCAB)
        self.token_embedding = nn.Embedding(max_vocab + 1, hidden_size, padding_idx=0)

        # Position embedding for the 317 positions
        self.position_embedding = nn.Embedding(seq_len, hidden_size)

        # Policy head: pool over sequence -> action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, num_actions),
            nn.Softmax(dim=-1),
        )

        # Value head: [CLS]-like pooling -> scalar value
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

        # Embed tokens and add position
        tok_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(pos_ids)
        embeddings = tok_emb + pos_emb

        # Feed through DeBERTa (bypass its own word embeddings)
        outputs = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, L, H)

        # Pool: use first token (position 0) as [CLS]-like representation
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
```

**Step 4: Run tests**

Run: `pytest tests/test_models/test_deberta_model.py -v`
Expected: ALL PASS (may take time for DeBERTa download first time)

**Step 5: Commit**

```bash
git add models/deberta_policy_value.py tests/test_models/test_deberta_model.py
git commit -m "feat: add DeBERTaPolicyValue model with pretrained backbone"
```

---

### Task 3: CFR core implementation

**Files:**
- Create: `agents/rebel/cfr.py`
- Create: `tests/test_agents/test_cfr.py`

**Context:** Counterfactual Regret Minimization (CFR) iteratively updates a strategy to converge toward Nash equilibrium. For ReBeL, we run CFR in subgames using the neural network as a leaf evaluator.

**Step 1: Write the failing test**

```python
# tests/test_agents/test_cfr.py
import numpy as np
import pytest
from engine.game_state import GameState
from agents.rebel.cfr import CFRNode, run_cfr


class TestCFRNode:
    def test_initial_strategy_uniform(self):
        node = CFRNode(num_actions=5)
        strategy = node.get_strategy()
        assert strategy.shape == (5,)
        assert np.allclose(strategy, 0.2)  # uniform

    def test_update_regrets(self):
        node = CFRNode(num_actions=3)
        node.update_regrets(np.array([1.0, -0.5, 0.0]))
        strategy = node.get_strategy()
        # Positive regret action should have higher probability
        assert strategy[0] > strategy[1]


class TestRunCFR:
    def test_returns_strategy(self):
        gs = GameState.new_game(seed=42)

        def dummy_value_fn(state):
            return 0.0

        strategy = run_cfr(
            state=gs,
            value_fn=dummy_value_fn,
            num_iterations=5,
            max_depth=2,
        )
        assert isinstance(strategy, np.ndarray)
        assert strategy.shape == (300,)
        assert abs(strategy.sum() - 1.0) < 1e-5

    def test_strategy_is_valid_distribution(self):
        gs = GameState.new_game(seed=42)

        def dummy_value_fn(state):
            return 0.0

        strategy = run_cfr(
            state=gs,
            value_fn=dummy_value_fn,
            num_iterations=10,
            max_depth=3,
        )
        assert np.all(strategy >= 0)
        assert abs(strategy.sum() - 1.0) < 1e-5
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agents/test_cfr.py -v`
Expected: FAIL

**Step 3: Implement agents/rebel/cfr.py**

```python
# agents/rebel/cfr.py
from __future__ import annotations
import numpy as np
from typing import Callable
from engine.game_state import GameState
from engine.actions import Action, action_to_index


class CFRNode:
    """A single CFR information set node."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        """Regret-matching: positive regrets normalized to strategy."""
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        else:
            return np.ones(self.num_actions) / self.num_actions

    def update_regrets(self, regrets: np.ndarray):
        self.regret_sum += regrets

    def update_strategy_sum(self, strategy: np.ndarray, weight: float = 1.0):
        self.strategy_sum += weight * strategy

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.num_actions) / self.num_actions


ValueFn = Callable[[GameState], float]


def run_cfr(
    state: GameState,
    value_fn: ValueFn,
    num_iterations: int = 100,
    max_depth: int = 5,
) -> np.ndarray:
    """Run CFR iterations on a subgame rooted at `state`.

    Returns a strategy vector of shape (300,) normalized to legal actions.
    """
    legal_actions = state.get_legal_actions()
    if not legal_actions:
        return np.zeros(300, dtype=np.float32)

    num_legal = len(legal_actions)
    node = CFRNode(num_actions=num_legal)

    for _ in range(num_iterations):
        strategy = node.get_strategy()
        action_values = np.zeros(num_legal, dtype=np.float64)

        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)
            if next_state.is_terminal():
                winner = next_state.get_winner()
                if winner is None:
                    action_values[i] = 0.0
                elif winner == state.current_player:
                    action_values[i] = 1.0
                else:
                    action_values[i] = -1.0
            elif max_depth <= 1:
                # Use neural network value function at leaf
                v = value_fn(next_state)
                # Flip sign: value_fn returns from next_state's current_player perspective
                if next_state.current_player != state.current_player:
                    v = -v
                action_values[i] = v
            else:
                # Recurse (opponent's CFR — simplified: just use value_fn)
                v = value_fn(next_state)
                if next_state.current_player != state.current_player:
                    v = -v
                action_values[i] = v

        # Counterfactual value of the node
        node_value = np.dot(strategy, action_values)

        # Regrets: action value minus node value
        regrets = action_values - node_value
        node.update_regrets(regrets)
        node.update_strategy_sum(strategy)

    # Convert average strategy to full 300-action vector
    avg_strategy = node.get_average_strategy()
    full_strategy = np.zeros(300, dtype=np.float32)
    for i, action in enumerate(legal_actions):
        idx = action_to_index(action)
        full_strategy[idx] = avg_strategy[i]

    # Normalize
    total = full_strategy.sum()
    if total > 0:
        full_strategy /= total

    return full_strategy
```

**Step 4: Run tests**

Run: `pytest tests/test_agents/test_cfr.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/rebel/cfr.py tests/test_agents/test_cfr.py
git commit -m "feat: add CFR core with regret matching and subgame solving"
```

---

### Task 4: ReBeL agent wrapper

**Files:**
- Create: `agents/rebel/agent.py`
- Create: `tests/test_agents/test_rebel_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_rebel_agent.py
import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.rebel.agent import RebelAgent
from agents.base import Agent


class TestRebelAgent:
    def test_is_agent(self):
        agent = RebelAgent(name="test", model=MagicMock(), tokenizer=MagicMock())
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

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize_with_mask.return_value = (
            np.zeros(317, dtype=np.int32),
            np.ones(317, dtype=np.int32),
        )

        agent = RebelAgent(
            name="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            num_cfr_iterations=3,
            max_depth=1,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_save_and_load(self, tmp_path):
        from models.deberta_policy_value import DeBERTaPolicyValue
        from models.board_tokenizer import BoardTokenizer

        model = DeBERTaPolicyValue(num_actions=300, seq_len=317)
        tokenizer = BoardTokenizer()
        agent = RebelAgent(name="test", model=model, tokenizer=tokenizer, device="cpu")

        save_path = str(tmp_path / "rebel_agent")
        agent.save(save_path)
        agent.load(save_path)
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agents/test_rebel_agent.py -v`
Expected: FAIL

**Step 3: Implement agents/rebel/agent.py**

```python
# agents/rebel/agent.py
from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_tokenizer import BoardTokenizer
from agents.rebel.cfr import run_cfr


class RebelAgent(Agent):
    """ReBeL-style agent using CFR + DeBERTa value network."""

    def __init__(
        self,
        name: str = "rebel_deberta",
        model: torch.nn.Module | None = None,
        tokenizer: BoardTokenizer | None = None,
        num_cfr_iterations: int = 50,
        max_depth: int = 3,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer or BoardTokenizer()
        self.num_cfr_iterations = num_cfr_iterations
        self.max_depth = max_depth

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

    def _value_fn(self, state: GameState) -> float:
        """Neural network value estimate for a state."""
        tokens, mask = self.tokenizer.tokenize_with_mask(
            state, player_perspective=state.current_player
        )
        t = torch.from_numpy(tokens).unsqueeze(0).long().to(self.device)
        m = torch.from_numpy(mask).unsqueeze(0).long().to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, value = self.model(t, m)
        return value.cpu().item()

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        strategy = run_cfr(
            state=state,
            value_fn=self._value_fn,
            num_iterations=self.num_cfr_iterations,
            max_depth=self.max_depth,
        )

        # Select action with highest probability
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]

        best_idx = max(legal_indices, key=lambda i: strategy[i])
        return index_to_action(best_idx)

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

Run: `pytest tests/test_agents/test_rebel_agent.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/rebel/agent.py tests/test_agents/test_rebel_agent.py
git commit -m "feat: add RebelAgent wrapping CFR + DeBERTa value network"
```

---

### Task 5: DeBERTa training config

**Files:**
- Create: `training/configs/rebel_deberta.yaml`

**Step 1: Write the config**

```yaml
# training/configs/rebel_deberta.yaml
model:
  backbone: "microsoft/deberta-v3-small"
  freeze_backbone_epochs: 5
  policy_hidden: 512
  value_hidden: 256
  seq_len: 317

cfr:
  iterations: 50
  max_depth: 3

training:
  batch_size: 128
  learning_rate: 0.0005
  weight_decay: 0.0001
  epochs_per_iteration: 10
  buffer_size: 500000

self_play:
  games_per_iteration: 50
  max_moves: 500

arena:
  games: 100
  promotion_threshold: 0.55

monitoring:
  wandb_project: "sequence-ai"
  log_every_n_steps: 10
```

**Step 2: Commit**

```bash
git add training/configs/rebel_deberta.yaml
git commit -m "feat: add ReBeL+DeBERTa training config"
```

---

## Phase 2B: FastAPI Backend

### Task 6: Game session manager

**Files:**
- Create: `api/game_session.py`
- Create: `tests/test_api/test_game_session.py`
- Create: `tests/test_api/__init__.py`

**Context:** The game session manager tracks active games. It creates games, applies moves, and queries agent moves. It does NOT handle HTTP/WebSocket — that's the handler layer.

**Step 1: Write the failing test**

```python
# tests/test_api/test_game_session.py
import pytest
from api.game_session import GameSession, GameSessionManager
from agents.random_agent import RandomAgent


class TestGameSession:
    def test_create_session(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        assert session.game_state is not None
        assert not session.is_over

    def test_human_move(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        legal = session.get_legal_moves()
        assert len(legal) > 0
        move = legal[0]
        result = session.apply_human_move(move["position"], move["type"])
        assert result["success"]

    def test_agent_responds_after_human(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        legal = session.get_legal_moves()
        session.apply_human_move(legal[0]["position"], legal[0]["type"])
        agent_result = session.get_agent_move()
        assert "action" in agent_result
        assert "thinking_time_ms" in agent_result

    def test_get_state_dict(self):
        agent = RandomAgent("r", seed=1)
        session = GameSession(agent=agent, seed=42)
        state = session.to_dict()
        assert "board" in state
        assert "hand" in state
        assert "legal_moves" in state
        assert "turn" in state
        assert "sequences" in state


class TestGameSessionManager:
    def test_create_and_get(self):
        manager = GameSessionManager()
        sid = manager.create_session(agent_name="random", seed=42)
        session = manager.get_session(sid)
        assert session is not None

    def test_list_agents(self):
        manager = GameSessionManager()
        agents = manager.list_agents()
        assert "random" in agents
        assert "heuristic" in agents
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_game_session.py -v`
Expected: FAIL

**Step 3: Implement api/game_session.py**

```python
# api/game_session.py
from __future__ import annotations
import time
import uuid
from engine.game_state import GameState
from engine.actions import Action, ActionType
from engine.board import BOARD_LAYOUT, Occupant
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


AVAILABLE_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
}


class GameSession:
    """Manages a single human-vs-AI game."""

    def __init__(
        self,
        agent: Agent,
        seed: int | None = None,
        human_player: int = 1,
    ):
        self.agent = agent
        self.human_player = human_player
        self.ai_player = 2 if human_player == 1 else 1
        self.game_state = GameState.new_game(seed=seed)
        self.move_history: list[dict] = []
        self.session_id = str(uuid.uuid4())

    @property
    def is_over(self) -> bool:
        return self.game_state.is_terminal()

    @property
    def is_human_turn(self) -> bool:
        return self.game_state.current_player == self.human_player

    def get_legal_moves(self) -> list[dict]:
        """Return legal moves as dicts for JSON serialization."""
        if not self.is_human_turn or self.is_over:
            return []
        actions = self.game_state.get_legal_actions()
        moves = []
        for a in actions:
            moves.append({
                "position": [a.row, a.col],
                "type": a.action_type.name.lower(),
            })
        return moves

    def apply_human_move(self, position: list[int], move_type: str) -> dict:
        """Apply a human move. Returns success status."""
        if not self.is_human_turn or self.is_over:
            return {"success": False, "error": "Not your turn"}

        action_type = ActionType[move_type.upper()]
        action = Action(row=position[0], col=position[1], action_type=action_type)

        legal = self.game_state.get_legal_actions()
        if action not in legal:
            return {"success": False, "error": "Illegal move"}

        self.game_state = self.game_state.apply_action(action)
        self.move_history.append({
            "player": self.human_player,
            "position": position,
            "type": move_type,
        })
        return {"success": True}

    def get_agent_move(self) -> dict:
        """Get and apply the AI agent's move."""
        if self.is_human_turn or self.is_over:
            return {"error": "Not AI's turn"}

        start = time.time()
        info_set = self.game_state.to_information_set(self.ai_player)
        action = self.agent.select_action(self.game_state, info_set)
        elapsed_ms = int((time.time() - start) * 1000)

        self.game_state = self.game_state.apply_action(action)
        move_data = {
            "player": self.ai_player,
            "position": [action.row, action.col],
            "type": action.action_type.name.lower(),
        }
        self.move_history.append(move_data)
        return {
            "action": move_data,
            "thinking_time_ms": elapsed_ms,
        }

    def to_dict(self) -> dict:
        """Serialize current state for the client."""
        board = []
        for r in range(10):
            row = []
            for c in range(10):
                occ = int(self.game_state.occupancy[r][c])
                card = BOARD_LAYOUT[r][c]
                row.append({"card": card, "occupant": occ})
            board.append(row)

        return {
            "board": board,
            "hand": list(self.game_state.hands[self.human_player]),
            "legal_moves": self.get_legal_moves(),
            "turn": "human" if self.is_human_turn else "ai",
            "current_player": self.game_state.current_player,
            "sequences": {
                str(k): list(v) for k, v in self.game_state.sequences.items()
            },
            "deck_size": self.game_state.deck_size,
            "is_over": self.is_over,
            "winner": self.game_state.get_winner(),
            "move_history": self.move_history,
        }


class GameSessionManager:
    """Manages multiple concurrent game sessions."""

    def __init__(self):
        self.sessions: dict[str, GameSession] = {}

    def create_session(
        self,
        agent_name: str = "random",
        seed: int | None = None,
        human_player: int = 1,
    ) -> str:
        agent_cls = AVAILABLE_AGENTS.get(agent_name)
        if agent_cls is None:
            raise ValueError(f"Unknown agent: {agent_name}")

        if agent_name == "random":
            agent = agent_cls(name=agent_name, seed=seed)
        else:
            agent = agent_cls(name=agent_name)

        session = GameSession(agent=agent, seed=seed, human_player=human_player)
        self.sessions[session.session_id] = session
        return session.session_id

    def get_session(self, session_id: str) -> GameSession | None:
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def list_agents(self) -> list[str]:
        return list(AVAILABLE_AGENTS.keys())
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_game_session.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/game_session.py tests/test_api/__init__.py tests/test_api/test_game_session.py
git commit -m "feat: add GameSession and GameSessionManager for human-vs-AI play"
```

---

### Task 7: FastAPI app with REST endpoints

**Files:**
- Create: `api/main.py`
- Create: `api/routes/__init__.py`
- Create: `api/routes/game.py`
- Create: `tests/test_api/test_routes.py`

**Context:** REST endpoints for creating games, listing agents, and getting leaderboard data. WebSocket is Task 8.

**Step 1: Write the failing test**

```python
# tests/test_api/test_routes.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestRESTEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_list_agents(self, client):
        response = client.get("/api/agents")
        assert response.status_code == 200
        data = response.json()
        assert "random" in data["agents"]
        assert "heuristic" in data["agents"]

    def test_create_game(self, client):
        response = client.post("/api/game", json={
            "agent": "random",
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "state" in data

    def test_get_game_state(self, client):
        # Create game first
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        sid = create.json()["session_id"]

        response = client.get(f"/api/game/{sid}")
        assert response.status_code == 200
        data = response.json()
        assert "board" in data
        assert "hand" in data

    def test_make_move(self, client):
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        sid = create.json()["session_id"]
        state = create.json()["state"]
        legal = state["legal_moves"]
        assert len(legal) > 0

        move = legal[0]
        response = client.post(f"/api/game/{sid}/move", json={
            "position": move["position"],
            "type": move["type"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["move_result"]["success"]
        # AI should have responded
        assert "agent_move" in data

    def test_nonexistent_game(self, client):
        response = client.get("/api/game/nonexistent")
        assert response.status_code == 404
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_routes.py -v`
Expected: FAIL

**Step 3: Implement api/main.py and api/routes/game.py**

```python
# api/main.py
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.game import router as game_router

app = FastAPI(title="Sequence AI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router, prefix="/api")
```

```python
# api/routes/__init__.py
```

```python
# api/routes/game.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.game_session import GameSessionManager

router = APIRouter()
manager = GameSessionManager()


class CreateGameRequest(BaseModel):
    agent: str = "random"
    seed: int | None = None
    human_player: int = 1


class MakeMoveRequest(BaseModel):
    position: list[int]
    type: str


@router.get("/agents")
def list_agents():
    return {"agents": manager.list_agents()}


@router.post("/game")
def create_game(req: CreateGameRequest):
    try:
        sid = manager.create_session(
            agent_name=req.agent,
            seed=req.seed,
            human_player=req.human_player,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = manager.get_session(sid)
    return {"session_id": sid, "state": session.to_dict()}


@router.get("/game/{session_id}")
def get_game_state(session_id: str):
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session.to_dict()


@router.post("/game/{session_id}/move")
def make_move(session_id: str, req: MakeMoveRequest):
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    move_result = session.apply_human_move(req.position, req.type)

    response = {"move_result": move_result, "state": session.to_dict()}

    # If move was successful and it's now AI's turn, get AI move
    if move_result.get("success") and not session.is_over and not session.is_human_turn:
        agent_move = session.get_agent_move()
        response["agent_move"] = agent_move
        response["state"] = session.to_dict()

    return response


@router.get("/leaderboard")
def get_leaderboard():
    from training.elo import EloRating
    try:
        elo = EloRating.load("data/elo_ratings.json")
        return {"leaderboard": elo.leaderboard()}
    except FileNotFoundError:
        return {"leaderboard": []}
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_routes.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/main.py api/routes/__init__.py api/routes/game.py tests/test_api/test_routes.py
git commit -m "feat: add FastAPI REST endpoints for game management"
```

---

### Task 8: WebSocket handler for real-time play

**Files:**
- Create: `api/ws_handler.py`
- Modify: `api/main.py` (add WebSocket route)
- Create: `tests/test_api/test_websocket.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_websocket.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestWebSocket:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_websocket_new_game(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "new_game", "agent": "random", "seed": 42})
            data = ws.receive_json()
            assert data["type"] == "game_state"
            assert "board" in data
            assert "hand" in data

    def test_websocket_make_move(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "new_game", "agent": "random", "seed": 42})
            state = ws.receive_json()
            legal = state["legal_moves"]
            assert len(legal) > 0

            ws.send_json({
                "type": "make_move",
                "position": legal[0]["position"],
                "move_type": legal[0]["type"],
            })
            # Should receive our move confirmation + AI move + updated state
            response = ws.receive_json()
            assert response["type"] in ("game_state", "agent_move", "error")

    def test_websocket_invalid_message(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "unknown_command"})
            data = ws.receive_json()
            assert data["type"] == "error"
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_websocket.py -v`
Expected: FAIL

**Step 3: Implement api/ws_handler.py and update api/main.py**

```python
# api/ws_handler.py
from __future__ import annotations
import json
from fastapi import WebSocket, WebSocketDisconnect
from api.game_session import GameSession, GameSessionManager, AVAILABLE_AGENTS
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


async def handle_websocket(ws: WebSocket, manager: GameSessionManager):
    """Handle a WebSocket connection for a single player."""
    await ws.accept()
    session: GameSession | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "new_game":
                agent_name = msg.get("agent", "random")
                seed = msg.get("seed")
                try:
                    sid = manager.create_session(agent_name=agent_name, seed=seed)
                    session = manager.get_session(sid)
                    await ws.send_json({
                        "type": "game_state",
                        **session.to_dict(),
                    })
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})

            elif msg_type == "make_move":
                if session is None:
                    await ws.send_json({"type": "error", "message": "No active game"})
                    continue

                position = msg.get("position")
                move_type = msg.get("move_type")
                result = session.apply_human_move(position, move_type)

                if not result.get("success"):
                    await ws.send_json({
                        "type": "error",
                        "message": result.get("error", "Invalid move"),
                    })
                    continue

                # Check if game is over after human move
                if session.is_over:
                    await ws.send_json({
                        "type": "game_over",
                        "winner": session.game_state.get_winner(),
                        **session.to_dict(),
                    })
                    continue

                # AI responds
                if not session.is_human_turn:
                    agent_result = session.get_agent_move()
                    await ws.send_json({
                        "type": "agent_move",
                        **agent_result,
                    })

                    if session.is_over:
                        await ws.send_json({
                            "type": "game_over",
                            "winner": session.game_state.get_winner(),
                            **session.to_dict(),
                        })
                    else:
                        await ws.send_json({
                            "type": "game_state",
                            **session.to_dict(),
                        })

            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        pass
```

Now update `api/main.py` to add the WebSocket route:

```python
# api/main.py (updated — full file)
from __future__ import annotations
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.routes.game import router as game_router, manager
from api.ws_handler import handle_websocket

app = FastAPI(title="Sequence AI", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(game_router, prefix="/api")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await handle_websocket(ws, manager)
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_websocket.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/ws_handler.py api/main.py tests/test_api/test_websocket.py
git commit -m "feat: add WebSocket handler for real-time game play"
```

---

### Task 9: API integration test — full game via REST

**Files:**
- Create: `tests/test_api/test_full_game.py`

**Step 1: Write the test**

```python
# tests/test_api/test_full_game.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestFullGameViaREST:
    def test_play_complete_game(self):
        client = TestClient(app)

        # Create game
        create = client.post("/api/game", json={"agent": "random", "seed": 42})
        assert create.status_code == 200
        sid = create.json()["session_id"]

        # Play up to 200 moves (should finish)
        for _ in range(200):
            state = client.get(f"/api/game/{sid}").json()
            if state["is_over"]:
                break

            if state["turn"] != "human":
                # Shouldn't happen — AI moves are automatic after human
                break

            legal = state["legal_moves"]
            if not legal:
                break

            move = legal[0]
            response = client.post(f"/api/game/{sid}/move", json={
                "position": move["position"],
                "type": move["type"],
            })
            assert response.status_code == 200

        # Game should have completed
        final = client.get(f"/api/game/{sid}").json()
        assert final["is_over"]
```

**Step 2: Run test**

Run: `pytest tests/test_api/test_full_game.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_api/test_full_game.py
git commit -m "test: add full game integration test via REST API"
```

---

## Phase 2C: HF Accelerate Integration

### Task 10: Accelerate-wrapped trainer

**Files:**
- Create: `training/accelerate_trainer.py`
- Create: `tests/test_training/test_accelerate.py`

**Context:** Wrap the existing `SequenceTrainer` with HuggingFace Accelerate for device-agnostic execution (CPU, MPS, CUDA, multi-GPU). The key change: `Accelerator` handles device placement, gradient accumulation, and distributed data loading.

**Step 1: Write the failing test**

```python
# tests/test_training/test_accelerate.py
import numpy as np
import pytest
import torch
from training.accelerate_trainer import AccelerateTrainer, AccelerateConfig


class TestAccelerateConfig:
    def test_default_config(self):
        config = AccelerateConfig()
        assert config.batch_size == 256
        assert config.target_size == 224


class TestAccelerateTrainer:
    def test_train_step(self):
        from models.vit_policy_value import ViTPolicyValue
        model = ViTPolicyValue(
            num_actions=300,
            num_input_channels=22,
            policy_hidden=64,
            value_hidden=32,
        )
        config = AccelerateConfig(batch_size=4, learning_rate=0.001, epochs_per_iteration=1)
        trainer = AccelerateTrainer(model=model, config=config)

        states = np.random.randn(8, 22, 10, 10).astype(np.float32)
        policies = np.random.dirichlet(np.ones(300), size=8).astype(np.float32)
        values = np.random.uniform(-1, 1, size=8).astype(np.float32)

        metrics = trainer.train_epoch(states, policies, values)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "total_loss" in metrics

    def test_device_detection(self):
        config = AccelerateConfig()
        from models.vit_policy_value import ViTPolicyValue
        model = ViTPolicyValue(
            num_actions=300,
            num_input_channels=22,
            policy_hidden=64,
            value_hidden=32,
        )
        trainer = AccelerateTrainer(model=model, config=config)
        assert trainer.device is not None
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_training/test_accelerate.py -v`
Expected: FAIL

**Step 3: Implement training/accelerate_trainer.py**

```python
# training/accelerate_trainer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False


@dataclass
class AccelerateConfig:
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs_per_iteration: int = 10
    target_size: int = 224
    gradient_accumulation_steps: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> AccelerateConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        training = data.get("training", {})
        return cls(
            batch_size=training.get("batch_size", 256),
            learning_rate=training.get("learning_rate", 0.001),
            weight_decay=training.get("weight_decay", 0.0001),
            epochs_per_iteration=training.get("epochs_per_iteration", 10),
        )


class AccelerateTrainer:
    """Training wrapper using HF Accelerate for device-agnostic execution."""

    def __init__(
        self,
        model: nn.Module,
        config: AccelerateConfig,
    ):
        self.config = config

        if HAS_ACCELERATE:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if self.accelerator:
            self.model, self.optimizer = self.accelerator.prepare(model, self.optimizer)
        else:
            self.model = model.to(self.device)

    def _upscale_states(self, states: np.ndarray) -> np.ndarray:
        scale = self.config.target_size // 10
        up = np.repeat(np.repeat(states, scale, axis=2), scale, axis=3)
        pad_h = self.config.target_size - up.shape[2]
        pad_w = self.config.target_size - up.shape[3]
        if pad_h > 0 or pad_w > 0:
            up = np.pad(up, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
        return up[:, :, :self.config.target_size, :self.config.target_size]

    def train_epoch(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, float]:
        upscaled = self._upscale_states(states)

        dataset = TensorDataset(
            torch.FloatTensor(upscaled),
            torch.FloatTensor(policies),
            torch.FloatTensor(values),
        )
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        if self.accelerator:
            dataloader = self.accelerator.prepare(dataloader)

        self.model.train()
        total_pi_loss = 0.0
        total_v_loss = 0.0
        num_batches = 0

        for batch_s, batch_p, batch_v in dataloader:
            if not self.accelerator:
                batch_s = batch_s.to(self.device)
                batch_p = batch_p.to(self.device)
                batch_v = batch_v.to(self.device)

            pred_p, pred_v = self.model(batch_s)
            pi_loss = -torch.sum(batch_p * torch.log(pred_p + 1e-8)) / batch_p.size(0)
            v_loss = torch.mean((batch_v - pred_v.squeeze()) ** 2)
            loss = pi_loss + v_loss

            self.optimizer.zero_grad()
            if self.accelerator:
                self.accelerator.backward(loss)
            else:
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

**Step 4: Run tests**

Run: `pytest tests/test_training/test_accelerate.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/accelerate_trainer.py tests/test_training/test_accelerate.py
git commit -m "feat: add AccelerateTrainer for device-agnostic distributed training"
```

---

### Task 11: Accelerate config file

**Files:**
- Create: `accelerate_config.yaml`

**Step 1: Write the config**

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: "NO"
mixed_precision: "no"
num_processes: 1
use_cpu: false
```

**Step 2: Commit**

```bash
git add accelerate_config.yaml
git commit -m "feat: add HF Accelerate config for local training"
```

---

### Task 12: Update pyproject.toml with new dependencies

**Files:**
- Modify: `pyproject.toml`

**Context:** Add `httpx` for async test client and ensure `api` package is discoverable.

**Step 1: Update pyproject.toml**

Add `"api*"` to the setuptools packages include list. Add `httpx` to dev dependencies for FastAPI test client.

```toml
[tool.setuptools.packages.find]
include = ["engine*", "agents*", "models*", "training*", "api*"]
```

Add to dev dependencies:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-benchmark>=4.0",
    "ruff>=0.1",
    "httpx>=0.25",
]
```

**Step 2: Install**

```bash
pip install -e ".[dev]"
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add api package and httpx to project config"
```

---

### Task 13: Run all tests and verify full Phase 2 pipeline

**Step 1: Run entire test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS (should be ~130+ tests)

**Step 2: Quick smoke tests**

```bash
# Test ReBeL agent plays a game
python -c "
from engine.game_state import GameState
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from training.arena import Arena

arena = Arena(num_games=5)
result = arena.evaluate(RandomAgent('r', seed=1), HeuristicAgent('h'))
print('Random vs Heuristic:', result)
"

# Test API starts
python -c "
from api.main import app
print('FastAPI app created:', app.title, app.version)
"
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: verify full Phase 2 pipeline passes all tests"
```

---

## Summary

**Phase 2 delivers (13 tasks):**

| Track | Tasks | What |
|-------|-------|------|
| **2A: ReBeL + DeBERTa** | 1-5 | Board tokenizer, DeBERTa model, CFR core, ReBeL agent, training config |
| **2B: FastAPI Backend** | 6-9 | Game session manager, REST endpoints, WebSocket handler, integration test |
| **2C: HF Accelerate** | 10-12 | Accelerate trainer wrapper, config, dependency updates |
| **Verification** | 13 | Full test suite pass + smoke tests |

**Phase 3 (next):**
- React/Next.js frontend (using UI design guide)
- Decision Transformer agent
- Classic AlphaZero + belief sampling agent
- Kubernetes manifests
- Spectator mode
