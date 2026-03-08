# Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Decision Transformer and AlphaZero+Belief agents, build the React/Next.js frontend with the full UI design system, add spectator mode, and create Kubernetes manifests for deployment.

**Architecture:** Phase 3 builds four tracks on top of the Phase 2 foundation (143 tests passing). Track A adds two new agent architectures (Decision Transformer using GPT-2, AlphaZero with belief sampling). Track B builds the complete React/Next.js frontend following the UI design guide. Track C adds AI-vs-AI spectator mode to both backend and frontend. Track D creates Kubernetes manifests for cloud deployment.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers (GPT-2), React 18, Next.js 14, TypeScript, Tailwind CSS, WebSocket, Kubernetes

**Existing codebase (Phase 1+2):**
- `engine/` — actions.py, board.py, deck.py, game_state.py, rules.py
- `agents/` — base.py, random_agent.py, heuristic_agent.py, ismcts/{mcts.py, ismcts.py, agent.py}, rebel/{cfr.py, agent.py}
- `models/` — board_encoder.py, board_tokenizer.py, vit_policy_value.py, deberta_policy_value.py
- `training/` — experience_buffer.py, self_play.py, trainer.py, accelerate_trainer.py, train.py, arena.py, elo.py
- `api/` — main.py, game_session.py, ws_handler.py, routes/game.py
- 143 tests across tests/test_engine/, tests/test_agents/, tests/test_models/, tests/test_training/, tests/test_api/

---

## Phase 3A: Decision Transformer Agent

### Task 1: Decision Transformer model

**Files:**
- Create: `models/decision_transformer.py`
- Create: `tests/test_models/test_decision_transformer.py`

**Context:** Decision Transformers treat RL as sequence modeling. Input is a sequence of (return-to-go, state, action) triples. We use GPT-2 backbone with custom embedding layers for board states (via BoardTokenizer), actions, and returns-to-go.

**Step 1: Write the failing test**

```python
# tests/test_models/test_decision_transformer.py
import torch
import pytest
from models.decision_transformer import DecisionTransformer


class TestDecisionTransformer:
    @pytest.fixture
    def model(self):
        return DecisionTransformer(
            state_dim=317,
            action_dim=300,
            max_timestep=500,
            context_length=20,
            hidden_size=256,
            n_layer=4,
            n_head=4,
        )

    def test_forward_shapes(self, model):
        B, T = 2, 10
        states = torch.randint(0, 10, (B, T, 317))
        actions = torch.randint(0, 300, (B, T))
        returns_to_go = torch.randn(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0).expand(B, -1)

        action_preds, value_preds = model(states, actions, returns_to_go, timesteps)
        assert action_preds.shape == (B, T, 300)
        assert value_preds.shape == (B, T, 1)

    def test_action_prediction_is_distribution(self, model):
        B, T = 1, 5
        states = torch.randint(0, 10, (B, T, 317))
        actions = torch.randint(0, 300, (B, T))
        returns_to_go = torch.ones(B, T, 1)
        timesteps = torch.arange(T).unsqueeze(0)

        action_preds, _ = model(states, actions, returns_to_go, timesteps)
        probs = action_preds[0, -1]  # last timestep
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_get_action(self, model):
        import numpy as np
        states = [torch.randint(0, 10, (317,)) for _ in range(3)]
        actions = [0, 5]
        returns_to_go = [1.0, 0.8, 0.6]
        timesteps = [0, 1, 2]

        action_probs = model.get_action(states, actions, returns_to_go, timesteps)
        assert action_probs.shape == (300,)
        assert abs(action_probs.sum().item() - 1.0) < 1e-5
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_models/test_decision_transformer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement models/decision_transformer.py**

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_models/test_decision_transformer.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add models/decision_transformer.py tests/test_models/test_decision_transformer.py
git commit -m "feat: add Decision Transformer model for sequence modeling RL"
```

---

### Task 2: Decision Transformer agent wrapper

**Files:**
- Create: `agents/decision_tf/__init__.py`
- Create: `agents/decision_tf/agent.py`
- Create: `tests/test_agents/test_decision_tf_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agents/test_decision_tf_agent.py
import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.decision_tf.agent import DecisionTFAgent
from agents.base import Agent


class TestDecisionTFAgent:
    def test_is_agent(self):
        agent = DecisionTFAgent(name="test", model=MagicMock(), tokenizer=MagicMock())
        assert isinstance(agent, Agent)

    def test_selects_legal_action(self):
        gs = GameState.new_game(seed=42)
        info = gs.to_information_set(player=gs.current_player)

        mock_model = MagicMock()
        mock_model.get_action.return_value = torch.ones(300) / 300
        mock_model.parameters.return_value = iter([torch.zeros(1)])

        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = np.zeros(317, dtype=np.int32)

        agent = DecisionTFAgent(
            name="test",
            model=mock_model,
            tokenizer=mock_tokenizer,
            target_return=1.0,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_reset_clears_history(self):
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        agent = DecisionTFAgent(name="test", model=mock_model, device="cpu")
        agent._state_history.append(torch.zeros(317))
        agent.reset()
        assert len(agent._state_history) == 0
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agents/test_decision_tf_agent.py -v`
Expected: FAIL

**Step 3: Implement agents/decision_tf/agent.py**

```python
# agents/decision_tf/agent.py
from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_tokenizer import BoardTokenizer


class DecisionTFAgent(Agent):
    """Decision Transformer agent — treats RL as sequence modeling."""

    def __init__(
        self,
        name: str = "decision_tf",
        model: torch.nn.Module | None = None,
        tokenizer: BoardTokenizer | None = None,
        target_return: float = 1.0,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.tokenizer = tokenizer or BoardTokenizer()
        self.target_return = target_return

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

        # Episode history
        self._state_history: list[torch.Tensor] = []
        self._action_history: list[int] = []
        self._return_history: list[float] = []
        self._timestep = 0

    def reset(self):
        """Reset episode history for a new game."""
        self._state_history.clear()
        self._action_history.clear()
        self._return_history.clear()
        self._timestep = 0

    def select_action(self, state: GameState, info_set: InformationSet) -> Action:
        tokens = self.tokenizer.tokenize(state, player_perspective=state.current_player)
        state_tensor = torch.from_numpy(tokens).long()
        self._state_history.append(state_tensor)

        # Compute return-to-go (decreasing from target)
        rtg = self.target_return - self._timestep * 0.002
        self._return_history.append(rtg)

        # Get action distribution from model
        action_probs = self.model.get_action(
            states=self._state_history,
            actions=self._action_history,
            returns_to_go=self._return_history,
            timesteps=list(range(len(self._state_history))),
        )

        # Mask to legal actions
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]
        masked = torch.zeros(300)
        for idx in legal_indices:
            masked[idx] = action_probs[idx]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            for idx in legal_indices:
                masked[idx] = 1.0 / len(legal_indices)

        best_idx = masked.argmax().item()
        self._action_history.append(best_idx)
        self._timestep += 1
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

Run: `pytest tests/test_agents/test_decision_tf_agent.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/decision_tf/__init__.py agents/decision_tf/agent.py tests/test_agents/test_decision_tf_agent.py
git commit -m "feat: add Decision Transformer agent wrapper"
```

---

### Task 3: AlphaZero + belief sampling agent

**Files:**
- Create: `agents/alphazero_belief/__init__.py`
- Create: `agents/alphazero_belief/agent.py`
- Create: `tests/test_agents/test_alphazero_belief.py`

**Context:** Classic AlphaZero approach but adapted for imperfect info via belief sampling (determinization). Uses the existing ViT model + MCTS, but with a cleaner interface and configurable exploration.

**Step 1: Write the failing test**

```python
# tests/test_agents/test_alphazero_belief.py
import pytest
from unittest.mock import MagicMock
import numpy as np
import torch
from engine.game_state import GameState
from agents.alphazero_belief.agent import AlphaZeroBeliefAgent
from agents.base import Agent


class TestAlphaZeroBeliefAgent:
    def test_is_agent(self):
        agent = AlphaZeroBeliefAgent(name="test", model=MagicMock(), encoder=MagicMock())
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

        agent = AlphaZeroBeliefAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            num_simulations=5,
            num_determinizations=2,
            device="cpu",
        )
        action = agent.select_action(gs, info)
        assert action in gs.get_legal_actions()

    def test_temperature_controls_exploration(self):
        mock_model = MagicMock()
        mock_model.return_value = (
            torch.ones(1, 300) / 300,
            torch.zeros(1, 1),
        )
        mock_model.eval = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.encode_torch.return_value = torch.randn(22, 224, 224)

        agent = AlphaZeroBeliefAgent(
            name="test",
            model=mock_model,
            encoder=mock_encoder,
            temperature=0.0,
            device="cpu",
        )
        assert agent.temperature == 0.0
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_agents/test_alphazero_belief.py -v`
Expected: FAIL

**Step 3: Implement agents/alphazero_belief/agent.py**

```python
# agents/alphazero_belief/agent.py
from __future__ import annotations
import os
import numpy as np
import torch
from agents.base import Agent
from engine.game_state import GameState, InformationSet
from engine.actions import Action, action_to_index, index_to_action
from models.board_encoder import BoardEncoder
from agents.ismcts.mcts import mcts_search
from agents.ismcts.ismcts import determinize


class AlphaZeroBeliefAgent(Agent):
    """AlphaZero-style agent with belief sampling for imperfect information."""

    def __init__(
        self,
        name: str = "alphazero_belief",
        model: torch.nn.Module | None = None,
        encoder: BoardEncoder | None = None,
        num_simulations: int = 200,
        num_determinizations: int = 20,
        cpuct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model = model
        self.encoder = encoder or BoardEncoder()
        self.num_simulations = num_simulations
        self.num_determinizations = num_determinizations
        self.cpuct = cpuct
        self.temperature = temperature
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
        current_player = state.current_player
        aggregated_visits: dict[int, int] = {}

        for d in range(self.num_determinizations):
            det_state = determinize(info_set, current_player, seed=d * 1000)
            _, visit_counts = mcts_search(
                state=det_state,
                policy_value_fn=self._policy_value_fn,
                num_simulations=self.num_simulations,
                cpuct=self.cpuct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
            )
            for action, visits in visit_counts.items():
                idx = action_to_index(action)
                aggregated_visits[idx] = aggregated_visits.get(idx, 0) + visits

        # Build policy from visit counts
        legal = state.get_legal_actions()
        legal_indices = [action_to_index(a) for a in legal]

        if self.temperature == 0.0:
            # Greedy
            best_idx = max(legal_indices, key=lambda i: aggregated_visits.get(i, 0))
        else:
            # Temperature-scaled
            visits = np.array([aggregated_visits.get(i, 0) for i in legal_indices], dtype=np.float64)
            if self.temperature != 1.0:
                visits = visits ** (1.0 / self.temperature)
            total = visits.sum()
            if total > 0:
                probs = visits / total
            else:
                probs = np.ones(len(legal_indices)) / len(legal_indices)
            chosen = np.random.choice(len(legal_indices), p=probs)
            best_idx = legal_indices[chosen]

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

Run: `pytest tests/test_agents/test_alphazero_belief.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add agents/alphazero_belief/__init__.py agents/alphazero_belief/agent.py tests/test_agents/test_alphazero_belief.py
git commit -m "feat: add AlphaZero+belief sampling agent for imperfect info"
```

---

### Task 4: Register new agents in GameSessionManager

**Files:**
- Modify: `api/game_session.py`
- Create: `tests/test_api/test_new_agents.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_new_agents.py
import pytest
from api.game_session import GameSessionManager


class TestNewAgents:
    def test_all_agents_listed(self):
        manager = GameSessionManager()
        agents = manager.list_agents()
        assert "random" in agents
        assert "heuristic" in agents
        assert "decision_tf" in agents
        assert "alphazero_belief" in agents
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_new_agents.py -v`
Expected: FAIL (decision_tf and alphazero_belief not registered)

**Step 3: Update api/game_session.py**

Add to the imports and `AVAILABLE_AGENTS`:

```python
# Add these imports at the top
from agents.decision_tf.agent import DecisionTFAgent
from agents.alphazero_belief.agent import AlphaZeroBeliefAgent

# Update AVAILABLE_AGENTS
AVAILABLE_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "decision_tf": DecisionTFAgent,
    "alphazero_belief": AlphaZeroBeliefAgent,
}
```

Update `create_session` to handle agents that need no seed:

```python
def create_session(self, agent_name, seed=None, human_player=1):
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
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_new_agents.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/game_session.py tests/test_api/test_new_agents.py
git commit -m "feat: register Decision Transformer and AlphaZero agents in session manager"
```

---

## Phase 3B: React/Next.js Frontend

### Task 5: Next.js project scaffolding with design tokens

**Files:**
- Create: `webapp/package.json`
- Create: `webapp/tsconfig.json`
- Create: `webapp/tailwind.config.ts`
- Create: `webapp/next.config.ts`
- Create: `webapp/src/app/layout.tsx`
- Create: `webapp/src/app/page.tsx`
- Create: `webapp/src/app/globals.css`
- Create: `webapp/src/styles/tokens.css`

**Step 1: Initialize Next.js project**

```bash
cd webapp && npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --no-import-alias --use-npm
```

**Step 2: Create design tokens CSS**

Create `webapp/src/styles/tokens.css` with the full color system, typography, spacing, and motion tokens from the UI design guide.

**Step 3: Create root layout with design system**

Update `webapp/src/app/layout.tsx` to import tokens and set up font loading (Space Grotesk, Inter, JetBrains Mono, Playfair Display).

**Step 4: Create landing page**

Create `webapp/src/app/page.tsx` with "Challenge the AI at Sequence" hero, agent picker, and [Start Playing] CTA.

**Step 5: Verify dev server**

```bash
cd webapp && npm run dev
```
Verify: http://localhost:3000 shows landing page

**Step 6: Commit**

```bash
git add webapp/
git commit -m "feat: scaffold Next.js app with design token system"
```

---

### Task 6: Board component

**Files:**
- Create: `webapp/src/components/Board.tsx`
- Create: `webapp/src/components/BoardCell.tsx`
- Create: `webapp/src/components/Token.tsx`

**Context:** The board is a 10x10 SVG/CSS grid following the UI design guide. Each cell shows card rank/suit, occupancy token, legal move highlights, and sequence indicators.

**Step 1: Create BoardCell component**

```typescript
// webapp/src/components/BoardCell.tsx
"use client";
import { Token } from "./Token";

interface BoardCellProps {
  card: string;
  occupant: number; // 0=empty, 1=player1, 2=player2, 3=corner
  isLegal: boolean;
  isLastMove: boolean;
  inSequence: boolean;
  onClick?: () => void;
  isHovered?: boolean;
}

export function BoardCell({
  card, occupant, isLegal, isLastMove, inSequence, onClick, isHovered,
}: BoardCellProps) {
  const isCorner = card === "XX";
  const suit = card.slice(-1);
  const rank = card.slice(0, -1);

  const suitSymbol: Record<string, string> = { S: "♠", H: "♥", D: "♦", C: "♣" };
  const suitColor = suit === "H" || suit === "D" ? "text-red-600" : "text-gray-900";

  return (
    <div
      className={`relative aspect-square rounded-sm border transition-all duration-150
        ${isCorner ? "bg-gradient-to-br from-green-700 to-green-800 border-green-600" : "bg-[var(--surface-card)] border-[var(--border-card)]"}
        ${isLegal ? "cursor-pointer ring-2 ring-green-400/40 hover:ring-green-400/70 hover:scale-[1.02]" : ""}
        ${isLastMove ? "ring-2 ring-amber-400/50" : ""}
      `}
      onClick={isLegal ? onClick : undefined}
    >
      {!isCorner && (
        <>
          <span className={`absolute top-0.5 left-1 text-[10px] font-semibold font-serif ${suitColor}`}>
            {rank}
          </span>
          <span className={`absolute bottom-0.5 right-1 text-[9px] font-serif ${suitColor}`}>
            {suitSymbol[suit] || ""}
          </span>
        </>
      )}
      {isCorner && (
        <span className="absolute inset-0 flex items-center justify-center text-green-300/60 text-lg">★</span>
      )}
      {(occupant === 1 || occupant === 2) && (
        <Token player={occupant} inSequence={inSequence} />
      )}
    </div>
  );
}
```

**Step 2: Create Token component**

```typescript
// webapp/src/components/Token.tsx
"use client";

interface TokenProps {
  player: number;
  inSequence?: boolean;
  animate?: boolean;
}

export function Token({ player, inSequence = false, animate = true }: TokenProps) {
  const isGold = player === 1;
  return (
    <div
      className={`absolute top-[15%] left-[15%] w-[70%] h-[70%] rounded-full border-2
        ${animate ? "animate-token-place" : ""}
        ${isGold
          ? "bg-gradient-radial from-amber-300 to-amber-500 border-amber-600"
          : "bg-gradient-radial from-blue-300 to-blue-500 border-blue-600"
        }
        ${inSequence ? (isGold ? "shadow-[0_0_12px_rgba(245,158,11,0.4)]" : "shadow-[0_0_12px_rgba(59,130,246,0.4)]") : ""}
      `}
      style={{
        boxShadow: `inset 0 2px 4px rgba(255,255,255,0.3), inset 0 -2px 4px rgba(0,0,0,0.15)${
          inSequence ? `, 0 0 12px ${isGold ? "rgba(245,158,11,0.4)" : "rgba(59,130,246,0.4)"}` : ""
        }`,
      }}
    />
  );
}
```

**Step 3: Create Board component**

```typescript
// webapp/src/components/Board.tsx
"use client";
import { BoardCell } from "./BoardCell";

interface CellData {
  card: string;
  occupant: number;
}

interface BoardProps {
  board: CellData[][];
  legalMoves: { position: number[]; type: string }[];
  lastMove?: { position: number[] } | null;
  sequences?: Record<string, number[][]>;
  onCellClick?: (row: number, col: number) => void;
}

export function Board({ board, legalMoves, lastMove, sequences, onCellClick }: BoardProps) {
  const legalSet = new Set(legalMoves.map(m => `${m.position[0]},${m.position[1]}`));
  const lastMoveKey = lastMove ? `${lastMove.position[0]},${lastMove.position[1]}` : "";

  // Collect cells in sequences
  const sequenceCells = new Set<string>();
  if (sequences) {
    Object.values(sequences).forEach(seqs => {
      if (Array.isArray(seqs)) {
        seqs.forEach(seq => {
          if (Array.isArray(seq)) {
            seq.forEach(cell => {
              if (Array.isArray(cell)) sequenceCells.add(`${cell[0]},${cell[1]}`);
            });
          }
        });
      }
    });
  }

  return (
    <div className="bg-[var(--surface-board)] p-3 rounded-xl shadow-2xl">
      <div className="grid grid-cols-10 gap-0.5">
        {board.map((row, r) =>
          row.map((cell, c) => (
            <BoardCell
              key={`${r}-${c}`}
              card={cell.card}
              occupant={cell.occupant}
              isLegal={legalSet.has(`${r},${c}`)}
              isLastMove={`${r},${c}` === lastMoveKey}
              inSequence={sequenceCells.has(`${r},${c}`)}
              onClick={() => onCellClick?.(r, c)}
            />
          ))
        )}
      </div>
    </div>
  );
}
```

**Step 4: Verify components render**

Add a test board to the landing page and verify visually.

**Step 5: Commit**

```bash
git add webapp/src/components/
git commit -m "feat: add Board, BoardCell, and Token components"
```

---

### Task 7: Hand, GameInfo, and MoveHistory components

**Files:**
- Create: `webapp/src/components/HandCard.tsx`
- Create: `webapp/src/components/Hand.tsx`
- Create: `webapp/src/components/GameInfo.tsx`
- Create: `webapp/src/components/MoveHistory.tsx`

**Step 1: Create HandCard component**

Card in hand with hover lift, selected state, unplayable dimming.

**Step 2: Create Hand component**

Row of 5 HandCards with selection state management.

**Step 3: Create GameInfo component**

Turn indicator, sequence counts, deck size, AI thinking indicator.

**Step 4: Create MoveHistory component**

Scrollable log of moves with player color indicators.

**Step 5: Commit**

```bash
git add webapp/src/components/
git commit -m "feat: add Hand, GameInfo, and MoveHistory UI components"
```

---

### Task 8: WebSocket hook and game state management

**Files:**
- Create: `webapp/src/hooks/useGameWebSocket.ts`
- Create: `webapp/src/hooks/useGameState.ts`
- Create: `webapp/src/types/game.ts`

**Step 1: Create game types**

```typescript
// webapp/src/types/game.ts
export interface CellData {
  card: string;
  occupant: number;
}

export interface LegalMove {
  position: number[];
  type: string;
}

export interface AgentMove {
  action: { player: number; position: number[]; type: string };
  thinking_time_ms: number;
}

export interface GameState {
  board: CellData[][];
  hand: string[];
  legal_moves: LegalMove[];
  turn: "human" | "ai";
  current_player: number;
  sequences: Record<string, number[][][]>;
  deck_size: number;
  is_over: boolean;
  winner: number | null;
  move_history: Array<{ player: number; position: number[]; type: string }>;
}
```

**Step 2: Create WebSocket hook**

```typescript
// webapp/src/hooks/useGameWebSocket.ts
"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { GameState, AgentMove } from "@/types/game";

export function useGameWebSocket(wsUrl: string) {
  const ws = useRef<WebSocket | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastAgentMove, setLastAgentMove] = useState<AgentMove | null>(null);

  useEffect(() => {
    const socket = new WebSocket(wsUrl);
    ws.current = socket;

    socket.onopen = () => setIsConnected(true);
    socket.onclose = () => setIsConnected(false);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "game_state") {
        setGameState(data);
        setAiThinking(false);
      } else if (data.type === "agent_move") {
        setLastAgentMove(data);
      } else if (data.type === "game_over") {
        setGameState(data);
        setAiThinking(false);
      }
    };

    return () => socket.close();
  }, [wsUrl]);

  const newGame = useCallback((agent: string, seed?: number) => {
    ws.current?.send(JSON.stringify({ type: "new_game", agent, seed }));
  }, []);

  const makeMove = useCallback((position: number[], moveType: string) => {
    ws.current?.send(JSON.stringify({
      type: "make_move", position, move_type: moveType,
    }));
    setAiThinking(true);
  }, []);

  return { gameState, isConnected, aiThinking, lastAgentMove, newGame, makeMove };
}
```

**Step 3: Commit**

```bash
git add webapp/src/hooks/ webapp/src/types/
git commit -m "feat: add WebSocket hook and game state types"
```

---

### Task 9: Play page — full game UI

**Files:**
- Create: `webapp/src/app/play/page.tsx`
- Modify: `webapp/src/app/page.tsx` (link to /play)

**Context:** The main game play page combining Board, Hand, GameInfo, and MoveHistory with the WebSocket hook. Three-column layout: Hand | Board | Info.

**Step 1: Create play page**

Full game page implementing the layout from the UI design guide with the move flow: select card → see matching positions → click cell → token places → AI responds.

**Step 2: Add game over modal**

Result overlay card showing winner, stats, and play-again CTA.

**Step 3: Update landing page**

Link [Start Playing] button to `/play`.

**Step 4: Verify end-to-end**

Start the FastAPI backend (`uvicorn api.main:app`) and the Next.js dev server (`npm run dev`), play a game through the UI.

**Step 5: Commit**

```bash
git add webapp/src/app/
git commit -m "feat: add play page with full game UI"
```

---

## Phase 3C: Spectator Mode

### Task 10: Backend spectator support

**Files:**
- Create: `api/spectator.py`
- Modify: `api/ws_handler.py` (add spectate message type)
- Modify: `api/main.py` (add spectator routes)
- Create: `tests/test_api/test_spectator.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_spectator.py
import pytest
from api.spectator import SpectatorManager
from agents.random_agent import RandomAgent


class TestSpectatorManager:
    def test_start_match(self):
        manager = SpectatorManager()
        match_id = manager.start_match(
            agent1_name="random",
            agent2_name="heuristic",
            seed=42,
        )
        assert match_id is not None
        assert manager.get_match(match_id) is not None

    def test_advance_match(self):
        manager = SpectatorManager()
        match_id = manager.start_match("random", "heuristic", seed=42)
        result = manager.advance_match(match_id)
        assert "board" in result or "error" in result

    def test_list_matches(self):
        manager = SpectatorManager()
        manager.start_match("random", "heuristic", seed=42)
        matches = manager.list_matches()
        assert len(matches) >= 1
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_spectator.py -v`
Expected: FAIL

**Step 3: Implement api/spectator.py**

```python
# api/spectator.py
from __future__ import annotations
import uuid
from engine.game_state import GameState
from engine.board import BOARD_LAYOUT
from agents.base import Agent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

SPECTATOR_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
}


class SpectatorMatch:
    def __init__(self, agent1: Agent, agent2: Agent, seed: int | None = None):
        self.match_id = str(uuid.uuid4())
        self.agents = {1: agent1, 2: agent2}
        self.game_state = GameState.new_game(seed=seed)
        self.move_count = 0

    @property
    def is_over(self) -> bool:
        return self.game_state.is_terminal()

    def advance(self) -> dict:
        if self.is_over:
            return {"done": True, "winner": self.game_state.get_winner()}
        current = self.game_state.current_player
        agent = self.agents[current]
        info_set = self.game_state.to_information_set(current)
        action = agent.select_action(self.game_state, info_set)
        self.game_state = self.game_state.apply_action(action)
        self.move_count += 1
        return {
            "move": {"player": current, "position": [action.row, action.col], "type": action.action_type.name.lower()},
            "board": [[{"card": BOARD_LAYOUT[r][c], "occupant": int(self.game_state.occupancy[r][c])} for c in range(10)] for r in range(10)],
            "move_count": self.move_count,
            "is_over": self.is_over,
            "winner": self.game_state.get_winner(),
        }


class SpectatorManager:
    def __init__(self):
        self.matches: dict[str, SpectatorMatch] = {}

    def start_match(self, agent1_name: str, agent2_name: str, seed: int | None = None) -> str:
        a1_cls = SPECTATOR_AGENTS.get(agent1_name)
        a2_cls = SPECTATOR_AGENTS.get(agent2_name)
        if not a1_cls or not a2_cls:
            raise ValueError(f"Unknown agent")
        a1 = a1_cls(name=agent1_name, seed=seed) if agent1_name == "random" else a1_cls(name=agent1_name)
        a2 = a2_cls(name=agent2_name, seed=(seed or 0) + 1) if agent2_name == "random" else a2_cls(name=agent2_name)
        match = SpectatorMatch(a1, a2, seed=seed)
        self.matches[match.match_id] = match
        return match.match_id

    def get_match(self, match_id: str) -> SpectatorMatch | None:
        return self.matches.get(match_id)

    def advance_match(self, match_id: str) -> dict:
        match = self.matches.get(match_id)
        if not match:
            return {"error": "Match not found"}
        return match.advance()

    def list_matches(self) -> list[dict]:
        return [
            {"match_id": m.match_id, "move_count": m.move_count, "is_over": m.is_over}
            for m in self.matches.values()
        ]
```

**Step 4: Update ws_handler.py to add spectate support**

Add `"spectate_start"`, `"spectate_advance"`, and `"spectate_list"` message types.

**Step 5: Run tests**

Run: `pytest tests/test_api/test_spectator.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add api/spectator.py api/ws_handler.py api/main.py tests/test_api/test_spectator.py
git commit -m "feat: add spectator mode backend with AI-vs-AI match management"
```

---

### Task 11: Spectator frontend page

**Files:**
- Create: `webapp/src/app/spectate/page.tsx`
- Create: `webapp/src/hooks/useSpectatorWebSocket.ts`

**Step 1: Create spectator WebSocket hook**

Hook that connects to the spectator endpoint, starts matches, and receives board updates at configurable speed.

**Step 2: Create spectate page**

Larger board (no hand panel), auto-play with speed controls (0.5s, 1s, 2s), play/pause toggle, agent name labels, move counter.

**Step 3: Commit**

```bash
git add webapp/src/app/spectate/ webapp/src/hooks/
git commit -m "feat: add spectator mode frontend page"
```

---

## Phase 3D: Kubernetes Manifests

### Task 12: Kubernetes manifests and Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `k8s/api-deployment.yaml`
- Create: `k8s/training-job.yaml`
- Create: `k8s/self-play-job.yaml`
- Create: `k8s/arena-job.yaml`
- Create: `k8s/pvc.yaml`
- Create: `k8s/configmap.yaml`

**Step 1: Create Dockerfile**

Multi-stage Docker build: Python 3.11 base, install dependencies, copy source.

**Step 2: Create K8s manifests**

- `api-deployment.yaml`: FastAPI deployment with HPA
- `training-job.yaml`: GPU training job with PVC mount
- `self-play-job.yaml`: CPU self-play workers
- `arena-job.yaml`: Arena evaluation job
- `pvc.yaml`: 50Gi persistent volume for checkpoints/buffer
- `configmap.yaml`: Training config YAML injection

**Step 3: Verify manifests**

```bash
kubectl --dry-run=client -f k8s/ apply
```

**Step 4: Commit**

```bash
git add Dockerfile k8s/
git commit -m "feat: add Dockerfile and Kubernetes manifests for deployment"
```

---

## Verification

### Task 13: Full Phase 3 verification

**Step 1: Run all Python tests**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS (~160+ tests)

**Step 2: Verify frontend builds**

```bash
cd webapp && npm run build
```

**Step 3: Smoke tests**

```bash
# New agents play
python -c "
from engine.game_state import GameState
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from training.arena import Arena
arena = Arena(num_games=3)
print('Arena:', arena.evaluate(RandomAgent('r', seed=1), HeuristicAgent('h')))
"

# API with new agents
python -c "
from api.game_session import GameSessionManager
m = GameSessionManager()
print('Agents:', m.list_agents())
"

# Spectator
python -c "
from api.spectator import SpectatorManager
sm = SpectatorManager()
mid = sm.start_match('random', 'heuristic', seed=42)
for _ in range(5):
    print(sm.advance_match(mid)['move_count'])
"
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: verify full Phase 3 pipeline"
```

---

## Summary

**Phase 3 delivers (13 tasks):**

| Track | Tasks | What |
|-------|-------|------|
| **3A: New Agents** | 1-4 | Decision Transformer model+agent, AlphaZero+belief agent, session registration |
| **3B: React Frontend** | 5-9 | Next.js scaffold, Board/Token/Hand/GameInfo components, WebSocket hooks, play page |
| **3C: Spectator Mode** | 10-11 | Backend SpectatorManager, frontend spectate page |
| **3D: Kubernetes** | 12 | Dockerfile, K8s manifests (api, training, self-play, arena, PVC, configmap) |
| **Verification** | 13 | Full test suite + frontend build + smoke tests |

**Phase 4 (next):**
- Polish, optimization, deployment
- HuggingFace Hub model publishing
- Documentation
