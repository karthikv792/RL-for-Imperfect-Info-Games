# Phase 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Polish the UI with responsive design and accessibility, add leaderboard and game replay features, set up CI/CD, clean up legacy code, and prepare for HuggingFace Hub model publishing.

**Architecture:** Phase 4 is the "polish, optimize, publish" phase. Track A handles frontend polish (responsive design, accessibility, animations). Track B adds remaining features (leaderboard, replay). Track C sets up CI/CD and cleans up legacy code. Track D prepares HuggingFace Hub publishing infrastructure.

**Tech Stack:** React 18, Next.js 14, TypeScript, Tailwind CSS, GitHub Actions, HuggingFace Hub, pytest

**Current state (Phase 3 complete):**
- 156 Python tests passing
- 4 AI agents: random, heuristic, decision_tf, alphazero_belief
- Full React/Next.js frontend with play + spectate pages
- FastAPI backend with REST + WebSocket
- Kubernetes manifests ready
- Legacy agent directories still present (BasePlayer/, BasePolicyAgent/, NeuralNetworkAgent/, RandomAgent/)

---

## Phase 4A: Frontend Polish

### Task 1: Responsive design and mobile support

**Files:**
- Modify: `webapp/src/app/globals.css`
- Modify: `webapp/src/components/Board.tsx`
- Modify: `webapp/src/components/Hand.tsx`
- Modify: `webapp/src/components/GameInfo.tsx`
- Modify: `webapp/src/components/MoveHistory.tsx`
- Modify: `webapp/src/app/play/page.tsx`
- Modify: `webapp/src/app/spectate/page.tsx`
- Modify: `webapp/src/app/page.tsx`

**Step 1: Add responsive utilities to globals.css**

Add responsive breakpoint utilities and mobile-first adjustments:
```css
/* Mobile-first responsive adjustments */
@media (max-width: 768px) {
  .game-layout {
    flex-direction: column;
  }
}
```

**Step 2: Make Board responsive**

Update Board.tsx to use responsive sizing — on mobile the board should take full width with smaller cells, on desktop maintain the current size. Add touch support (tap instead of hover).

**Step 3: Make Hand responsive**

On mobile, cards should scroll horizontally. On desktop, maintain the current centered row.

**Step 4: Make play page layout responsive**

Change from side-by-side (Board | Info) to stacked layout (Board → Hand → Info) on mobile. Use CSS grid or flex-wrap.

**Step 5: Make landing page responsive**

Ensure the hero section and agent picker work on small screens (stack agent cards vertically on mobile).

**Step 6: Make spectate page responsive**

Adjust controls layout for mobile.

**Step 7: Commit**

```bash
git add webapp/
git commit -m "feat: add responsive design and mobile support"
```

---

### Task 2: Accessibility improvements

**Files:**
- Modify: `webapp/src/components/BoardCell.tsx`
- Modify: `webapp/src/components/Board.tsx`
- Modify: `webapp/src/components/HandCard.tsx`
- Modify: `webapp/src/components/GameInfo.tsx`
- Modify: `webapp/src/app/play/page.tsx`

**Step 1: Add ARIA labels to board cells**

Each BoardCell should have an `aria-label` describing the cell state, e.g. "Row 3, Column 5: 4 of Hearts, empty" or "Row 1, Column 2: King of Spades, occupied by Gold, legal move available".

**Step 2: Add keyboard navigation to Board**

Allow arrow key navigation between cells. Enter/Space to place token on highlighted cell. Tab to move focus to board, hand, and info panels.

**Step 3: Add ARIA labels to HandCard**

Each card button should have aria-label like "7 of Hearts, selected" or "Jack of Spades, not playable".

**Step 4: Add live region for game status announcements**

Add `aria-live="polite"` region that announces: turn changes, AI moves, game over status.

**Step 5: Add focus indicators**

Ensure all interactive elements have visible focus indicators (2px ring on `:focus-visible`).

**Step 6: Commit**

```bash
git add webapp/src/
git commit -m "feat: add accessibility improvements with ARIA labels and keyboard navigation"
```

---

### Task 3: Enhanced animations and visual polish

**Files:**
- Modify: `webapp/src/app/globals.css`
- Modify: `webapp/src/components/Token.tsx`
- Modify: `webapp/src/components/BoardCell.tsx`
- Modify: `webapp/src/components/HandCard.tsx`
- Create: `webapp/src/components/ThinkingIndicator.tsx`

**Step 1: Add card deal animation**

```css
@keyframes card-deal {
  0% { transform: translateX(100px) rotate(10deg); opacity: 0; }
  100% { transform: translateX(0) rotate(0); opacity: 1; }
}
.animate-card-deal {
  animation: card-deal 0.4s var(--ease-out) forwards;
}
```

**Step 2: Add sequence completion animation**

When a sequence is completed, add a brief glow pulse to the sequence cells.

```css
@keyframes sequence-glow {
  0%, 100% { box-shadow: 0 0 8px currentColor; }
  50% { box-shadow: 0 0 20px currentColor; }
}
```

**Step 3: Create ThinkingIndicator component**

A pulsing dot animation with "AI is thinking..." text for the AI thinking state, replacing the simple pulse in GameInfo.

```typescript
// webapp/src/components/ThinkingIndicator.tsx
"use client";

export function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 text-amber-400">
      <div className="flex gap-1">
        {[0, 1, 2].map(i => (
          <div
            key={i}
            className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-bounce"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
      <span className="text-sm">AI is thinking...</span>
    </div>
  );
}
```

**Step 4: Improve HandCard hover/select animations**

Add smooth lift on hover (`transition: transform 200ms ease-out`) and golden ring on select.

**Step 5: Add prefers-reduced-motion support**

Wrap all animations in `@media (prefers-reduced-motion: no-preference)`.

**Step 6: Commit**

```bash
git add webapp/src/
git commit -m "feat: add enhanced animations and visual polish"
```

---

## Phase 4B: Feature Completion

### Task 4: Leaderboard backend

**Files:**
- Modify: `api/routes/game.py`
- Create: `api/leaderboard.py`
- Create: `tests/test_api/test_leaderboard.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_leaderboard.py
import pytest
from api.leaderboard import LeaderboardManager


class TestLeaderboardManager:
    def test_record_result(self):
        lb = LeaderboardManager()
        lb.record_result(agent1="random", agent2="heuristic", winner="heuristic")
        stats = lb.get_stats()
        assert "heuristic" in stats
        assert stats["heuristic"]["wins"] == 1

    def test_head_to_head(self):
        lb = LeaderboardManager()
        lb.record_result("random", "heuristic", "heuristic")
        lb.record_result("random", "heuristic", "random")
        h2h = lb.get_head_to_head("random", "heuristic")
        assert h2h["random"] == 1
        assert h2h["heuristic"] == 1

    def test_elo_ratings(self):
        lb = LeaderboardManager()
        for _ in range(5):
            lb.record_result("random", "heuristic", "heuristic")
        stats = lb.get_stats()
        assert stats["heuristic"]["elo"] > stats["random"]["elo"]
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_leaderboard.py -v`
Expected: FAIL

**Step 3: Implement api/leaderboard.py**

```python
# api/leaderboard.py
from __future__ import annotations
from training.elo import EloRating


class LeaderboardManager:
    """Tracks agent performance across games."""

    def __init__(self):
        self.elo = EloRating()
        self.results: list[dict] = []

    def record_result(self, agent1: str, agent2: str, winner: str | None) -> None:
        self.results.append({"agent1": agent1, "agent2": agent2, "winner": winner})
        if winner == agent1:
            self.elo.update(agent1, agent2, winner=1)
        elif winner == agent2:
            self.elo.update(agent1, agent2, winner=2)
        else:
            self.elo.update(agent1, agent2, winner=0)

    def get_stats(self) -> dict[str, dict]:
        stats: dict[str, dict] = {}
        for r in self.results:
            for agent in [r["agent1"], r["agent2"]]:
                if agent not in stats:
                    stats[agent] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
                stats[agent]["games"] += 1
            if r["winner"]:
                stats[r["winner"]]["wins"] += 1
                loser = r["agent2"] if r["winner"] == r["agent1"] else r["agent1"]
                stats[loser]["losses"] += 1
            else:
                stats[r["agent1"]]["draws"] += 1
                stats[r["agent2"]]["draws"] += 1

        for agent in stats:
            stats[agent]["elo"] = self.elo.get_rating(agent)

        return stats

    def get_head_to_head(self, agent1: str, agent2: str) -> dict[str, int]:
        h2h = {agent1: 0, agent2: 0, "draws": 0}
        for r in self.results:
            if set([r["agent1"], r["agent2"]]) == set([agent1, agent2]):
                if r["winner"] == agent1:
                    h2h[agent1] += 1
                elif r["winner"] == agent2:
                    h2h[agent2] += 1
                else:
                    h2h["draws"] += 1
        return h2h

    def get_leaderboard(self) -> list[dict]:
        stats = self.get_stats()
        return sorted(
            [{"agent": k, **v} for k, v in stats.items()],
            key=lambda x: x["elo"],
            reverse=True,
        )
```

**Step 4: Update routes/game.py to use LeaderboardManager**

Wire the `/api/leaderboard` endpoint to use the new LeaderboardManager instead of returning an empty list.

**Step 5: Run tests**

Run: `pytest tests/test_api/test_leaderboard.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add api/leaderboard.py api/routes/game.py tests/test_api/test_leaderboard.py
git commit -m "feat: add leaderboard backend with Elo ratings and head-to-head stats"
```

---

### Task 5: Leaderboard frontend page

**Files:**
- Create: `webapp/src/app/leaderboard/page.tsx`
- Modify: `webapp/src/app/page.tsx` (add link to leaderboard)

**Step 1: Create leaderboard page**

Table showing agent rankings with Elo, win/loss/draw counts. Fetches from `/api/leaderboard`.

**Step 2: Add navigation link from landing page**

Add a "View Leaderboard" link to the landing page.

**Step 3: Commit**

```bash
git add webapp/src/app/leaderboard/ webapp/src/app/page.tsx
git commit -m "feat: add leaderboard frontend page"
```

---

### Task 6: WebSocket reconnection and error handling

**Files:**
- Modify: `webapp/src/hooks/useGameWebSocket.ts`
- Modify: `webapp/src/hooks/useSpectatorWebSocket.ts`
- Create: `webapp/src/components/ConnectionStatus.tsx`

**Step 1: Add auto-reconnection to WebSocket hooks**

Implement exponential backoff reconnection (1s, 2s, 4s, max 30s). Track reconnect attempts.

**Step 2: Create ConnectionStatus component**

A small banner that appears when disconnected showing "Reconnecting..." with a progress indicator.

**Step 3: Add error handling for failed moves**

Handle server error responses in the WebSocket message handler, show user-friendly error messages.

**Step 4: Commit**

```bash
git add webapp/src/hooks/ webapp/src/components/ConnectionStatus.tsx
git commit -m "feat: add WebSocket reconnection and error handling"
```

---

## Phase 4C: CI/CD and Cleanup

### Task 7: GitHub Actions CI pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Lint
        run: ruff check .
      - name: Test
        run: pytest tests/ -v --tb=short

  frontend-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - name: Install dependencies
        run: cd webapp && npm ci
      - name: Build
        run: cd webapp && npm run build

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t sequence-ai:test .
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions CI pipeline for tests, frontend build, and Docker"
```

---

### Task 8: Legacy code cleanup

**Files:**
- Delete: `agents/BasePlayer/` (entire directory)
- Delete: `agents/BasePolicyAgent/` (entire directory)
- Delete: `agents/NeuralNetworkAgent/` (entire directory)
- Delete: `agents/RandomAgent/` (entire directory)
- Delete: `main.py` (standalone test script)
- Delete: `main2.py` (standalone test script)
- Delete: `requirements.txt` (superseded by pyproject.toml)
- Delete: `.DS_Store`

**Step 1: Verify nothing imports from legacy directories**

Search for any imports from the legacy directories. Confirm none exist.

**Step 2: Remove legacy files**

```bash
rm -rf agents/BasePlayer agents/BasePolicyAgent agents/NeuralNetworkAgent agents/RandomAgent
rm -f main.py main2.py requirements.txt .DS_Store
```

**Step 3: Add .DS_Store to .gitignore**

Create or update `.gitignore` at root:
```
.DS_Store
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.pt
*.pth
wandb/
checkpoints/
```

**Step 4: Verify tests still pass**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS (156 tests)

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove legacy code and add .gitignore"
```

---

### Task 9: Ruff linting fixes

**Files:**
- Multiple Python files across the codebase

**Step 1: Run ruff check**

```bash
ruff check . --fix
```

**Step 2: Fix any remaining lint issues manually**

**Step 3: Verify tests still pass**

Run: `pytest tests/ -v --tb=short`

**Step 4: Commit**

```bash
git add -A
git commit -m "style: fix linting issues across codebase"
```

---

## Phase 4D: HuggingFace Hub Publishing

### Task 10: Model publishing utilities

**Files:**
- Create: `training/hub_publisher.py`
- Create: `tests/test_training/test_hub_publisher.py`

**Step 1: Write the failing test**

```python
# tests/test_training/test_hub_publisher.py
import pytest
from unittest.mock import MagicMock, patch
from training.hub_publisher import ModelCard, create_model_card


class TestModelCard:
    def test_create_card(self):
        card = create_model_card(
            model_name="sequence-ai-vit",
            architecture="ViT + MCTS",
            description="Vision Transformer policy-value network for Sequence",
            metrics={"elo": 1200, "win_rate_vs_random": 0.85},
        )
        assert "sequence-ai-vit" in card
        assert "ViT + MCTS" in card
        assert "1200" in card

    def test_card_has_required_sections(self):
        card = create_model_card(
            model_name="test",
            architecture="test",
            description="test",
            metrics={},
        )
        assert "## Model Description" in card
        assert "## Training" in card
        assert "## Usage" in card
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_training/test_hub_publisher.py -v`

**Step 3: Implement training/hub_publisher.py**

```python
# training/hub_publisher.py
from __future__ import annotations
import os
import json
import torch


def create_model_card(
    model_name: str,
    architecture: str,
    description: str,
    metrics: dict,
    training_config: dict | None = None,
) -> str:
    """Generate a HuggingFace model card in markdown format."""
    metrics_table = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    config_section = ""
    if training_config:
        config_section = f"\n### Training Configuration\n```json\n{json.dumps(training_config, indent=2)}\n```\n"

    return f"""---
tags:
  - sequence-game
  - board-game
  - reinforcement-learning
  - imperfect-information
library_name: pytorch
---

# {model_name}

{description}

## Model Description

- **Architecture:** {architecture}
- **Game:** Sequence (10x10 board, card-based strategy game)
- **Type:** Policy-Value network for imperfect information game

## Training

Trained via self-play with AlphaZero-style reinforcement learning.
{config_section}

## Metrics

| Metric | Value |
|--------|-------|
{metrics_table}

## Usage

```python
import torch
from models.vit_policy_value import ViTPolicyValue

model = ViTPolicyValue()
model.load_state_dict(torch.load("model.pt", weights_only=True))
```

## Citation

```bibtex
@misc{{sequence-ai,
  title={{{model_name}}},
  year={{2026}},
}}
```
"""


def publish_model(
    model: torch.nn.Module,
    model_name: str,
    repo_id: str,
    card_content: str,
    token: str | None = None,
) -> str:
    """Save model and push to HuggingFace Hub."""
    save_dir = f"/tmp/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    # Save model card
    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card_content)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=save_dir, repo_id=repo_id)
        return f"https://huggingface.co/{repo_id}"
    except ImportError:
        return save_dir
```

**Step 4: Run tests**

Run: `pytest tests/test_training/test_hub_publisher.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add training/hub_publisher.py tests/test_training/test_hub_publisher.py
git commit -m "feat: add HuggingFace Hub model publishing utilities"
```

---

### Task 11: Training config for all agents

**Files:**
- Create: `training/configs/decision_tf.yaml`
- Create: `training/configs/alphazero_belief.yaml`

**Step 1: Create Decision Transformer training config**

```yaml
# training/configs/decision_tf.yaml
model:
  type: decision_transformer
  state_dim: 317
  action_dim: 300
  max_timestep: 500
  context_length: 20
  hidden_size: 256
  n_layer: 6
  n_head: 8
  dropout: 0.1

training:
  learning_rate: 1.0e-4
  weight_decay: 0.01
  batch_size: 64
  num_epochs: 100
  gradient_accumulation_steps: 4
  warmup_steps: 1000
  max_grad_norm: 1.0
  save_every_epochs: 10
  checkpoint_dir: checkpoints/decision_tf

self_play:
  num_games_per_iteration: 500
  num_iterations: 50
  target_return: 1.0

arena:
  num_eval_games: 100
  promotion_threshold: 0.55

monitoring:
  wandb_project: sequence-ai
  wandb_run_name: decision-tf-v1
  log_every_steps: 50
```

**Step 2: Create AlphaZero+belief training config**

```yaml
# training/configs/alphazero_belief.yaml
model:
  type: alphazero_belief
  backbone: google/vit-base-patch16-224
  num_channels: 22
  action_dim: 300

search:
  num_simulations: 200
  num_determinizations: 20
  cpuct: 1.5
  temperature: 1.0
  temperature_drop_move: 30
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

training:
  learning_rate: 2.0e-4
  weight_decay: 0.01
  batch_size: 32
  num_epochs: 100
  gradient_accumulation_steps: 8
  warmup_steps: 500
  max_grad_norm: 1.0
  save_every_epochs: 5
  checkpoint_dir: checkpoints/alphazero_belief

self_play:
  num_games_per_iteration: 200
  num_iterations: 100
  buffer_size: 100000

arena:
  num_eval_games: 50
  promotion_threshold: 0.55

monitoring:
  wandb_project: sequence-ai
  wandb_run_name: alphazero-belief-v1
  log_every_steps: 25
```

**Step 3: Commit**

```bash
git add training/configs/
git commit -m "feat: add training configs for Decision Transformer and AlphaZero agents"
```

---

## Verification

### Task 12: Full Phase 4 verification

**Step 1: Run all Python tests**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS (~165+ tests)

**Step 2: Run ruff lint**

```bash
ruff check .
```
Expected: No errors

**Step 3: Verify frontend builds**

```bash
cd webapp && npm run build
```
Expected: Build succeeds with all pages

**Step 4: Smoke tests**

```bash
# All agents registered
python -c "
from api.game_session import GameSessionManager
m = GameSessionManager()
print('Agents:', m.list_agents())
assert len(m.list_agents()) == 4
"

# Leaderboard works
python -c "
from api.leaderboard import LeaderboardManager
lb = LeaderboardManager()
lb.record_result('random', 'heuristic', 'heuristic')
print('Leaderboard:', lb.get_leaderboard())
"

# Model card generation
python -c "
from training.hub_publisher import create_model_card
card = create_model_card('test', 'ViT', 'Test model', {'elo': 1200})
print('Card generated:', len(card), 'chars')
"
```

**Step 5: Verify no legacy code remains**

```bash
# Should not exist
ls agents/BasePlayer 2>&1 || echo "OK: Legacy cleaned"
ls agents/NeuralNetworkAgent 2>&1 || echo "OK: Legacy cleaned"
```

---

## Summary

**Phase 4 delivers (12 tasks):**

| Track | Tasks | What |
|-------|-------|------|
| **4A: Frontend Polish** | 1-3 | Responsive design, accessibility (ARIA + keyboard), animations |
| **4B: Features** | 4-6 | Leaderboard backend + frontend, WebSocket reconnection |
| **4C: CI/CD + Cleanup** | 7-9 | GitHub Actions, legacy removal, ruff linting |
| **4D: HF Publishing** | 10-11 | Model card generator, publishing utilities, training configs |
| **Verification** | 12 | Full test suite + lint + frontend build + smoke tests |
