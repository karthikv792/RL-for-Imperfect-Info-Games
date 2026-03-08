# Phase 5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add game replay persistence and viewer, performance benchmarks, comprehensive README, model checkpoint selection, and end-to-end tests to make the project production-ready.

**Architecture:** Phase 5 is the final "production hardening" phase. Track A adds persistent game storage and a replay viewer. Track B adds performance benchmarks and model versioning. Track C writes comprehensive documentation. Track D adds end-to-end tests.

**Tech Stack:** Python 3.11+, SQLite (via built-in sqlite3), pytest-benchmark, Playwright, FastAPI

**Current state (Phase 4 complete):**
- 161 Python tests passing, ruff clean
- 5 frontend pages: /, /play, /spectate, /leaderboard, /_not-found
- 4 AI agents registered, all with training configs
- CI/CD pipeline, K8s manifests, HF Hub publisher ready

---

## Phase 5A: Game Replay Persistence & Viewer

### Task 1: Game replay storage backend

**Files:**
- Create: `api/replay_store.py`
- Create: `tests/test_api/test_replay_store.py`
- Modify: `api/game_session.py` — auto-save completed games

**Step 1: Write the failing test**

```python
# tests/test_api/test_replay_store.py
import pytest
from api.replay_store import ReplayStore


class TestReplayStore:
    def test_save_and_load(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        replay_id = store.save_replay(
            agent1="human",
            agent2="heuristic",
            winner="heuristic",
            moves=[
                {"player": 1, "position": [1, 2], "type": "place"},
                {"player": 2, "position": [3, 4], "type": "place"},
            ],
            seed=42,
        )
        replay = store.get_replay(replay_id)
        assert replay is not None
        assert replay["winner"] == "heuristic"
        assert len(replay["moves"]) == 2

    def test_list_replays(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        store.save_replay("human", "random", "human", [], seed=1)
        store.save_replay("human", "heuristic", "heuristic", [], seed=2)
        replays = store.list_replays(limit=10)
        assert len(replays) == 2

    def test_list_by_agent(self, tmp_path):
        store = ReplayStore(db_path=str(tmp_path / "replays.db"))
        store.save_replay("human", "random", "human", [], seed=1)
        store.save_replay("human", "heuristic", "heuristic", [], seed=2)
        replays = store.list_replays(agent="random")
        assert len(replays) == 1
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_api/test_replay_store.py -v`
Expected: FAIL

**Step 3: Implement api/replay_store.py**

```python
# api/replay_store.py
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone


class ReplayStore:
    """Persists game replays to SQLite."""

    def __init__(self, db_path: str = "data/replays.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replays (
                id TEXT PRIMARY KEY,
                agent1 TEXT NOT NULL,
                agent2 TEXT NOT NULL,
                winner TEXT,
                moves TEXT NOT NULL,
                seed INTEGER,
                num_moves INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def save_replay(
        self,
        agent1: str,
        agent2: str,
        winner: str | None,
        moves: list[dict],
        seed: int | None = None,
    ) -> str:
        replay_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO replays (id, agent1, agent2, winner, moves, seed, num_moves, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (replay_id, agent1, agent2, winner, json.dumps(moves), seed, len(moves), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
        return replay_id

    def get_replay(self, replay_id: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM replays WHERE id = ?", (replay_id,)).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "id": row["id"],
            "agent1": row["agent1"],
            "agent2": row["agent2"],
            "winner": row["winner"],
            "moves": json.loads(row["moves"]),
            "seed": row["seed"],
            "num_moves": row["num_moves"],
            "created_at": row["created_at"],
        }

    def list_replays(self, limit: int = 20, agent: str | None = None) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if agent:
            rows = conn.execute(
                "SELECT id, agent1, agent2, winner, num_moves, created_at FROM replays WHERE agent1 = ? OR agent2 = ? ORDER BY created_at DESC LIMIT ?",
                (agent, agent, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, agent1, agent2, winner, num_moves, created_at FROM replays ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
```

**Step 4: Run tests**

Run: `pytest tests/test_api/test_replay_store.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add api/replay_store.py tests/test_api/test_replay_store.py
git commit -m "feat: add SQLite-backed game replay storage"
```

---

### Task 2: Replay REST endpoints

**Files:**
- Modify: `api/routes/game.py` — add replay endpoints
- Modify: `api/game_session.py` — auto-save games on completion
- Create: `tests/test_api/test_replay_routes.py`

**Step 1: Write the failing test**

```python
# tests/test_api/test_replay_routes.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestReplayRoutes:
    def test_list_replays_empty(self):
        client = TestClient(app)
        resp = client.get("/api/replays")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_replay_not_found(self):
        client = TestClient(app)
        resp = client.get("/api/replays/nonexistent")
        assert resp.status_code == 404
```

**Step 2: Add endpoints to routes/game.py**

```python
@router.get("/replays")
def list_replays(agent: str | None = None, limit: int = 20):
    return replay_store.list_replays(limit=limit, agent=agent)

@router.get("/replays/{replay_id}")
def get_replay(replay_id: str):
    replay = replay_store.get_replay(replay_id)
    if replay is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Replay not found")
    return replay
```

**Step 3: Run tests, commit**

```bash
git add api/routes/game.py api/game_session.py tests/test_api/test_replay_routes.py
git commit -m "feat: add replay REST endpoints and auto-save on game completion"
```

---

### Task 3: Replay viewer frontend

**Files:**
- Create: `webapp/src/app/replay/[id]/page.tsx`
- Create: `webapp/src/hooks/useReplayPlayer.ts`
- Modify: `webapp/src/app/page.tsx` — add "Recent Games" link

**Step 1: Create useReplayPlayer hook**

A hook that takes a replay's moves array and provides play/pause/step/rewind controls with a timeline slider. Reconstructs board state at each step.

**Step 2: Create replay page**

Shows the board with timeline controls (like a video player), move-by-move navigation, and game metadata (agents, winner, date).

**Step 3: Add link from landing page**

**Step 4: Commit**

```bash
git add webapp/src/
git commit -m "feat: add game replay viewer with timeline controls"
```

---

## Phase 5B: Benchmarks & Model Versioning

### Task 4: Performance benchmark suite

**Files:**
- Create: `tests/test_benchmarks/test_performance.py`
- Create: `tests/test_benchmarks/__init__.py`

**Step 1: Write benchmark tests**

```python
# tests/test_benchmarks/test_performance.py
import pytest
from engine.game_state import GameState
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


class TestPerformanceBenchmarks:
    def test_legal_actions_speed(self, benchmark):
        gs = GameState.new_game(seed=42)
        benchmark(gs.get_legal_actions)

    def test_apply_action_speed(self, benchmark):
        gs = GameState.new_game(seed=42)
        actions = gs.get_legal_actions()

        def apply():
            gs.apply_action(actions[0])

        benchmark(apply)

    def test_random_agent_game(self, benchmark):
        def play_game():
            gs = GameState.new_game(seed=42)
            agent = RandomAgent("r", seed=42)
            moves = 0
            while not gs.is_terminal() and moves < 200:
                info = gs.to_information_set(gs.current_player)
                action = agent.select_action(gs, info)
                gs = gs.apply_action(action)
                moves += 1

        benchmark(play_game)

    def test_heuristic_agent_game(self, benchmark):
        def play_game():
            gs = GameState.new_game(seed=42)
            agent = HeuristicAgent("h")
            moves = 0
            while not gs.is_terminal() and moves < 200:
                info = gs.to_information_set(gs.current_player)
                action = agent.select_action(gs, info)
                gs = gs.apply_action(action)
                moves += 1

        benchmark(play_game)

    def test_tokenizer_speed(self, benchmark):
        from models.board_tokenizer import BoardTokenizer
        gs = GameState.new_game(seed=42)
        tok = BoardTokenizer()
        benchmark(tok.tokenize, gs, player_perspective=1)
```

**Step 2: Run benchmarks**

Run: `pytest tests/test_benchmarks/ -v --benchmark-only`

**Step 3: Commit**

```bash
git add tests/test_benchmarks/
git commit -m "feat: add performance benchmark suite with pytest-benchmark"
```

---

### Task 5: Model checkpoint selection API

**Files:**
- Create: `api/model_registry.py`
- Create: `tests/test_api/test_model_registry.py`
- Modify: `api/routes/game.py` — add model listing endpoint

**Step 1: Write the failing test**

```python
# tests/test_api/test_model_registry.py
import pytest
from api.model_registry import ModelRegistry


class TestModelRegistry:
    def test_list_models_empty(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        models = registry.list_models()
        assert models == []

    def test_register_and_list(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        (tmp_path / "ismcts_v1").mkdir()
        (tmp_path / "ismcts_v1" / "model.pt").write_bytes(b"fake")
        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "ismcts_v1"

    def test_get_model_path(self, tmp_path):
        registry = ModelRegistry(checkpoint_dir=str(tmp_path))
        (tmp_path / "rebel_v2").mkdir()
        (tmp_path / "rebel_v2" / "model.pt").write_bytes(b"fake")
        path = registry.get_model_path("rebel_v2")
        assert path is not None
        assert "rebel_v2" in path
```

**Step 2: Implement api/model_registry.py**

```python
# api/model_registry.py
from __future__ import annotations
import os
from pathlib import Path


class ModelRegistry:
    """Discovers and serves model checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

    def list_models(self) -> list[dict]:
        if not self.checkpoint_dir.exists():
            return []
        models = []
        for d in sorted(self.checkpoint_dir.iterdir()):
            if d.is_dir() and (d / "model.pt").exists():
                size = (d / "model.pt").stat().st_size
                models.append({
                    "name": d.name,
                    "size_bytes": size,
                    "path": str(d / "model.pt"),
                })
        return models

    def get_model_path(self, name: str) -> str | None:
        p = self.checkpoint_dir / name / "model.pt"
        return str(p) if p.exists() else None
```

**Step 3: Add endpoint**

```python
@router.get("/models")
def list_models():
    return model_registry.list_models()
```

**Step 4: Run tests, commit**

```bash
git add api/model_registry.py tests/test_api/test_model_registry.py api/routes/game.py
git commit -m "feat: add model checkpoint registry and discovery API"
```

---

## Phase 5C: Documentation

### Task 6: Comprehensive README

**Files:**
- Modify: `README.md`

**Step 1: Write comprehensive README**

Structure:
- Project title, badges (CI, Python version, license)
- One-line description
- Architecture diagram (ASCII)
- Quick start (3-step setup)
- Project structure overview
- AI Agents section (table of 4 agents with descriptions)
- Training section (how to train, configs)
- Web application section (how to run)
- API reference (key endpoints)
- Deployment section (Docker, K8s)
- Contributing section
- License

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with setup, training, and deployment guides"
```

---

### Task 7: API documentation with OpenAPI

**Files:**
- Modify: `api/main.py` — enhance FastAPI metadata
- Modify: `api/routes/game.py` — add docstrings and response models

**Step 1: Add OpenAPI metadata to FastAPI app**

Add description, tags, and versioning. Add response models to endpoints for auto-generated docs.

**Step 2: Verify docs page**

FastAPI auto-generates docs at `/docs` (Swagger UI) and `/redoc`.

**Step 3: Commit**

```bash
git add api/
git commit -m "docs: enhance API documentation with OpenAPI metadata and response models"
```

---

## Phase 5D: End-to-End Testing

### Task 8: API integration tests

**Files:**
- Create: `tests/test_integration/test_full_flow.py`
- Create: `tests/test_integration/__init__.py`

**Step 1: Write integration tests**

```python
# tests/test_integration/test_full_flow.py
import pytest
from fastapi.testclient import TestClient
from api.main import app


class TestFullFlow:
    def test_create_game_play_move_check_leaderboard(self):
        """End-to-end: create game, play a move, verify state updates."""
        client = TestClient(app)

        # List agents
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert "random" in agents

        # Create game
        resp = client.post("/api/game", json={"agent_name": "random", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["session_id"]

        # Get game state
        resp = client.get(f"/api/game/{session_id}")
        assert resp.status_code == 200
        state = resp.json()
        assert state["turn"] == "human"
        assert len(state["hand"]) > 0
        assert len(state["legal_moves"]) > 0

        # Make a move
        move = state["legal_moves"][0]
        resp = client.post(f"/api/game/{session_id}/move", json={
            "position": move["position"],
            "move_type": move["type"],
        })
        assert resp.status_code == 200

    def test_spectator_flow(self):
        """End-to-end: start spectator match, advance 5 moves."""
        client = TestClient(app)

        # Start match
        resp = client.post("/api/spectate?agent1=random&agent2=heuristic&seed=42")
        assert resp.status_code == 200
        match_id = resp.json()["match_id"]

        # Advance 5 times
        for _ in range(5):
            resp = client.post(f"/api/spectate/{match_id}/advance")
            assert resp.status_code == 200

    def test_models_endpoint(self):
        client = TestClient(app)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
```

**Step 2: Run tests, commit**

```bash
git add tests/test_integration/
git commit -m "test: add end-to-end API integration tests"
```

---

### Task 9: Remove legacy Sequence/ directory

**Files:**
- Delete: `Sequence/` (entire legacy directory)

**Context:** The `Sequence/` directory contains old pre-Phase 1 code (1200+ lines with pygame, etc.) that's completely superseded by the modern codebase. It causes ruff lint warnings and adds confusion.

**Step 1: Verify no modern code imports from Sequence/**

```bash
grep -r "from Sequence" engine/ agents/ models/ training/ api/ tests/
grep -r "import Sequence" engine/ agents/ models/ training/ api/ tests/
```

**Step 2: Remove it**

```bash
rm -rf Sequence/
```

**Step 3: Verify tests pass**

```bash
pytest tests/ -q --tb=short
```

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove legacy Sequence/ directory"
```

---

## Verification

### Task 10: Full Phase 5 verification

**Step 1: Run all Python tests**

```bash
pytest tests/ -v --tb=short
```
Expected: ALL PASS (~175+ tests)

**Step 2: Run benchmarks**

```bash
pytest tests/test_benchmarks/ -v --benchmark-only
```

**Step 3: Verify frontend builds**

```bash
cd webapp && npm run build
```

**Step 4: Run ruff lint**

```bash
ruff check . --exclude Sequence/
```

**Step 5: Smoke tests**

```bash
# Replay store
python -c "
from api.replay_store import ReplayStore
store = ReplayStore(db_path='/tmp/test_replays.db')
rid = store.save_replay('human', 'random', 'human', [{'move': 1}], seed=42)
print('Saved replay:', rid)
print('Loaded:', store.get_replay(rid)['winner'])
print('Listed:', len(store.list_replays()))
"

# Model registry
python -c "
from api.model_registry import ModelRegistry
reg = ModelRegistry(checkpoint_dir='/tmp/no_checkpoints')
print('Models:', reg.list_models())
"

# Full agent list
python -c "
from api.game_session import GameSessionManager
print('Agents:', GameSessionManager().list_agents())
"
```

---

## Summary

**Phase 5 delivers (10 tasks):**

| Track | Tasks | What |
|-------|-------|------|
| **5A: Replay System** | 1-3 | SQLite replay store, REST endpoints, frontend replay viewer |
| **5B: Benchmarks & Models** | 4-5 | pytest-benchmark suite, model checkpoint registry |
| **5C: Documentation** | 6-7 | Comprehensive README, OpenAPI docs |
| **5D: Testing & Cleanup** | 8-9 | E2E integration tests, remove legacy Sequence/ dir |
| **Verification** | 10 | Full test suite + benchmarks + build + smoke tests |
