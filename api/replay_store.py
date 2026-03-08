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
