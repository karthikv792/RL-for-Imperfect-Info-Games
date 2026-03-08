from __future__ import annotations
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
