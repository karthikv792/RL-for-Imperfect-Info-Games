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
