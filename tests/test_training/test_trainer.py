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
