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
