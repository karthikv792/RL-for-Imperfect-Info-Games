# training/trainer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
        """Upscale (B, 22, 10, 10) to (B, 22, target_size, target_size) via nearest neighbor."""
        scale = self.config.target_size // 10
        # np.repeat along spatial dims
        up = np.repeat(np.repeat(states, scale, axis=2), scale, axis=3)
        # Pad if needed to reach exact target size (e.g. 220 -> 224)
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
