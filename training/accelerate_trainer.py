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
