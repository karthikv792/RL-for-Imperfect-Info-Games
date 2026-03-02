# training/experience_buffer.py
from __future__ import annotations
from collections import deque
import numpy as np


class ExperienceBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.states: deque[np.ndarray] = deque(maxlen=max_size)
        self.policies: deque[np.ndarray] = deque(maxlen=max_size)
        self.values: deque[float] = deque(maxlen=max_size)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        states = np.stack([self.states[i] for i in indices])
        policies = np.stack([self.policies[i] for i in indices])
        values = np.array([self.values[i] for i in indices], dtype=np.float32)
        return states, policies, values

    def save(self, path: str):
        np.savez_compressed(
            path,
            states=np.stack(list(self.states)),
            policies=np.stack(list(self.policies)),
            values=np.array(list(self.values), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str, max_size: int = 500_000) -> ExperienceBuffer:
        data = np.load(path)
        buf = cls(max_size=max_size)
        for s, p, v in zip(data["states"], data["policies"], data["values"]):
            buf.add(s, p, float(v))
        return buf
