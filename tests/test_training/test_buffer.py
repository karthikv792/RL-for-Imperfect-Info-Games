# tests/test_training/test_buffer.py
import numpy as np
from training.experience_buffer import ExperienceBuffer


class TestExperienceBuffer:
    def test_add_and_size(self):
        buf = ExperienceBuffer(max_size=100)
        state = np.zeros((22, 10, 10))
        policy = np.ones(300) / 300
        buf.add(state, policy, 1.0)
        assert len(buf) == 1

    def test_max_size(self):
        buf = ExperienceBuffer(max_size=5)
        for i in range(10):
            buf.add(np.zeros((22, 10, 10)), np.ones(300) / 300, float(i))
        assert len(buf) == 5

    def test_sample(self):
        buf = ExperienceBuffer(max_size=100)
        for i in range(20):
            buf.add(np.random.randn(22, 10, 10).astype(np.float32), np.ones(300) / 300, 1.0)
        states, policies, values = buf.sample(batch_size=8)
        assert states.shape == (8, 22, 10, 10)
        assert policies.shape == (8, 300)
        assert values.shape == (8,)

    def test_save_and_load(self, tmp_path):
        buf = ExperienceBuffer(max_size=100)
        for i in range(10):
            buf.add(np.random.randn(22, 10, 10).astype(np.float32), np.ones(300) / 300, 1.0)
        path = str(tmp_path / "buffer.npz")
        buf.save(path)
        buf2 = ExperienceBuffer.load(path, max_size=100)
        assert len(buf2) == len(buf)
