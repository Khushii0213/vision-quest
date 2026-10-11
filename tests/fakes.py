"""Test-only stand-ins. These are NEVER imported by the application."""
from __future__ import annotations

import numpy as np

from src.dataset import CIFAR10_CLASSES, CifarDataset
from src.model import Prediction


def fake_dataset(per_class: int = 20) -> CifarDataset:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), per_class)
    data = rng.integers(0, 255, size=(len(labels), 32, 32, 3), dtype=np.uint8)
    return CifarDataset(data, labels, CIFAR10_CLASSES)


class FakeClassifier:
    """Deterministic classifier used only to exercise the UI in tests."""

    def predict(self, image, top_k=5):
        m = float(np.asarray(image.convert("L")).mean())
        base = int(m) % 900
        return [Prediction(base + i, f"class_{base + i}", 0.5 / (i + 1)) for i in range(top_k)]
