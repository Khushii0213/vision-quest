"""CIFAR-10 test split (10,000 labelled 32x32 images).

IMPORTANT: these labels (airplane, cat, ...) are the *ground truth* used for game scoring.
They are NOT ImageNet labels. The AI model predicts ImageNet-1K classes, a different label
space, and is never used to decide whether a game answer is correct.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
_EXTRACTED = "cifar-10-batches-py"


class DatasetUnavailableError(RuntimeError):
    """The dataset could not be downloaded or read."""


class CifarDataset:
    def __init__(self, data: np.ndarray, labels: np.ndarray, class_names=CIFAR10_CLASSES):
        if data.ndim != 4 or data.shape[0] != len(labels):
            raise ValueError("data must be (N, H, W, 3) with one label per image")
        self._data, self._labels, self.class_names = data, np.asarray(labels), tuple(class_names)
        self._index: dict[int, list[int]] | None = None

    def __len__(self) -> int:
        return len(self._labels)

    def label(self, idx: int) -> int:
        return int(self._labels[idx])

    def label_name(self, idx: int) -> str:
        return self.class_names[self.label(idx)]

    def image(self, idx: int) -> Image.Image:
        if not 0 <= idx < len(self):
            raise IndexError(f"image index {idx} out of range")
        return Image.fromarray(self._data[idx]).convert("RGB")

    def index_by_class(self) -> dict[int, list[int]]:
        if self._index is None:
            self._index = {c: np.flatnonzero(self._labels == c).tolist() for c in range(len(self.class_names))}
        return self._index

    @staticmethod
    def is_downloaded(root: Path) -> bool:
        return (Path(root) / _EXTRACTED / "test_batch").is_file()

    @classmethod
    def load(cls, root: Path, download: bool = True) -> "CifarDataset":
        try:
            from torchvision.datasets import CIFAR10
        except ImportError as exc:
            raise DatasetUnavailableError("torchvision is not installed. Run: pip install -r requirements.txt") from exc
        try:
            Path(root).mkdir(parents=True, exist_ok=True)
            ds = CIFAR10(root=str(root), train=False, download=download)
        except (OSError, RuntimeError) as exc:  # URLError is an OSError; bad checksum is a RuntimeError
            raise DatasetUnavailableError(
                "CIFAR-10 could not be downloaded or verified. Check your internet connection and try again "
                f"(about 170 MB, stored in {root}). Details: {exc}") from exc
        names = tuple(ds.classes)
        if names != CIFAR10_CLASSES:
            raise DatasetUnavailableError(f"Unexpected CIFAR-10 class list: {names}")
        return cls(np.asarray(ds.data), np.asarray(ds.targets), names)
