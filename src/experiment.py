"""Prediction-stability experiment: how much do MobileNetV3's ImageNet predictions move under augmentation?

Metrics (all computed on this machine when you run it; nothing is pre-filled):
  top1_unchanged   share of images whose top-1 ImageNet class is the same before and after the transform
  top5_overlap     mean size of the top-5 intersection divided by 5
  top1_conf_change mean change in confidence of each image's ORIGINAL top-1 class (counted as 0 if it left the top 5)
  ms_per_image     mean inference time per image
These measure STABILITY, not accuracy: CIFAR-10 labels are not ImageNet labels, so no accuracy is computed.
"""
from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .augmentation import AugmentParams, apply_augmentations
from .config import NOISE_SEED, WORK_SIZE
from .imageio import upscale_square

SETTINGS: list[tuple[str, AugmentParams]] = [
    ("brightness 0.5", AugmentParams(brightness=0.5)),
    ("brightness 1.5", AugmentParams(brightness=1.5)),
    ("rotation 30", AugmentParams(rotation=30)),
    ("rotation 90", AugmentParams(rotation=90)),
    ("blur 2", AugmentParams(blur=2.0)),
    ("contrast 0.5", AugmentParams(contrast=0.5)),
    ("noise 25", AugmentParams(noise=25)),
    ("gamma 2 (low light)", AugmentParams(gamma=2.0)),
]


@dataclass(frozen=True)
class Row:
    setting: str
    images: int
    top1_unchanged: float
    top5_overlap: float
    top1_conf_change: float
    ms_per_image: float


def run_experiment(classifier, dataset, n_images: int = 200, seed: int = 0, settings=None) -> list[Row]:
    if n_images < 1:
        raise ValueError("n_images must be at least 1")
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_images, len(dataset)))
    images = [upscale_square(dataset.image(i), WORK_SIZE) for i in indices]
    base = [classifier.predict(img, 5) for img in images]
    rows = []
    for name, params in (settings or SETTINGS):
        same = overlap = delta = 0.0
        t0 = time.perf_counter()
        for k, (img, b) in enumerate(zip(images, base)):
            a = classifier.predict(apply_augmentations(img, params, seed=NOISE_SEED + k), 5)
            same += a[0].class_index == b[0].class_index
            overlap += len({p.class_index for p in a} & {p.class_index for p in b}) / 5
            delta += a[0].confidence - b[0].confidence if a[0].class_index == b[0].class_index else \
                next((p.confidence for p in a if p.class_index == b[0].class_index), 0.0) - b[0].confidence
        n = len(images)
        rows.append(Row(name, n, same / n, overlap / n, delta / n, (time.perf_counter() - t0) * 1000 / n))
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "images", "top1_unchanged", "top5_overlap", "top1_conf_change", "ms_per_image"])
        for r in rows:
            w.writerow([r.setting, r.images, f"{r.top1_unchanged:.4f}", f"{r.top5_overlap:.4f}",
                        f"{r.top1_conf_change:.4f}", f"{r.ms_per_image:.1f}"])
