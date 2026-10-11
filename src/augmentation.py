"""Image augmentations (Pillow + NumPy).

Order of operations (fixed, so results are reproducible):
rotation -> contrast -> brightness -> gamma -> blur -> noise.
Noise is last so it is not smoothed away by the blur.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RANGES: dict[str, tuple[float, float]] = {
    "brightness": (0.2, 2.0),    # multiplicative factor, 1.0 = unchanged
    "rotation": (-180.0, 180.0),  # degrees, counter-clockwise
    "blur": (0.0, 6.0),          # Gaussian radius in pixels
    "contrast": (0.2, 2.0),      # multiplicative factor, 1.0 = unchanged
    "noise": (0.0, 60.0),        # Gaussian std-dev on the 0-255 scale
    "gamma": (0.4, 3.0),         # out = in ** gamma; > 1 darkens (low light)
}
NEUTRAL: dict[str, float] = {
    "brightness": 1.0, "rotation": 0.0, "blur": 0.0,
    "contrast": 1.0, "noise": 0.0, "gamma": 1.0,
}


@dataclass(frozen=True)
class AugmentParams:
    brightness: float = 1.0
    rotation: float = 0.0
    blur: float = 0.0
    contrast: float = 1.0
    noise: float = 0.0
    gamma: float = 1.0

    def validate(self) -> None:
        for name, (lo, hi) in RANGES.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if not lo <= value <= hi:
                raise ValueError(f"{name}={value} is outside the allowed range {lo}..{hi}")

    def changed(self) -> list[str]:
        """Names of parameters that differ from the neutral value."""
        return [f.name for f in fields(self) if getattr(self, f.name) != NEUTRAL[f.name]]

    @property
    def is_neutral(self) -> bool:
        return not self.changed()


def _rotate_reflect(img: Image.Image, angle: float) -> Image.Image:
    """Rotate about the centre; fill the corners by reflecting the image (no black wedges)."""
    w, h = img.size
    pad = int(max(w, h) * 0.3) + 1
    padded = np.pad(np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    rotated = Image.fromarray(padded).rotate(angle, resample=Image.Resampling.BICUBIC)
    return rotated.crop((pad, pad, pad + w, pad + h))


def apply_augmentations(image: Image.Image, params: AugmentParams, seed: int | None = None) -> Image.Image:
    """Return a new RGB image with ``params`` applied. The input is never modified."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    params.validate()
    img = image.convert("RGB")
    if img.width < 2 or img.height < 2:
        raise ValueError("image is too small to augment")

    if params.rotation % 360 != 0:
        img = _rotate_reflect(img, params.rotation)
    if params.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(params.contrast)
    if params.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(params.brightness)
    if params.gamma != 1.0:
        lut = [round(255 * (i / 255) ** params.gamma) for i in range(256)]
        img = img.point(lut * 3)
    if params.blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(params.blur))
    if params.noise > 0:
        rng = np.random.default_rng(seed)
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, params.noise, arr.shape).astype(np.float32)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def random_params(rng: random.Random, strength: float) -> AugmentParams:
    """Random per-tile transform for the game. ``strength`` in 0..1 grows with the round."""
    s = min(max(strength, 0.0), 1.0)
    p = 0.25 + 0.5 * s  # chance that each individual transform is applied
    vals = dict(NEUTRAL)
    if rng.random() < p:
        vals["rotation"] = rng.uniform(-1, 1) * 180 * s
    if rng.random() < p:
        vals["brightness"] = rng.uniform(1 - 0.65 * s, 1 + 0.8 * s)
    if rng.random() < p:
        vals["contrast"] = rng.uniform(1 - 0.6 * s, 1 + 0.6 * s)
    if rng.random() < p:
        vals["blur"] = rng.uniform(0, 3.0 * s)
    if rng.random() < p:
        vals["noise"] = rng.uniform(0, 35 * s)
    if rng.random() < p:
        vals["gamma"] = rng.uniform(1.0, 1 + 1.2 * s)
    clamped = {k: round(min(max(v, RANGES[k][0]), RANGES[k][1]), 2) for k, v in vals.items()}
    return AugmentParams(**clamped)
