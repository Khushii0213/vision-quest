import random

import numpy as np
import pytest
from PIL import Image

from src.augmentation import NEUTRAL, RANGES, AugmentParams, apply_augmentations, random_params


def img(color=(120, 90, 200), size=(64, 64)):
    return Image.new("RGB", size, color)


def gradient():
    a = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    return Image.fromarray(np.stack([a, a[::-1], a.T], axis=2))


def mean(i):
    return float(np.asarray(i).mean())


def test_neutral_is_identity():
    src = gradient()
    out = apply_augmentations(src, AugmentParams())
    assert np.array_equal(np.asarray(src), np.asarray(out))


def test_input_not_modified():
    src = gradient()
    before = np.asarray(src).copy()
    apply_augmentations(src, AugmentParams(brightness=1.5, rotation=30, blur=2, noise=20), seed=1)
    assert np.array_equal(before, np.asarray(src))


def test_brightness_directions():
    src = img()
    assert mean(apply_augmentations(src, AugmentParams(brightness=1.5))) > mean(src)
    assert mean(apply_augmentations(src, AugmentParams(brightness=0.5))) < mean(src)


def test_gamma_above_one_darkens():
    assert mean(apply_augmentations(gradient(), AugmentParams(gamma=2.0))) < mean(gradient())


def test_contrast_reduces_spread():
    out = apply_augmentations(gradient(), AugmentParams(contrast=0.3))
    assert np.asarray(out).std() < np.asarray(gradient()).std()


def test_blur_reduces_high_frequency():
    checker = Image.fromarray(((np.indices((64, 64)).sum(0) % 2) * 255).astype(np.uint8)).convert("RGB")
    out = apply_augmentations(checker, AugmentParams(blur=2))
    assert np.asarray(out).std() < np.asarray(checker).std()


def test_noise_seeded_reproducible_and_changes_image():
    p = AugmentParams(noise=25)
    a, b = apply_augmentations(img(), p, seed=5), apply_augmentations(img(), p, seed=5)
    c = apply_augmentations(img(), p, seed=6)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))
    assert not np.array_equal(np.asarray(a), np.asarray(img()))


def test_rotation_keeps_size_and_has_no_black_corners():
    out = apply_augmentations(img((200, 200, 200)), AugmentParams(rotation=45))
    assert out.size == (64, 64)
    assert np.asarray(out).min() > 150  # reflect-fill, not black


def test_rotation_180_matches_flip():
    src = gradient()
    out = apply_augmentations(src, AugmentParams(rotation=180))
    flipped = src.transpose(Image.Transpose.ROTATE_180)
    assert np.abs(np.asarray(out, dtype=int) - np.asarray(flipped, dtype=int)).mean() < 3


@pytest.mark.parametrize("name", list(RANGES))
def test_out_of_range_rejected(name):
    lo, hi = RANGES[name]
    with pytest.raises(ValueError):
        apply_augmentations(img(), AugmentParams(**{name: hi + 1}))
    with pytest.raises(ValueError):
        apply_augmentations(img(), AugmentParams(**{name: lo - 1}))


def test_nan_rejected():
    with pytest.raises(ValueError):
        apply_augmentations(img(), AugmentParams(brightness=float("nan")))


def test_non_image_rejected():
    with pytest.raises(TypeError):
        apply_augmentations("not an image", AugmentParams())


def test_grayscale_and_rgba_inputs_become_rgb():
    assert apply_augmentations(Image.new("L", (32, 32), 100), AugmentParams(blur=1)).mode == "RGB"
    assert apply_augmentations(Image.new("RGBA", (32, 32), (1, 2, 3, 4)), AugmentParams(noise=5)).mode == "RGB"


def test_random_params_in_range_and_strength_grows():
    rng = random.Random(3)
    for s in (0.0, 0.5, 1.0, 5.0, -1.0):
        for _ in range(50):
            random_params(rng, s).validate()
    zero = [random_params(random.Random(i), 0.0) for i in range(50)]
    assert all(p.rotation == 0 and p.blur == 0 and p.noise == 0 for p in zero)


def test_changed_fields():
    assert AugmentParams(blur=1, gamma=1.5).changed() == ["blur", "gamma"]
    assert NEUTRAL["brightness"] == 1.0
