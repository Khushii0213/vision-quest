import io

import numpy as np
import pytest
from PIL import Image

from src.compare import compare_predictions
from src.dataset import CIFAR10_CLASSES, CifarDataset
from src.imageio import InvalidImageError, fit_longest_side, load_image, upscale_square
from src.model import Prediction
from tests.fakes import fake_dataset


def png(size=(10, 10), mode="RGB"):
    b = io.BytesIO()
    Image.new(mode, size, 128).save(b, "PNG")
    return b.getvalue()


def test_load_valid_and_modes():
    assert load_image(png()).mode == "RGB"
    assert load_image(png(mode="RGBA")).mode == "RGB"


@pytest.mark.parametrize("bad", [b"", b"hello world", png()[:40]])
def test_load_invalid(bad):
    with pytest.raises(InvalidImageError):
        load_image(bad)


def test_oversize_pixels_rejected(monkeypatch):
    monkeypatch.setattr("src.imageio.MAX_PIXELS", 50)
    with pytest.raises(InvalidImageError):
        load_image(png((20, 20)))


def test_resize_helpers():
    big = Image.new("RGB", (2000, 1000))
    assert fit_longest_side(big, 1000).size == (1000, 500)
    small = Image.new("RGB", (50, 50))
    assert fit_longest_side(small, 1000) is small
    assert upscale_square(Image.new("RGB", (32, 32)), 128).size == (128, 128)


def P(i, c):
    return Prediction(i, f"c{i}", c)


def test_compare_statuses():
    orig = [P(1, .5), P(2, .2), P(3, .1), P(4, .05), P(5, .02)]
    aug = [P(2, .4), P(1, .3), P(9, .1), P(3, .05), P(8, .02)]
    cmp = compare_predictions(orig, aug)
    st = {r.class_index: r.status for r in cmp.rows}
    assert st == {1: "moved", 2: "moved", 3: "moved", 4: "dropped", 5: "dropped", 9: "entered", 8: "entered"}
    assert cmp.top1_changed and "changed from 'c1'" in cmp.describe()
    row1 = next(r for r in cmp.rows if r.class_index == 1)
    assert row1.delta == pytest.approx(-0.2)
    assert next(r for r in cmp.rows if r.class_index == 4).delta is None


def test_compare_identical_is_all_same():
    preds = [P(i, 0.5 / (i + 1)) for i in range(5)]
    cmp = compare_predictions(preds, preds)
    assert not cmp.top1_changed and all(r.status == "same" for r in cmp.rows)


def test_compare_empty_rejected():
    with pytest.raises(ValueError):
        compare_predictions([], [P(1, 1)])


def test_dataset_labels_and_index():
    ds = fake_dataset(per_class=7)
    assert len(ds) == 70 and ds.label_name(0) == "airplane" and ds.label_name(69) == "truck"
    idx = ds.index_by_class()
    assert set(idx) == set(range(10)) and all(len(v) == 7 for v in idx.values())
    assert all(ds.label(i) == c for c, v in idx.items() for i in v)
    assert ds.image(3).size == (32, 32)
    with pytest.raises(IndexError):
        ds.image(70)
    assert ds.class_names == CIFAR10_CLASSES


def test_dataset_shape_validation():
    with pytest.raises(ValueError):
        CifarDataset(np.zeros((3, 32, 32, 3), np.uint8), np.zeros(2), CIFAR10_CLASSES)


def test_is_downloaded(tmp_path):
    assert not CifarDataset.is_downloaded(tmp_path)
    (tmp_path / "cifar-10-batches-py").mkdir()
    (tmp_path / "cifar-10-batches-py" / "test_batch").write_bytes(b"x")
    assert CifarDataset.is_downloaded(tmp_path)
