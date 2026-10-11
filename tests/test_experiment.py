import pytest

from src.augmentation import AugmentParams
from src.experiment import run_experiment, write_csv
from src.model import Prediction
from tests.fakes import FakeClassifier, fake_dataset


class ConstantClassifier:
    def predict(self, image, top_k=5):
        return [Prediction(i, f"c{i}", 0.5 / (i + 1)) for i in range(top_k)]


def test_constant_classifier_is_perfectly_stable():
    rows = run_experiment(ConstantClassifier(), fake_dataset(5), n_images=10, settings=[("blur", AugmentParams(blur=2))])
    r = rows[0]
    assert r.images == 10 and r.top1_unchanged == 1.0 and r.top5_overlap == 1.0 and r.top1_conf_change == 0.0


def test_brightness_dependent_classifier_is_less_stable():
    rows = run_experiment(FakeClassifier(), fake_dataset(5), n_images=10,
                          settings=[("bright", AugmentParams(brightness=0.3))])
    assert rows[0].top1_unchanged < 1.0


def test_invalid_n_and_csv(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(ConstantClassifier(), fake_dataset(5), n_images=0)
    rows = run_experiment(ConstantClassifier(), fake_dataset(5), n_images=3, settings=[("blur", AugmentParams(blur=1))])
    write_csv(rows, tmp_path / "x" / "s.csv")
    assert (tmp_path / "x" / "s.csv").read_text().startswith("setting,images")
