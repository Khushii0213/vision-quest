"""Run the prediction-stability experiment on real CIFAR-10 images with the real model.

    python scripts/run_experiment.py --images 200 --seed 0

Writes data/experiments/stability.csv. Report the numbers together with: machine, Python/torch versions,
number of images and seed. Needs the dataset and weights (run scripts/preflight.py --download first).
"""
import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DATASET_DIR  # noqa: E402
from src.dataset import CifarDataset, DatasetUnavailableError  # noqa: E402
from src.experiment import run_experiment, write_csv  # noqa: E402
from src.model import Classifier, ModelUnavailableError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        ds, clf = CifarDataset.load(DATASET_DIR), Classifier.load()
    except (DatasetUnavailableError, ModelUnavailableError) as exc:
        print(f"Cannot run: {exc}")
        return 1
    import torch
    print(f"Machine: {platform.platform()} | Python {platform.python_version()} | torch {torch.__version__}")
    rows = run_experiment(clf, ds, args.images, args.seed)
    print(f"{'setting':22} {'top1 same':>10} {'top5 overlap':>13} {'conf change':>12} {'ms/img':>8}")
    for r in rows:
        print(f"{r.setting:22} {r.top1_unchanged:10.1%} {r.top5_overlap:13.1%} {r.top1_conf_change:+12.3f} {r.ms_per_image:8.1f}")
    out = DATA_DIR / "experiments" / "stability.csv"
    write_csv(rows, out)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
