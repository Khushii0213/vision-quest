"""Pre-flight check and first-run initialisation.

    python scripts/preflight.py             # check Python and packages only
    python scripts/preflight.py --download  # also download CIFAR-10 and the model weights (needs internet)
    python scripts/preflight.py --smoke     # also run ONE real inference on a CIFAR-10 image and print it

Exit code 0 = everything requested succeeded, 1 = a check failed (a message explains how to fix it).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED = ["streamlit", "torch", "torchvision", "PIL", "numpy"]


def check_python() -> bool:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    print(f"[{'ok' if ok else 'WARN'}] Python {v.major}.{v.minor}.{v.micro}"
          + ("" if ok else "  (3.11 or 3.12 recommended; torch wheels may be missing for other versions)"))
    return (v.major, v.minor) >= (3, 10)


def check_packages() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"[ok] {name} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"[FAIL] {name} is not installed. Activate the virtual environment and run: pip install -r requirements.txt")
            ok = False
    return ok


def download() -> bool:
    from src.config import DATASET_DIR
    from src.dataset import CifarDataset, DatasetUnavailableError
    from src.model import Classifier, ModelUnavailableError
    try:
        print("Downloading/verifying CIFAR-10 (about 170 MB, first run only)...")
        ds = CifarDataset.load(DATASET_DIR)
        print(f"[ok] CIFAR-10 ready: {len(ds)} test images")
        print("Downloading/loading MobileNetV3-Small weights (about 10 MB, first run only)...")
        Classifier.load()
        print("[ok] Model ready")
        return True
    except (DatasetUnavailableError, ModelUnavailableError) as exc:
        print(f"[FAIL] {exc}")
        return False


def smoke() -> bool:
    from src.config import DATASET_DIR, TOP_K, WORK_SIZE
    from src.dataset import CifarDataset
    from src.imageio import upscale_square
    from src.model import Classifier
    ds, clf = CifarDataset.load(DATASET_DIR), Classifier.load()
    img = upscale_square(ds.image(0), WORK_SIZE)
    t0 = time.perf_counter()
    preds = clf.predict(img, TOP_K)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Dataset image 0 has CIFAR-10 label '{ds.label_name(0)}'. Model top-{TOP_K} (ImageNet labels):")
    for p in preds:
        print(f"  {p.confidence:6.1%}  {p.label}")
    total = sum(p.confidence for p in preds)
    print(f"[ok] one inference took {ms:.0f} ms (includes first-call warm-up); top-5 mass {total:.1%}")
    return 0 < total <= 1.0 + 1e-6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ok = check_python() & check_packages()
    if ok and (args.download or args.smoke):
        ok = download()
    if ok and args.smoke:
        ok = smoke()
    print("\nAll requested checks passed." if ok else "\nSome checks failed. See messages above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
