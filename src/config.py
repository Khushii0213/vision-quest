"""Project-wide paths and constants.

All paths are derived from this file's location, so the project works from any
folder and any user account. Set VISION_QUEST_DATA_DIR to move the cache.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = Path(os.environ.get("VISION_QUEST_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
DATASET_DIR = DATA_DIR / "datasets"
STATE_FILE = DATA_DIR / "vision_quest.json"

# Keep downloaded model weights inside the project's data folder (not in the ZIP).
os.environ.setdefault("TORCH_HOME", str(DATA_DIR / "torch"))

TILE_SIZE = 192      # pixels, arcade tile working resolution
WORK_SIZE = 256      # pixels, AI Lab working resolution for dataset images
MAX_UPLOAD_SIDE = 1024
TOP_K = 5
NOISE_SEED = 42      # fixed so the AI Lab noise does not flicker between reruns
