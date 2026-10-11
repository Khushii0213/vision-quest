#!/bin/bash
# One-shot setup + launch for macOS. Run from the project folder:  bash scripts/setup_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3 not found. Install 3.11 or 3.12 from https://www.python.org/downloads/"; exit 1; fi
echo "Using $($PY --version)"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/preflight.py --download
echo "Starting Vision Quest at http://localhost:8501 (Ctrl+C to stop)"
streamlit run app.py
