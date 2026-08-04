#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python data/download.py
python -m pytest -q
python scripts/run_experiment.py --config "$CONFIG"
python scripts/render_readme.py

