#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10 à 3.12 requis"'
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python data/download.py
python -m pytest -q
python scripts/run_experiment.py --config "$CONFIG"
python scripts/render_readme.py
