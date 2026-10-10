#!/bin/bash
# Sets up an isolated Python environment with mlx-lm and runs the benchmark.
set -e
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This needs a Mac with Apple silicon."; exit 1
fi
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet --upgrade mlx-lm
python scripts/bench.py
