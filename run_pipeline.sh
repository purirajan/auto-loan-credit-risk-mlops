#!/bin/zsh
set -e

echo "== 1. Training =="
python train.py

echo "== 2. Tests =="
pytest -q

echo "== 3. Batch scoring =="
python score.py

echo "Pipeline finished successfully."