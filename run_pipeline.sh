#!/bin/zsh
set -e

echo "== 1. Training =="
python -m src.models.training

echo "== 2. Tests =="
pytest -q

echo "== 3. Batch scoring =="
python -m src.models.batch_scoring

echo "Pipeline finished successfully."