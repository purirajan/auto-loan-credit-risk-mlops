#!/usr/bin/env bash
set -e

echo "== 1. Training =="
python -m src.models.training

echo "== 2. Tests =="
pytest -q

echo "== 3. Batch scoring =="
python -m src.models.batch_scoring

echo "== 4. Drift monitoring =="
python -m src.models.monitor_batch

echo "Pipeline finished successfully."