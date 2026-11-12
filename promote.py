"""Backward-compatible entry point for promoting an MLflow model version."""

import sys

from src.models.model_registry import promote


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python promote.py <model-version>")
    promote(sys.argv[1])
