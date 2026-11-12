"""Assign the Unity Catalog model's champion alias to a registered version."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.settings import MODEL_NAME
from src.models.model_registry import promote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Registered Unity Catalog model version")
    args = parser.parse_args()
    promote(args.version)


if __name__ == "__main__":
    main()
