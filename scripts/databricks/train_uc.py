"""Entry point for training and registering a model in Unity Catalog."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

catalog = os.environ.get("DATABRICKS_CATALOG", "main")
schema = os.environ.get("DATABRICKS_SCHEMA", "mlops")
model_name = os.environ.get("DATABRICKS_MODEL_NAME", "auto_pd")
os.environ.setdefault("MODEL_NAME", f"{catalog}.{schema}.{model_name}")

from src.models.mlflow_training import main


if __name__ == "__main__":
    main()
