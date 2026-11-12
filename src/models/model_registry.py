"""Manage and inspect registered model versions and the champion alias."""

import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from configs.settings import MLFLOW_TRACKING_URI, MODEL_NAME


def promote(version: str) -> None:
    """Assign the champion alias to the specified registered model version."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    model_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"{MODEL_NAME} v{model_version.version} is now champion (from run {model_version.run_id})")


def show_models() -> None:
    """Print registered versions and mark the current champion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    except MlflowException:
        champion = None
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for model_version in sorted(versions, key=lambda version: int(version.version)):
        run = client.get_run(model_version.run_id)
        regularization = run.data.params["C"]
        auc = run.data.metrics["test_auc"]
        marker = "  <- champion" if model_version.version == champion else ""
        print(f"v{model_version.version}   C={regularization:<6}  AUC={auc:.3f}{marker}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_models()
    else:
        promote(sys.argv[1])
