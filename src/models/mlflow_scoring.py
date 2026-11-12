"""Validate and score applications using the champion MLflow model."""

import sys
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from configs.settings import MLFLOW_TRACKING_URI, MODEL_NAME, OUTPUT_DIR
from src.data.ingestion import get_history_and_new_apps
from src.evaluation.validation import validate

ALIAS = "champion"


def main() -> None:
    """Score the incoming batch and write lineage to the output CSV."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    schema = mlflow.artifacts.load_dict(f"runs:/{model_version.run_id}/input_schema.json")
    features = list(schema["numeric"]) + list(schema["categorical"])
    _, new_apps = get_history_and_new_apps()

    errors, warnings = validate(new_apps, schema)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("VALIDATION FAILED. Batch was NOT scored:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    pd_scores = model.predict_proba(new_apps[features])[:, 1]
    out = pd.DataFrame({
        "loan_id": new_apps["loan_id"].values,
        "pd": pd_scores.round(4),
        "model_name": MODEL_NAME,
        "model_version": model_version.version,
        "run_id": model_version.run_id,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "scored_applications_mlflow.csv", index=False)
    print(out.head())
    print(f"Scored {len(out)} applications with {MODEL_NAME} v{model_version.version} ({ALIAS})")


if __name__ == "__main__":
    main()
