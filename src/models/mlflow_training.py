"""Train, track, and register an experiment with MLflow."""

import hashlib
import platform
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from configs.settings import (
    FEATURES,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    NUM_FEATURES,
    CAT_FEATURES,
)
from src.data.ingestion import get_history_and_new_apps
from src.evaluation.metrics import evaluate_binary_classifier
from src.evaluation.validation import build_schema
from src.features.pipeline import build_model_pipeline


def main() -> None:
    """Run one parameterized experiment and register its fitted model."""
    regularization = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    history, _ = get_history_and_new_apps()
    X, y = history[FEATURES], history[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(history, index=True).values
    ).hexdigest()[:16]
    model = build_model_pipeline(C=regularization)

    with mlflow.start_run(run_name=f"logreg_C={regularization}") as run:
        model.fit(X_train, y_train)
        metrics = evaluate_binary_classifier(model, X_test, y_test)
        mlflow.log_params({
            "model_type": "logistic_regression",
            "C": regularization,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metrics({
            "test_auc": metrics["auc"],
            "test_gini": metrics["gini"],
            "mean_predicted_pd": float(metrics["predicted_pd"].mean()),
            "observed_default_rate": float(y_test.mean()),
        })
        mlflow.set_tags({
            "data_source": "OpenML credit-g v1, car loans only",
            "data_hash": data_hash,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        })
        mlflow.log_dict(build_schema(X_train, NUM_FEATURES, CAT_FEATURES), "input_schema.json")
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
        )
        print(f"C={regularization}  AUC={metrics['auc']:.3f}  run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
