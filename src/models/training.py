"""Train and package the baseline auto-loan probability-of-default model."""

import hashlib
import json
import platform
from datetime import date

import joblib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from configs.settings import (
    ARTIFACTS_DIR,
    FEATURES,
    MODEL_NAME,
    MODEL_VERSION,
    NUM_FEATURES,
    CAT_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.data.ingestion import get_history_and_new_apps
from src.evaluation.metrics import evaluate_binary_classifier
from src.evaluation.validation import build_schema
from src.features.pipeline import build_model_pipeline


def train_model(X, y, random_state: int = RANDOM_STATE):
    """Split the data reproducibly, fit the pipeline, and return model and split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )
    model = build_model_pipeline()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


def main() -> None:
    """Train the model and write its artifact, schema, and metadata."""
    history, _ = get_history_and_new_apps()
    X, y = history[FEATURES], history[TARGET]
    model, X_train, X_test, y_test = train_model(X, y)
    metrics = evaluate_binary_classifier(model, X_test, y_test)
    auc = metrics["auc"]
    print(f"Test AUC = {auc:.3f}, Gini = {metrics['gini']:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS_DIR / "pd_model.joblib")
    schema = build_schema(X_train, NUM_FEATURES, CAT_FEATURES)
    (ARTIFACTS_DIR / "input_schema.json").write_text(json.dumps(schema, indent=2))

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(history, index=True).values
    ).hexdigest()[:16]
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_on": str(date.today()),
        "data_source": "OpenML credit-g, purpose in [new car, used car]",
        "data_hash": data_hash,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "training_rows": len(X_train),
        "features": FEATURES,
        "test_auc": round(auc, 4),
    }
    (ARTIFACTS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model and metadata to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
