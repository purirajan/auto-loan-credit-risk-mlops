import json
from datetime import date
import hashlib
import platform

import pandas as pd
import sklearn

import joblib
from validate import build_schema
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_source import get_history_and_new_apps

NUM_FEATURES = ["duration", "credit_amount", "installment_commitment",
                "age", "existing_credits"]
CAT_FEATURES = ["purpose", "checking_status", "credit_history",
                "savings_status", "employment", "housing"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "default_flag"

def train_model(X, y, random_state=42):
    """Split the data reproducibly, fit the pipeline, and return model and split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


def main():
    history, _ = get_history_and_new_apps()
    X, y = history[FEATURES], history[TARGET]
    model, X_train, X_test, y_test = train_model(X, y)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"Test AUC = {auc:.3f}, Gini = {2 * auc - 1:.3f}")

    joblib.dump(model, "artifacts/pd_model.joblib")
    schema = build_schema(X_train, NUM_FEATURES, CAT_FEATURES)
    with open("artifacts/input_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(history, index=True).values
    ).hexdigest()[:16]
    metadata = {
        "model_name": "auto_pd",
        "model_version": "1.0",
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
    with open("artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved model and metadata to artifacts/")


if __name__ == "__main__":
    main()