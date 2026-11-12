"""Validate and batch-score new applications using the local model artifact."""

import json
import sys
from datetime import datetime

import joblib
import pandas as pd

from configs.settings import ARTIFACTS_DIR, OUTPUT_DIR
from src.data.ingestion import get_history_and_new_apps
from src.evaluation.validation import validate


def main() -> None:
    """Run validation, save an audit report, and write batch PD scores."""
    model = joblib.load(ARTIFACTS_DIR / "pd_model.joblib")
    meta = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())
    schema = json.loads((ARTIFACTS_DIR / "input_schema.json").read_text())
    _, new_apps = get_history_and_new_apps()

    errors, warnings = validate(new_apps, schema)
    report = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "model_version": meta["model_version"],
        "rows": len(new_apps),
        "status": "FAILED" if errors else "PASSED",
        "errors": errors,
        "warnings": warnings,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("VALIDATION FAILED. Batch was NOT scored:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    pd_scores = model.predict_proba(new_apps[meta["features"]])[:, 1]
    out = pd.DataFrame({
        "loan_id": new_apps["loan_id"].values,
        "purpose": new_apps["purpose"].values,
        "pd": pd_scores.round(4),
        "model_version": meta["model_version"],
        "scored_at": datetime.now().isoformat(timespec="seconds"),
    })
    out.to_csv(OUTPUT_DIR / "scored_applications.csv", index=False)
    print(f"Validation passed. Scored {len(out)} applications with model v{meta['model_version']}")


if __name__ == "__main__":
    main()
