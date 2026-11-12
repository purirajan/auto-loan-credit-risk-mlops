"""Prepare a CSV of applications as SageMaker Batch Transform JSON Lines."""

import argparse
import json
from pathlib import Path

import pandas as pd

from configs.settings import ARTIFACTS_DIR, FEATURES
from src.data.ingestion import get_history_and_new_apps
from src.evaluation.validation import validate


def prepare_batch_data(input_csv: Path, output_jsonl: Path) -> int:
    """Validate input applications and write one JSON object per line."""
    schema = json.loads((ARTIFACTS_DIR / "input_schema.json").read_text())
    applications = pd.read_csv(input_csv)
    errors, warnings = validate(applications, schema)
    if errors:
        raise ValueError("Invalid batch input: " + "; ".join(errors))
    for warning in warnings:
        print(f"WARNING: {warning}")

    required = ["loan_id", *FEATURES]
    missing = [column for column in required if column not in applications.columns]
    if missing:
        raise ValueError(f"Missing columns required for batch scoring: {missing}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as output_file:
        for row in applications[required].to_dict(orient="records"):
            output_file.write(json.dumps(row, allow_nan=False) + "\n")
    return len(applications)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Applications CSV containing loan_id and all model features; defaults to simulated OpenML applications.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/aws_batch/applications.jsonl"),
        help="Destination JSON Lines file for uploading to S3.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    if input_csv is None:
        _, applications = get_history_and_new_apps()
        input_csv = args.output_jsonl.parent / "applications.csv"
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        applications.to_csv(input_csv, index=False)

    row_count = prepare_batch_data(input_csv, args.output_jsonl)
    print(f"Prepared {row_count} applications at {args.output_jsonl}")


if __name__ == "__main__":
    main()
