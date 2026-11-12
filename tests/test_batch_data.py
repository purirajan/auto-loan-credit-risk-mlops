"""Tests for the SageMaker Batch Transform input preparation utility."""

import json

import pandas as pd
import pytest

from configs.settings import ARTIFACTS_DIR, FEATURES
from src.models.prepare_batch_data import prepare_batch_data


@pytest.fixture

def schema():
    """Load the model's real input schema artifact for batch-data tests."""
    return json.loads((ARTIFACTS_DIR / "input_schema.json").read_text())


def clean_application(schema):
    row = {
        feature: (bounds["train_min"] + bounds["train_max"]) / 2
        for feature, bounds in schema["numeric"].items()
    }
    row.update({feature: categories[0] for feature, categories in schema["categorical"].items()})
    row["loan_id"] = 5001
    return row


def test_prepare_batch_data_writes_one_json_object_per_line(tmp_path, schema):
    """Valid applications are exported as one JSON object per input line."""
    input_csv = tmp_path / "applications.csv"
    output_jsonl = tmp_path / "nested" / "applications.jsonl"
    pd.DataFrame([clean_application(schema), {**clean_application(schema), "loan_id": 5002}]).to_csv(
        input_csv, index=False
    )

    count = prepare_batch_data(input_csv, output_jsonl)
    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]

    assert count == 2
    assert [row["loan_id"] for row in rows] == [5001, 5002]
    assert all(set(row) == {"loan_id", *FEATURES} for row in rows)


def test_prepare_batch_data_rejects_invalid_values(tmp_path, schema):
    """Invalid model inputs fail before a Batch Transform file is written."""
    input_csv = tmp_path / "invalid.csv"
    output_jsonl = tmp_path / "applications.jsonl"
    invalid_application = clean_application(schema)
    invalid_application["purpose"] = "boat"
    pd.DataFrame([invalid_application]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="unknown categories"):
        prepare_batch_data(input_csv, output_jsonl)

    assert not output_jsonl.exists()
