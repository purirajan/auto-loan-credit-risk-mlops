import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from configs.settings import CAT_FEATURES, NUM_FEATURES
from src.evaluation.validation import HARD_LIMITS, validate
from src.models.training import train_model


@pytest.fixture(scope="module")
def validation_data():
    """Load the generated schema once and create one schema-valid sample row."""
    schema_path = Path(__file__).resolve().parents[1] / "artifacts" / "input_schema.json"
    with schema_path.open() as schema_file:
        schema = json.load(schema_file)

    row = {
        feature: (bounds["train_min"] + bounds["train_max"]) / 2
        for feature, bounds in schema["numeric"].items()
    }
    row.update({feature: categories[0] for feature, categories in schema["categorical"].items()})
    return schema, pd.DataFrame([row])


# A sample with valid schema values passes without warnings.
def test_clean_batch_passes(validation_data):
    schema, clean_batch = validation_data

    errors, warnings = validate(clean_batch.copy(), schema)

    assert errors == []
    assert warnings == []


# Removing a required feature is reported as a validation error.
def test_missing_column_fails(validation_data):
    schema, clean_batch = validation_data
    batch = clean_batch.copy().drop(columns="duration")

    errors, _ = validate(batch, schema)

    assert any("duration" in error and "Missing required columns" in error for error in errors)


@pytest.mark.parametrize(
    "feature, value, should_fail",
    [
        (feature, value, should_fail)
        for feature, (lower, upper) in HARD_LIMITS.items()
        for value, should_fail in (
            (lower - 1, True),
            (lower, False),
            (upper, False),
            (upper + 1, True),
        )
    ],
    ids=[
        f"{feature}-{label}"
        for feature, (lower, upper) in HARD_LIMITS.items()
        for label in ("below-lower", "at-lower", "at-upper", "above-upper")
    ],
)
def test_hard_limits_are_exact(validation_data, feature, value, should_fail):
    """Each hard-limit edge is inclusive and values immediately outside fail."""
    schema, clean_batch = validation_data
    batch = clean_batch.copy()
    batch.loc[batch.index[0], feature] = value

    errors, _ = validate(batch, schema)
    range_errors = [
        error
        for error in errors
        if error.startswith(f"{feature}:") and "outside allowed range" in error
    ]

    assert bool(range_errors) == should_fail


# A category absent from the model schema is rejected.
def test_unknown_category_fails(validation_data):
    schema, clean_batch = validation_data
    batch = clean_batch.copy()
    batch.loc[batch.index[0], "purpose"] = "boat"

    errors, _ = validate(batch, schema)

    assert any("purpose" in error and "unknown categories" in error for error in errors)


# Text in a numeric feature is reported as a non-numeric value.
def test_non_numeric_text_fails(validation_data):
    schema, clean_batch = validation_data
    batch = clean_batch.copy()
    batch["duration"] = batch["duration"].astype(object)
    batch.loc[batch.index[0], "duration"] = "sixty"

    errors, _ = validate(batch, schema)

    assert any("duration" in error and "non-numeric" in error for error in errors)


# A plausible value outside training bounds warns but does not block scoring.
def test_out_of_training_range_warns_but_passes(validation_data):
    schema, clean_batch = validation_data
    batch = clean_batch.copy()
    batch.loc[batch.index[0], "credit_amount"] = (
        schema["numeric"]["credit_amount"]["train_max"] + 1
    )

    errors, warnings = validate(batch, schema)

    assert errors == []
    assert any("credit_amount" in warning and "outside training range" in warning for warning in warnings)


# Fitting twice with the same split seed produces exactly the same predicted PDs.
def test_training_is_reproducible(validation_data):
    schema, clean_batch = validation_data
    rows = []
    labels = []
    for index in range(40):
        row = clean_batch.iloc[0].to_dict()
        fraction = (index % 20) / 19
        for feature in NUM_FEATURES:
            bounds = schema["numeric"][feature]
            row[feature] = bounds["train_min"] + fraction * (
                bounds["train_max"] - bounds["train_min"]
            )
        for feature in CAT_FEATURES:
            categories = schema["categorical"][feature]
            row[feature] = categories[index % len(categories)]
        rows.append(row)
        labels.append(int(index % 20 >= 10))

    X = pd.DataFrame(rows)[NUM_FEATURES + CAT_FEATURES]
    y = pd.Series(labels)
    first_model, *_ = train_model(X, y, random_state=42)
    second_model, *_ = train_model(X, y, random_state=42)

    np.testing.assert_array_equal(
        first_model.predict_proba(X)[:, 1],
        second_model.predict_proba(X)[:, 1],
    )