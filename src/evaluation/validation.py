"""Build and enforce the input data contract."""

import pandas as pd

INF = float("inf")

# Business rules: values outside these ranges are impossible -> ERROR.
HARD_LIMITS = {
    "age": (18, 100),
    "duration": (1, 96),
    "credit_amount": (1, 1_000_000),
    "installment_commitment": (1, 4),
    "existing_credits": (0, 20),
}


def build_schema(X, num_features, cat_features):
    """Record numeric training ranges and known categorical values."""
    schema = {"numeric": {}, "categorical": {}}
    for col in num_features:
        schema["numeric"][col] = {
            "train_min": float(X[col].min()),
            "train_max": float(X[col].max()),
        }
    for col in cat_features:
        schema["categorical"][col] = sorted(X[col].dropna().astype(str).unique().tolist())
    return schema


def validate(df, schema):
    """Return hard errors and out-of-training-range warnings for a batch."""
    errors, warnings = [], []

    expected = list(schema["numeric"]) + list(schema["categorical"])
    missing = [col for col in expected if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    for col, rng in schema["numeric"].items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(values.isna().sum())
        if n_bad:
            errors.append(f"{col}: {n_bad} missing or non-numeric values")

        lo, hi = HARD_LIMITS.get(col, (-INF, INF))
        n_invalid = int(((values < lo) | (values > hi)).sum())
        if n_invalid:
            errors.append(f"{col}: {n_invalid} values outside allowed range [{lo}, {hi}]")

        in_limits = (values >= lo) & (values <= hi)
        outside_train = (values < rng["train_min"]) | (values > rng["train_max"])
        n_ood = int((in_limits & outside_train).sum())
        if n_ood:
            warnings.append(
                f"{col}: {n_ood} values outside training range "
                f"[{rng['train_min']}, {rng['train_max']}] (model is extrapolating)"
            )

    for col, allowed in schema["categorical"].items():
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            errors.append(f"{col}: {n_null} missing values")
        unknown = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if unknown:
            errors.append(f"{col}: unknown categories {unknown}; allowed are {allowed}")

    return errors, warnings
