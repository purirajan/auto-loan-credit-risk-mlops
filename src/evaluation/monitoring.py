"""Population stability and delayed-label calibration monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

PSI_WARNING_THRESHOLD = 0.1
PSI_CRITICAL_THRESHOLD = 0.25
CALIBRATION_WARNING_THRESHOLD = 0.05
MIN_CALIBRATION_ROWS = 30
MISSING_BUCKET = "__MISSING__"
OTHER_BUCKET = "__OTHER__"


def _numeric_bins(values: pd.Series, bin_count: int = 10) -> tuple[list[float], list[float]]:
    """Return finite cut points and training proportions for quantile bins."""
    all_numeric = pd.to_numeric(values, errors="coerce")
    missing_share = float(all_numeric.isna().mean())
    numeric = all_numeric.dropna()
    if numeric.empty:
        raise ValueError("Cannot build a monitoring profile from an empty numeric feature")
    quantiles = np.linspace(0, 1, bin_count + 1)[1:-1]
    cuts = sorted(set(float(value) for value in numeric.quantile(quantiles).tolist()))
    edges = [-float("inf"), *cuts, float("inf")]
    buckets = pd.cut(numeric, bins=edges, include_lowest=True, duplicates="drop")
    proportions = buckets.value_counts(sort=False, normalize=True).astype(float).tolist()
    proportions = [value * (1 - missing_share) for value in proportions]
    proportions.append(missing_share)
    return cuts, proportions


def build_monitoring_profile(
    training_features: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    bin_count: int = 10,
) -> dict[str, Any]:
    """Build feature-distribution references from training rows only."""
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}, "training_rows": len(training_features)}
    for feature in numeric_features:
        cuts, proportions = _numeric_bins(training_features[feature], bin_count)
        profile["numeric"][feature] = {"cuts": cuts, "proportions": proportions}
    for feature in categorical_features:
        values = training_features[feature].fillna(MISSING_BUCKET).astype(str)
        shares = values.value_counts(normalize=True).sort_index()
        profile["categorical"][feature] = {
            key: float(value) for key, value in shares.items()
        }
        profile["categorical"][feature].setdefault(OTHER_BUCKET, 0.0)
    return profile


def _psi(expected: list[float], actual: list[float]) -> float:
    """Compute population stability index using a small zero-count adjustment."""
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    epsilon = 1e-6
    expected_array = np.clip(expected_array, epsilon, None)
    actual_array = np.clip(actual_array, epsilon, None)
    expected_array /= expected_array.sum()
    actual_array /= actual_array.sum()
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


def _current_numeric_proportions(values: pd.Series, cuts: list[float]) -> list[float]:
    numeric = pd.to_numeric(values, errors="coerce")
    missing_share = float(numeric.isna().mean())
    valid = numeric.dropna()
    edges = [-float("inf"), *cuts, float("inf")]
    buckets = pd.cut(valid, bins=edges, include_lowest=True, duplicates="drop")
    proportions = buckets.value_counts(sort=False, normalize=True).astype(float).tolist()
    if not proportions:
        proportions = [0.0] * (len(cuts) + 1)
    proportions = [value * (1 - missing_share) for value in proportions]
    proportions.append(missing_share)
    return proportions


def calculate_monitoring_report(
    scored: pd.DataFrame,
    profile: dict[str, Any],
    psi_warning_threshold: float = PSI_WARNING_THRESHOLD,
    psi_critical_threshold: float = PSI_CRITICAL_THRESHOLD,
    calibration_warning_threshold: float = CALIBRATION_WARNING_THRESHOLD,
    min_calibration_rows: int = MIN_CALIBRATION_ROWS,
) -> dict[str, Any]:
    """Calculate feature PSI and, when outcomes mature, calibration metrics."""
    if "pd" not in scored.columns:
        raise ValueError("Scored data must contain a 'pd' column")
    if scored.empty:
        raise ValueError("Scored data is empty")
    pd_values = pd.to_numeric(scored["pd"], errors="coerce")
    if pd_values.isna().any() or ((pd_values < 0) | (pd_values > 1)).any():
        raise ValueError("'pd' values must be numeric probabilities between 0 and 1")

    feature_reports: dict[str, dict[str, float]] = {}
    alerts: list[str] = []
    warnings: list[str] = []
    for feature, reference in profile["numeric"].items():
        if feature not in scored:
            raise ValueError(f"Scored data is missing feature '{feature}'")
        expected = list(reference["proportions"])
        actual = _current_numeric_proportions(scored[feature], reference["cuts"])
        if len(actual) != len(expected):
            raise ValueError(f"Invalid monitoring profile bins for '{feature}'")
        value = _psi(expected, actual)
        feature_reports[feature] = {"psi": value}
        if value >= psi_critical_threshold:
            alerts.append(f"{feature}: critical population shift (PSI={value:.3f})")
        elif value >= psi_warning_threshold:
            warnings.append(f"{feature}: population shift warning (PSI={value:.3f})")

    for feature, reference in profile["categorical"].items():
        if feature not in scored:
            raise ValueError(f"Scored data is missing feature '{feature}'")
        expected_keys = list(reference)
        expected = [float(reference[key]) for key in expected_keys]
        actual_values = scored[feature].fillna(MISSING_BUCKET).astype(str)
        actual = [float((actual_values == key).mean()) for key in expected_keys]
        other_index = expected_keys.index(OTHER_BUCKET)
        actual[other_index] += float((~actual_values.isin(expected_keys)).mean())
        value = _psi(expected, actual)
        feature_reports[feature] = {"psi": value}
        if value >= psi_critical_threshold:
            alerts.append(f"{feature}: critical population shift (PSI={value:.3f})")
        elif value >= psi_warning_threshold:
            warnings.append(f"{feature}: population shift warning (PSI={value:.3f})")

    calibration: dict[str, Any]
    if "default_flag" not in scored:
        calibration = {"status": "pending_labels", "rows": 0}
    else:
        observed = pd.to_numeric(scored["default_flag"], errors="coerce")
        if observed.isna().any() or not observed.isin([0, 1]).all():
            raise ValueError("'default_flag' must contain only binary 0/1 outcomes")
        rows = len(scored)
        if rows < min_calibration_rows:
            calibration = {"status": "insufficient_rows", "rows": rows, "minimum_rows": min_calibration_rows}
        else:
            observed_rate = float(observed.mean())
            mean_pd = float(pd_values.mean())
            difference = observed_rate - mean_pd
            calibration = {
                "status": "evaluated",
                "rows": rows,
                "observed_default_rate": observed_rate,
                "mean_predicted_pd": mean_pd,
                "observed_minus_predicted": difference,
                "brier_score": float(brier_score_loss(observed, pd_values)),
            }
            if observed.nunique() == 2:
                calibration["auc"] = float(roc_auc_score(observed, pd_values))
            if abs(difference) >= calibration_warning_threshold:
                alerts.append(
                    "calibration: observed default rate differs from mean PD "
                    f"by {difference:+.3f}"
                )

    return {
        "rows": len(scored),
        "features": feature_reports,
        "calibration": calibration,
        "warnings": warnings,
        "alerts": alerts,
        "retraining_recommended": bool(alerts),
        "retraining_automatic": False,
    }
