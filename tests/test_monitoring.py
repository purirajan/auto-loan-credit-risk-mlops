"""Tests for PSI and delayed-label calibration monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.monitoring import (
    build_monitoring_profile,
    calculate_monitoring_report,
)


@pytest.fixture

def training_frame():
    """Create stable numeric and categorical reference data for PSI tests."""
    return pd.DataFrame({
        "age": np.linspace(20, 70, 100),
        "credit_amount": np.linspace(500, 10000, 100),
        "purpose": ["new car", "used car"] * 50,
    })


def test_unchanged_population_has_low_psi_and_pending_calibration(training_frame):
    """An unchanged scored population has low PSI and waits for outcome labels."""
    profile = build_monitoring_profile(training_frame, ["age", "credit_amount"], ["purpose"])
    scored = training_frame.copy()
    scored["pd"] = np.linspace(0.1, 0.4, len(scored))

    report = calculate_monitoring_report(scored, profile)

    assert all(feature["psi"] < 0.1 for feature in report["features"].values())
    assert report["calibration"]["status"] == "pending_labels"
    assert report["retraining_recommended"] is False
    assert report["retraining_automatic"] is False


def test_population_shift_recommends_review(training_frame):
    """A sufficiently shifted feature population raises an alert and recommends review."""
    profile = build_monitoring_profile(training_frame, ["age", "credit_amount"], ["purpose"])
    scored = training_frame.copy()
    scored["age"] = 65
    scored["pd"] = 0.2

    report = calculate_monitoring_report(scored, profile)

    assert report["features"]["age"]["psi"] >= 0.25
    assert report["retraining_recommended"] is True
    assert any("age" in alert for alert in report["alerts"])


def test_matured_outcomes_report_calibration_metrics(training_frame):
    """Matured outcomes produce calibration metrics and flag large PD gaps."""
    profile = build_monitoring_profile(training_frame, ["age", "credit_amount"], ["purpose"])
    scored = training_frame.copy()
    scored["pd"] = 0.1
    scored["default_flag"] = 1

    report = calculate_monitoring_report(scored, profile)

    assert report["calibration"]["status"] == "evaluated"
    assert report["calibration"]["observed_default_rate"] == 1.0
    assert "brier_score" in report["calibration"]
    assert report["retraining_recommended"] is True


def test_invalid_probability_is_rejected(training_frame):
    """Malformed PD values are rejected before monitoring calculations."""
    profile = build_monitoring_profile(training_frame, ["age", "credit_amount"], ["purpose"])
    scored = training_frame.copy()
    scored["pd"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_monitoring_report(scored, profile)
