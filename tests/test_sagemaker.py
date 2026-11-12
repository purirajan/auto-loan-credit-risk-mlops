"""Tests for the SageMaker-compatible inference endpoints."""

from src.models.api_service import Application, invocations, ping


VALID_APPLICATION = {
    "loan_id": 5001,
    "duration": 36,
    "credit_amount": 5000,
    "installment_commitment": 3,
    "age": 35,
    "existing_credits": 1,
    "purpose": "used car",
    "checking_status": "0<=X<200",
    "credit_history": "existing paid",
    "savings_status": "<100",
    "employment": "1<=X<4",
    "housing": "own",
}


def test_ping_reports_healthy():
    """SageMaker's health route reports a ready process."""
    assert ping() == {"status": "ok"}


def test_invocations_scores_single_application():
    """The SageMaker inference route accepts a single application object."""
    result = invocations(Application(**VALID_APPLICATION))

    assert result["loan_id"] == VALID_APPLICATION["loan_id"]
    assert 0 <= result["pd"] <= 1
    assert result["model_version"] == "1.0"


def test_invocations_scores_application_batch():
    """The SageMaker inference route accepts a JSON-style list of applications."""
    result = invocations([Application(**VALID_APPLICATION), Application(**{**VALID_APPLICATION, "loan_id": 5002})])

    assert [row["loan_id"] for row in result] == [5001, 5002]
    assert all(0 <= row["pd"] <= 1 for row in result)
