"""Backward-compatible exports for input-contract validation."""

from src.evaluation.validation import HARD_LIMITS, build_schema, validate

__all__ = ["HARD_LIMITS", "build_schema", "validate"]
