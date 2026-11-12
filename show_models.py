"""Backward-compatible entry point for listing MLflow model versions."""

from src.models.model_registry import show_models


if __name__ == "__main__":
    show_models()
