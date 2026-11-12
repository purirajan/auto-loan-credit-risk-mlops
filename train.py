"""Backward-compatible entry point for model training."""

from configs.settings import CAT_FEATURES, FEATURES, NUM_FEATURES, TARGET
from src.models.training import main, train_model

__all__ = ["CAT_FEATURES", "FEATURES", "NUM_FEATURES", "TARGET", "main", "train_model"]


if __name__ == "__main__":
    main()
