"""Backward-compatible exports for the data-ingestion package."""

from src.data.ingestion import CAR_PURPOSES, get_history_and_new_apps, load_auto_loans

__all__ = ["CAR_PURPOSES", "get_history_and_new_apps", "load_auto_loans"]


if __name__ == "__main__":
    history, new_apps = get_history_and_new_apps()
    print(f"History: {len(history)} loans, default rate = {history['default_flag'].mean():.1%}")
    print(f"New applications: {len(new_apps)}")
    print(history["purpose"].value_counts())
    print(history.head())
