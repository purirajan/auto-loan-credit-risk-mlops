"""Load auto-loan data and split historical loans from new applications."""

import pandas as pd
from sklearn.datasets import fetch_openml

CAR_PURPOSES = ["new car", "used car"]


def load_auto_loans() -> pd.DataFrame:
    """Fetch German Credit from OpenML and keep only auto loans."""
    raw = fetch_openml("credit-g", version=1, as_frame=True).frame
    df = raw[raw["purpose"].isin(CAR_PURPOSES)].copy()

    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)

    df["default_flag"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns="class").reset_index(drop=True)
    df.insert(0, "loan_id", range(1, len(df) + 1))
    return df


def get_history_and_new_apps(new_share: float = 0.3, seed: int = 42):
    """Split outcomes-known historical loans from applications to be scored."""
    df = load_auto_loans()
    new_apps = df.sample(frac=new_share, random_state=seed)
    history = df.drop(new_apps.index)
    return history, new_apps.drop(columns="default_flag")
