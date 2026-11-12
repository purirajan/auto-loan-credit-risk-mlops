"""Build the shared preprocessing and classification pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.settings import CAT_FEATURES, NUM_FEATURES


def build_model_pipeline(C: float = 1.0) -> Pipeline:
    """Create a pipeline that scales numeric and encodes categorical features."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ])),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])
