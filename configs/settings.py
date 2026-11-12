"""Shared configuration and paths for training and serving."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = Path(os.environ.get("MODEL_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
OUTPUT_DIR = PROJECT_ROOT / "output"

NUM_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "age",
    "existing_credits",
]
CAT_FEATURES = [
    "purpose",
    "checking_status",
    "credit_history",
    "savings_status",
    "employment",
    "housing",
]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "default_flag"
RANDOM_STATE = 42
TEST_SIZE = 0.3
MODEL_NAME = os.environ.get("MODEL_NAME", "auto_pd")
MODEL_VERSION = "1.0"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
MLFLOW_REGISTRY_URI = os.environ.get("MLFLOW_REGISTRY_URI")
MLFLOW_EXPERIMENT = "auto-pd"
