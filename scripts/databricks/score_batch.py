"""Score a validated UC Volume CSV with a champion model using Spark."""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp, isnan, lit, pandas_udf, sum as spark_sum, when

os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.settings import (  # noqa: E402
    CAT_FEATURES,
    MLFLOW_REGISTRY_URI,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    NUM_FEATURES,
)
from src.evaluation.validation import HARD_LIMITS  # noqa: E402


def validate_spark_input(data: DataFrame, schema: dict[str, Any]) -> None:
    """Validate required columns and all rows before distributed scoring."""
    expected = list(schema["numeric"]) + list(schema["categorical"])
    missing = [name for name in expected if name not in data.columns]
    if "loan_id" not in data.columns:
        missing.append("loan_id")
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    checks = {}
    descriptions = {}
    warning_descriptions = {}
    checks["loan_id_null"] = col("loan_id").isNull()
    descriptions["loan_id_null"] = "loan_id: missing values"
    for index, (name, bounds) in enumerate(schema["numeric"].items()):
        value = col(name).cast("double")
        hard_min, hard_max = HARD_LIMITS.get(name, (float("-inf"), float("inf")))
        checks[f"numeric_null_{index}"] = col(name).isNull() | value.isNull() | isnan(value)
        descriptions[f"numeric_null_{index}"] = f"{name}: missing or non-numeric values"
        checks[f"numeric_range_{index}"] = (value < hard_min) | (value > hard_max)
        descriptions[f"numeric_range_{index}"] = f"{name}: values outside allowed range [{hard_min}, {hard_max}]"
        warning_checks = (
            (value >= hard_min)
            & (value <= hard_max)
            & ((value < bounds["train_min"]) | (value > bounds["train_max"]))
        )
        warning_descriptions[f"ood_{index}"] = (warning_checks, name, bounds)

    for index, (name, allowed) in enumerate(schema["categorical"].items()):
        checks[f"category_null_{index}"] = col(name).isNull()
        descriptions[f"category_null_{index}"] = f"{name}: missing values"
        checks[f"category_unknown_{index}"] = ~col(name).cast("string").isin(allowed)
        descriptions[f"category_unknown_{index}"] = f"{name}: unknown categories"

    expressions = [
        spark_sum(when(condition, 1).otherwise(0)).alias(key)
        for key, condition in checks.items()
    ]
    expressions.extend(
        spark_sum(when(condition, 1).otherwise(0)).alias(key)
        for key, (condition, _, _) in warning_descriptions.items()
    )
    counts = data.agg(*expressions).first().asDict()
    errors = [
        f"{descriptions[key]} ({counts[key]})"
        for key in descriptions
        if counts[key]
    ]
    if errors:
        raise ValueError("Invalid batch input: " + "; ".join(errors))
    for key, (_, name, bounds) in warning_descriptions.items():
        if counts[key]:
            print(
                f"WARNING: {name}: {counts[key]} values outside training range "
                f"[{bounds['train_min']}, {bounds['train_max']}] (model is extrapolating)"
            )


def score_batch(input_path: str, output_path: str, spark: SparkSession) -> None:
    """Read a CSV batch from a UC Volume and save scored rows as Delta."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    schema = mlflow.artifacts.load_dict(
        f"runs:/{model_version.run_id}/input_schema.json"
    )

    data = spark.read.option("header", True).option("inferSchema", True).csv(input_path)
    validate_spark_input(data, schema)
    broadcast_model = spark.sparkContext.broadcast(model)
    feature_order = NUM_FEATURES + CAT_FEATURES

    @pandas_udf("double")
    def predict_pd(*columns: pd.Series) -> pd.Series:
        feature_frame = pd.concat(
            [series.reset_index(drop=True) for series in columns], axis=1
        )
        feature_frame.columns = feature_order
        predictions = broadcast_model.value.predict_proba(feature_frame)[:, 1]
        return pd.Series(predictions)

    scored = (
        data.withColumn("pd", predict_pd(*[col(name) for name in feature_order]))
        .withColumn("model_name", lit(MODEL_NAME))
        .withColumn("model_version", lit(str(model_version.version)))
        .withColumn("run_id", lit(model_version.run_id))
        .withColumn("scored_at", current_timestamp())
    )
    run_suffix = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_output_path = f"{output_path.rstrip('/')}/{run_suffix}"
    scored.write.format("delta").mode("errorifexists").save(run_output_path)
    print(
        f"Scored batch with {MODEL_NAME} v{model_version.version}; "
        f"Delta output: {run_output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, help="CSV path in a Unity Catalog Volume")
    parser.add_argument("--output-path", required=True, help="Delta output path in a Unity Catalog Volume")
    args = parser.parse_args()
    score_batch(args.input_path, args.output_path, SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()
