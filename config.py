"""Portable configuration for the Digital Twin anomaly detection service."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DIGITAL_TWIN_DATA_DIR", str(BASE_DIR / "data")))
MODEL_BUNDLE_PATH = Path(os.getenv(
    "DIGITAL_TWIN_MODEL_BUNDLE_PATH",
    str(BASE_DIR / "LightGBM hourly with future" / "two_stage.pkl"),
))

# Packaged feature tables produced by src/anomaly/export.py.
DAILY_FEATURES_PATH = DATA_DIR / "anomaly_daily.parquet"
HOURLY_FEATURES_PATH = DATA_DIR / "anomaly_hourly_with_neighbours.parquet"
BUILDINGS_METADATA_PATH = DATA_DIR / "lead_buildings.csv"
PREDICTION_OUTPUT_PATH = DATA_DIR / "anomaly_predictions_30_buildings.parquet"

# The packaged table represents electricity readings and has no meter_type field.
DEFAULT_METER_TYPE = os.getenv("DEFAULT_METER_TYPE", "electricity")
READING_INTERVAL_SECONDS = int(os.getenv("READING_INTERVAL_SECONDS", "30"))
PREDICTION_INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "30"))