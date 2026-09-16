"""Data loading, cached scoring, and payload services for the Digital Twin API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder

import config
from app.constants import PUBLIC_BUILDING_METADATA_COLUMNS


@dataclass
class ServiceState:
    bundle: dict[str, Any] | None = None
    model_error: str | None = None
    hourly_frame: pd.DataFrame | None = None
    stream_error: str | None = None
    metadata_file: Path | None = None
    prediction_cache_status: str = "not_checked"
    prediction_error: str | None = None


service_state = ServiceState()
prediction_lock = Lock()


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file format: {path.suffix}")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _load_buildings_metadata() -> pd.DataFrame:
    path = Path(config.BUILDINGS_METADATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Building metadata file not found: {path}")
    metadata = _read_dataframe(path)
    if "building_id" not in metadata.columns:
        raise ValueError(f"{path.name} must include building_id")
    if metadata["building_id"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate building_id values")
    return metadata


def _enrich_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = _load_buildings_metadata()
    metadata_columns = [
        column for column in metadata.columns
        if column == "building_id" or column not in frame.columns
    ]
    return frame.merge(
        metadata[metadata_columns],
        on="building_id",
        how="left",
        validate="many_to_one",
    )


def load_hourly_features() -> tuple[pd.DataFrame, Path]:
    path = Path(config.HOURLY_FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Hourly feature table not found: {path}")
    frame = _read_dataframe(path)
    required = {"building_id", "timestamp", "meter_reading", "site_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    frame = _enrich_hourly_frame(frame)
    if "meter_type" not in frame.columns:
        frame["meter_type"] = config.DEFAULT_METER_TYPE
    for column in ("x", "y"):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame.sort_values(["timestamp", "building_id"]).reset_index(drop=True), path


def load_model_bundle() -> dict[str, Any]:
    path = Path(config.MODEL_BUNDLE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    required = {"daily_model", "hourly_model", "day_cutoff", "hour_cutoff", "daily_features", "hourly_features"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing required keys: {missing}")
    return bundle


def _prediction_output_is_current() -> bool:
    output = Path(config.PREDICTION_OUTPUT_PATH)
    inputs = [
        Path(config.MODEL_BUNDLE_PATH),
        Path(config.DAILY_FEATURES_PATH),
        Path(config.HOURLY_FEATURES_PATH),
        Path(config.BUILDINGS_METADATA_PATH),
    ]
    return (
        output.exists()
        and all(path.exists() for path in inputs)
        and all(output.stat().st_mtime >= path.stat().st_mtime for path in inputs)
    )


def initialize_services() -> None:
    try:
        service_state.bundle = load_model_bundle()
        service_state.model_error = None
    except Exception as exc:
        service_state.bundle = None
        service_state.model_error = str(exc)

    try:
        service_state.hourly_frame, _ = load_hourly_features()
        service_state.metadata_file = Path(config.BUILDINGS_METADATA_PATH)
        service_state.stream_error = None
    except Exception as exc:
        service_state.hourly_frame = None
        service_state.metadata_file = None
        service_state.stream_error = str(exc)

    if service_state.bundle is None:
        service_state.prediction_cache_status = "model_unavailable"
        return
    service_state.prediction_cache_status = "cache_available" if _prediction_output_is_current() else "generating"
    service_state.prediction_error = None


def _public_building_metadata(row: pd.Series) -> dict[str, Any]:
    return {
        column: _sanitize_value(row[column])
        for column in PUBLIC_BUILDING_METADATA_COLUMNS
        if column in row.index and not pd.isna(row[column])
    }


def _prepare_payload(row: pd.Series, anomaly_status: str | None = None) -> dict[str, Any]:
    payload = {
        "building_id": _sanitize_value(row.get("building_id")),
        "timestamp": _sanitize_value(row.get("timestamp")),
        "meter_type": _sanitize_value(row.get("meter_type")),
        "meter_reading": _sanitize_value(row.get("meter_reading")),
        "x": _sanitize_value(row.get("x")),
        "y": _sanitize_value(row.get("y")),
        "building_metadata": _public_building_metadata(row),
    }
    if anomaly_status is not None:
        payload = {"anomaly_status": anomaly_status, **payload}
    return jsonable_encoder(payload)


def prepare_stream_payload(row: pd.Series) -> dict[str, Any]:
    return _prepare_payload(row)


def load_prediction_results() -> pd.DataFrame:
    path = Path(config.PREDICTION_OUTPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Prediction output not found: {path}")
    frame = _read_dataframe(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def prepare_prediction_payload(row: pd.Series) -> dict[str, Any]:
    return _prepare_payload(row, anomaly_status=str(row["anomaly_status"]))


def _load_daily_features() -> pd.DataFrame:
    path = Path(config.DAILY_FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Daily feature table not found: {path}")
    frame = _read_dataframe(path)
    if not {"building_id", "timestamp"}.issubset(frame.columns):
        raise ValueError(f"{path.name} must include building_id and timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    return frame


def _validate_features(frame: pd.DataFrame, features: list[str], source: Path) -> None:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"{source.name} is missing required model features: {missing}")


def predict_dataset() -> dict[str, Any]:
    """Score the packaged daily/hourly feature tables exactly as predict.py does."""
    if service_state.bundle is None:
        raise RuntimeError(service_state.model_error or "The model bundle is not loaded.")
    if service_state.hourly_frame is None:
        raise RuntimeError(service_state.stream_error or "The hourly feature table is not loaded.")

    bundle = service_state.bundle
    daily = _load_daily_features()
    hourly = service_state.hourly_frame.copy()
    daily_features = list(bundle["daily_features"])
    hourly_features = list(bundle["hourly_features"])
    _validate_features(daily, daily_features, Path(config.DAILY_FEATURES_PATH))
    _validate_features(hourly, hourly_features, Path(config.HOURLY_FEATURES_PATH))

    daily_scores = bundle["daily_model"].predict_proba(daily[daily_features])[:, 1]
    gate = daily.loc[daily_scores >= float(bundle["day_cutoff"]), ["building_id", "timestamp"]].copy()
    gate["day"] = gate["timestamp"].dt.floor("D")
    gate = gate[["building_id", "day"]].drop_duplicates()

    hourly["_row_id"] = np.arange(len(hourly))
    hourly["day"] = hourly["timestamp"].dt.floor("D")
    hourly["_hour_score"] = bundle["hourly_model"].predict_proba(hourly[hourly_features])[:, 1]
    eligible = hourly.merge(gate, on=["building_id", "day"], how="inner")
    anomaly_ids = set(eligible.loc[eligible["_hour_score"] >= float(bundle["hour_cutoff"]), "_row_id"])

    output_columns = ["building_id", "timestamp", "meter_type", "meter_reading", "x", "y"]
    output_columns += [column for column in PUBLIC_BUILDING_METADATA_COLUMNS if column in hourly.columns]
    output = hourly[output_columns].copy()
    output["anomaly_status"] = np.where(hourly["_row_id"].isin(anomaly_ids), "anomaly", "normal")
    output = output.sort_values(["timestamp", "building_id"]).reset_index(drop=True)
    output.to_parquet(config.PREDICTION_OUTPUT_PATH, index=False)

    return {
        "prediction_status": "completed",
        "total_readings": int(len(output)),
        "anomaly_count": int((output["anomaly_status"] == "anomaly").sum()),
        "output_file": str(config.PREDICTION_OUTPUT_PATH),
    }


def ensure_prediction_output() -> dict[str, Any]:
    """Load a fresh cache or build it once when model/data inputs have changed."""
    with prediction_lock:
        try:
            if _prediction_output_is_current():
                cached = load_prediction_results()
                service_state.prediction_cache_status = "loaded_from_cache"
                service_state.prediction_error = None
                return {
                    "prediction_status": "loaded_from_cache",
                    "total_readings": int(len(cached)),
                    "anomaly_count": int((cached["anomaly_status"] == "anomaly").sum()),
                    "output_file": str(config.PREDICTION_OUTPUT_PATH),
                }
            service_state.prediction_cache_status = "generating"
            result = predict_dataset()
            service_state.prediction_cache_status = "generated"
            service_state.prediction_error = None
            return result
        except Exception as exc:
            service_state.prediction_cache_status = "unavailable"
            service_state.prediction_error = str(exc)
            raise