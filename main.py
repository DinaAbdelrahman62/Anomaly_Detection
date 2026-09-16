"""FastAPI runtime entry point that serves precomputed anomaly predictions."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

import config
from app.constants import PUBLIC_BUILDING_METADATA_COLUMNS


@dataclass
class RuntimeState:
    predictions_frame: pd.DataFrame | None = None
    prediction_error: str | None = None


state = RuntimeState()


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


def load_prediction_results() -> pd.DataFrame:
    path = Path(config.PREDICTION_OUTPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Prediction output not found: {path}")
    frame = _read_dataframe(path)
    required = {"building_id", "timestamp", "anomaly_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    return frame.sort_values(["timestamp", "building_id"]).reset_index(drop=True)


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


def prepare_prediction_payload(row: pd.Series) -> dict[str, Any]:
    return _prepare_payload(row, anomaly_status=str(row["anomaly_status"]))


def _timestamp_batches(
    frame: pd.DataFrame,
    payload_builder: Callable[[pd.Series], dict[str, Any]],
    items_key: str,
):
    for timestamp, rows in frame.groupby("timestamp", sort=True):
        items = [payload_builder(row) for _, row in rows.sort_values("building_id").iterrows()]
        yield {
            "timestamp": timestamp.isoformat(),
            items_key: items,
        }


def _timestamp_batch_at_index(
    frame: pd.DataFrame,
    batch_index: int,
    payload_builder: Callable[[pd.Series], dict[str, Any]],
    items_key: str,
) -> dict[str, Any] | None:
    if batch_index < 0:
        return None
    for index, batch in enumerate(_timestamp_batches(frame, payload_builder, items_key)):
        if index == batch_index:
            return batch
    return None


def _prediction_summary() -> dict[str, Any]:
    predictions = state.predictions_frame if state.predictions_frame is not None else load_prediction_results()
    return {
        "prediction_status": "loaded_from_file",
        "total_readings": int(len(predictions)),
        "anomaly_count": int((predictions["anomaly_status"] == "anomaly").sum()),
        "output_file": str(config.PREDICTION_OUTPUT_PATH),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.predictions_frame = load_prediction_results()
        state.prediction_error = None
    except Exception as exc:
        state.predictions_frame = None
        state.prediction_error = str(exc)

    yield


app = FastAPI(title="Digital Twin Anomaly Detection Service", version="1.2.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    prediction_cache_status = "loaded_from_file" if state.prediction_error is None else "unavailable"
    return {
        "service_status": "ok",
        "model_loading_status": "not_used",
        "configured_data_path": str(config.DATA_DIR),
        "configured_reading_interval_seconds": config.READING_INTERVAL_SECONDS,
        "configured_prediction_interval_seconds": config.PREDICTION_INTERVAL_SECONDS,
        "prediction_cache_status": prediction_cache_status,
        "prediction_output_file": str(config.PREDICTION_OUTPUT_PATH),
        "errors": {
            "model": None,
            "data": state.prediction_error,
            "prediction": state.prediction_error,
        },
    }


@app.get("/api/readings/stream", response_class=StreamingResponse)
async def stream_readings() -> StreamingResponse:
    """Push precomputed readings over Server-Sent Events without prediction status."""
    if state.predictions_frame is None:
        raise HTTPException(status_code=503, detail={"error": "missing_prediction_output", "message": state.prediction_error})

    async def events():
        try:
            for batch in _timestamp_batches(state.predictions_frame, prepare_stream_payload, "readings"):
                yield f"event: reading\ndata: {json.dumps(batch)}\n\n"
                await asyncio.sleep(config.READING_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/readings/{reading_index}")
async def get_reading(reading_index: int) -> dict[str, Any]:
    if state.predictions_frame is None:
        raise HTTPException(status_code=503, detail={"error": "missing_prediction_output", "message": state.prediction_error})
    batch = _timestamp_batch_at_index(state.predictions_frame, reading_index, prepare_stream_payload, "readings")
    if batch is None:
        raise HTTPException(status_code=404, detail={"error": "reading_not_found"})
    return batch


@app.post("/api/anomaly/predict")
async def predict_anomaly() -> dict[str, Any]:
    """Return the status of the precomputed predictions file without running inference."""
    try:
        return _prediction_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={"error": "missing_prediction_output", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_prediction_output", "message": str(exc)}) from exc


@app.get("/api/anomaly/predictions/{prediction_index}")
async def get_prediction(prediction_index: int) -> dict[str, Any]:
    try:
        predictions = state.predictions_frame if state.predictions_frame is not None else load_prediction_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={"error": "missing_prediction_output", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_prediction_output", "message": str(exc)}) from exc
    batch = _timestamp_batch_at_index(predictions, prediction_index, prepare_prediction_payload, "predictions")
    if batch is None:
        raise HTTPException(status_code=404, detail={"error": "prediction_not_found"})
    return batch


@app.get("/api/predictions/stream", response_class=StreamingResponse)
async def stream_predictions() -> StreamingResponse:
    """Push precomputed predictions over Server-Sent Events."""
    try:
        predictions = state.predictions_frame if state.predictions_frame is not None else load_prediction_results()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={"error": "missing_prediction_output", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_prediction_output", "message": str(exc)}) from exc

    async def events():
        try:
            for batch in _timestamp_batches(predictions, prepare_prediction_payload, "predictions"):
                yield f"event: prediction\ndata: {json.dumps(batch)}\n\n"
                await asyncio.sleep(config.PREDICTION_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            return

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
