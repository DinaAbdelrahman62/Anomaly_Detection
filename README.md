# Digital Twin Anomaly Detection API

A FastAPI service that streams packaged Digital Twin electricity readings and
serves anomaly results produced by the saved Two-Stage LightGBM model with
neighbour features.

The API does not accept raw readings for scoring. It uses the packaged feature
tables in `data/`, builds `data/anomaly_predictions.parquet` on its first
successful startup, and reuses that cache on later runs. Readings can begin
streaming while the prediction cache is being built.

## Runtime files

The service needs these files:

| Path | Purpose |
| --- | --- |
| `LightGBM hourly with future/two_stage.pkl` | Saved two-stage LightGBM model bundle |
| `data/anomaly_daily.parquet` | Daily features used by the first-stage model |
| `data/anomaly_hourly_with_neighbours.parquet` | Hourly and neighbour features used by the second-stage model |
| `data/lead_buildings.csv` | Building metadata and coordinates |

The generated file `data/anomaly_predictions.parquet` is a reusable prediction
cache. Delete it only when the model or packaged feature data changes and you
want to force a new scoring run.

## WSL setup

From WSL, open the project directory and create a virtual environment:

```bash
cd "/mnt/d/Alkan_QSIT/Digital_Twin/Anomaly_Detection/Two_Stage_LightGBM_Neighbour_Features/Digital_Twin_4_AI_Features/Anomaly_Detection"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the model and Parquet dependencies are available:

```bash
python -c "import lightgbm, pyarrow, sklearn; print('Dependencies are ready')"
```

## Configuration

All configuration is in `config.py`. The default paths are relative to the
project, making the service portable. Override them when needed with
environment variables:

```bash
export DIGITAL_TWIN_DATA_DIR="/path/to/data"
export DIGITAL_TWIN_MODEL_BUNDLE_PATH="/path/to/two_stage.pkl"
export READING_INTERVAL_SECONDS=30
export PREDICTION_INTERVAL_SECONDS=30
```

`READING_INTERVAL_SECONDS` controls reading SSE events. `PREDICTION_INTERVAL_SECONDS`
controls prediction SSE events; the two streams are independent.

## Run the API

Activate the environment, then start Uvicorn from the project root:

```bash
source .venv/bin/activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive API documentation at `http://localhost:8000/docs` and
check startup readiness at `http://localhost:8000/health`.

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service, model, source-data, and prediction-cache status |
| `GET` | `/api/readings/stream` | SSE stream of raw enriched readings |
| `GET` | `/api/readings/{reading_index}` | One raw enriched reading by zero-based index |
| `POST` | `/api/anomaly/predict` | Builds or returns the prediction cache summary |
| `GET` | `/api/anomaly/predictions/{prediction_index}` | One saved prediction by zero-based index |
| `GET` | `/api/predictions/stream` | SSE stream of cached prediction results |

Public reading and prediction payloads include the building identifier,
timestamp, meter reading, coordinates, and available building metadata. They
intentionally exclude `site_id` and `timezone`.

## Examples

Get the health status:

```bash
curl http://localhost:8000/health
```

Get a single raw reading:

```bash
curl http://localhost:8000/api/readings/0
```

Request prediction-cache creation or its existing result:

```bash
curl -X POST http://localhost:8000/api/anomaly/predict
```

Get a known anomalous prediction from the current cached dataset:

```bash
curl http://localhost:8000/api/anomaly/predictions/124
```

Connect to the reading SSE stream:

```bash
curl -N http://localhost:8000/api/readings/stream
```

Connect to the prediction SSE stream:

```bash
curl -N http://localhost:8000/api/predictions/stream
```

In a browser frontend, consume either stream with `EventSource`:

```javascript
const readings = new EventSource('/api/readings/stream');
readings.addEventListener('reading', (event) => {
  const reading = JSON.parse(event.data);
  console.log(reading);
});

const predictions = new EventSource('/api/predictions/stream');
predictions.addEventListener('prediction', (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
});
```

The prediction stream sends `prediction_status` events while the initial cache
is unavailable or still being built, then sends `prediction` events when ready.

## Project runtime structure

```text
main.py             FastAPI application and endpoint definitions
config.py           Portable paths and streaming intervals
app/constants.py    Public response-column definitions
app/services.py     Model loading, data loading, scoring, cache, payload logic
```
