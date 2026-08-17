"""FastAPI inference service with safe logging and Prometheus metrics."""

from __future__ import annotations

import csv
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from catsdogs.model import load_model, predict_image

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("catsdogs.api")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs_logreg.pkl")
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", "reports/monitoring/feedback.csv"))

REQUEST_COUNT = Counter(
    "catsdogs_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "catsdogs_http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
PREDICTIONS = Counter("catsdogs_predictions_total", "Predictions", ["label"])
FEEDBACK = Counter("catsdogs_feedback_total", "Feedback labels", ["correct"])
FEEDBACK_LOCK = Lock()


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_version: int
    data_provenance: str


class PredictionResponse(BaseModel):
    prediction_id: str
    label: str
    probabilities: dict[str, float]


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=100)
    predicted_label: str = Field(pattern="^(cat|dog)$")
    true_label: str = Field(pattern="^(cat|dog)$")


class FeedbackResponse(BaseModel):
    accepted: bool
    correct: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(MODEL_PATH)
    LOGGER.info(
        "event=model_loaded model_type=%s format_version=%s",
        app.state.model["model_type"],
        app.state.model["format_version"],
    )
    yield


app = FastAPI(
    title="Cats vs Dogs Inference API",
    version="1.0.0",
    description="A reproducible baseline service for the MLOps assignment.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:100]
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        LOGGER.exception(
            "event=request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        REQUEST_COUNT.labels(request.method, path_label, str(status_code)).inc()
        REQUEST_LATENCY.labels(request.method, path_label).observe(duration)
        LOGGER.info(
            "event=request_complete request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            path_label,
            status_code,
            duration * 1000,
        )
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(UnidentifiedImageError)
async def invalid_image_handler(_request: Request, _exc: UnidentifiedImageError):
    return JSONResponse(status_code=400, content={"detail": "Uploaded file is not a valid image"})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    return HealthResponse(
        status="healthy",
        model_type=model["model_type"],
        model_version=model["format_version"],
        data_provenance=model.get("data_provenance", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Use JPEG, PNG, WEBP, or BMP")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    try:
        result = predict_image(request.app.state.model, payload)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    prediction_id = str(uuid.uuid4())
    PREDICTIONS.labels(result["label"]).inc()
    return PredictionResponse(prediction_id=prediction_id, **result)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(item: FeedbackRequest) -> FeedbackResponse:
    """Store non-sensitive ground truth for post-deployment performance checks."""
    correct = item.predicted_label == item.true_label
    FEEDBACK.labels(str(correct).lower()).inc()
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOCK:
        exists = FEEDBACK_PATH.exists()
        with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["prediction_id", "predicted_label", "true_label", "correct"],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "prediction_id": item.prediction_id,
                    "predicted_label": item.predicted_label,
                    "true_label": item.true_label,
                    "correct": correct,
                }
            )
    return FeedbackResponse(accepted=True, correct=correct)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
