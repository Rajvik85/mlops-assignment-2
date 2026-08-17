import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app.main as api_module
from catsdogs.model import MODEL_FORMAT_VERSION, save_model


def test_health_and_prediction_endpoints(tmp_path, monkeypatch):
    feature_count = 2 * 2 * 3
    model = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "binary_logistic_regression",
        "weights": np.zeros(feature_count, dtype=np.float32),
        "bias": 0.0,
        "feature_mean": np.zeros(feature_count, dtype=np.float32),
        "feature_std": np.ones(feature_count, dtype=np.float32),
        "image_size": 224,
        "feature_size": 2,
        "class_names": ["cat", "dog"],
    }
    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)
    monkeypatch.setattr(api_module, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(api_module, "FEEDBACK_PATH", tmp_path / "feedback.csv")

    buffer = io.BytesIO()
    Image.new("RGB", (60, 80), color="blue").save(buffer, "JPEG")
    with TestClient(api_module.app) as client:
        health = client.get("/health")
        prediction = client.post(
            "/predict", files={"image": ("sample.jpg", buffer.getvalue(), "image/jpeg")}
        )

    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert prediction.status_code == 200
    assert prediction.json()["label"] in {"cat", "dog"}


def test_prediction_rejects_non_image(tmp_path, monkeypatch):
    feature_count = 12
    model = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "binary_logistic_regression",
        "weights": np.zeros(feature_count, dtype=np.float32),
        "bias": 0.0,
        "feature_mean": np.zeros(feature_count, dtype=np.float32),
        "feature_std": np.ones(feature_count, dtype=np.float32),
        "image_size": 224,
        "feature_size": 2,
        "class_names": ["cat", "dog"],
    }
    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)
    monkeypatch.setattr(api_module, "MODEL_PATH", str(model_path))
    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict", files={"image": ("bad.txt", b"not an image", "text/plain")}
        )
    assert response.status_code == 415
