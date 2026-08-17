import io

import numpy as np
from PIL import Image

from catsdogs.model import MODEL_FORMAT_VERSION, predict_features, predict_image


def make_model(feature_size=2):
    feature_count = feature_size * feature_size * 3
    return {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "binary_logistic_regression",
        "weights": np.zeros(feature_count, dtype=np.float32),
        "bias": 0.0,
        "feature_mean": np.zeros(feature_count, dtype=np.float32),
        "feature_std": np.ones(feature_count, dtype=np.float32),
        "image_size": 224,
        "feature_size": feature_size,
        "class_names": ["cat", "dog"],
    }


def test_model_utility_returns_normalized_probabilities():
    model = make_model()
    result = predict_features(model, np.zeros(12, dtype=np.float32))

    assert result["label"] == "dog"
    assert result["probabilities"] == {"cat": 0.5, "dog": 0.5}
    assert sum(result["probabilities"].values()) == 1.0


def test_image_inference_accepts_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (30, 50), color="orange").save(buffer, "PNG")
    result = predict_image(make_model(), buffer.getvalue())

    assert set(result["probabilities"]) == {"cat", "dog"}

