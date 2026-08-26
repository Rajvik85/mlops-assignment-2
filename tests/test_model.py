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


class DummyOnnxSession:
    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 224, 224)
        return [np.asarray([[1.0, 3.0]], dtype=np.float32)]


def test_cnn_inference_normalizes_probabilities():
    model = {
        "format_version": 1,
        "model_type": "simple_cnn_onnx",
        "class_names": ["cat", "dog"],
        "image_size": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "session": DummyOnnxSession(),
        "input_name": "images",
    }
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color="gray").save(buffer, "PNG")

    result = predict_image(model, buffer.getvalue())

    assert result["label"] == "dog"
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-7
    assert result["probabilities"]["dog"] > result["probabilities"]["cat"]
