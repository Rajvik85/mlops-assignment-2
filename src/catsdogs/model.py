"""Serializable NumPy logistic-regression model and inference utilities."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from catsdogs.preprocessing import preprocess_image

MODEL_FORMAT_VERSION = 1
DEFAULT_CLASS_NAMES = ("cat", "dog")


def extract_features(image: np.ndarray, feature_size: int = 32) -> np.ndarray:
    """Average-pool an RGB square image and flatten it into a compact vector."""
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] != image.shape[1]:
        raise ValueError("Expected a square HxWx3 RGB image")
    if feature_size <= 0 or image.shape[0] % feature_size != 0:
        raise ValueError("feature_size must evenly divide the input image size")
    block = image.shape[0] // feature_size
    pooled = image.reshape(feature_size, block, feature_size, block, 3).mean(
        axis=(1, 3)
    )
    return pooled.astype(np.float32).reshape(-1)


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    """Numerically stable sigmoid."""
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    negative_exp = np.exp(array[~positive])
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def validate_model(model: dict[str, Any]) -> None:
    required = {
        "format_version",
        "weights",
        "bias",
        "image_size",
        "feature_size",
        "class_names",
        "feature_mean",
        "feature_std",
    }
    missing = required - set(model)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    if model["format_version"] != MODEL_FORMAT_VERSION:
        raise ValueError(f"Unsupported model format: {model['format_version']}")
    expected = int(model["feature_size"]) ** 2 * 3
    weights = np.asarray(model["weights"])
    if weights.shape != (expected,):
        raise ValueError(f"Expected {expected} weights, found {weights.shape}")
    if np.asarray(model["feature_mean"]).shape != (expected,):
        raise ValueError("feature_mean has the wrong shape")
    if np.asarray(model["feature_std"]).shape != (expected,):
        raise ValueError("feature_std has the wrong shape")
    if tuple(model["class_names"]) != DEFAULT_CLASS_NAMES:
        raise ValueError("This service expects class names ['cat', 'dog']")


def save_model(model: dict[str, Any], path: str | Path) -> None:
    validate_model(model)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a trusted local model artifact and validate its schema."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with model_path.open("rb") as file:
        model = pickle.load(file)  # noqa: S301 - only load trusted project artifacts.
    if not isinstance(model, dict):
        raise ValueError("Model artifact must contain a dictionary")
    validate_model(model)
    return model


def predict_features(model: dict[str, Any], features: np.ndarray) -> dict[str, Any]:
    validate_model(model)
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    weights = np.asarray(model["weights"], dtype=np.float32)
    if vector.shape != weights.shape:
        raise ValueError(f"Feature shape {vector.shape} does not match {weights.shape}")
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    std = np.asarray(model["feature_std"], dtype=np.float32)
    normalized = (vector - mean) / std
    dog_probability = float(sigmoid(float(normalized @ weights + model["bias"])))
    probabilities = {"cat": 1.0 - dog_probability, "dog": dog_probability}
    label = "dog" if dog_probability >= 0.5 else "cat"
    return {"label": label, "probabilities": probabilities}


def predict_image(
    model: dict[str, Any], source: str | Path | BinaryIO | bytes
) -> dict[str, Any]:
    """Preprocess an image and return its predicted label and class probabilities."""
    readable: str | Path | BinaryIO
    readable = io.BytesIO(source) if isinstance(source, bytes) else source
    image = preprocess_image(readable, int(model["image_size"]))
    features = extract_features(image, int(model["feature_size"]))
    return predict_features(model, features)
