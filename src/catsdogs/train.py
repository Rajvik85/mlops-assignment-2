"""Train, evaluate, serialize, and track a Cats-vs-Dogs baseline model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from catsdogs.model import MODEL_FORMAT_VERSION, extract_features, save_model, sigmoid
from catsdogs.preprocessing import preprocess_image


def _data_provenance(data_dir: Path) -> str:
    raw_dir = data_dir.parent / "raw"
    if (raw_dir / "DEMO_DATASET.json").is_file():
        return "synthetic_demo"
    source_path = raw_dir / "DATASET_SOURCE.json"
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("dataset_handle") == "bhavikjikadara/dog-and-cat-classification-dataset":
            return "assignment_kaggle_dataset"
    return "unverified_dataset"


def load_split(
    data_dir: str | Path, split: str, feature_size: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one manifest split as pooled features, labels, and relative paths."""
    root = Path(data_dir)
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}; run the preprocessing stage before training"
        )
    rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"Split '{split}' contains no images")
    features, labels, paths = [], [], []
    for row in rows:
        path = root / row["path"]
        image = preprocess_image(path, 224)
        features.append(extract_features(image, feature_size))
        labels.append(int(row["label_index"]))
        paths.append(row["path"])
    return np.stack(features), np.asarray(labels, dtype=np.float32), paths


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= 0.5).astype(np.int32)
    truth = labels.astype(np.int32)
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    accuracy = (tp + tn) / max(1, len(truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    loss = float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))
    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "loss": round(loss, 6),
        "true_cat": tn,
        "cat_as_dog": fp,
        "dog_as_cat": fn,
        "true_dog": tp,
        "samples": len(truth),
    }


def _augment_features(
    features: np.ndarray, feature_size: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = features.reshape(-1, feature_size, feature_size, 3).copy()
    flip_mask = rng.random(len(blocks)) < 0.5
    blocks[flip_mask] = blocks[flip_mask, :, ::-1, :]
    brightness = rng.uniform(0.85, 1.15, size=(len(blocks), 1, 1, 1))
    return np.clip(blocks * brightness, 0, 1).reshape(len(blocks), -1)


def _loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _draw_loss_curve(history: list[dict[str, float]], output: Path) -> None:
    width, height, margin = 900, 520, 70
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Training and validation loss", fill="black")
    draw.line((margin, height - margin, width - 25, height - margin), fill="black", width=2)
    draw.line((margin, 45, margin, height - margin), fill="black", width=2)
    values = [row[key] for row in history for key in ("train_loss", "val_loss")]
    lower = min(values) if values else 0.0
    upper = max(values) if values else 1.0
    padding = max((upper - lower) * 0.15, 0.005)
    lower = max(0.0, lower - padding)
    upper += padding

    def point(index: int, value: float) -> tuple[float, float]:
        x = margin + index * (width - margin - 35) / max(1, len(history) - 1)
        y = height - margin - (value - lower) / max(1e-9, upper - lower) * (height - margin - 55)
        return x, y

    for key, color in (("train_loss", "#1565c0"), ("val_loss", "#d84315")):
        points = [point(i, row[key]) for i, row in enumerate(history)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((width - 250, 18), "blue: train   orange: validation", fill="black")
    draw.text((width // 2 - 25, height - 35), "epoch", fill="black")
    draw.text((8, 48), f"{upper:.3f}", fill="black")
    draw.text((8, height - margin - 7), f"{lower:.3f}", fill="black")
    for index, row in enumerate(history):
        if index == 0 or index == len(history) - 1 or (index + 1) % 5 == 0:
            x, _ = point(index, lower)
            draw.line((x, height - margin, x, height - margin + 5), fill="black")
            draw.text((x - 5, height - margin + 9), str(int(row["epoch"])), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _draw_confusion_matrix(metrics: dict[str, float | int], output: Path) -> None:
    canvas = Image.new("RGB", (720, 600), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((230, 35), "Confusion matrix (test set)", fill="black")
    values = [
        int(metrics["true_cat"]),
        int(metrics["cat_as_dog"]),
        int(metrics["dog_as_cat"]),
        int(metrics["true_dog"]),
    ]
    maximum = max(values + [1])
    draw.text((260, 90), "Predicted cat", fill="black")
    draw.text((460, 90), "Predicted dog", fill="black")
    draw.text((40, 230), "Actual cat", fill="black")
    draw.text((40, 420), "Actual dog", fill="black")
    cells = [(220, 140), (420, 140), (220, 330), (420, 330)]
    for (x, y), value in zip(cells, values):
        intensity = 245 - int(150 * value / maximum)
        color = (intensity, intensity, 255)
        draw.rectangle((x, y, x + 170, y + 160), fill=color, outline="black", width=2)
        draw.text((x + 78, y + 72), str(value), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def train_model(
    data_dir: str | Path,
    model_output: str | Path,
    reports_dir: str | Path,
    feature_size: int = 32,
    epochs: int = 40,
    learning_rate: float = 0.0025,
    batch_size: int = 64,
    l2: float = 0.0001,
    seed: int = 42,
    augmentation: bool = True,
    patience: int = 6,
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, learning_rate, batch_size, and patience must be positive")
    train_x, train_y, _ = load_split(data_dir, "train", feature_size)
    val_x, val_y, _ = load_split(data_dir, "val", feature_size)
    test_x, test_y, _ = load_split(data_dir, "test", feature_size)
    feature_mean = train_x.mean(axis=0).astype(np.float32)
    feature_std = train_x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    norm_val = (val_x - feature_mean) / feature_std
    norm_test = (test_x - feature_mean) / feature_std

    rng = np.random.default_rng(seed)
    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0
    history: list[dict[str, float]] = []
    best: tuple[float, np.ndarray, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_x = _augment_features(train_x, feature_size, rng) if augmentation else train_x
        epoch_x = (epoch_x - feature_mean) / feature_std
        order = rng.permutation(len(epoch_x))
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x, batch_y = epoch_x[indices], train_y[indices]
            probabilities = sigmoid(batch_x @ weights + bias).astype(np.float32)
            error = probabilities - batch_y
            weights -= learning_rate * (
                batch_x.T @ error / len(indices) + l2 * weights
            )
            bias -= learning_rate * float(error.mean())
        train_prob = sigmoid(((train_x - feature_mean) / feature_std) @ weights + bias)
        val_prob = sigmoid(norm_val @ weights + bias)
        train_loss, val_loss = _loss(train_y, train_prob), _loss(val_y, val_prob)
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )
        if best is None or val_loss < best[0]:
            best = (val_loss, weights.copy(), bias)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best is not None
    _, weights, bias = best
    test_probabilities = sigmoid(norm_test @ weights + bias)
    test_metrics = binary_metrics(test_y, test_probabilities)
    model: dict[str, Any] = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_type": "binary_logistic_regression",
        "weights": weights,
        "bias": float(bias),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "image_size": 224,
        "feature_size": feature_size,
        "class_names": ["cat", "dog"],
        "decision_threshold": 0.5,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": _data_provenance(Path(data_dir)),
        "training": {
            "epochs_configured": epochs,
            "epochs_completed": len(history),
            "early_stopping_patience": patience,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "l2": l2,
            "seed": seed,
            "augmentation": augmentation,
            "train_samples": len(train_y),
            "validation_samples": len(val_y),
            "test_samples": len(test_y),
        },
        "test_metrics": test_metrics,
    }
    save_model(model, model_output)

    report_root = Path(reports_dir)
    figures = report_root / "figures"
    _draw_loss_curve(history, figures / "loss_curve.png")
    _draw_confusion_matrix(test_metrics, figures / "confusion_matrix.png")
    report = {
        "model_type": model["model_type"],
        "data_provenance": model["data_provenance"],
        "test": test_metrics,
        "history": history,
        "parameters": model["training"],
    }
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if enable_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is not installed. Install requirements.txt or pass --no-mlflow."
            ) from exc
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("cats-vs-dogs-baseline")
        context = mlflow.start_run(run_name="numpy-logistic-baseline")
    else:
        context = nullcontext()
    with context:
        if enable_mlflow:
            mlflow.log_params(
                {
                    "model_type": model["model_type"],
                    "data_provenance": model["data_provenance"],
                    "feature_size": feature_size,
                    "epochs": epochs,
                    "epochs_completed": len(history),
                    "early_stopping_patience": patience,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "l2": l2,
                    "seed": seed,
                    "augmentation": augmentation,
                    "train_samples": len(train_y),
                    "validation_samples": len(val_y),
                    "test_samples": len(test_y),
                }
            )
            mlflow.log_metrics(
                {f"test_{key}": value for key, value in test_metrics.items() if isinstance(value, float)}
            )
            for row in history:
                mlflow.log_metrics(
                    {"train_loss": row["train_loss"], "val_loss": row["val_loss"]},
                    step=int(row["epoch"]),
                )
            mlflow.log_artifact(str(model_output), artifact_path="model")
            mlflow.log_artifacts(str(figures), artifact_path="figures")
            mlflow.log_artifact(str(report_root / "metrics.json"), artifact_path="reports")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model-output", default="models/cats_dogs_logreg.pkl")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--feature-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_model(
        data_dir=args.data,
        model_output=args.model_output,
        reports_dir=args.reports_dir,
        feature_size=args.feature_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        l2=args.l2,
        seed=args.seed,
        augmentation=args.augmentation,
        patience=args.patience,
        enable_mlflow=not args.no_mlflow,
    )
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
