"""Train, evaluate, export, and track a simple Cats-vs-Dogs CNN from scratch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from catsdogs.cnn import SimpleCNN, parameter_count
from catsdogs.train import (
    _data_provenance,
    _draw_confusion_matrix,
    _draw_loss_curve,
    binary_metrics,
)

CLASS_NAMES = ("cat", "dog")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, Apple Metal, or CPU with an explicit override for debugging."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int, augmentation: bool) -> tuple[Any, Any]:
    """Create training and deterministic evaluation transforms."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    evaluation = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    if not augmentation:
        return evaluation, evaluation
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def build_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augmentation: bool,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    training_transform, evaluation_transform = build_transforms(image_size, augmentation)
    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=training_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=evaluation_transform),
        "test": datasets.ImageFolder(data_dir / "test", transform=evaluation_transform),
    }
    for split, dataset in datasets_by_split.items():
        if tuple(dataset.classes) != CLASS_NAMES:
            raise ValueError(
                f"{split} classes must be {list(CLASS_NAMES)}, found {dataset.classes}"
            )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_by_split.items()
    }
    counts = {split: len(dataset) for split, dataset in datasets_by_split.items()}
    return loaders, counts


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(probabilities.cpu().numpy())
    labels_array = np.concatenate(labels_all).astype(np.int32)
    probabilities_array = np.concatenate(probabilities_all).astype(np.float32)
    accuracy = float(np.mean((probabilities_array >= 0.5) == labels_array))
    return total_loss / len(labels_array), accuracy, labels_array, probabilities_array


def _export_onnx(model: nn.Module, output: Path, image_size: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        cpu_model,
        example,
        output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(output))


def train_cnn(
    data_dir: str | Path,
    model_output: str | Path,
    checkpoint_output: str | Path,
    reports_dir: str | Path,
    image_size: int = 224,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_learning_rate: float = 0.003,
    weight_decay: float = 0.0001,
    dropout: float = 0.4,
    patience: int = 5,
    seed: int = 42,
    num_workers: int = 2,
    augmentation: bool = True,
    device_name: str = "auto",
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    """Train a scratch CNN and return its auditable report."""
    if min(epochs, batch_size, patience) <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if min(learning_rate, max_learning_rate) <= 0:
        raise ValueError("learning rates must be positive")
    _seed_everything(seed)
    data_root = Path(data_dir)
    loaders, counts = build_loaders(
        data_root, image_size, batch_size, num_workers, augmentation, seed
    )
    device = select_device(device_name)
    model = SimpleCNN(class_count=2, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        epochs=epochs,
        steps_per_epoch=len(loaders["train"]),
        pct_start=0.2,
        anneal_strategy="cos",
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy, _, _ = _run_epoch(
            model, loaders["train"], criterion, device, optimizer, scheduler
        )
        val_loss, val_accuracy, _, _ = _run_epoch(
            model, loaders["val"], criterion, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}",
            flush=True,
        )
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    test_loss, _, test_labels, test_probabilities = _run_epoch(
        model, loaders["test"], criterion, device
    )
    test_metrics = binary_metrics(test_labels, test_probabilities)
    test_metrics["loss"] = round(test_loss, 6)

    model_path = Path(model_output)
    checkpoint_path = Path(checkpoint_output)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "SimpleCNN",
            "state_dict": {key: value.cpu() for key, value in best_state.items()},
            "class_names": list(CLASS_NAMES),
            "image_size": image_size,
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
        },
        checkpoint_path,
    )
    _export_onnx(model, model_path, image_size)

    provenance = _data_provenance(data_root)
    parameters = {
        "architecture": "five_block_simple_cnn",
        "pretrained": False,
        "transfer_learning": False,
        "trainable_parameters": parameter_count(model),
        "epochs_configured": epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": patience,
        "learning_rate": learning_rate,
        "max_learning_rate": max_learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "seed": seed,
        "augmentation": augmentation,
        "device": str(device),
        "train_samples": counts["train"],
        "validation_samples": counts["val"],
        "test_samples": counts["test"],
    }
    metadata = {
        "format_version": 1,
        "model_type": "simple_cnn_onnx",
        "architecture": "SimpleCNN",
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "decision_threshold": 0.5,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": provenance,
        "training": parameters,
        "test_metrics": test_metrics,
    }
    metadata_path = model_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_root = Path(reports_dir)
    figures = report_root / "figures"
    _draw_loss_curve(history, figures / "loss_curve.png")
    _draw_confusion_matrix(test_metrics, figures / "confusion_matrix.png")
    report = {
        "model_type": metadata["model_type"],
        "architecture": metadata["architecture"],
        "data_provenance": provenance,
        "test": test_metrics,
        "history": history,
        "parameters": parameters,
    }
    report_root.mkdir(parents=True, exist_ok=True)
    metrics_path = report_root / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if enable_mlflow:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is not installed. Install requirements.txt or pass --no-mlflow."
            ) from exc
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment("cats-vs-dogs-cnn")
        context = mlflow.start_run(run_name="scratch-simple-cnn")
    else:
        context = nullcontext()
    with context:
        if enable_mlflow:
            mlflow.log_params(parameters)
            mlflow.log_metrics(
                {
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                    if isinstance(value, float)
                }
            )
            for row in history:
                mlflow.log_metrics(
                    {
                        "train_loss": float(row["train_loss"]),
                        "train_accuracy": float(row["train_accuracy"]),
                        "val_loss": float(row["val_loss"]),
                        "val_accuracy": float(row["val_accuracy"]),
                    },
                    step=int(row["epoch"]),
                )
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(metadata_path), artifact_path="model")
            mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
            mlflow.log_artifacts(str(figures), artifact_path="figures")
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model-output", default="models/cats_dogs_cnn.onnx")
    parser.add_argument("--checkpoint-output", default="models/cats_dogs_cnn.pt")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_cnn(
        data_dir=args.data,
        model_output=args.model_output,
        checkpoint_output=args.checkpoint_output,
        reports_dir=args.reports_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_learning_rate=args.max_learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        device_name=args.device,
        enable_mlflow=not args.no_mlflow,
    )
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
