"""Dataset validation, deterministic splitting, and 224x224 RGB preprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES = ("cat", "dog")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_image(
    source: str | Path | BinaryIO,
    image_size: int = 224,
) -> np.ndarray:
    """Load an image, convert it to RGB, center-crop it, and scale to [0, 1]."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (image_size, image_size, 3):
        raise ValueError(f"Unexpected processed shape: {array.shape}")
    return array


def _find_class_directory(input_dir: Path, class_name: str) -> Path:
    matches = [
        path
        for path in input_dir.rglob("*")
        if path.is_dir() and path.name.casefold() == class_name.casefold()
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a '{class_name}' directory below {input_dir}. "
            "Expected data/raw/Cat and data/raw/Dog (capitalization is flexible)."
        )
    return min(matches, key=lambda item: len(item.parts))


def discover_images(input_dir: str | Path) -> dict[str, list[Path]]:
    """Find supported images below Cat and Dog directories."""
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {root}")
    result: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        class_dir = _find_class_directory(root, class_name)
        result[class_name] = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        )
        if not result[class_name]:
            raise ValueError(f"No supported images found in {class_dir}")
    return result


def split_paths(
    paths: list[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Create a reproducible class-stratified train/validation/test split."""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("train_ratio must be in (0,1), val_ratio in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be below 1")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, object]:
    """Validate, preprocess, and split the dataset; return an auditable summary."""
    source_root, target_root = Path(input_dir), Path(output_dir)
    if source_root.resolve() == target_root.resolve():
        raise ValueError("Input and output directories must be different")
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)

    discovered = discover_images(source_root)
    manifest_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    for class_index, class_name in enumerate(CLASS_NAMES):
        partitions = split_paths(
            discovered[class_name], train_ratio, val_ratio, seed + class_index
        )
        for split_name, paths in partitions.items():
            class_output = target_root / split_name / class_name
            class_output.mkdir(parents=True, exist_ok=True)
            for source_path in paths:
                destination = class_output / f"{source_path.stem}.jpg"
                suffix = 1
                while destination.exists():
                    destination = class_output / f"{source_path.stem}_{suffix}.jpg"
                    suffix += 1
                try:
                    processed = preprocess_image(source_path, image_size)
                    Image.fromarray((processed * 255).astype(np.uint8)).save(
                        destination, "JPEG", quality=92
                    )
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    skipped.append({"path": str(source_path), "reason": str(exc)})
                    continue
                relative = destination.relative_to(target_root)
                manifest_rows.append(
                    {
                        "path": relative.as_posix(),
                        "split": split_name,
                        "label": class_name,
                        "label_index": str(class_index),
                    }
                )
                counts[f"{split_name}_{class_name}"] += 1

    if not manifest_rows:
        raise ValueError("Every candidate image was invalid; no dataset was created")
    with (target_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["path", "split", "label", "label_index"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary: dict[str, object] = {
        "input_directory": str(source_root),
        "image_size": [image_size, image_size, 3],
        "ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": round(1.0 - train_ratio - val_ratio, 10),
        },
        "seed": seed,
        "processed_images": len(manifest_rows),
        "skipped_images": len(skipped),
        "counts": dict(sorted(counts.items())),
        "skipped": skipped,
    }
    (target_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="Raw dataset root")
    parser.add_argument("--output", default="data/processed", help="Output root")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_dataset(
        args.input,
        args.output,
        args.image_size,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

