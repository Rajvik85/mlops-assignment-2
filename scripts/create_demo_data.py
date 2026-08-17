"""Create a tiny synthetic dataset only for installation and smoke-test verification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def create_demo_dataset(output_dir: str | Path, per_class: int = 30, seed: int = 42) -> None:
    root = Path(output_dir)
    rng = random.Random(seed)
    for label in ("Cat", "Dog"):
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            size = rng.randint(180, 300)
            warm = label == "Cat"
            base = (210, 125, 60) if warm else (70, 130, 205)
            image = Image.new("RGB", (size, size), base)
            draw = ImageDraw.Draw(image)
            for offset in range(0, size, 18):
                jitter = rng.randint(-12, 12)
                color = (245, 195, 95) if warm else (150, 210, 245)
                if warm:
                    draw.line((0, offset + jitter, size, offset + jitter), fill=color, width=7)
                else:
                    draw.line((offset + jitter, 0, offset + jitter, size), fill=color, width=7)
            if warm:
                draw.polygon([(size * 0.3, size * 0.3), (size * 0.4, size * 0.1), (size * 0.5, size * 0.3)], fill=(80, 50, 30))
                draw.polygon([(size * 0.5, size * 0.3), (size * 0.6, size * 0.1), (size * 0.7, size * 0.3)], fill=(80, 50, 30))
            else:
                draw.ellipse((size * 0.25, size * 0.2, size * 0.75, size * 0.75), fill=(80, 65, 50))
            image.save(directory / f"{label.lower()}_{index:03d}.jpg", quality=90)
    metadata = {
        "demo_only": True,
        "warning": "Synthetic patterns verify the pipeline; they are not assignment evidence or a real cats/dogs model.",
        "images_per_class": per_class,
        "seed": seed,
    }
    (root / "DEMO_DATASET.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_demo_dataset(args.output, args.per_class, args.seed)
    print(f"Created demo data in {args.output}")


if __name__ == "__main__":
    main()

