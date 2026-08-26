"""Validate and create the final source/config/model submission ZIP."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


REQUIRED = [
    "README.md",
    "STUDY_NOTES.md",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "dvc.yaml",
    "dvc.lock",
    "params.yaml",
    "data/raw.dvc",
    ".dvc/config",
    ".github/workflows/ci-cd.yml",
    "app/main.py",
    "models/cats_dogs_cnn.onnx",
    "models/cats_dogs_cnn.json",
    "models/cats_dogs_cnn.pt",
    "reports/metrics.json",
    "reports/figures/confusion_matrix.png",
    "reports/figures/loss_curve.png",
]
EXCLUDED_NAMES = {
    ".DS_Store",
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "tmp",
    "mlruns",
    "dist",
    "dvc-storage",
}
EXCLUDED_PREFIXES = {".dvc/cache", ".dvc/tmp", "data/raw", "data/processed", "data/demo-raw-backup"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/mlops-assignment-2-submission.zip")
    args = parser.parse_args()
    root = Path.cwd()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"Cannot package; missing required files: {missing}")
    from catsdogs.model import load_model

    model = load_model(root / "models/cats_dogs_cnn.onnx")
    if model.get("data_provenance") != "assignment_kaggle_dataset":
        raise SystemExit("Cannot package: model is not verified as assignment Kaggle data")
    metrics = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
    if metrics.get("data_provenance") != "assignment_kaggle_dataset":
        raise SystemExit("Cannot package: metrics provenance is not the assignment Kaggle data")
    if float(metrics.get("test", {}).get("accuracy", 0.0)) < 0.90:
        raise SystemExit("Cannot package: verified test accuracy is below 90%")
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            relative_text = relative.as_posix()
            if not path.is_file() or any(part in EXCLUDED_NAMES for part in relative.parts):
                continue
            if any(part.endswith(".egg-info") for part in relative.parts):
                continue
            if any(relative_text == prefix or relative_text.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES):
                continue
            if relative_text == args.output:
                continue
            archive.write(path, relative_text)
    manifest = {"archive": str(output), "bytes": output.stat().st_size}
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
