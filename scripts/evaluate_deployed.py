"""Send labeled test images, record feedback, and calculate deployed accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx


def evaluate(base_url: str, data_dir: str | Path, limit: int, output_dir: str | Path) -> dict:
    root = Path(data_dir)
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"][:limit]
    if not rows:
        raise ValueError("No test rows found in the processed manifest")
    records = []
    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{base_url.rstrip('/')}/health")
        health.raise_for_status()
        data_provenance = health.json().get("data_provenance", "unknown")
        for row in rows:
            image_path = root / row["path"]
            with image_path.open("rb") as image_file:
                response = client.post(
                    f"{base_url.rstrip('/')}/predict",
                    files={"image": (image_path.name, image_file, "image/jpeg")},
                )
            response.raise_for_status()
            prediction = response.json()
            feedback = client.post(
                f"{base_url.rstrip('/')}/feedback",
                json={
                    "prediction_id": prediction["prediction_id"],
                    "predicted_label": prediction["label"],
                    "true_label": row["label"],
                },
            )
            feedback.raise_for_status()
            records.append(
                {
                    "path": row["path"],
                    "true_label": row["label"],
                    "predicted_label": prediction["label"],
                    "cat_probability": prediction["probabilities"]["cat"],
                    "dog_probability": prediction["probabilities"]["dog"],
                    "correct": prediction["label"] == row["label"],
                }
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "deployed_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "data_provenance": data_provenance,
        "samples": len(records),
        "correct": sum(item["correct"] for item in records),
        "accuracy": sum(item["correct"] for item in records) / len(records),
    }
    (output / "deployed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="reports/monitoring")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.base_url, args.data, args.limit, args.output), indent=2))


if __name__ == "__main__":
    main()
