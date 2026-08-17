"""Fail-fast post-deployment health and prediction smoke test."""

from __future__ import annotations

import argparse
import io
import sys

import httpx
from PIL import Image, ImageDraw


def sample_image() -> bytes:
    image = Image.new("RGB", (224, 224), (210, 125, 60))
    draw = ImageDraw.Draw(image)
    for y in range(0, 224, 18):
        draw.line((0, y, 224, y), fill=(245, 195, 95), width=7)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def run(base_url: str, timeout: float = 15.0) -> None:
    base = base_url.rstrip("/")
    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        if health.json().get("status") != "healthy":
            raise RuntimeError(f"Unhealthy response: {health.text}")
        prediction = client.post(
            f"{base}/predict",
            files={"image": ("smoke.jpg", sample_image(), "image/jpeg")},
        )
        prediction.raise_for_status()
        body = prediction.json()
        if body.get("label") not in {"cat", "dog"}:
            raise RuntimeError(f"Invalid label: {body}")
        probabilities = body.get("probabilities", {})
        if set(probabilities) != {"cat", "dog"}:
            raise RuntimeError(f"Missing probabilities: {body}")
        if abs(sum(probabilities.values()) - 1.0) > 1e-5:
            raise RuntimeError(f"Probabilities do not sum to one: {body}")
    print(f"Smoke test passed: health=healthy label={body['label']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    try:
        run(args.base_url, args.timeout)
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

