"""Download and normalize the assignment's Kaggle dataset using KaggleHub."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DATASET_HANDLE = "bhavikjikadara/dog-and-cat-classification-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--handle", default=DATASET_HANDLE)
    args = parser.parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Install KaggleHub first: python -m pip install kagglehub"
        ) from exc
    downloaded = Path(kagglehub.dataset_download(args.handle))
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}; move or remove it first")
    shutil.copytree(downloaded, output)
    source = {
        "dataset_handle": args.handle,
        "source_url": f"https://www.kaggle.com/datasets/{args.handle}",
        "requested_version": "latest",
        "license": "Apache-2.0",
    }
    (output / "DATASET_SOURCE.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    print(f"Downloaded {args.handle} to {output}")
    print("Next: dvc add data/raw && dvc push")


if __name__ == "__main__":
    main()
