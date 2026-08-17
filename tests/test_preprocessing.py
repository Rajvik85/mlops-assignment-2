from pathlib import Path

import numpy as np
from PIL import Image

from catsdogs.preprocessing import preprocess_image, split_paths


def test_preprocess_converts_to_224_rgb_and_normalizes(tmp_path):
    source = tmp_path / "grayscale.png"
    Image.new("L", (80, 160), color=128).save(source)

    result = preprocess_image(source, image_size=224)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert 0.49 < float(result.mean()) < 0.51
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_split_is_deterministic_and_roughly_80_10_10():
    paths = [Path(f"image-{index}.jpg") for index in range(100)]
    first = split_paths(paths, seed=42)
    second = split_paths(paths, seed=42)

    assert first == second
    assert {name: len(items) for name, items in first.items()} == {
        "train": 80,
        "val": 10,
        "test": 10,
    }
    assert not (set(first["train"]) & set(first["test"]))

