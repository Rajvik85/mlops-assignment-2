import torch

from catsdogs.cnn import SimpleCNN, parameter_count
from catsdogs.train_cnn import build_transforms


def test_simple_cnn_output_shape_and_size():
    model = SimpleCNN()
    output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, 2)
    assert 500_000 < parameter_count(model) < 5_000_000


def test_cnn_transforms_keep_required_shape():
    from PIL import Image

    training, evaluation = build_transforms(224, augmentation=True)
    image = Image.new("RGB", (224, 224), color="purple")

    assert training(image).shape == (3, 224, 224)
    assert evaluation(image).shape == (3, 224, 224)
