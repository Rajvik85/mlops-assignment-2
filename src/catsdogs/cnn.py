"""Small convolutional neural network trained from scratch for Cats vs Dogs."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two convolution layers followed by 2x downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class SimpleCNN(nn.Module):
    """A beginner-readable CNN with no pretrained or transferred weights."""

    def __init__(self, class_count: int = 2, dropout: float = 0.4) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be at least 2")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, class_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def parameter_count(model: nn.Module) -> int:
    """Return the trainable parameter count for reporting and tests."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
