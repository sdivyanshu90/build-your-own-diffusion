"""Datasets and dataloaders.

All datasets yield single image tensors (labels are dropped -- this is an
unconditional model) of shape ``(C, image_size, image_size)`` normalised to
``[-1, 1]``, the model-space convention used throughout the library.

Besides the torchvision datasets, a procedural ``synthetic`` dataset of
random shapes is provided. It requires no download and each sample is a
deterministic function of its index, which makes it the workhorse of the
test suite and of CI smoke-training runs.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torchvision import transforms

from diffusionlab.config import DATASET_CHANNELS, DataConfig


class ImagesOnly(Dataset):
    """Wrap a ``(image, label)`` dataset to yield only the image tensor."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.base[index]
        return image


class SyntheticShapes(Dataset):
    """Procedural dataset of anti-alias-free shapes on plain backgrounds.

    Each sample is a filled circle or axis-aligned rectangle with random
    colour, size, and position, generated on the fly from a per-index seed:
    ``dataset[i]`` is identical across processes, epochs, and machines.

    Args:
        size: Number of samples.
        image_size: Square side length in pixels.
        channels: 1 (grayscale) or 3 (RGB).
        seed: Base seed combined with the index for per-sample RNG.
    """

    def __init__(self, size: int = 1024, image_size: int = 32, channels: int = 3, seed: int = 0):
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        self.size = size
        self.image_size = image_size
        self.channels = channels
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.size:
            raise IndexError(index)
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
        s = self.image_size

        def rand(low: float, high: float) -> float:
            return float(torch.empty(1).uniform_(low, high, generator=gen).item())

        background = torch.empty(self.channels).uniform_(-1.0, -0.2, generator=gen)
        foreground = torch.empty(self.channels).uniform_(0.2, 1.0, generator=gen)
        image = background[:, None, None].expand(self.channels, s, s).clone()

        ys = torch.arange(s, dtype=torch.float32)[:, None].expand(s, s)
        xs = torch.arange(s, dtype=torch.float32)[None, :].expand(s, s)
        cx, cy = rand(0.25 * s, 0.75 * s), rand(0.25 * s, 0.75 * s)
        half = rand(0.15 * s, 0.35 * s)
        if torch.rand(1, generator=gen).item() < 0.5:  # circle
            mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= half**2
        else:  # rectangle
            mask = ((xs - cx).abs() <= half) & ((ys - cy).abs() <= half)
        image[:, mask] = foreground[:, None]
        return image


def build_dataset(config: DataConfig) -> Dataset:
    """Instantiate the dataset named in ``config``.

    Torchvision datasets are resized to ``config.image_size``, augmented
    with horizontal flips when enabled, and normalised to ``[-1, 1]``.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    name = config.dataset
    if name == "synthetic":
        return SyntheticShapes(
            size=config.synthetic_size,
            image_size=config.image_size,
            channels=DATASET_CHANNELS["synthetic"],
        )

    factories = {
        "mnist": tv_datasets.MNIST,
        "fashion_mnist": tv_datasets.FashionMNIST,
        "cifar10": tv_datasets.CIFAR10,
    }
    if name not in factories:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASET_CHANNELS)}")

    steps: list[object] = [
        transforms.Resize(config.image_size, antialias=True),
        transforms.CenterCrop(config.image_size),
    ]
    if config.horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip())
    channels = DATASET_CHANNELS[name]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ]
    transform = transforms.Compose(steps)
    base = factories[name](
        root=config.data_dir, train=True, transform=transform, download=config.download
    )
    return ImagesOnly(base)


def build_dataloader(config: DataConfig, device: torch.device, seed: int = 0) -> DataLoader:
    """Build a shuffling, epoch-cycling-friendly training dataloader.

    ``drop_last=True`` keeps every step's batch statistics identical, and a
    seeded generator makes the shuffle order reproducible.
    """
    dataset = build_dataset(config)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )
