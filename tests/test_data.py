"""Tests for datasets and dataloaders (synthetic only -- no downloads)."""

from __future__ import annotations

import pytest
import torch

from diffusionlab.config import DataConfig
from diffusionlab.data import ImagesOnly, SyntheticShapes, build_dataloader, build_dataset


def synthetic_config(**overrides: object) -> DataConfig:
    defaults: dict[str, object] = {
        "dataset": "synthetic",
        "image_size": 16,
        "batch_size": 4,
        "num_workers": 0,
        "synthetic_size": 10,
        "download": False,
    }
    defaults.update(overrides)
    return DataConfig(**defaults)  # type: ignore[arg-type]


def test_synthetic_shapes_properties() -> None:
    dataset = SyntheticShapes(size=8, image_size=16, channels=3)
    assert len(dataset) == 8
    sample = dataset[0]
    assert sample.shape == (3, 16, 16)
    assert sample.dtype == torch.float32
    assert sample.min().item() >= -1.0
    assert sample.max().item() <= 1.0


def test_synthetic_shapes_deterministic_per_index() -> None:
    a = SyntheticShapes(size=8, image_size=16, channels=3, seed=5)
    b = SyntheticShapes(size=8, image_size=16, channels=3, seed=5)
    for i in range(8):
        assert torch.equal(a[i], b[i])
    different_seed = SyntheticShapes(size=8, image_size=16, channels=3, seed=6)
    assert not torch.equal(a[0], different_seed[0])


def test_synthetic_shapes_vary_across_indices() -> None:
    dataset = SyntheticShapes(size=4, image_size=16)
    assert not torch.equal(dataset[0], dataset[1])


def test_synthetic_shapes_contain_foreground_and_background() -> None:
    sample = SyntheticShapes(size=1, image_size=32)[0]
    assert (sample > 0.2).any(), "expected bright foreground pixels"
    assert (sample < -0.2).any(), "expected dark background pixels"


def test_synthetic_shapes_index_and_arg_validation() -> None:
    dataset = SyntheticShapes(size=2)
    with pytest.raises(IndexError):
        _ = dataset[2]
    with pytest.raises(ValueError, match="size"):
        SyntheticShapes(size=0)
    with pytest.raises(ValueError, match="channels"):
        SyntheticShapes(size=1, channels=2)


def test_images_only_drops_labels() -> None:
    class Labelled(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            return torch.zeros(1, 4, 4), index

    wrapped = ImagesOnly(Labelled())
    assert len(wrapped) == 3
    assert isinstance(wrapped[1], torch.Tensor)


def test_build_dataset_synthetic_uses_config() -> None:
    dataset = build_dataset(synthetic_config(synthetic_size=7, image_size=32))
    assert len(dataset) == 7
    assert dataset[0].shape == (3, 32, 32)


def test_build_dataset_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset(synthetic_config(dataset="imagenet"))


def test_build_dataloader_batches_and_drops_last() -> None:
    loader = build_dataloader(
        synthetic_config(synthetic_size=10, batch_size=4), torch.device("cpu"), seed=0
    )
    batches = list(loader)
    assert len(batches) == 2  # 10 // 4, last partial batch dropped
    assert all(batch.shape == (4, 3, 16, 16) for batch in batches)


def test_build_dataloader_shuffle_is_seeded() -> None:
    config = synthetic_config(synthetic_size=16, batch_size=16)
    first = next(iter(build_dataloader(config, torch.device("cpu"), seed=1)))
    second = next(iter(build_dataloader(config, torch.device("cpu"), seed=1)))
    third = next(iter(build_dataloader(config, torch.device("cpu"), seed=2)))
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
