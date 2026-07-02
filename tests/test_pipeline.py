"""Tests for the sampling pipeline and safe checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from diffusionlab import DiffusionPipeline, UNet, build_scheduler
from diffusionlab.pipeline import load_checkpoint
from tests.conftest import make_tiny_config


def test_from_checkpoint_ema_and_online_weights(trained_run: dict[str, Any]) -> None:
    ema_pipe = DiffusionPipeline.from_checkpoint(
        trained_run["checkpoint"], device="cpu", use_ema=True
    )
    online_pipe = DiffusionPipeline.from_checkpoint(
        trained_run["checkpoint"], device="cpu", use_ema=False
    )
    ema_params = dict(ema_pipe.model.named_parameters())
    online_params = dict(online_pipe.model.named_parameters())
    assert any(not torch.equal(ema_params[name], online_params[name]) for name in ema_params)
    assert ema_pipe.image_size == 16
    assert ema_pipe.image_channels == 3


def test_sample_shape_range_and_determinism(trained_run: dict[str, Any]) -> None:
    pipeline = DiffusionPipeline.from_checkpoint(trained_run["checkpoint"], device="cpu")
    a = pipeline.sample(2, generator=torch.Generator().manual_seed(0))
    b = pipeline.sample(2, generator=torch.Generator().manual_seed(0))
    c = pipeline.sample(2, generator=torch.Generator().manual_seed(1))
    assert a.shape == (2, 3, 16, 16)
    assert a.min().item() >= -1.0 and a.max().item() <= 1.0
    assert torch.equal(a, b), "same seed must give identical samples (eta=0 DDIM)"
    assert not torch.equal(a, c)


def test_sample_with_step_override_and_progress(trained_run: dict[str, Any]) -> None:
    pipeline = DiffusionPipeline.from_checkpoint(trained_run["checkpoint"], device="cpu")
    images = pipeline.sample(1, num_inference_steps=3, progress=True)
    assert images.shape == (1, 3, 16, 16)


def test_sample_with_ddpm_sampler_override(trained_run: dict[str, Any]) -> None:
    pipeline = DiffusionPipeline.from_checkpoint(
        trained_run["checkpoint"], device="cpu", sampler="ddpm"
    )
    # DDPM must walk the full training chain (50 steps for the tiny config).
    images = pipeline.sample(1, num_inference_steps=50)
    assert images.shape == (1, 3, 16, 16)
    assert torch.isfinite(images).all()


def test_sample_rejects_bad_num_images(trained_run: dict[str, Any]) -> None:
    pipeline = DiffusionPipeline.from_checkpoint(trained_run["checkpoint"], device="cpu")
    with pytest.raises(ValueError, match="num_images"):
        pipeline.sample(0)


def test_pipeline_direct_construction(tmp_path: Path) -> None:
    config = make_tiny_config(tmp_path)
    pipeline = DiffusionPipeline(
        model=UNet(config.model),
        scheduler=build_scheduler(config.diffusion),
        image_size=16,
        image_channels=3,
        device="cpu",
    )
    assert not pipeline.model.training  # eval mode enforced
    images = pipeline.sample(1)
    assert images.shape == (1, 3, 16, 16)


def test_load_checkpoint_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_load_checkpoint_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="missing required keys"):
        load_checkpoint(path)
