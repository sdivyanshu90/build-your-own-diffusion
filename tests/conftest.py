"""Shared fixtures: a miniature configuration and a pre-trained tiny run.

Everything here runs on CPU with a model small enough that the full suite
finishes in seconds; the ``trained_run`` fixture is session-scoped so the
trainer/pipeline/CLI tests share a single short training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from diffusionlab import Config, Trainer

TINY_CONFIG_DICT: dict[str, Any] = {
    "run_name": "tiny",
    "model": {
        "in_channels": 3,
        "base_channels": 16,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "attention_levels": [1],
        "num_heads": 2,
        "dropout": 0.0,
        "num_groups": 4,
    },
    "diffusion": {
        "num_train_timesteps": 50,
        "beta_schedule": "cosine",
        "prediction_type": "epsilon",
        "clip_sample": True,
        "sampler": "ddim",
        "num_inference_steps": 5,
        "ddim_eta": 0.0,
    },
    "data": {
        "dataset": "synthetic",
        "image_size": 16,
        "batch_size": 4,
        "num_workers": 0,
        "horizontal_flip": False,
        "download": False,
        "synthetic_size": 32,
    },
    "optim": {
        "lr": 1.0e-3,
        "warmup_steps": 2,
        "grad_clip_norm": 1.0,
        "ema_decay": 0.9,
    },
    "training": {
        "max_steps": 4,
        "log_interval": 2,
        "sample_interval": 1000,
        "checkpoint_interval": 1000,
        "num_sample_images": 2,
        "seed": 0,
        "device": "cpu",
    },
}


def make_tiny_config(tmp_dir: Path, **section_overrides: dict[str, Any]) -> Config:
    """Build the tiny test config with its output redirected to ``tmp_dir``.

    ``section_overrides`` maps section names to dicts of field overrides,
    e.g. ``make_tiny_config(tmp, training={"max_steps": 2})``.
    """
    raw = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in TINY_CONFIG_DICT.items()
    }
    raw["training"]["output_dir"] = str(tmp_dir / "runs")
    for section, fields in section_overrides.items():
        raw.setdefault(section, {}).update(fields)
    return Config.from_dict(raw)


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    return make_tiny_config(tmp_path)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Train the tiny model for a few steps once; share the artifacts."""
    tmp_dir = tmp_path_factory.mktemp("trained_run")
    config = make_tiny_config(
        tmp_dir,
        training={"max_steps": 4, "checkpoint_interval": 2, "sample_interval": 4},
    )
    trainer = Trainer(config)
    metrics = trainer.train()
    trainer.close()
    checkpoint = trainer.checkpoint_dir / "last.pt"
    assert checkpoint.is_file()
    return {
        "config": config,
        "trainer": trainer,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "run_dir": trainer.run_dir,
    }


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Give every test the same starting RNG state."""
    torch.manual_seed(1234)
