"""Tests for the Trainer: loop mechanics, artifacts, checkpointing, resume."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from diffusionlab import Trainer
from diffusionlab.pipeline import load_checkpoint
from tests.conftest import make_tiny_config


def test_training_run_produces_artifacts(trained_run: dict[str, Any]) -> None:
    run_dir: Path = trained_run["run_dir"]
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "train.log").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "step_00000002.pt").is_file()
    assert (run_dir / "checkpoints" / "step_00000004.pt").is_file()
    assert (run_dir / "samples" / "step_00000004.png").is_file()


def test_training_metrics_are_finite_and_ordered(trained_run: dict[str, Any]) -> None:
    lines = (trained_run["run_dir"] / "metrics.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert records, "expected at least one metrics record"
    steps = [r["step"] for r in records]
    assert steps == sorted(steps)
    for record in records:
        assert math.isfinite(record["loss"])
        assert record["lr"] > 0
        assert math.isfinite(record["grad_norm"])


def test_trainer_reaches_max_steps(trained_run: dict[str, Any]) -> None:
    assert trained_run["trainer"].step == 4
    assert trained_run["metrics"]["step"] == 4


def test_checkpoint_loads_with_weights_only(trained_run: dict[str, Any]) -> None:
    """The security contract: checkpoints must never require pickle."""
    checkpoint = torch.load(trained_run["checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 4
    assert checkpoint["format_version"] == 1
    assert {"model", "ema", "optimizer", "scaler", "config", "rng"} <= checkpoint.keys()
    assert isinstance(checkpoint["config"], dict)


def test_ema_shadow_differs_from_online_weights(trained_run: dict[str, Any]) -> None:
    checkpoint = load_checkpoint(trained_run["checkpoint"])
    model_sd = checkpoint["model"]
    ema_sd = checkpoint["ema"]["shadow"]
    assert set(model_sd) >= set(ema_sd)
    assert any(not torch.equal(model_sd[name], ema_sd[name]) for name in ema_sd), (
        "EMA should lag behind the online weights after a few steps"
    )


def test_resume_continues_from_checkpoint(tmp_path: Path, trained_run: dict[str, Any]) -> None:
    config = make_tiny_config(tmp_path, training={"max_steps": 6})
    trainer = Trainer(config, resume_from=trained_run["checkpoint"])
    assert trainer.step == 4
    trainer.train()
    trainer.close()
    assert trainer.step == 6


def test_train_is_noop_when_max_steps_reached(tmp_path: Path, trained_run: dict[str, Any]) -> None:
    config = make_tiny_config(tmp_path, training={"max_steps": 4})
    trainer = Trainer(config, resume_from=trained_run["checkpoint"])
    result = trainer.train()
    trainer.close()
    assert trainer.step == 4
    assert result == {"step": 4.0}


def test_fp16_requires_cuda(tmp_path: Path) -> None:
    config = make_tiny_config(tmp_path, training={"mixed_precision": "fp16", "device": "cpu"})
    with pytest.raises(ValueError, match="fp16"):
        Trainer(config)


def test_gradient_accumulation_and_no_clipping(tmp_path: Path) -> None:
    config = make_tiny_config(
        tmp_path,
        training={"max_steps": 2, "gradient_accumulation_steps": 2},
        optim={"grad_clip_norm": 0.0},
    )
    trainer = Trainer(config)
    metrics = trainer.train()
    trainer.close()
    assert trainer.step == 2
    assert math.isfinite(metrics["loss"])


def test_warmup_learning_rate_schedule(tmp_path: Path) -> None:
    config = make_tiny_config(tmp_path, optim={"warmup_steps": 4, "lr": 1.0})
    trainer = Trainer(config)
    try:
        assert trainer._lr_for_step(0) == pytest.approx(0.25)
        assert trainer._lr_for_step(1) == pytest.approx(0.5)
        assert trainer._lr_for_step(3) == pytest.approx(1.0)
        assert trainer._lr_for_step(100) == pytest.approx(1.0)
    finally:
        trainer.close()


def test_no_warmup_uses_full_lr(tmp_path: Path) -> None:
    config = make_tiny_config(tmp_path, optim={"warmup_steps": 0, "lr": 0.5})
    trainer = Trainer(config)
    try:
        assert trainer._lr_for_step(0) == 0.5
    finally:
        trainer.close()


def test_config_snapshot_matches_run(tmp_path: Path) -> None:
    from diffusionlab.config import Config

    config = make_tiny_config(tmp_path, training={"max_steps": 1})
    trainer = Trainer(config)
    trainer.close()
    snapshot = Config.from_yaml(trainer.run_dir / "config.yaml")
    assert snapshot.to_dict() == config.to_dict()
