"""Targeted tests for error handlers and rarely-hit branches."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

import diffusionlab.data as data_module
from diffusionlab import Trainer
from diffusionlab.cli import entrypoint, main
from diffusionlab.config import DataConfig, DiffusionConfig
from diffusionlab.schedulers import DDIMScheduler, build_scheduler
from tests.conftest import make_tiny_config


# ----------------------------------------------------------------------
# torchvision dataset construction (stubbed -- no downloads in tests)
# ----------------------------------------------------------------------
def _make_fake_torchvision_dataset(mode: str, size: tuple[int, int]) -> type:
    """Build a stand-in torchvision dataset class yielding PIL images."""

    class FakeDataset:
        calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, root: str, train: bool, transform: Any, download: bool) -> None:
            type(self).calls.append({"root": root, "train": train, "download": download})
            self.transform = transform

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> tuple[Any, int]:
            from PIL import Image

            color = 200 if mode == "L" else (200, 100, 50)
            return self.transform(Image.new(mode, size, color=color)), index

    return FakeDataset


@pytest.mark.parametrize(
    ("name", "attr", "mode", "source_size", "channels"),
    [
        ("mnist", "MNIST", "L", (28, 28), 1),
        ("fashion_mnist", "FashionMNIST", "L", (28, 28), 1),
        ("cifar10", "CIFAR10", "RGB", (32, 32), 3),
    ],
)
def test_build_dataset_torchvision_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    attr: str,
    mode: str,
    source_size: tuple[int, int],
    channels: int,
) -> None:
    fake = _make_fake_torchvision_dataset(mode, source_size)
    monkeypatch.setattr(data_module.tv_datasets, attr, fake)
    config = DataConfig(
        dataset=name, image_size=16, horizontal_flip=True, download=False, data_dir="/tmp/x"
    )
    dataset = data_module.build_dataset(config)
    sample = dataset[0]
    assert sample.shape == (channels, 16, 16)
    assert sample.min().item() >= -1.0 and sample.max().item() <= 1.0
    assert fake.calls[-1]["download"] is False


# ----------------------------------------------------------------------
# Scheduler edge branches
# ----------------------------------------------------------------------
def test_ddim_stochastic_step_uses_generator() -> None:
    scheduler = DDIMScheduler(DiffusionConfig(num_train_timesteps=50, ddim_eta=0.5))
    scheduler.set_timesteps(10)
    x = torch.randn(2, 3, 8, 8)
    output = torch.randn_like(x)
    t = int(scheduler.timesteps[0])
    a = scheduler.step(output, t, x, generator=torch.Generator().manual_seed(3))
    b = scheduler.step(output, t, x, generator=torch.Generator().manual_seed(3))
    c = scheduler.step(output, t, x, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a.prev_sample, b.prev_sample)
    assert not torch.equal(a.prev_sample, c.prev_sample)


def test_unknown_prediction_type_raises_at_use() -> None:
    scheduler = build_scheduler(DiffusionConfig(num_train_timesteps=10))
    scheduler.prediction_type = "bogus"
    x = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="prediction_type"):
        scheduler.training_target(x, x, torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="prediction_type"):
        scheduler.predict_original_sample(x, 0, x)


# ----------------------------------------------------------------------
# Trainer failure paths
# ----------------------------------------------------------------------
def test_trainer_raises_on_non_finite_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_tiny_config(tmp_path, training={"max_steps": 1})
    trainer = Trainer(config)
    monkeypatch.setattr(
        trainer,
        "_compute_loss",
        lambda batch: torch.tensor(float("nan"), requires_grad=True),
    )
    with pytest.raises(RuntimeError, match="non-finite loss"):
        trainer.train()
    trainer.close()


# ----------------------------------------------------------------------
# CLI plumbing
# ----------------------------------------------------------------------
def test_main_translates_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import diffusionlab.cli as cli_module

    def interrupt(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module.Config, "from_yaml", interrupt)
    config = tmp_path / "c.yaml"
    config.write_text("run_name: x\n")
    assert main(["validate-config", "--config", str(config)]) == 130


def test_entrypoint_exits_with_main_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["diffusionlab", "validate-config", "--config", "nope"])
    with pytest.raises(SystemExit) as excinfo:
        entrypoint()
    assert excinfo.value.code == 2


def test_python_dash_m_invocation_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "diffusionlab", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "diffusionlab" in result.stdout
