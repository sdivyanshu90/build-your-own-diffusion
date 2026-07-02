"""Tests for utility helpers: imaging, seeding, devices, metrics logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from diffusionlab.utils import (
    JsonlMetricsWriter,
    count_parameters,
    get_device,
    save_image_grid,
    seed_everything,
    setup_logging,
    to_uint8,
)


# ----------------------------------------------------------------------
# Imaging
# ----------------------------------------------------------------------
def test_to_uint8_maps_model_range_to_bytes() -> None:
    images = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    out = to_uint8(images)
    assert out.dtype == torch.uint8
    assert out.flatten().tolist() == [0, 128, 255]


def test_to_uint8_clamps_out_of_range_values() -> None:
    out = to_uint8(torch.tensor([[[[-5.0, 5.0]]]]))
    assert out.flatten().tolist() == [0, 255]


def test_save_image_grid_creates_file_and_parents(tmp_path: Path) -> None:
    images = torch.rand(4, 3, 8, 8) * 2 - 1
    path = save_image_grid(images, tmp_path / "nested" / "grid.png", nrow=2)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_save_image_grid_grayscale(tmp_path: Path) -> None:
    path = save_image_grid(torch.rand(2, 1, 8, 8) * 2 - 1, tmp_path / "gray.png")
    assert path.is_file()


# ----------------------------------------------------------------------
# Seeding & devices
# ----------------------------------------------------------------------
def test_seed_everything_reproduces_torch_randoms() -> None:
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_get_device_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="CUDA is not available"):
        get_device("cuda")


def test_count_parameters() -> None:
    model = torch.nn.Linear(10, 5)  # 10*5 weights + 5 biases
    assert count_parameters(model) == 55
    model.bias.requires_grad_(False)
    assert count_parameters(model) == 50
    assert count_parameters(model, trainable_only=False) == 55


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def test_setup_logging_is_idempotent(tmp_path: Path) -> None:
    logger = setup_logging(log_file=tmp_path / "a.log")
    handlers_first = len(logger.handlers)
    logger = setup_logging(log_file=tmp_path / "a.log")
    assert len(logger.handlers) == handlers_first == 2
    logger.info("hello file")
    for handler in logger.handlers:
        handler.flush()
    assert "hello file" in (tmp_path / "a.log").read_text()
    # Restore console-only logging for other tests.
    setup_logging(level=logging.INFO)


def test_jsonl_metrics_writer_appends_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with JsonlMetricsWriter(path) as writer:
        writer.log({"step": 1, "loss": 0.5})
        writer.log({"step": 2, "loss": 0.25})
    lines = path.read_text().strip().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]


def test_jsonl_metrics_writer_close_is_idempotent(tmp_path: Path) -> None:
    writer = JsonlMetricsWriter(tmp_path / "m.jsonl")
    writer.log({"a": 1})
    writer.close()
    writer.close()
