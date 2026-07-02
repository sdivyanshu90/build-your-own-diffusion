"""End-to-end CLI tests: train -> checkpoint -> sample, plus error paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from diffusionlab.cli import main
from tests.conftest import TINY_CONFIG_DICT


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    raw = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in TINY_CONFIG_DICT.items()
    }
    raw["run_name"] = "cli-run"
    raw["training"]["output_dir"] = str(tmp_path / "runs")
    raw["training"]["max_steps"] = 2
    raw["training"]["checkpoint_interval"] = 2
    raw["training"]["sample_interval"] = 1000
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_validate_config_ok(cli_config: Path) -> None:
    assert main(["validate-config", "--config", str(cli_config)]) == 0


def test_validate_config_missing_file(tmp_path: Path) -> None:
    assert main(["validate-config", "--config", str(tmp_path / "nope.yaml")]) == 2


def test_validate_config_bad_override(cli_config: Path) -> None:
    assert (
        main(
            [
                "validate-config",
                "--config",
                str(cli_config),
                "--set",
                "training.max_steps=0",
            ]
        )
        == 2
    )


def test_train_then_sample_end_to_end(cli_config: Path, tmp_path: Path) -> None:
    assert main(["train", "--config", str(cli_config)]) == 0
    checkpoint = tmp_path / "runs" / "cli-run" / "checkpoints" / "last.pt"
    assert checkpoint.is_file()

    output = tmp_path / "out" / "grid.png"
    assert (
        main(
            [
                "sample",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--num-images",
                "2",
                "--steps",
                "3",
                "--seed",
                "0",
                "--device",
                "cpu",
                "--grid-cols",
                "2",
            ]
        )
        == 0
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_train_with_overrides_and_resume(cli_config: Path, tmp_path: Path) -> None:
    assert main(["train", "--config", str(cli_config)]) == 0
    checkpoint = tmp_path / "runs" / "cli-run" / "checkpoints" / "last.pt"
    assert (
        main(
            [
                "train",
                "--config",
                str(cli_config),
                "--set",
                "training.max_steps=3",
                "--resume",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert (tmp_path / "runs" / "cli-run" / "checkpoints" / "step_00000003.pt").is_file()


def test_sample_missing_checkpoint_returns_error(tmp_path: Path) -> None:
    assert main(["sample", "--checkpoint", str(tmp_path / "missing.pt"), "--device", "cpu"]) == 2


def test_sample_rejects_bad_num_images(trained_run: dict) -> None:
    assert (
        main(
            [
                "sample",
                "--checkpoint",
                str(trained_run["checkpoint"]),
                "--num-images",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 2
    )


def test_sample_no_ema_flag(trained_run: dict, tmp_path: Path) -> None:
    output = tmp_path / "no_ema.png"
    assert (
        main(
            [
                "sample",
                "--checkpoint",
                str(trained_run["checkpoint"]),
                "--output",
                str(output),
                "--num-images",
                "1",
                "--steps",
                "2",
                "--no-ema",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert output.is_file()
