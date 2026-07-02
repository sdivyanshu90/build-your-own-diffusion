"""Tests for the configuration system: loading, overrides, validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from diffusionlab.config import Config, ConfigError


def test_defaults_are_valid() -> None:
    Config().validate()


def test_from_dict_roundtrip() -> None:
    config = Config.from_dict({"run_name": "x", "model": {"base_channels": 64}})
    assert config.run_name == "x"
    assert config.model.base_channels == 64
    assert Config.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_tuples_survive_yaml_roundtrip(tmp_path: Path) -> None:
    config = Config.from_dict({"model": {"channel_multipliers": [1, 2, 4]}})
    assert config.model.channel_multipliers == (1, 2, 4)
    path = tmp_path / "config.yaml"
    config.save_yaml(path)
    reloaded = Config.from_yaml(path)
    assert reloaded.model.channel_multipliers == (1, 2, 4)
    assert reloaded.to_dict() == config.to_dict()


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"modle": {}})


def test_unknown_nested_key_rejected_with_section_name() -> None:
    with pytest.raises(ConfigError, match="model"):
        Config.from_dict({"model": {"bacth_size": 3}})


def test_non_mapping_section_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        Config.from_dict({"model": [1, 2]})


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="top level"):
        Config.from_yaml(path)


def test_empty_yaml_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert Config.from_yaml(path).to_dict() == Config().to_dict()


def test_overrides_parse_yaml_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"run_name": "base"}))
    config = Config.from_yaml(
        path,
        overrides=[
            "training.max_steps=7",
            "diffusion.clip_sample=false",
            "model.channel_multipliers=[1, 2]",
            "run_name=overridden",
        ],
    )
    assert config.training.max_steps == 7
    assert config.diffusion.clip_sample is False
    assert config.model.channel_multipliers == (1, 2)
    assert config.run_name == "overridden"


@pytest.mark.parametrize("bad", ["no_equals_sign", ".=3", "=5"])
def test_malformed_override_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("run_name: base\n")
    with pytest.raises(ConfigError):
        Config.from_yaml(path, overrides=[bad])


@pytest.mark.parametrize(
    ("section", "field", "value", "pattern"),
    [
        ("model", "in_channels", 0, "in_channels"),
        ("model", "base_channels", 30, "divisible"),  # 30 % 8 != 0
        ("model", "channel_multipliers", [], "non-empty"),
        ("model", "num_res_blocks", 0, "num_res_blocks"),
        ("model", "attention_levels", [9], "attention_levels"),
        ("model", "num_heads", 7, "num_heads"),  # 128 % 7 != 0
        ("model", "dropout", 1.5, "dropout"),
        ("diffusion", "num_train_timesteps", 1, "num_train_timesteps"),
        ("diffusion", "beta_schedule", "quadratic", "beta_schedule"),
        ("diffusion", "beta_start", 0.5, "beta_start"),
        ("diffusion", "prediction_type", "noise", "prediction_type"),
        ("diffusion", "variance_type", "learned", "variance_type"),
        ("diffusion", "sampler", "euler", "sampler"),
        ("diffusion", "num_inference_steps", 10_000, "num_inference_steps"),
        ("diffusion", "ddim_eta", -0.1, "ddim_eta"),
        ("diffusion", "timestep_spacing", "middle", "timestep_spacing"),
        ("data", "dataset", "imagenet", "dataset"),
        ("data", "image_size", 30, "image_size"),  # not divisible by 8
        ("data", "batch_size", 0, "batch_size"),
        ("data", "num_workers", -1, "num_workers"),
        ("optim", "lr", 0, "lr"),
        ("optim", "betas", [0.9, 1.0], "betas"),
        ("optim", "grad_clip_norm", -1, "grad_clip_norm"),
        ("optim", "ema_decay", 1.0, "ema_decay"),
        ("training", "max_steps", 0, "max_steps"),
        ("training", "mixed_precision", "fp8", "mixed_precision"),
        ("training", "log_interval", 0, "log_interval"),
        ("training", "device", "tpu", "device"),
    ],
)
def test_field_validation(section: str, field: str, value: object, pattern: str) -> None:
    with pytest.raises(ConfigError, match=pattern):
        Config.from_dict({section: {field: value}})


def test_run_name_validation() -> None:
    with pytest.raises(ConfigError, match="run_name"):
        Config.from_dict({"run_name": "bad/name"})


def test_ddpm_sampler_requires_full_steps() -> None:
    with pytest.raises(ConfigError, match="ddpm sampler"):
        Config.from_dict({"diffusion": {"sampler": "ddpm", "num_inference_steps": 50}})
    Config.from_dict({"diffusion": {"sampler": "ddpm", "num_inference_steps": 1000}}).validate()


def test_dataset_channels_must_match_model() -> None:
    with pytest.raises(ConfigError, match="in_channels"):
        Config.from_dict({"data": {"dataset": "mnist"}})  # model default is 3 channels
    Config.from_dict({"data": {"dataset": "mnist"}, "model": {"in_channels": 1}}).validate()
