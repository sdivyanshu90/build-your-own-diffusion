"""Typed, validated, YAML-backed configuration for diffusionlab.

Design notes
------------
- Plain :mod:`dataclasses` are used instead of a third-party validation
  library to keep the dependency surface minimal; validation lives in
  :meth:`Config.validate` and is invoked by every loading entry point.
- Loading is *strict*: unknown keys in a YAML file raise :class:`ConfigError`
  instead of being silently ignored, which catches typos such as
  ``bacth_size`` before a multi-hour training run starts.
- Command-line overrides use dotted paths (``training.max_steps=100``) whose
  values are parsed with ``yaml.safe_load`` so that ``true``, ``0.1`` and
  ``[1, 2, 4]`` all coerce to their natural Python types.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


#: Channel count of each built-in dataset; used to validate ``model.in_channels``.
DATASET_CHANNELS: dict[str, int] = {
    "mnist": 1,
    "fashion_mnist": 1,
    "cifar10": 3,
    "synthetic": 3,
}

_BETA_SCHEDULES = ("linear", "scaled_linear", "cosine")
_PREDICTION_TYPES = ("epsilon", "sample", "v_prediction")
_VARIANCE_TYPES = ("fixed_small", "fixed_large")
_SAMPLERS = ("ddpm", "ddim")
_TIMESTEP_SPACINGS = ("leading", "trailing")
_PRECISIONS = ("none", "fp16", "bf16")


@dataclass
class ModelConfig:
    """Architecture of the UNet noise-prediction network.

    Attributes:
        in_channels: Number of image channels (1 for grayscale, 3 for RGB).
            The network output has the same channel count.
        base_channels: Width of the first UNet level. Every other level's
            width is ``base_channels * channel_multipliers[level]``.
        channel_multipliers: Per-level width multipliers; the number of
            entries determines the UNet depth (each extra level halves the
            spatial resolution once).
        num_res_blocks: Residual blocks per level in the encoder (the decoder
            uses ``num_res_blocks + 1`` to consume skip connections).
        attention_levels: Level indices (0-based) at which self-attention is
            inserted after every residual block. The bottleneck always has
            attention regardless of this setting.
        num_heads: Attention heads for every attention block.
        dropout: Dropout probability inside residual blocks.
        num_groups: Group count for all GroupNorm layers; must divide
            ``base_channels``.
    """

    in_channels: int = 3
    base_channels: int = 128
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = (1,)
    num_heads: int = 4
    dropout: float = 0.1
    num_groups: int = 8


@dataclass
class DiffusionConfig:
    """Forward/reverse diffusion process and sampler settings.

    Attributes:
        num_train_timesteps: Length ``T`` of the discrete diffusion chain.
        beta_schedule: Noise schedule; one of ``linear``, ``scaled_linear``
            (Stable-Diffusion style) or ``cosine`` (Nichol & Dhariwal).
        beta_start: First beta for the (scaled_)linear schedules.
        beta_end: Last beta for the (scaled_)linear schedules.
        prediction_type: What the network is trained to predict --
            ``epsilon`` (the added noise), ``sample`` (the clean image x0) or
            ``v_prediction`` (Salimans & Ho velocity).
        clip_sample: Clip the predicted x0 to ``[-1, 1]`` during sampling,
            which stabilises early denoising steps for image data.
        variance_type: DDPM reverse-process variance -- ``fixed_small``
            (posterior variance) or ``fixed_large`` (beta_t).
        sampler: Default sampler used for image generation.
        num_inference_steps: Reverse steps used at sampling time. Must equal
            ``num_train_timesteps`` for the DDPM sampler; DDIM may use fewer.
        ddim_eta: DDIM stochasticity in [0, 1]; 0 is fully deterministic and
            1 matches the DDPM posterior variance.
        timestep_spacing: How DDIM selects its timestep subsequence
            (``leading`` matches the original DDIM code; ``trailing`` starts
            exactly at T-1 and is often slightly better at few steps).
    """

    num_train_timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    variance_type: str = "fixed_small"
    sampler: str = "ddim"
    num_inference_steps: int = 50
    ddim_eta: float = 0.0
    timestep_spacing: str = "leading"


@dataclass
class DataConfig:
    """Dataset and dataloader settings.

    Attributes:
        dataset: One of ``mnist``, ``fashion_mnist``, ``cifar10`` or
            ``synthetic`` (a procedural, download-free dataset of shapes).
        data_dir: Root directory for downloaded datasets.
        image_size: Square side length images are resized to; must be
            divisible by ``2 ** (len(channel_multipliers) - 1)``.
        batch_size: Per-step batch size (before gradient accumulation).
        num_workers: DataLoader worker processes.
        horizontal_flip: Apply random horizontal flips (disable for digits).
        download: Allow torchvision to download missing datasets.
        synthetic_size: Number of samples in the synthetic dataset.
    """

    dataset: str = "cifar10"
    data_dir: str = "./data"
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 4
    horizontal_flip: bool = True
    download: bool = True
    synthetic_size: int = 1024


@dataclass
class OptimConfig:
    """Optimizer, learning-rate warmup, and EMA settings.

    Attributes:
        lr: Peak AdamW learning rate (after warmup).
        weight_decay: AdamW decoupled weight decay.
        betas: AdamW momentum coefficients.
        warmup_steps: Linear LR warmup duration in optimizer steps.
        grad_clip_norm: Global gradient-norm clip; ``0`` disables clipping.
        ema_decay: Exponential-moving-average decay for evaluation weights.
    """

    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999


@dataclass
class TrainingConfig:
    """Run-level training loop settings.

    Attributes:
        max_steps: Total optimizer steps to train for.
        gradient_accumulation_steps: Micro-batches accumulated per step.
        mixed_precision: ``none``, ``fp16`` (CUDA only, with loss scaling) or
            ``bf16``.
        log_interval: Steps between metric log lines / JSONL records.
        sample_interval: Steps between EMA sample grids written to disk.
        checkpoint_interval: Steps between numbered checkpoints; a rolling
            ``last.pt`` is refreshed at the same cadence.
        num_sample_images: Images per in-training sample grid.
        output_dir: Root under which ``<output_dir>/<run_name>`` is created.
        seed: Global random seed for reproducibility.
        device: ``auto`` (CUDA if available), ``cpu``, ``cuda`` or ``cuda:N``.
    """

    max_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "none"
    log_interval: int = 50
    sample_interval: int = 2_000
    checkpoint_interval: int = 10_000
    num_sample_images: int = 64
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    """Top-level experiment configuration.

    Attributes:
        run_name: Name of the run; artifacts land in
            ``<training.output_dir>/<run_name>``.
        model: UNet architecture settings.
        diffusion: Diffusion process and sampler settings.
        data: Dataset and dataloader settings.
        optim: Optimizer/EMA settings.
        training: Training-loop settings.
    """

    run_name: str = "diffusion-run"
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a validated :class:`Config` from a nested plain dict.

        Unknown keys at any level raise :class:`ConfigError`.
        """
        config = typing.cast(Config, _build_dataclass(cls, data, path=""))
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        """Load a config from a YAML file, then apply dotted overrides.

        Args:
            path: YAML file produced by hand or by :meth:`save_yaml`.
            overrides: Strings of the form ``section.key=value`` (values are
                YAML-parsed, so lists and booleans work).
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"top level of {path} must be a mapping, got {type(raw).__name__}")
        for override in overrides or []:
            _apply_override(raw, override)
        return cls.from_dict(raw)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested dict of YAML/JSON-safe primitives."""
        return _to_primitive_dict(dataclasses.asdict(self))

    def save_yaml(self, path: str | Path) -> None:
        """Write the config as YAML (used to snapshot each training run)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check every field and cross-field constraint.

        Raises:
            ConfigError: With a message naming the offending field.
        """
        m, d, dt, o, t = self.model, self.diffusion, self.data, self.optim, self.training

        if not self.run_name or "/" in self.run_name:
            raise ConfigError(f"run_name must be a non-empty name without '/': {self.run_name!r}")

        # --- model ---
        _require(m.in_channels >= 1, "model.in_channels must be >= 1", m.in_channels)
        _require(m.base_channels >= 1, "model.base_channels must be >= 1", m.base_channels)
        _require(m.num_groups >= 1, "model.num_groups must be >= 1", m.num_groups)
        _require(
            m.base_channels % m.num_groups == 0,
            "model.base_channels must be divisible by model.num_groups",
            (m.base_channels, m.num_groups),
        )
        _require(
            len(m.channel_multipliers) >= 1 and all(c >= 1 for c in m.channel_multipliers),
            "model.channel_multipliers must be a non-empty tuple of positive ints",
            m.channel_multipliers,
        )
        _require(m.num_res_blocks >= 1, "model.num_res_blocks must be >= 1", m.num_res_blocks)
        levels = range(len(m.channel_multipliers))
        _require(
            all(a in levels for a in m.attention_levels),
            f"model.attention_levels must be level indices in [0, {len(m.channel_multipliers)})",
            m.attention_levels,
        )
        _require(m.num_heads >= 1, "model.num_heads must be >= 1", m.num_heads)
        attn_channels = [m.base_channels * m.channel_multipliers[a] for a in m.attention_levels]
        attn_channels.append(m.base_channels * m.channel_multipliers[-1])  # bottleneck
        _require(
            all(c % m.num_heads == 0 for c in attn_channels),
            "every attention layer's channel count must be divisible by model.num_heads",
            (attn_channels, m.num_heads),
        )
        _require(0.0 <= m.dropout < 1.0, "model.dropout must be in [0, 1)", m.dropout)

        # --- diffusion ---
        _require(
            d.num_train_timesteps >= 2,
            "diffusion.num_train_timesteps must be >= 2",
            d.num_train_timesteps,
        )
        _require_choice("diffusion.beta_schedule", d.beta_schedule, _BETA_SCHEDULES)
        _require(
            0.0 < d.beta_start < d.beta_end < 1.0,
            "diffusion must satisfy 0 < beta_start < beta_end < 1",
            (d.beta_start, d.beta_end),
        )
        _require_choice("diffusion.prediction_type", d.prediction_type, _PREDICTION_TYPES)
        _require_choice("diffusion.variance_type", d.variance_type, _VARIANCE_TYPES)
        _require_choice("diffusion.sampler", d.sampler, _SAMPLERS)
        _require_choice("diffusion.timestep_spacing", d.timestep_spacing, _TIMESTEP_SPACINGS)
        _require(
            1 <= d.num_inference_steps <= d.num_train_timesteps,
            "diffusion.num_inference_steps must be in [1, num_train_timesteps]",
            d.num_inference_steps,
        )
        if d.sampler == "ddpm":
            _require(
                d.num_inference_steps == d.num_train_timesteps,
                "the ddpm sampler requires num_inference_steps == num_train_timesteps "
                "(use sampler=ddim for accelerated sampling)",
                d.num_inference_steps,
            )
        _require(d.ddim_eta >= 0.0, "diffusion.ddim_eta must be >= 0", d.ddim_eta)

        # --- data ---
        _require_choice("data.dataset", dt.dataset, tuple(DATASET_CHANNELS))
        expected_channels = DATASET_CHANNELS[dt.dataset]
        _require(
            m.in_channels == expected_channels,
            f"model.in_channels must be {expected_channels} for dataset {dt.dataset!r}",
            m.in_channels,
        )
        downsample_factor = 2 ** (len(m.channel_multipliers) - 1)
        _require(
            dt.image_size >= 4 and dt.image_size % downsample_factor == 0,
            f"data.image_size must be >= 4 and divisible by {downsample_factor} "
            f"(2 ** (len(channel_multipliers) - 1))",
            dt.image_size,
        )
        _require(dt.batch_size >= 1, "data.batch_size must be >= 1", dt.batch_size)
        _require(dt.num_workers >= 0, "data.num_workers must be >= 0", dt.num_workers)
        _require(dt.synthetic_size >= 1, "data.synthetic_size must be >= 1", dt.synthetic_size)

        # --- optim ---
        _require(o.lr > 0, "optim.lr must be > 0", o.lr)
        _require(o.weight_decay >= 0, "optim.weight_decay must be >= 0", o.weight_decay)
        _require(
            len(o.betas) == 2 and all(0.0 <= b < 1.0 for b in o.betas),
            "optim.betas must be two floats in [0, 1)",
            o.betas,
        )
        _require(o.warmup_steps >= 0, "optim.warmup_steps must be >= 0", o.warmup_steps)
        _require(
            o.grad_clip_norm >= 0,
            "optim.grad_clip_norm must be >= 0 (0 disables)",
            o.grad_clip_norm,
        )
        _require(0.0 <= o.ema_decay < 1.0, "optim.ema_decay must be in [0, 1)", o.ema_decay)

        # --- training ---
        _require(t.max_steps >= 1, "training.max_steps must be >= 1", t.max_steps)
        _require(
            t.gradient_accumulation_steps >= 1,
            "training.gradient_accumulation_steps must be >= 1",
            t.gradient_accumulation_steps,
        )
        _require_choice("training.mixed_precision", t.mixed_precision, _PRECISIONS)
        for name in ("log_interval", "sample_interval", "checkpoint_interval"):
            _require(getattr(t, name) >= 1, f"training.{name} must be >= 1", getattr(t, name))
        _require(
            t.num_sample_images >= 1, "training.num_sample_images must be >= 1", t.num_sample_images
        )
        _require(
            t.device == "auto" or t.device == "cpu" or t.device.startswith("cuda"),
            "training.device must be 'auto', 'cpu', 'cuda' or 'cuda:N'",
            t.device,
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _require(condition: bool, message: str, value: Any) -> None:
    if not condition:
        raise ConfigError(f"{message} (got {value!r})")


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ConfigError(f"{name} must be one of {choices} (got {value!r})")


def _build_dataclass(cls: type, data: Any, path: str) -> Any:
    """Recursively construct dataclass ``cls`` from a plain dict, strictly."""
    if not isinstance(data, dict):
        raise ConfigError(
            f"section {path or '<root>'} must be a mapping, got {type(data).__name__}"
        )
    hints = typing.get_type_hints(cls)
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in section {path or '<root>'}; "
            f"valid keys: {sorted(valid)}"
        )
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        child_path = f"{path}.{f.name}" if path else f.name
        if dataclasses.is_dataclass(hint):
            kwargs[f.name] = _build_dataclass(hint, value, child_path)
        elif typing.get_origin(hint) is tuple:
            if not isinstance(value, list | tuple):
                raise ConfigError(f"{child_path} must be a list, got {type(value).__name__}")
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _to_primitive_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Convert tuples to lists recursively so YAML/JSON stay round-trippable."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[key] = _to_primitive_dict(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``dotted.path=value`` override to the raw config dict."""
    if "=" not in override:
        raise ConfigError(f"override must look like 'section.key=value', got {override!r}")
    dotted, _, literal = override.partition("=")
    keys = [k for k in dotted.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"override has an empty key path: {override!r}")
    try:
        value = yaml.safe_load(literal)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse override value {literal!r}: {exc}") from exc
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override path {dotted!r} collides with a non-mapping value")
    node[keys[-1]] = value
