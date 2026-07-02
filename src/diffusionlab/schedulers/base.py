"""Scheduler base class: the forward process and prediction conversions.

Notation (following Ho et al., 2020)
------------------------------------
- ``beta_t``: per-step noise variance, ``t = 0 .. T-1``.
- ``alpha_t = 1 - beta_t``; ``alpha_bar_t = prod_{s<=t} alpha_s``
  (called ``alphas_cumprod`` in code).
- Forward (noising) process:
  ``q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)``, sampled
  in closed form by :meth:`BaseScheduler.add_noise`.
- Velocity (Salimans & Ho, 2022):
  ``v = sqrt(alpha_bar_t) eps - sqrt(1 - alpha_bar_t) x_0``.

Everything a *sampler* needs beyond this -- the reverse-step rule -- lives in
the concrete subclasses (:class:`~diffusionlab.schedulers.ddpm.DDPMScheduler`
and :class:`~diffusionlab.schedulers.ddim.DDIMScheduler`).
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import torch

from diffusionlab.config import DiffusionConfig
from diffusionlab.schedulers.schedules import get_beta_schedule


class SchedulerOutput(NamedTuple):
    """Result of one reverse diffusion step.

    Attributes:
        prev_sample: ``x_{t-1}``, the input for the next reverse step.
        pred_original_sample: The model's implied estimate of ``x_0`` at this
            step (useful for visualisation and for guidance techniques).
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor


def _extract(
    values: torch.Tensor, t: int | torch.Tensor, broadcast_to: torch.Tensor
) -> torch.Tensor:
    """Gather per-timestep coefficients and reshape for broadcasting.

    Args:
        values: 1-D float32 tensor of length T (CPU-resident).
        t: A python int (uniform timestep across the batch, the sampling
            case) or an integer tensor of shape ``(B,)`` (the training case).
        broadcast_to: The image tensor whose shape the result must broadcast
            against, e.g. ``(B, C, H, W)``.

    Returns:
        Tensor of shape ``(B, 1, 1, ...)`` or ``(1, 1, 1, ...)`` on the same
        device/dtype as ``broadcast_to``.
    """
    if isinstance(t, int):
        t = torch.tensor([t])
    index = t.detach().to(device=values.device, dtype=torch.long)
    coeff = values[index].to(device=broadcast_to.device, dtype=broadcast_to.dtype)
    return coeff.reshape(coeff.shape[0], *([1] * (broadcast_to.ndim - 1)))


class BaseScheduler(abc.ABC):
    """Shared diffusion-process math for all samplers.

    All schedule tensors are precomputed in float64 at construction and
    stored as float32 CPU tensors; :func:`_extract` moves the (tiny) indexed
    slices to the data device on demand.

    Args:
        config: Validated :class:`~diffusionlab.config.DiffusionConfig`.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.prediction_type = config.prediction_type
        self.clip_sample = config.clip_sample

        betas64 = get_beta_schedule(
            config.beta_schedule,
            config.num_train_timesteps,
            config.beta_start,
            config.beta_end,
        )
        alphas64 = 1.0 - betas64
        alphas_cumprod64 = torch.cumprod(alphas64, dim=0)

        self.betas = betas64.float()
        self.alphas = alphas64.float()
        self.alphas_cumprod = alphas_cumprod64.float()
        self.sqrt_alphas_cumprod = alphas_cumprod64.sqrt().float()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod64).sqrt().float()

        self._timesteps: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward (noising) process -- used in training
    # ------------------------------------------------------------------
    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: int | torch.Tensor
    ) -> torch.Tensor:
        """Sample ``q(x_t | x_0)`` in closed form.

        ``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps``
        """
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, timesteps, original_samples)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples)
        return sqrt_ab * original_samples + sqrt_1m_ab * noise

    def get_velocity(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: int | torch.Tensor
    ) -> torch.Tensor:
        """Velocity target ``v = sqrt(alpha_bar_t) eps - sqrt(1-alpha_bar_t) x_0``."""
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, timesteps, original_samples)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples)
        return sqrt_ab * noise - sqrt_1m_ab * original_samples

    def training_target(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """The regression target implied by ``prediction_type``."""
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return original_samples
        if self.prediction_type == "v_prediction":
            return self.get_velocity(original_samples, noise, timesteps)
        raise ValueError(f"unknown prediction_type {self.prediction_type!r}")

    # ------------------------------------------------------------------
    # Prediction conversions -- used by both samplers
    # ------------------------------------------------------------------
    def predict_original_sample(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        clip: bool | None = None,
    ) -> torch.Tensor:
        """Recover the implied ``x_0`` from the model output at ``timestep``.

        Args:
            model_output: Raw network output (epsilon, x0, or v).
            timestep: Timestep(s) at which the output was produced.
            sample: The noisy input ``x_t`` the network saw.
            clip: Override for ``config.clip_sample``.
        """
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, timestep, sample)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, timestep, sample)
        if self.prediction_type == "epsilon":
            pred_x0 = (sample - sqrt_1m_ab * model_output) / sqrt_ab
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = sqrt_ab * sample - sqrt_1m_ab * model_output
        else:
            raise ValueError(f"unknown prediction_type {self.prediction_type!r}")
        clip = self.clip_sample if clip is None else clip
        if clip:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        return pred_x0

    def predict_epsilon(
        self, sample: torch.Tensor, timestep: int | torch.Tensor, pred_original_sample: torch.Tensor
    ) -> torch.Tensor:
        """Epsilon consistent with ``x_t`` and a given (possibly clipped) x0."""
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, timestep, sample)
        sqrt_1m_ab = _extract(self.sqrt_one_minus_alphas_cumprod, timestep, sample)
        return (sample - sqrt_ab * pred_original_sample) / sqrt_1m_ab

    # ------------------------------------------------------------------
    # Sampling interface -- implemented by subclasses
    # ------------------------------------------------------------------
    @property
    def timesteps(self) -> torch.Tensor:
        """Descending timestep sequence set by :meth:`set_timesteps`."""
        if self._timesteps is None:
            raise RuntimeError("call set_timesteps(num_inference_steps) before sampling")
        return self._timesteps

    @abc.abstractmethod
    def set_timesteps(self, num_inference_steps: int) -> None:
        """Prepare the reverse-process timestep sequence."""

    @abc.abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput:
        """Perform one reverse step ``x_t -> x_{t_prev}``."""
