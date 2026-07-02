"""DDIM sampler (Song, Meng & Ermon, 2021).

DDIM defines a *non-Markovian* family of reverse processes that share the
DDPM training objective, which is why a model trained once can be sampled
with far fewer steps. For a timestep pair ``(t, t_prev)`` from a subsequence
of the training chain:

``x_{t_prev} = sqrt(ab_prev) x0_hat
             + sqrt(1 - ab_prev - sigma_t^2) eps_hat
             + sigma_t z``,  ``z ~ N(0, I)``

with ``sigma_t = eta * sqrt((1-ab_prev)/(1-ab_t)) * sqrt(1 - ab_t/ab_prev)``.

- ``eta = 0`` gives the deterministic DDIM ODE-like sampler.
- ``eta = 1`` with the full chain recovers exactly the DDPM posterior
  variance (verified in the test suite).
"""

from __future__ import annotations

import torch

from diffusionlab.config import DiffusionConfig
from diffusionlab.schedulers.base import BaseScheduler, SchedulerOutput


class DDIMScheduler(BaseScheduler):
    """Deterministic-to-stochastic accelerated sampler."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__(config)
        self.eta = config.ddim_eta
        self.timestep_spacing = config.timestep_spacing
        self._prev_timestep: dict[int, int] = {}

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Choose a descending subsequence of the training chain.

        ``leading`` spacing (``0, k, 2k, ...`` reversed) matches the original
        DDIM implementation; ``trailing`` (``T-1, T-1-k, ...``) always starts
        at the final training timestep, which empirically helps at very low
        step counts (Lin et al., 2024).

        Raises:
            ValueError: If the step count is outside ``[1, T]``.
        """
        T = self.num_train_timesteps
        if not 1 <= num_inference_steps <= T:
            raise ValueError(f"num_inference_steps must be in [1, {T}], got {num_inference_steps}")
        if self.timestep_spacing == "leading":
            step_ratio = T // num_inference_steps
            timesteps = (torch.arange(num_inference_steps, dtype=torch.long) * step_ratio).flip(0)
        else:  # trailing
            timesteps = (torch.arange(T, 0, -T / num_inference_steps).round().long() - 1).clamp(
                min=0
            )
        self._timesteps = timesteps
        sequence = timesteps.tolist()
        # Map each timestep to its successor in the reverse walk (-1 = done).
        self._prev_timestep = {
            t: (sequence[i + 1] if i + 1 < len(sequence) else -1) for i, t in enumerate(sequence)
        }

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput:
        """One DDIM step from ``timestep`` to its predecessor in the sequence.

        Args:
            model_output: Network output at ``(sample, timestep)``.
            timestep: Current timestep; must belong to the sequence prepared
                by :meth:`set_timesteps`.
            sample: Current noisy images ``x_t``.
            generator: Optional RNG (only used when ``eta > 0``).
        """
        t = int(timestep)
        if t not in self._prev_timestep:
            raise ValueError(
                f"timestep {t} is not part of the prepared sequence; call set_timesteps first"
            )
        prev_t = self._prev_timestep[t]

        alpha_bar_t = self.alphas_cumprod[t].to(sample.device, sample.dtype)
        alpha_bar_prev = (
            self.alphas_cumprod[prev_t].to(sample.device, sample.dtype)
            if prev_t >= 0
            else torch.ones((), device=sample.device, dtype=sample.dtype)
        )

        pred_x0 = self.predict_original_sample(model_output, t, sample)
        # Re-derive epsilon from the (possibly clipped) x0 so that clipping
        # keeps x_t, x0 and epsilon mutually consistent.
        pred_eps = self.predict_epsilon(sample, t, pred_x0)

        variance = (
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
        )
        sigma = self.eta * variance.sqrt()

        direction = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0).sqrt() * pred_eps
        prev_sample = alpha_bar_prev.sqrt() * pred_x0 + direction
        if self.eta > 0 and prev_t >= 0:
            noise = torch.randn(
                sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
            )
            prev_sample = prev_sample + sigma * noise
        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
