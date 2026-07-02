"""DDPM ancestral sampler (Ho et al., 2020).

The reverse step samples from the closed-form Gaussian posterior

``q(x_{t-1} | x_t, x_0) = N(mu_t(x_t, x_0), sigma_t^2 I)`` with

- ``mu_t = coef1_t * x_0 + coef2_t * x_t``
- ``coef1_t = beta_t sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)``
- ``coef2_t = (1 - alpha_bar_{t-1}) sqrt(alpha_t) / (1 - alpha_bar_t)``
- ``sigma_t^2 = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)``
  (``fixed_small``) or ``beta_t`` (``fixed_large``),

where ``x_0`` is replaced by the model's estimate. DDPM steps are only
valid between *adjacent* timesteps, so this sampler always walks the full
chain ``T-1 .. 0``; for accelerated sampling use
:class:`~diffusionlab.schedulers.ddim.DDIMScheduler`.
"""

from __future__ import annotations

import torch

from diffusionlab.config import DiffusionConfig
from diffusionlab.schedulers.base import BaseScheduler, SchedulerOutput, _extract


class DDPMScheduler(BaseScheduler):
    """Stochastic ancestral sampler over the full training chain."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__(config)
        self.variance_type = config.variance_type

        alphas_cumprod64 = self.alphas_cumprod.double()
        betas64 = self.betas.double()
        alphas64 = self.alphas.double()
        alphas_cumprod_prev64 = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod64[:-1]]
        )

        posterior_variance64 = betas64 * (1.0 - alphas_cumprod_prev64) / (1.0 - alphas_cumprod64)
        self.alphas_cumprod_prev = alphas_cumprod_prev64.float()
        self.posterior_variance = posterior_variance64.float()
        # The t=0 posterior variance is 0; clamp before log so fixed_small's
        # log-variance is finite everywhere (t=0 adds no noise regardless).
        self.posterior_log_variance_clipped = (
            posterior_variance64.clamp(min=posterior_variance64[1]).log().float()
        )
        self.posterior_mean_coef1 = (
            betas64 * alphas_cumprod_prev64.sqrt() / (1.0 - alphas_cumprod64)
        ).float()
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev64) * alphas64.sqrt() / (1.0 - alphas_cumprod64)
        ).float()

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Prepare the full descending chain; DDPM cannot skip steps.

        Raises:
            ValueError: If ``num_inference_steps != num_train_timesteps``.
        """
        if num_inference_steps != self.num_train_timesteps:
            raise ValueError(
                f"DDPM requires num_inference_steps == num_train_timesteps "
                f"({self.num_train_timesteps}), got {num_inference_steps}; "
                f"use the DDIM sampler for accelerated sampling"
            )
        self._timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput:
        """One ancestral step ``x_t -> x_{t-1}``.

        Args:
            model_output: Network output at ``(sample, timestep)``.
            timestep: Current timestep ``t`` (python int).
            sample: Current noisy images ``x_t``.
            generator: Optional RNG for reproducible sampling.
        """
        t = int(timestep)
        pred_x0 = self.predict_original_sample(model_output, t, sample)
        mean = (
            _extract(self.posterior_mean_coef1, t, sample) * pred_x0
            + _extract(self.posterior_mean_coef2, t, sample) * sample
        )
        if t == 0:
            return SchedulerOutput(prev_sample=mean, pred_original_sample=pred_x0)

        if self.variance_type == "fixed_small":
            log_variance = _extract(self.posterior_log_variance_clipped, t, sample)
        else:  # fixed_large
            log_variance = _extract(self.betas, t, sample).log()
        noise = torch.randn(
            sample.shape, generator=generator, device=sample.device, dtype=sample.dtype
        )
        prev_sample = mean + (0.5 * log_variance).exp() * noise
        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)
