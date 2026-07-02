"""Beta (noise) schedules.

All schedules are computed in float64 -- the cumulative products that derive
from the betas lose visible precision in float32 for T = 1000 -- and the
consumer (:class:`~diffusionlab.schedulers.base.BaseScheduler`) casts the
derived quantities down to float32 once, at construction time.

Schedules
---------
``linear``
    ``linspace(beta_start, beta_end, T)`` -- the original DDPM schedule
    (Ho et al., 2020) with its published endpoints as defaults.
``scaled_linear``
    Linear in sqrt(beta): ``linspace(sqrt(beta_start), sqrt(beta_end), T)**2``.
    Used by Stable Diffusion / latent-space models.
``cosine``
    The Nichol & Dhariwal (2021) schedule, defined through the cumulative
    signal ``alpha_bar(t) = cos^2(((t/T)+s)/(1+s) * pi/2)`` and clipped at
    ``beta <= 0.999`` to avoid singularities at the end of the chain. It
    destroys information more gradually than linear and typically improves
    log-likelihood and sample quality on small images.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def linear_beta_schedule(
    num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear schedule from Ho et al. (2020)."""
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)


def scaled_linear_beta_schedule(
    num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Stable-Diffusion-style schedule, linear in the square root of beta."""
    return (
        torch.linspace(
            math.sqrt(beta_start), math.sqrt(beta_end), num_timesteps, dtype=torch.float64
        )
        ** 2
    )


def cosine_beta_schedule(
    num_timesteps: int, s: float = 0.008, max_beta: float = 0.999
) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021).

    Args:
        num_timesteps: Chain length T.
        s: Small offset preventing beta from vanishing at t=0.
        max_beta: Clip value keeping the final betas away from 1.
    """
    steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
    t = steps / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(min=0.0, max=max_beta)


_SCHEDULE_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "linear": linear_beta_schedule,
    "scaled_linear": scaled_linear_beta_schedule,
    "cosine": cosine_beta_schedule,
}


def get_beta_schedule(
    name: str, num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Return the float64 beta schedule ``name`` of length ``num_timesteps``.

    ``beta_start``/``beta_end`` only apply to the (scaled_)linear schedules;
    the cosine schedule is fully determined by its offset ``s``.

    Raises:
        ValueError: If the schedule name is unknown.
    """
    if name not in _SCHEDULE_REGISTRY:
        raise ValueError(f"unknown beta schedule {name!r}; available: {sorted(_SCHEDULE_REGISTRY)}")
    if name == "cosine":
        return cosine_beta_schedule(num_timesteps)
    return _SCHEDULE_REGISTRY[name](num_timesteps, beta_start, beta_end)
