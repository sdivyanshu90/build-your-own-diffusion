"""Diffusion schedulers: the forward process, DDPM and DDIM reverse samplers."""

from __future__ import annotations

from diffusionlab.config import DiffusionConfig
from diffusionlab.schedulers.base import BaseScheduler, SchedulerOutput
from diffusionlab.schedulers.ddim import DDIMScheduler
from diffusionlab.schedulers.ddpm import DDPMScheduler
from diffusionlab.schedulers.schedules import get_beta_schedule

_SCHEDULER_REGISTRY: dict[str, type[BaseScheduler]] = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
}


def build_scheduler(config: DiffusionConfig, sampler: str | None = None) -> BaseScheduler:
    """Instantiate the scheduler described by ``config``.

    Args:
        config: Diffusion process settings.
        sampler: Optional override of ``config.sampler`` (used by the CLI so
            a model trained with one default can be sampled with another --
            the training math is sampler-independent).

    Returns:
        A ready-to-use scheduler.

    Raises:
        ValueError: If the sampler name is unknown.
    """
    name = sampler if sampler is not None else config.sampler
    try:
        cls = _SCHEDULER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown sampler {name!r}; available: {sorted(_SCHEDULER_REGISTRY)}"
        ) from None
    return cls(config)


__all__ = [
    "BaseScheduler",
    "DDIMScheduler",
    "DDPMScheduler",
    "SchedulerOutput",
    "build_scheduler",
    "get_beta_schedule",
]
