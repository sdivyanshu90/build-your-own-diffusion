"""diffusionlab: a from-scratch, production-quality diffusion model library.

The package implements the two core components of a denoising diffusion
probabilistic model and everything needed to train and use one:

- :mod:`diffusionlab.models` -- the UNet noise-prediction network.
- :mod:`diffusionlab.schedulers` -- the forward/reverse diffusion processes
  (DDPM ancestral sampling and DDIM accelerated sampling).
- :mod:`diffusionlab.training` -- a step-based trainer with EMA, AMP,
  gradient accumulation, checkpointing, and resumable runs.
- :mod:`diffusionlab.pipeline` -- a high-level sampling pipeline that turns a
  trained checkpoint into images.
- :mod:`diffusionlab.config` -- typed, validated, YAML-backed configuration.
- :mod:`diffusionlab.cli` -- the ``diffusionlab`` command-line interface.
"""

from diffusionlab.config import (
    Config,
    ConfigError,
    DataConfig,
    DiffusionConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
)
from diffusionlab.models import UNet
from diffusionlab.pipeline import DiffusionPipeline
from diffusionlab.schedulers import (
    BaseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    SchedulerOutput,
    build_scheduler,
)
from diffusionlab.training import ExponentialMovingAverage, Trainer

__version__ = "1.0.0"

__all__ = [
    "BaseScheduler",
    "Config",
    "ConfigError",
    "DDIMScheduler",
    "DDPMScheduler",
    "DataConfig",
    "DiffusionConfig",
    "DiffusionPipeline",
    "ExponentialMovingAverage",
    "ModelConfig",
    "OptimConfig",
    "SchedulerOutput",
    "Trainer",
    "TrainingConfig",
    "UNet",
    "__version__",
    "build_scheduler",
]
