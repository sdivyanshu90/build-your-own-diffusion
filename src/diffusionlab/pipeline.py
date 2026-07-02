"""High-level sampling pipeline: checkpoint -> images.

:class:`DiffusionPipeline` owns the reverse-diffusion loop shared by the CLI
``sample`` command and the in-training preview grids. Checkpoints are loaded
with ``torch.load(weights_only=True)``: because the trainer serialises the
config as plain primitives (never pickled Python objects), loading an
untrusted checkpoint cannot execute arbitrary code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from diffusionlab.config import Config
from diffusionlab.models import UNet
from diffusionlab.schedulers import BaseScheduler, build_scheduler
from diffusionlab.utils import get_device

logger = logging.getLogger("diffusionlab.pipeline")


class DiffusionPipeline:
    """Bundle of a denoising model and a scheduler that generates images.

    Args:
        model: Trained (or EMA) UNet; will be moved to ``device`` and set to
            eval mode.
        scheduler: Any :class:`~diffusionlab.schedulers.BaseScheduler`.
        image_size: Side length of generated square images.
        image_channels: Channel count of generated images.
        device: Target device (defaults to CUDA when available).
    """

    def __init__(
        self,
        model: UNet,
        scheduler: BaseScheduler,
        image_size: int,
        image_channels: int,
        device: torch.device | str = "auto",
    ) -> None:
        self.device = get_device(str(device)) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self.scheduler = scheduler
        self.image_size = image_size
        self.image_channels = image_channels

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "auto",
        use_ema: bool = True,
        sampler: str | None = None,
    ) -> DiffusionPipeline:
        """Reconstruct a pipeline from a trainer checkpoint.

        Args:
            path: ``.pt`` file written by :class:`~diffusionlab.training.Trainer`.
            device: Device spec (``auto``/``cpu``/``cuda``/``cuda:N``).
            use_ema: Load the EMA weights (recommended: they sample markedly
                better than the raw online weights) when present.
            sampler: Optional sampler override (``ddpm``/``ddim``).
        """
        checkpoint = load_checkpoint(path)
        config = Config.from_dict(checkpoint["config"])
        model = UNet(config.model)
        if use_ema and "ema" in checkpoint:
            model.load_state_dict(checkpoint["ema"]["shadow"])
            logger.info("loaded EMA weights from %s (step %s)", path, checkpoint.get("step"))
        else:
            model.load_state_dict(checkpoint["model"])
            logger.info("loaded online weights from %s (step %s)", path, checkpoint.get("step"))
        scheduler = build_scheduler(config.diffusion, sampler=sampler)
        return cls(
            model=model,
            scheduler=scheduler,
            image_size=config.data.image_size,
            image_channels=config.model.in_channels,
            device=device,
        )

    @torch.no_grad()
    def sample(
        self,
        num_images: int,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Generate images by running the reverse diffusion process.

        Args:
            num_images: Batch size of the generation.
            num_inference_steps: Reverse steps; defaults to the scheduler
                config's ``num_inference_steps``.
            generator: Optional RNG for reproducible generation. Must live on
                the pipeline device.
            progress: Show a tqdm progress bar over reverse steps.

        Returns:
            Tensor of shape ``(num_images, C, H, W)`` in model space
            ``[-1, 1]`` on the pipeline device.
        """
        if num_images < 1:
            raise ValueError(f"num_images must be >= 1, got {num_images}")
        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.scheduler.config.num_inference_steps
        )
        self.scheduler.set_timesteps(steps)

        shape = (num_images, self.image_channels, self.image_size, self.image_size)
        x = torch.randn(shape, generator=generator, device=self.device)
        iterator: Any = self.scheduler.timesteps.tolist()
        if progress:
            iterator = tqdm(iterator, desc="sampling", leave=False)
        for t in iterator:
            model_output = self.model(x, torch.tensor(t, device=self.device))
            x = self.scheduler.step(model_output, t, x, generator=generator).prev_sample
        return x.clamp(-1.0, 1.0)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a trainer checkpoint safely (``weights_only=True``).

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError-free ValueError: If required keys are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"model", "config"} - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return checkpoint
