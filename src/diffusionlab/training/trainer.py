"""Step-based diffusion trainer.

Responsibilities
----------------
- Owns the full training loop: data cycling, loss computation, gradient
  accumulation, optional mixed precision (fp16 with loss scaling / bf16),
  gradient clipping, LR warmup, and EMA updates.
- Emits human logs (console + per-run ``train.log``) and machine-readable
  metrics (``metrics.jsonl``).
- Writes atomic, safely-loadable checkpoints (numbered + rolling
  ``last.pt``) that contain everything needed to resume bit-compatibly:
  model, EMA, optimizer, scaler, step counter, RNG states, and the config
  serialised as plain primitives (so ``torch.load(weights_only=True)``
  works and untrusted checkpoints cannot execute code).
- Renders EMA sample grids during training for qualitative monitoring.

The loop is *step-based* rather than epoch-based because diffusion training
is conventionally budgeted in optimizer steps and datasets differ wildly in
size; an infinite cycling iterator hides the epoch boundary.
"""

from __future__ import annotations

import logging
import math
import os
import random
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusionlab.config import Config
from diffusionlab.data import build_dataloader
from diffusionlab.models import UNet
from diffusionlab.pipeline import DiffusionPipeline, load_checkpoint
from diffusionlab.schedulers import build_scheduler
from diffusionlab.training.ema import ExponentialMovingAverage
from diffusionlab.utils import (
    JsonlMetricsWriter,
    count_parameters,
    get_device,
    save_image_grid,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger("diffusionlab.trainer")


class Trainer:
    """Trains a UNet denoiser according to a validated :class:`Config`.

    Args:
        config: Full experiment configuration (already validated).
        resume_from: Optional checkpoint path. Training state (weights, EMA,
            optimizer, scaler, RNG, step counter) is restored from the
            checkpoint while hyperparameters come from ``config``, so a run
            can be extended by raising ``training.max_steps``.

    Attributes:
        run_dir: ``<output_dir>/<run_name>``; holds ``config.yaml``,
            ``train.log``, ``metrics.jsonl``, ``checkpoints/``, ``samples/``.
        step: Number of completed optimizer steps.
    """

    def __init__(self, config: Config, resume_from: str | Path | None = None) -> None:
        config.validate()
        self.config = config
        self.device = get_device(config.training.device)
        self._validate_precision()

        self.run_dir = Path(config.training.output_dir) / config.run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.sample_dir = self.run_dir / "samples"
        for directory in (self.checkpoint_dir, self.sample_dir):
            directory.mkdir(parents=True, exist_ok=True)

        setup_logging(log_file=self.run_dir / "train.log")
        self.metrics = JsonlMetricsWriter(self.run_dir / "metrics.jsonl")
        config.save_yaml(self.run_dir / "config.yaml")

        seed_everything(config.training.seed)
        self.model = UNet(config.model).to(self.device)
        self.scheduler = build_scheduler(config.diffusion)
        self.ema = ExponentialMovingAverage(self.model, decay=config.optim.ema_decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optim.lr,
            betas=config.optim.betas,
            weight_decay=config.optim.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision == "fp16")
        self.dataloader: DataLoader = build_dataloader(
            config.data, self.device, seed=config.training.seed
        )
        self.step = 0

        if resume_from is not None:
            self._load_state(resume_from)

        logger.info(
            "trainer ready: %s parameters, device=%s, dataset=%s (%d samples), start step=%d",
            f"{count_parameters(self.model):,}",
            self.device,
            config.data.dataset,
            len(self.dataloader.dataset),  # type: ignore[arg-type]
            self.step,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> dict[str, float]:
        """Run the loop until ``training.max_steps`` optimizer steps.

        Returns:
            Final metrics record (``step``, ``loss``, ``lr``, ``grad_norm``).
        """
        cfg = self.config.training
        if self.step >= cfg.max_steps:
            logger.info("nothing to do: step %d >= max_steps %d", self.step, cfg.max_steps)
            return {"step": float(self.step)}

        self.model.train()
        data_iter = self._cycle(self.dataloader)
        last_record: dict[str, float] = {}
        tic = time.monotonic()

        for step in range(self.step, cfg.max_steps):
            stats = self._train_step(data_iter, step)
            self.step = step + 1

            if self.step % cfg.log_interval == 0 or self.step == cfg.max_steps:
                toc = time.monotonic()
                interval_steps = cfg.log_interval if self.step % cfg.log_interval == 0 else 1
                last_record = {
                    "step": self.step,
                    **stats,
                    "seconds_per_step": round((toc - tic) / interval_steps, 4),
                }
                tic = toc
                self.metrics.log(last_record)
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | grad_norm %.3f",
                    self.step,
                    cfg.max_steps,
                    stats["loss"],
                    stats["lr"],
                    stats["grad_norm"],
                )
            if self.step % cfg.sample_interval == 0:
                self._write_sample_grid()
            if self.step % cfg.checkpoint_interval == 0 or self.step == cfg.max_steps:
                self.save_checkpoint()

        logger.info("training complete at step %d", self.step)
        return last_record

    def save_checkpoint(self) -> Path:
        """Write ``checkpoints/step_XXXXXXXX.pt`` atomically; refresh ``last.pt``.

        The file is written to a temp path and moved into place with
        ``os.replace`` so a crash mid-write can never corrupt a checkpoint.
        """
        payload: dict[str, Any] = {
            "format_version": 1,
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config.to_dict(),
            "rng": self._rng_state(),
        }
        path = self.checkpoint_dir / f"step_{self.step:08d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        shutil.copy2(path, self.checkpoint_dir / "last.pt")
        logger.info("checkpoint saved: %s", path)
        return path

    def close(self) -> None:
        """Release the metrics file handle."""
        self.metrics.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate_precision(self) -> None:
        precision = self.config.training.mixed_precision
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("mixed_precision='fp16' requires a CUDA device (use 'bf16' or 'none')")

    def _autocast(self) -> torch.autocast:
        precision = self.config.training.mixed_precision
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": torch.float32}[precision]
        return torch.autocast(
            device_type=self.device.type, dtype=dtype, enabled=precision != "none"
        )

    @staticmethod
    def _cycle(loader: DataLoader) -> Iterator[torch.Tensor]:
        while True:
            yield from loader

    def _lr_for_step(self, step: int) -> float:
        """Linear warmup to ``optim.lr`` over ``optim.warmup_steps`` steps."""
        cfg = self.config.optim
        if cfg.warmup_steps <= 0:
            return cfg.lr
        return cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """Denoising loss: MSE against the target implied by prediction_type."""
        x0 = batch.to(self.device, non_blocking=True)
        noise = torch.randn_like(x0)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (x0.shape[0],), device=self.device
        )
        noisy = self.scheduler.add_noise(x0, noise, timesteps)
        with self._autocast():
            prediction = self.model(noisy, timesteps)
        target = self.scheduler.training_target(x0, noise, timesteps)
        return F.mse_loss(prediction.float(), target.float())

    def _train_step(self, data_iter: Iterator[torch.Tensor], step: int) -> dict[str, float]:
        cfg = self.config
        lr = self._lr_for_step(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)

        accumulation = cfg.training.gradient_accumulation_steps
        loss_sum = 0.0
        for _ in range(accumulation):
            loss = self._compute_loss(next(data_iter)) / accumulation
            self.scaler.scale(loss).backward()
            loss_sum += loss.item() * accumulation
        loss_mean = loss_sum / accumulation
        if not math.isfinite(loss_mean):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss_mean}. Common causes: learning rate "
                f"too high, fp16 overflow (try bf16), or corrupt input data."
            )

        self.scaler.unscale_(self.optimizer)
        max_norm = cfg.optim.grad_clip_norm if cfg.optim.grad_clip_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update(self.model)
        return {"loss": loss_mean, "lr": lr, "grad_norm": float(grad_norm)}

    @torch.no_grad()
    def _write_sample_grid(self) -> None:
        """Render an EMA preview grid from a fixed noise seed.

        The generator is reseeded identically every time so consecutive
        grids show the *same* latent codes denoised by progressively better
        weights -- far easier to judge than fresh random samples.
        """
        cfg = self.config
        self.ema.store(self.model)
        self.ema.copy_to(self.model)
        self.model.eval()
        try:
            pipeline = DiffusionPipeline(
                model=self.model,
                scheduler=build_scheduler(cfg.diffusion),
                image_size=cfg.data.image_size,
                image_channels=cfg.model.in_channels,
                device=self.device,
            )
            generator = torch.Generator(device=self.device.type).manual_seed(cfg.training.seed)
            images = pipeline.sample(cfg.training.num_sample_images, generator=generator)
            path = self.sample_dir / f"step_{self.step:08d}.png"
            save_image_grid(images, path)
            logger.info("sample grid saved: %s", path)
        finally:
            self.model.train()
            self.ema.restore(self.model)

    def _rng_state(self) -> dict[str, Any]:
        version, internal, gauss_next = random.getstate()
        state: dict[str, Any] = {
            "python": [version, list(internal), gauss_next],
            "torch": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_state(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path)
        if checkpoint["config"] != self.config.to_dict():
            logger.warning(
                "resume config differs from checkpoint config; proceeding with the "
                "provided config (weights/optimizer state come from the checkpoint)"
            )
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.ema.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = int(checkpoint["step"])
        rng = checkpoint.get("rng")
        if rng is not None:
            version, internal, gauss_next = rng["python"]
            random.setstate((version, tuple(internal), gauss_next))
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if self.device.type == "cuda" and "cuda" in rng:
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
        logger.info("resumed from %s at step %d", path, self.step)
