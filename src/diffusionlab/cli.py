"""Command-line interface.

Three subcommands::

    diffusionlab train --config configs/cifar10.yaml [--set k=v ...] [--resume ckpt]
    diffusionlab sample --checkpoint runs/x/checkpoints/last.pt --output out.png
    diffusionlab validate-config --config configs/cifar10.yaml

``main`` returns an exit code instead of calling ``sys.exit`` so it is
directly unit-testable; ``entrypoint`` is the console-script shim.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import torch

from diffusionlab import __version__
from diffusionlab.config import Config, ConfigError
from diffusionlab.pipeline import DiffusionPipeline
from diffusionlab.training import Trainer
from diffusionlab.utils import save_image_grid, setup_logging

logger = logging.getLogger("diffusionlab.cli")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for docs generation/tests)."""
    parser = argparse.ArgumentParser(
        prog="diffusionlab",
        description="Train and sample denoising diffusion models (UNet + DDPM/DDIM).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a model from a YAML config")
    train.add_argument("--config", required=True, help="path to a YAML config file")
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set training.max_steps=1000 (repeatable)",
    )
    train.add_argument("--resume", default=None, help="checkpoint to resume from")

    sample = sub.add_parser("sample", help="generate images from a trained checkpoint")
    sample.add_argument("--checkpoint", required=True, help="path to a trainer checkpoint (.pt)")
    sample.add_argument("--output", default="samples.png", help="output PNG path")
    sample.add_argument("--num-images", type=int, default=16, help="number of images")
    sample.add_argument("--steps", type=int, default=None, help="inference steps override")
    sample.add_argument(
        "--sampler", choices=["ddpm", "ddim"], default=None, help="sampler override"
    )
    sample.add_argument("--seed", type=int, default=None, help="seed for reproducible output")
    sample.add_argument("--grid-cols", type=int, default=4, help="images per grid row")
    sample.add_argument("--device", default="auto", help="device: auto, cpu, cuda, or cuda:N")
    sample.add_argument(
        "--no-ema",
        action="store_true",
        help="sample the raw online weights instead of the EMA weights",
    )

    validate = sub.add_parser("validate-config", help="validate a YAML config and exit")
    validate.add_argument("--config", required=True, help="path to a YAML config file")
    validate.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    config = Config.from_yaml(args.config, overrides=args.overrides)
    trainer = Trainer(config, resume_from=args.resume)
    try:
        trainer.train()
    finally:
        trainer.close()
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    if args.num_images < 1:
        raise ConfigError(f"--num-images must be >= 1, got {args.num_images}")
    pipeline = DiffusionPipeline.from_checkpoint(
        args.checkpoint, device=args.device, use_ema=not args.no_ema, sampler=args.sampler
    )
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipeline.device.type).manual_seed(args.seed)
    images = pipeline.sample(
        args.num_images, num_inference_steps=args.steps, generator=generator, progress=True
    )
    path = save_image_grid(images, args.output, nrow=args.grid_cols)
    logger.info("wrote %d images to %s", args.num_images, path)
    return 0


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = Config.from_yaml(args.config, overrides=args.overrides)
    logger.info("config OK: run_name=%s", config.run_name)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch; returns a process exit code."""
    setup_logging()
    args = build_parser().parse_args(argv)
    handlers = {
        "train": _cmd_train,
        "sample": _cmd_sample,
        "validate-config": _cmd_validate_config,
    }
    try:
        return handlers[args.command](args)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130


def entrypoint() -> None:
    """Console-script entry point (``diffusionlab`` command)."""
    sys.exit(main())
