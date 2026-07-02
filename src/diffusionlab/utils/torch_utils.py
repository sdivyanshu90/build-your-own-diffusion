"""Torch helpers: reproducible seeding, device resolution, model stats."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python, PyTorch CPU, and all CUDA devices for reproducibility.

    Note that full bit-for-bit determinism on GPU additionally requires
    deterministic kernels (``torch.use_deterministic_algorithms``), which
    we deliberately do not force because several diffusion-relevant ops
    have no deterministic CUDA implementation and it costs throughput.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec (``auto``/``cpu``/``cuda``/``cuda:N``).

    Raises:
        ValueError: If CUDA is requested but not available.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {spec!r} requested but CUDA is not available")
    return device


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Total (or trainable-only) parameter count of a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
