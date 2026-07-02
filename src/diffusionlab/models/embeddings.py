"""Timestep embeddings.

Diffusion models condition the denoiser on the (integer) diffusion timestep.
Following the Transformer positional-encoding recipe used by DDPM, timesteps
are mapped to a vector of sines and cosines at geometrically spaced
frequencies, which downstream MLPs can turn into per-channel scales/shifts.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal embedding of scalar timesteps.

    Args:
        dim: Output embedding dimension (must be even).
        max_period: Longest wavelength in the frequency bank; 10_000 matches
            the Transformer/DDPM convention and comfortably covers the
            0..num_train_timesteps range used in practice.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        # Buffer (not parameter): moves with .to(device) but is never trained.
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            timesteps: Integer or float tensor of shape ``(B,)``.

        Returns:
            Tensor of shape ``(B, dim)``.
        """
        if timesteps.ndim != 1:
            raise ValueError(f"expected timesteps of shape (B,), got {tuple(timesteps.shape)}")
        args = timesteps.float()[:, None] * self.freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
