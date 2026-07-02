"""UNet building blocks: residual blocks, attention, and resampling layers.

Design choices (all standard for DDPM-family UNets):

- **GroupNorm + SiLU** everywhere: GroupNorm is batch-size independent
  (important with small per-GPU batches) and SiLU is the DDPM default.
- **Zero-initialised residual branches**: the last convolution of every
  residual block and the attention output projection start at zero, so each
  block is initially the identity. Deep UNets then start training from a
  well-conditioned near-identity function, which measurably stabilises
  early optimisation (Goyal et al.; used by both DDPM and ADM).
- **Additive timestep conditioning**: the time embedding is projected and
  added between the two convolutions of each residual block.
"""

from __future__ import annotations

import abc

import torch
import torch.nn.functional as F
from torch import nn


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters of ``module`` in place and return it."""
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class TimestepBlock(nn.Module, abc.ABC):
    """A module whose forward pass also consumes the timestep embedding."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` conditioned on embedding ``emb``."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential container that routes the timestep embedding.

    Children that are :class:`TimestepBlock` receive ``(x, emb)``; all other
    children (attention, resampling) receive only ``x``. This lets the UNet
    keep a flat, order-preserving module list without per-layer plumbing.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResidualBlock(TimestepBlock):
    """Pre-activation residual block with additive timestep conditioning.

    Structure::

        x ──► GN ─► SiLU ─► Conv3x3 ─► (+ time proj) ─► GN ─► SiLU ─► Drop ─► Conv3x3(zero) ──►(+)
        └────────────────────────── identity / Conv1x1 ────────────────────────────────────────┘

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count (a 1x1 shortcut is added if it
            differs from ``in_channels``).
        time_dim: Dimension of the shared timestep embedding.
        dropout: Dropout probability before the second convolution.
        num_groups: GroupNorm group count (must divide both channel counts).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.shortcut(x) + h


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions with residual output.

    Uses ``F.scaled_dot_product_attention`` so PyTorch can dispatch to
    memory-efficient/flash kernels where available.

    Args:
        channels: Feature channels (must be divisible by ``num_heads``).
        num_heads: Attention heads.
        num_groups: GroupNorm group count.
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        # (B, 3C, H, W) -> three tensors of shape (B, heads, H*W, C/heads)
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.transpose(-1, -2).unbind(dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """2x spatial downsampling via a stride-2 3x3 convolution (learned)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling: nearest-neighbour resize followed by a 3x3 conv.

    Resize-then-convolve avoids the checkerboard artefacts of transposed
    convolutions (Odena et al., 2016).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
