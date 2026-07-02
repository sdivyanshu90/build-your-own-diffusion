"""The UNet noise-prediction network.

Architecture (DDPM / ADM style)
-------------------------------
An encoder-decoder with skip connections operating at ``len(channel
multipliers)`` resolutions. Each encoder level applies ``num_res_blocks``
residual blocks (optionally followed by self-attention) and then halves the
spatial resolution; the bottleneck is ResBlock -> Attention -> ResBlock; each
decoder level applies ``num_res_blocks + 1`` residual blocks, consuming one
skip connection per block, and then doubles the resolution. The network is
conditioned on the diffusion timestep through a sinusoidal embedding refined
by a two-layer MLP and injected into every residual block.

The output has the same shape as the input and its interpretation depends on
``diffusion.prediction_type``: the added noise (epsilon), the clean image
(sample), or the velocity (v_prediction).
"""

from __future__ import annotations

import torch
from torch import nn

from diffusionlab.config import ModelConfig
from diffusionlab.models.blocks import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    TimestepEmbedSequential,
    Upsample,
)
from diffusionlab.models.embeddings import SinusoidalPositionalEmbedding


class UNet(nn.Module):
    """Timestep-conditioned UNet for noise/velocity/sample prediction.

    Args:
        config: Validated :class:`~diffusionlab.config.ModelConfig`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        time_dim = base * 4
        num_levels = len(config.channel_multipliers)
        self.downsample_factor = 2 ** (num_levels - 1)

        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.conv_in = nn.Conv2d(config.in_channels, base, kernel_size=3, padding=1)

        def res_block(in_ch: int, out_ch: int) -> ResidualBlock:
            return ResidualBlock(in_ch, out_ch, time_dim, config.dropout, config.num_groups)

        def attn_block(ch: int) -> AttentionBlock:
            return AttentionBlock(ch, config.num_heads, config.num_groups)

        # ---- Encoder ------------------------------------------------------
        # skip_channels mirrors, at build time, the channel count of every
        # feature map the forward pass will push onto its skip stack.
        self.down_blocks = nn.ModuleList()
        skip_channels: list[int] = [base]
        ch = base
        for level, mult in enumerate(config.channel_multipliers):
            out_ch = base * mult
            for _ in range(config.num_res_blocks):
                layers: list[nn.Module] = [res_block(ch, out_ch)]
                ch = out_ch
                if level in config.attention_levels:
                    layers.append(attn_block(ch))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                skip_channels.append(ch)
            if level != num_levels - 1:
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                skip_channels.append(ch)

        # ---- Bottleneck ---------------------------------------------------
        self.middle = TimestepEmbedSequential(
            res_block(ch, ch),
            attn_block(ch),
            res_block(ch, ch),
        )

        # ---- Decoder ------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            out_ch = base * config.channel_multipliers[level]
            for block_index in range(config.num_res_blocks + 1):
                layers = [res_block(ch + skip_channels.pop(), out_ch)]
                ch = out_ch
                if level in config.attention_levels:
                    layers.append(attn_block(ch))
                if level != 0 and block_index == config.num_res_blocks:
                    layers.append(Upsample(ch))
                self.up_blocks.append(TimestepEmbedSequential(*layers))
        assert not skip_channels, "internal error: unconsumed skip connections"

        self.out_norm = nn.GroupNorm(config.num_groups, ch)
        self.out_conv = nn.Conv2d(ch, config.in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict noise/velocity/x0 for noisy images ``x`` at ``timesteps``.

        Args:
            x: Noisy images, shape ``(B, in_channels, H, W)`` with H and W
                divisible by ``2 ** (num_levels - 1)``.
            timesteps: Integer timesteps, shape ``(B,)`` or scalar ``()``.

        Returns:
            Prediction tensor with the same shape as ``x``.
        """
        if x.shape[-1] % self.downsample_factor or x.shape[-2] % self.downsample_factor:
            raise ValueError(
                f"input spatial size {tuple(x.shape[-2:])} must be divisible by "
                f"{self.downsample_factor}"
            )
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x.shape[0])
        emb = self.time_embed(timesteps.to(x.device))

        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, emb)
            skips.append(h)
        h = self.middle(h, emb)
        for block in self.up_blocks:
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
