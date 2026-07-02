"""Tests for embeddings, UNet building blocks, and the assembled UNet."""

from __future__ import annotations

import pytest
import torch

from diffusionlab.config import ModelConfig
from diffusionlab.models import (
    AttentionBlock,
    Downsample,
    ResidualBlock,
    SinusoidalPositionalEmbedding,
    TimestepEmbedSequential,
    UNet,
    Upsample,
)
from diffusionlab.models.blocks import zero_module


def tiny_model_config(**overrides: object) -> ModelConfig:
    defaults: dict[str, object] = {
        "in_channels": 3,
        "base_channels": 16,
        "channel_multipliers": (1, 2),
        "num_res_blocks": 1,
        "attention_levels": (1,),
        "num_heads": 2,
        "dropout": 0.0,
        "num_groups": 4,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Sinusoidal embeddings
# ----------------------------------------------------------------------
def test_embedding_shape_and_range() -> None:
    emb = SinusoidalPositionalEmbedding(32)
    out = emb(torch.arange(10))
    assert out.shape == (10, 32)
    assert out.abs().max().item() <= 1.0


def test_embedding_distinguishes_timesteps() -> None:
    emb = SinusoidalPositionalEmbedding(64)
    out = emb(torch.tensor([0, 1, 500, 999]))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(out[i], out[j])


def test_embedding_t0_is_cos_one_sin_zero() -> None:
    emb = SinusoidalPositionalEmbedding(8)
    out = emb(torch.tensor([0]))
    torch.testing.assert_close(out[0, :4], torch.ones(4))
    torch.testing.assert_close(out[0, 4:], torch.zeros(4))


def test_embedding_rejects_odd_dim_and_bad_shape() -> None:
    with pytest.raises(ValueError, match="even"):
        SinusoidalPositionalEmbedding(7)
    with pytest.raises(ValueError, match="shape"):
        SinusoidalPositionalEmbedding(8)(torch.zeros(2, 2))


# ----------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------
def test_zero_module_zeros_all_parameters() -> None:
    conv = zero_module(torch.nn.Conv2d(4, 4, 3))
    assert all(torch.equal(p, torch.zeros_like(p)) for p in conv.parameters())


def test_residual_block_is_identity_at_init() -> None:
    """conv2 is zero-initialised, so a same-channel block starts as identity."""
    block = ResidualBlock(8, 8, time_dim=16, num_groups=4)
    x = torch.randn(2, 8, 8, 8)
    emb = torch.randn(2, 16)
    torch.testing.assert_close(block(x, emb), x)


def test_residual_block_channel_change_uses_shortcut() -> None:
    block = ResidualBlock(8, 16, time_dim=16, num_groups=4)
    assert isinstance(block.shortcut, torch.nn.Conv2d)
    out = block(torch.randn(2, 8, 8, 8), torch.randn(2, 16))
    assert out.shape == (2, 16, 8, 8)


def test_residual_block_uses_time_embedding() -> None:
    block = ResidualBlock(8, 8, time_dim=16, num_groups=4)
    torch.nn.init.normal_(block.conv2.weight)  # break the zero init
    x = torch.randn(2, 8, 8, 8)
    out_a = block(x, torch.zeros(2, 16))
    out_b = block(x, torch.ones(2, 16))
    assert not torch.allclose(out_a, out_b)


def test_attention_block_is_identity_at_init() -> None:
    block = AttentionBlock(8, num_heads=2, num_groups=4)
    x = torch.randn(2, 8, 4, 4)
    torch.testing.assert_close(block(x), x)


def test_attention_block_shape_and_head_validation() -> None:
    block = AttentionBlock(16, num_heads=4, num_groups=4)
    torch.nn.init.normal_(block.proj.weight)
    out = block(torch.randn(2, 16, 8, 8))
    assert out.shape == (2, 16, 8, 8)
    with pytest.raises(ValueError, match="divisible"):
        AttentionBlock(10, num_heads=4)


def test_down_and_upsample_change_resolution() -> None:
    x = torch.randn(2, 8, 16, 16)
    assert Downsample(8)(x).shape == (2, 8, 8, 8)
    assert Upsample(8)(x).shape == (2, 8, 32, 32)


def test_timestep_embed_sequential_routes_embedding() -> None:
    seq = TimestepEmbedSequential(
        ResidualBlock(8, 8, time_dim=16, num_groups=4),
        Downsample(8),
    )
    out = seq(torch.randn(2, 8, 8, 8), torch.randn(2, 16))
    assert out.shape == (2, 8, 4, 4)


# ----------------------------------------------------------------------
# UNet
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("in_channels", "multipliers", "image_size"),
    [(3, (1, 2), 16), (1, (1, 2, 4), 16), (3, (1,), 8)],
)
def test_unet_output_matches_input_shape(
    in_channels: int, multipliers: tuple[int, ...], image_size: int
) -> None:
    model = UNet(tiny_model_config(in_channels=in_channels, channel_multipliers=multipliers))
    x = torch.randn(2, in_channels, image_size, image_size)
    out = model(x, torch.randint(0, 100, (2,)))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_unet_accepts_scalar_timestep() -> None:
    model = UNet(tiny_model_config())
    x = torch.randn(2, 3, 16, 16)
    out = model(x, torch.tensor(5))
    assert out.shape == x.shape


def test_unet_rejects_indivisible_spatial_size() -> None:
    model = UNet(tiny_model_config(channel_multipliers=(1, 2, 4)))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 10, 10), torch.tensor(0))


def test_unet_rectangular_input() -> None:
    model = UNet(tiny_model_config())
    out = model(torch.randn(1, 3, 16, 32), torch.tensor(3))
    assert out.shape == (1, 3, 16, 32)


def test_unet_all_parameters_participate_in_backward() -> None:
    model = UNet(tiny_model_config())
    out = model(torch.randn(2, 3, 16, 16), torch.randint(0, 100, (2,)))
    out.square().mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradients: {missing}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_unet_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(0)
    model = UNet(tiny_model_config(dropout=0.5)).eval()
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([1, 2])
    torch.testing.assert_close(model(x, t), model(x, t))


def test_unet_same_seed_same_initialisation() -> None:
    torch.manual_seed(7)
    a = UNet(tiny_model_config())
    torch.manual_seed(7)
    b = UNet(tiny_model_config())
    for (name_a, pa), (name_b, pb) in zip(a.named_parameters(), b.named_parameters(), strict=False):
        assert name_a == name_b
        assert torch.equal(pa, pb)


def test_unet_output_depends_on_timestep() -> None:
    """At init the zero-initialised residual branches mute the conditioning
    (by design), so perturb them before checking timestep sensitivity."""
    model = UNet(tiny_model_config()).eval()
    for module in model.modules():
        if isinstance(module, ResidualBlock):
            torch.nn.init.normal_(module.conv2.weight, std=0.1)
    x = torch.randn(1, 3, 16, 16)
    out_a = model(x, torch.tensor([0]))
    out_b = model(x, torch.tensor([40]))
    assert not torch.allclose(out_a, out_b)


def test_unet_can_overfit_epsilon_prediction() -> None:
    """A behavioural sanity check: a few Adam steps on a fixed batch must
    reduce the denoising loss (catches broken gradient flow / conditioning)."""
    from diffusionlab.config import DiffusionConfig
    from diffusionlab.schedulers import build_scheduler

    torch.manual_seed(0)
    model = UNet(tiny_model_config())
    scheduler = build_scheduler(DiffusionConfig(num_train_timesteps=50, beta_schedule="cosine"))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    x0 = torch.rand(8, 3, 16, 16) * 2 - 1

    losses = []
    for _ in range(30):
        noise = torch.randn_like(x0)
        t = torch.randint(0, 50, (8,))
        pred = model(scheduler.add_noise(x0, noise, t), t)
        loss = torch.nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
