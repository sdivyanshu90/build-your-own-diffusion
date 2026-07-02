"""Tests for the beta schedules: shape, bounds, monotonicity, precision."""

from __future__ import annotations

import pytest
import torch

from diffusionlab.schedulers.schedules import (
    cosine_beta_schedule,
    get_beta_schedule,
    linear_beta_schedule,
    scaled_linear_beta_schedule,
)


@pytest.mark.parametrize("name", ["linear", "scaled_linear", "cosine"])
@pytest.mark.parametrize("timesteps", [10, 100, 1000])
def test_schedule_shape_bounds_dtype(name: str, timesteps: int) -> None:
    betas = get_beta_schedule(name, timesteps)
    assert betas.shape == (timesteps,)
    assert betas.dtype == torch.float64
    assert (betas > 0).all() and (betas < 1).all()


@pytest.mark.parametrize("name", ["linear", "scaled_linear", "cosine"])
def test_schedule_is_nondecreasing(name: str) -> None:
    betas = get_beta_schedule(name, 1000)
    assert (betas.diff() >= -1e-12).all()


def test_linear_endpoints() -> None:
    betas = linear_beta_schedule(1000, 1e-4, 0.02)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)


def test_scaled_linear_endpoints() -> None:
    betas = scaled_linear_beta_schedule(1000, 1e-4, 0.02)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)


def test_cosine_is_clipped() -> None:
    betas = cosine_beta_schedule(1000)
    assert betas.max().item() <= 0.999


def test_cosine_alpha_bar_matches_definition() -> None:
    """The cumulative product of (1 - beta) must reproduce alpha_bar(t)."""
    import math

    timesteps, s = 100, 0.008
    betas = cosine_beta_schedule(timesteps, s=s)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def alpha_bar(t: float) -> float:
        return (
            math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            / math.cos(s / (1 + s) * math.pi / 2) ** 2
        )

    for t in [0, 10, 50, 98]:  # away from the clipped tail
        assert alphas_cumprod[t].item() == pytest.approx(alpha_bar((t + 1) / timesteps), rel=1e-9)


def test_alphas_cumprod_strictly_decreasing_to_near_zero() -> None:
    betas = get_beta_schedule("cosine", 1000)
    acp = torch.cumprod(1.0 - betas, dim=0)
    assert (acp.diff() < 0).all()
    assert acp[0] > 0.99
    assert acp[-1] < 1e-3


def test_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError, match="unknown beta schedule"):
        get_beta_schedule("sigmoid", 100)
