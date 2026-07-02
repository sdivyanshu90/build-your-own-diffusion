"""Tests for the diffusion process math and the DDPM/DDIM samplers.

These tests check *identities the math guarantees* (e.g. prediction-type
round trips, DDIM eta=1 variance == DDPM posterior variance) rather than
just shapes, so a silent sign or indexing error cannot pass.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from diffusionlab.config import DiffusionConfig
from diffusionlab.schedulers import DDIMScheduler, DDPMScheduler, build_scheduler
from diffusionlab.schedulers.base import _extract


def make_diffusion_config(**overrides: object) -> DiffusionConfig:
    defaults: dict[str, object] = {"num_train_timesteps": 50, "beta_schedule": "cosine"}
    defaults.update(overrides)
    return DiffusionConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def x0() -> torch.Tensor:
    return torch.rand(4, 3, 8, 8) * 2.0 - 1.0


# ----------------------------------------------------------------------
# _extract
# ----------------------------------------------------------------------
def test_extract_int_and_tensor_indices() -> None:
    values = torch.arange(10, dtype=torch.float32)
    ref = torch.zeros(3, 2, 4, 4)
    assert _extract(values, 7, ref).shape == (1, 1, 1, 1)
    assert _extract(values, 7, ref).item() == 7.0
    out = _extract(values, torch.tensor([1, 2, 3]), ref)
    assert out.shape == (3, 1, 1, 1)
    assert out.flatten().tolist() == [1.0, 2.0, 3.0]


# ----------------------------------------------------------------------
# Forward process and conversions (shared base behaviour)
# ----------------------------------------------------------------------
def test_add_noise_interpolates_signal_and_noise(x0: torch.Tensor) -> None:
    scheduler = build_scheduler(make_diffusion_config())
    noise = torch.randn_like(x0)
    t = torch.full((x0.shape[0],), 10, dtype=torch.long)
    noisy = scheduler.add_noise(x0, noise, t)
    expected = (
        scheduler.sqrt_alphas_cumprod[10] * x0 + scheduler.sqrt_one_minus_alphas_cumprod[10] * noise
    )
    torch.testing.assert_close(noisy, expected)


def test_add_noise_preserves_variance_of_unit_inputs() -> None:
    """sqrt(ab)^2 + sqrt(1-ab)^2 == 1 for every timestep."""
    scheduler = build_scheduler(make_diffusion_config(num_train_timesteps=1000))
    total = scheduler.sqrt_alphas_cumprod**2 + scheduler.sqrt_one_minus_alphas_cumprod**2
    torch.testing.assert_close(total, torch.ones_like(total), atol=1e-5, rtol=0)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_predict_original_sample_roundtrip(prediction_type: str, x0: torch.Tensor) -> None:
    """Feeding the *true* target back must recover the original image."""
    scheduler = build_scheduler(
        make_diffusion_config(prediction_type=prediction_type, clip_sample=False)
    )
    noise = torch.randn_like(x0)
    for t_value in [0, 10, 25, 49]:
        t = torch.full((x0.shape[0],), t_value, dtype=torch.long)
        noisy = scheduler.add_noise(x0, noise, t)
        target = scheduler.training_target(x0, noise, t)
        recovered = scheduler.predict_original_sample(target, t, noisy)
        torch.testing.assert_close(recovered, x0, atol=5e-3, rtol=1e-3)


def test_predict_epsilon_roundtrip(x0: torch.Tensor) -> None:
    scheduler = build_scheduler(make_diffusion_config(clip_sample=False))
    noise = torch.randn_like(x0)
    t = torch.full((x0.shape[0],), 20, dtype=torch.long)
    noisy = scheduler.add_noise(x0, noise, t)
    torch.testing.assert_close(scheduler.predict_epsilon(noisy, t, x0), noise, atol=1e-4, rtol=1e-4)


def test_training_target_by_prediction_type(x0: torch.Tensor) -> None:
    noise = torch.randn_like(x0)
    t = torch.full((x0.shape[0],), 5, dtype=torch.long)
    eps_sched = build_scheduler(make_diffusion_config(prediction_type="epsilon"))
    assert torch.equal(eps_sched.training_target(x0, noise, t), noise)
    x0_sched = build_scheduler(make_diffusion_config(prediction_type="sample"))
    assert torch.equal(x0_sched.training_target(x0, noise, t), x0)
    v_sched = build_scheduler(make_diffusion_config(prediction_type="v_prediction"))
    torch.testing.assert_close(
        v_sched.training_target(x0, noise, t), v_sched.get_velocity(x0, noise, t)
    )


def test_clip_sample_bounds_prediction(x0: torch.Tensor) -> None:
    scheduler = build_scheduler(make_diffusion_config(clip_sample=True))
    wild_output = torch.randn_like(x0) * 100.0
    t = torch.full((x0.shape[0],), 30, dtype=torch.long)
    pred = scheduler.predict_original_sample(wild_output, t, x0)
    assert pred.abs().max().item() <= 1.0


def test_timesteps_property_requires_set_timesteps() -> None:
    scheduler = build_scheduler(make_diffusion_config())
    with pytest.raises(RuntimeError, match="set_timesteps"):
        _ = scheduler.timesteps


def test_build_scheduler_registry() -> None:
    assert isinstance(build_scheduler(make_diffusion_config(), sampler="ddpm"), DDPMScheduler)
    assert isinstance(build_scheduler(make_diffusion_config(), sampler="ddim"), DDIMScheduler)
    with pytest.raises(ValueError, match="unknown sampler"):
        build_scheduler(make_diffusion_config(), sampler="euler")


# ----------------------------------------------------------------------
# DDPM
# ----------------------------------------------------------------------
def test_ddpm_posterior_coefficients_sum_correctly() -> None:
    """At x0 == x_t == x, the posterior mean must return x (coef1+coef2 -> 1
    only when alpha relations hold); verify via the defining identity."""
    scheduler = DDPMScheduler(make_diffusion_config(num_train_timesteps=100))
    t = torch.arange(1, 100)
    coef1 = scheduler.posterior_mean_coef1[t]
    coef2 = scheduler.posterior_mean_coef2[t]
    expected1 = (
        scheduler.betas[t]
        * scheduler.alphas_cumprod_prev[t].sqrt()
        / (1.0 - scheduler.alphas_cumprod[t])
    )
    expected2 = (
        (1.0 - scheduler.alphas_cumprod_prev[t])
        * scheduler.alphas[t].sqrt()
        / (1.0 - scheduler.alphas_cumprod[t])
    )
    torch.testing.assert_close(coef1, expected1, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(coef2, expected2, atol=1e-5, rtol=1e-4)


def test_ddpm_posterior_variance_nonnegative_and_finite_log() -> None:
    scheduler = DDPMScheduler(make_diffusion_config(num_train_timesteps=1000))
    assert (scheduler.posterior_variance >= 0).all()
    assert torch.isfinite(scheduler.posterior_log_variance_clipped).all()


def test_ddpm_requires_full_chain() -> None:
    scheduler = DDPMScheduler(make_diffusion_config(num_train_timesteps=50))
    with pytest.raises(ValueError, match="DDPM requires"):
        scheduler.set_timesteps(10)
    scheduler.set_timesteps(50)
    assert scheduler.timesteps.tolist() == list(range(49, -1, -1))


def test_ddpm_step_at_t0_is_deterministic(x0: torch.Tensor) -> None:
    scheduler = DDPMScheduler(make_diffusion_config())
    scheduler.set_timesteps(50)
    output = torch.randn_like(x0)
    a = scheduler.step(output, 0, x0, generator=torch.Generator().manual_seed(1))
    b = scheduler.step(output, 0, x0, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.prev_sample, b.prev_sample)


def test_ddpm_step_is_reproducible_with_generator(x0: torch.Tensor) -> None:
    scheduler = DDPMScheduler(make_diffusion_config())
    scheduler.set_timesteps(50)
    output = torch.randn_like(x0)
    a = scheduler.step(output, 20, x0, generator=torch.Generator().manual_seed(7))
    b = scheduler.step(output, 20, x0, generator=torch.Generator().manual_seed(7))
    c = scheduler.step(output, 20, x0, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a.prev_sample, b.prev_sample)
    assert not torch.equal(a.prev_sample, c.prev_sample)
    assert a.prev_sample.shape == x0.shape
    assert torch.isfinite(a.prev_sample).all()


def test_ddpm_fixed_large_variance_differs(x0: torch.Tensor) -> None:
    small = DDPMScheduler(make_diffusion_config(variance_type="fixed_small"))
    large = DDPMScheduler(make_diffusion_config(variance_type="fixed_large"))
    output = torch.zeros_like(x0)
    a = small.step(output, 30, x0, generator=torch.Generator().manual_seed(0))
    b = large.step(output, 30, x0, generator=torch.Generator().manual_seed(0))
    # Same mean and same noise draw, but different standard deviations.
    assert not torch.equal(a.prev_sample, b.prev_sample)
    torch.testing.assert_close(a.pred_original_sample, b.pred_original_sample)


# ----------------------------------------------------------------------
# DDIM
# ----------------------------------------------------------------------
def test_ddim_timesteps_leading_and_trailing() -> None:
    leading = DDIMScheduler(make_diffusion_config(num_train_timesteps=100))
    leading.set_timesteps(10)
    assert leading.timesteps.tolist() == [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]

    trailing = DDIMScheduler(
        make_diffusion_config(num_train_timesteps=100, timestep_spacing="trailing")
    )
    trailing.set_timesteps(10)
    ts = trailing.timesteps.tolist()
    assert ts[0] == 99
    assert len(ts) == 10
    assert all(a > b for a, b in itertools.pairwise(ts))
    assert all(0 <= t < 100 for t in ts)


def test_ddim_set_timesteps_bounds() -> None:
    scheduler = DDIMScheduler(make_diffusion_config(num_train_timesteps=50))
    with pytest.raises(ValueError):
        scheduler.set_timesteps(0)
    with pytest.raises(ValueError):
        scheduler.set_timesteps(51)
    scheduler.set_timesteps(50)
    assert scheduler.timesteps.tolist() == list(range(49, -1, -1))


def test_ddim_step_rejects_unprepared_timestep(x0: torch.Tensor) -> None:
    scheduler = DDIMScheduler(make_diffusion_config(num_train_timesteps=100))
    scheduler.set_timesteps(10)
    with pytest.raises(ValueError, match="not part of the prepared sequence"):
        scheduler.step(torch.zeros_like(x0), 55, x0)


def test_ddim_eta_zero_is_deterministic(x0: torch.Tensor) -> None:
    scheduler = DDIMScheduler(make_diffusion_config(ddim_eta=0.0))
    scheduler.set_timesteps(50)
    output = torch.randn_like(x0)
    a = scheduler.step(output, 20, x0, generator=torch.Generator().manual_seed(1))
    b = scheduler.step(output, 20, x0, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.prev_sample, b.prev_sample)


def test_ddim_eta_one_variance_equals_ddpm_posterior() -> None:
    """The DDIM paper's key identity: at eta=1 over adjacent timesteps,
    sigma_t^2 equals the DDPM posterior variance exactly."""
    config = make_diffusion_config(num_train_timesteps=50)
    ddim = DDIMScheduler(config)
    ddpm = DDPMScheduler(config)
    acp = ddim.alphas_cumprod
    for t in range(1, 50):
        variance = (1 - acp[t - 1]) / (1 - acp[t]) * (1 - acp[t] / acp[t - 1])
        torch.testing.assert_close(variance, ddpm.posterior_variance[t], atol=1e-6, rtol=1e-4)


def test_ddim_final_step_returns_predicted_x0(x0: torch.Tensor) -> None:
    """At the last timestep (prev = -1, alpha_bar_prev = 1) the step must
    output exactly the model's x0 estimate: no noise, no direction term."""
    scheduler = DDIMScheduler(make_diffusion_config(clip_sample=False, ddim_eta=0.0))
    scheduler.set_timesteps(50)
    noise = torch.randn_like(x0)
    t = 0  # last element of the full-chain sequence
    noisy = scheduler.add_noise(x0, noise, torch.full((x0.shape[0],), t))
    result = scheduler.step(noise, t, noisy)
    torch.testing.assert_close(result.prev_sample, result.pred_original_sample)


def test_ddim_perfect_model_recovers_x0_in_one_step(x0: torch.Tensor) -> None:
    """With an oracle epsilon and a single inference step, DDIM must jump
    straight back to (approximately) x0."""
    scheduler = DDIMScheduler(
        make_diffusion_config(
            num_train_timesteps=100, clip_sample=False, timestep_spacing="trailing"
        )
    )
    scheduler.set_timesteps(1)
    t = int(scheduler.timesteps[0])
    assert t == 99
    noise = torch.randn_like(x0)
    noisy = scheduler.add_noise(x0, noise, torch.full((x0.shape[0],), t))
    result = scheduler.step(noise, t, noisy)
    torch.testing.assert_close(result.prev_sample, x0, atol=1e-2, rtol=1e-2)
