"""Tests for the exponential moving average of model parameters."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from diffusionlab.training import ExponentialMovingAverage


def make_model(value: float = 1.0) -> nn.Linear:
    model = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)
    return model


def test_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError, match="decay"):
        ExponentialMovingAverage(make_model(), decay=1.0)


def test_update_math_is_exact() -> None:
    model = make_model(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    ema.update(model)
    for shadow in ema.shadow.values():
        torch.testing.assert_close(shadow, torch.full_like(shadow, 0.9 * 1.0 + 0.1 * 2.0))


def test_repeated_updates_converge_to_parameters() -> None:
    model = make_model(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(3.0)
    for _ in range(50):
        ema.update(model)
    for shadow in ema.shadow.values():
        torch.testing.assert_close(shadow, torch.full_like(shadow, 3.0))


def test_copy_to_store_restore_cycle() -> None:
    model = make_model(1.0)
    ema = ExponentialMovingAverage(model, decay=0.0)  # shadow == latest params
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(5.0)
    ema.update(model)  # shadow becomes 5.0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(7.0)  # online weights move on

    ema.store(model)
    ema.copy_to(model)
    assert all(torch.equal(p, torch.full_like(p, 5.0)) for p in model.parameters())
    ema.restore(model)
    assert all(torch.equal(p, torch.full_like(p, 7.0)) for p in model.parameters())


def test_restore_without_store_raises() -> None:
    ema = ExponentialMovingAverage(make_model(), decay=0.9)
    with pytest.raises(RuntimeError, match="store"):
        ema.restore(make_model())


def test_state_dict_roundtrip() -> None:
    model = make_model(1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(4.0)
    ema.update(model)

    fresh = ExponentialMovingAverage(make_model(0.0), decay=0.5)
    fresh.load_state_dict(ema.state_dict())
    assert fresh.decay == 0.9
    for name in ema.shadow:
        torch.testing.assert_close(fresh.shadow[name], ema.shadow[name])


def test_load_state_dict_rejects_mismatched_parameters() -> None:
    ema = ExponentialMovingAverage(make_model(), decay=0.9)
    other = ExponentialMovingAverage(nn.Linear(3, 3), decay=0.9)
    with pytest.raises(ValueError, match="does not match"):
        ema.load_state_dict(other.state_dict())


def test_frozen_parameters_are_not_tracked() -> None:
    model = make_model()
    model.bias.requires_grad_(False)
    ema = ExponentialMovingAverage(model, decay=0.9)
    assert "weight" in ema.shadow
    assert "bias" not in ema.shadow
