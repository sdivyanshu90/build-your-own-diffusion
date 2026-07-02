"""Exponential moving average of model parameters.

Diffusion sample quality is substantially better when sampling from an EMA
of the training weights rather than the raw online weights (observed by
Ho et al. and every major implementation since): the average irons out the
noise of late-training SGD steps. The trainer updates the EMA after every
optimizer step and uses it for preview grids and as the default weights at
inference time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintains shadow copies ``s <- decay * s + (1 - decay) * p``.

    Only trainable floating-point parameters are tracked; the models in this
    library have no buffers that need averaging (GroupNorm is stateless).

    Args:
        model: Model whose parameters define the shadow set.
        decay: EMA decay in ``[0, 1)``; 0.9999 is the diffusion standard for
            long runs (halving time of ~7k steps).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and param.dtype.is_floating_point
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the model's current parameters into the shadow copies."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's parameters with the shadow copies."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Back up the model's current parameters (pair with :meth:`restore`)."""
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if name in self.shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore parameters saved by :meth:`store`.

        Raises:
            RuntimeError: If :meth:`store` was not called first.
        """
        if self._backup is None:
            raise RuntimeError("restore() called without a preceding store()")
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = None

    def to(self, device: torch.device) -> ExponentialMovingAverage:
        """Move the shadow copies to ``device`` (in place); returns self."""
        self.shadow = {name: tensor.to(device) for name, tensor in self.shadow.items()}
        return self

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state (decay + shadow tensors)."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict` output.

        Raises:
            ValueError: If the shadow parameter names do not match.
        """
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA state does not match the tracked parameter set")
        for name, tensor in state["shadow"].items():
            if tensor.shape != self.shadow[name].shape:
                raise ValueError(
                    f"EMA state does not match: parameter {name!r} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(self.shadow[name].shape)}"
                )
        self.decay = float(state["decay"])
        for name, tensor in state["shadow"].items():
            self.shadow[name] = tensor.clone().to(self.shadow[name].device)
