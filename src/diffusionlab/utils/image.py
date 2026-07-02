"""Image conversion and grid-saving helpers.

The library-wide convention is that model-space images live in ``[-1, 1]``
(matching the standard normalisation of the diffusion literature); these
helpers convert to displayable ranges at the edges of the system.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert a ``[-1, 1]`` float tensor to ``uint8`` in ``[0, 255]``.

    Args:
        images: Tensor of shape ``(N, C, H, W)`` or ``(C, H, W)``.
    """
    images = images.detach().float().clamp(-1.0, 1.0)
    return ((images + 1.0) * 127.5).round().to(torch.uint8)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> Path:
    """Save a batch of ``[-1, 1]`` images as a single PNG grid.

    Args:
        images: Tensor of shape ``(N, C, H, W)`` in model space ``[-1, 1]``.
        path: Output file path; parent directories are created.
        nrow: Images per grid row.

    Returns:
        The resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        images.detach().float().clamp(-1.0, 1.0).cpu(),
        nrow=nrow,
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    save_image(grid, str(path))
    return path
