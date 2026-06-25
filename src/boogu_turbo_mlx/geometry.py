from __future__ import annotations

from .constants import LATENT_CHANNELS, OUTPUT_ALIGNMENT, VAE_SCALE_FACTOR


def aligned_size(height: int, width: int, alignment: int = OUTPUT_ALIGNMENT) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    aligned_height = height - (height % alignment)
    aligned_width = width - (width % alignment)
    if aligned_height == 0 or aligned_width == 0:
        raise ValueError(
            f"height and width must be at least {alignment} after alignment"
        )
    return aligned_height, aligned_width


def latent_shape(
    height: int,
    width: int,
    batch_size: int = 1,
    channels: int = LATENT_CHANNELS,
) -> tuple[int, int, int, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")

    aligned_height, aligned_width = aligned_size(height, width)
    return (
        batch_size,
        channels,
        aligned_height // VAE_SCALE_FACTOR,
        aligned_width // VAE_SCALE_FACTOR,
    )


def image_token_grid(height: int, width: int) -> tuple[int, int]:
    aligned_height, aligned_width = aligned_size(height, width)
    return (
        aligned_height // OUTPUT_ALIGNMENT,
        aligned_width // OUTPUT_ALIGNMENT,
    )

