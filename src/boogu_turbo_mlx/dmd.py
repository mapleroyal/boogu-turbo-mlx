from __future__ import annotations

from collections.abc import Sequence

from .constants import DEFAULT_DMD_CONDITIONING_SIGMA, DEFAULT_STEPS


def sigma_schedule(
    num_inference_steps: int = DEFAULT_STEPS,
    conditioning_sigma: float = DEFAULT_DMD_CONDITIONING_SIGMA,
    timesteps: Sequence[float] | None = None,
) -> list[float]:
    """Return the official Turbo DMD sigma schedule as plain Python floats."""

    if timesteps is not None:
        sigmas = [float(timestep) for timestep in timesteps]
        if not sigmas:
            raise ValueError("timesteps must not be empty")
        if max(sigmas) > 1.0:
            sigmas = [sigma / 1000.0 for sigma in sigmas]
        return sigmas

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if not 0.0 <= conditioning_sigma <= 1.0:
        raise ValueError("conditioning_sigma must be between 0.0 and 1.0")

    step = (1.0 - conditioning_sigma) / num_inference_steps
    return [conditioning_sigma + i * step for i in range(num_inference_steps)]


def predict_dmd_update(latents, sigma: float, model_prediction):
    """Apply the official DMD student prediction update."""

    return latents + (1.0 - float(sigma)) * model_prediction


def renoise_dmd_latents(latents, next_sigma: float, noise):
    """Apply the official DMD student renoise step for a non-final update."""

    sigma = float(next_sigma)
    return (1.0 - sigma) * noise + sigma * latents
