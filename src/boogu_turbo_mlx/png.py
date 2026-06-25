from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ._version import __version__
from .errors import BooguTurboMlxError

PNG_METADATA_KEY = "boogu-turbo-mlx"
PNG_PARAMETERS_KEY = "parameters"


def save_generation_png(
    result: Any,
    output_path: str | Path,
    *,
    model: str | Path | None = None,
    memory_mode: str | None = None,
    denoise_batch_size: int | None = None,
    decode_batch_size: int | None = None,
) -> None:
    """Save a generation result as PNG with the CLI-compatible metadata keys."""

    try:
        from PIL.PngImagePlugin import PngInfo
    except ImportError as exc:  # pragma: no cover - pipeline normally raises first.
        raise BooguTurboMlxError(
            "PNG metadata output requires pillow>=10. Install "
            "`boogu-turbo-mlx[runtime]`."
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = generation_metadata_payload(
        result,
        model=model,
        memory_mode=memory_mode,
        denoise_batch_size=denoise_batch_size,
        decode_batch_size=decode_batch_size,
    )
    pnginfo = PngInfo()
    pnginfo.add_itxt(PNG_METADATA_KEY, json.dumps(payload, sort_keys=True))
    pnginfo.add_itxt(PNG_PARAMETERS_KEY, generation_parameters_text(payload))
    result.image.save(path, pnginfo=pnginfo)


def generation_metadata_payload(
    result: Any,
    *,
    model: str | Path | None = None,
    memory_mode: str | None = None,
    denoise_batch_size: int | None = None,
    decode_batch_size: int | None = None,
) -> dict[str, Any]:
    payload = dict(result.to_metadata())
    payload.update(
        {
            "schema_version": 1,
            "generator": "boogu-turbo-mlx",
            "generator_version": __version__,
            "model": None if model is None else str(model),
            "memory_mode": memory_mode,
            "denoise_batch_size": denoise_batch_size,
            "decode_batch_size": decode_batch_size,
        }
    )
    return payload


def generation_parameters_text(payload: dict[str, Any]) -> str:
    seed = payload["seed"] if payload["seed"] is not None else "unspecified"
    fields = [
        f"Steps: {payload['steps']}",
        f"Seed: {seed}",
        f"Size: {payload['width']}x{payload['height']}",
        f"Model: {payload['model']}",
        f"Max sequence length: {payload['max_sequence_length']}",
        f"Truncate instruction sequence: {payload['truncate_instruction_sequence']}",
        f"DMD conditioning sigma: {payload['dmd_conditioning_sigma']}",
        f"Sigmas: {_format_number_list(payload['sigmas'])}",
        f"Generator: {payload['generator']} {payload['generator_version']}",
    ]
    if payload["timesteps"] is not None:
        fields.insert(1, f"Timesteps: {_format_number_list(payload['timesteps'])}")
    return f"{payload['prompt']}\n" + ", ".join(fields)


def _format_number_list(values: Sequence[float]) -> str:
    return ",".join(format(float(value), ".12g") for value in values)
