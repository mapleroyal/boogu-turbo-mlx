"""MLX-native runtime for Boogu Image 0.1 Turbo."""

from __future__ import annotations

import sys

if sys.version_info < (3, 10):  # pragma: no cover - cannot run under test Python.
    raise RuntimeError(
        "boogu-turbo-mlx requires Python 3.10 or newer. "
        f"Found Python {sys.version_info.major}.{sys.version_info.minor}."
    )

from ._version import __version__
from .constants import (
    DEFAULT_DMD_CONDITIONING_SIGMA,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_STEPS,
)
from .dmd import sigma_schedule
from .encoding import (
    InstructionEncoding,
    Qwen3VLInstructionEncoder,
    encode_instruction,
)
from .errors import BooguTurboGenerationCancelled
from .pipeline import (
    BooguTurboPipeline,
    BooguTurboGenerationBatch,
    BooguTurboGenerationResult,
    PipelineProgressCallback,
    PipelineProgressEvent,
)
from .png import save_generation_png
from .quantization import quantize_artifact
from .transformer import (
    BooguImageTransformer,
    BooguImageTransformerConfig,
    build_transformer_freqs_cis,
)
from .vae import AutoencoderKLDecoder, AutoencoderKLDecoderConfig

__all__ = [
    "BooguTurboPipeline",
    "BooguTurboGenerationBatch",
    "BooguTurboGenerationResult",
    "BooguTurboGenerationCancelled",
    "BooguImageTransformer",
    "BooguImageTransformerConfig",
    "InstructionEncoding",
    "PipelineProgressCallback",
    "PipelineProgressEvent",
    "Qwen3VLInstructionEncoder",
    "AutoencoderKLDecoder",
    "AutoencoderKLDecoderConfig",
    "DEFAULT_DMD_CONDITIONING_SIGMA",
    "DEFAULT_MAX_SEQUENCE_LENGTH",
    "DEFAULT_STEPS",
    "build_transformer_freqs_cis",
    "encode_instruction",
    "quantize_artifact",
    "save_generation_png",
    "sigma_schedule",
]
