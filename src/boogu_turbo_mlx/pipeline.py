from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping, Sequence

from .artifacts import (
    component_config_path,
    component_weights_index_path,
    processor_path,
    read_artifact,
)
from .constants import (
    DEFAULT_DMD_CONDITIONING_SIGMA,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_STEPS,
    MAX_GENERATION_PIXELS,
    MAX_GENERATION_SEED,
    MAX_GENERATION_SIZE,
    OUTPUT_ALIGNMENT,
    VAE_SCALE_FACTOR,
)
from .dmd import predict_dmd_update, renoise_dmd_latents, sigma_schedule
from .encoding import (
    InstructionEncoding,
    Qwen3VLInstructionEncoder,
    _normalize_prompts,
)
from .errors import BooguTurboGenerationCancelled, BooguTurboMlxError
from .transformer import (
    BooguImageTransformer,
    build_transformer_freqs_cis,
)
from .vae import AutoencoderKLDecoder

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from PIL import Image


MAX_DENOISE_PIXELS = MAX_GENERATION_PIXELS
MEMORY_MODE_RESIDENT = "resident"
MEMORY_MODE_LOW_MEMORY = "low_memory"
VALID_MEMORY_MODES = frozenset({MEMORY_MODE_RESIDENT, MEMORY_MODE_LOW_MEMORY})

PipelineProgressKind = Literal[
    "load_start",
    "load_component_start",
    "load_component_end",
    "load_complete",
    "unload",
    "encode_start",
    "encode_end",
    "denoise_step_start",
    "denoise_step_end",
    "denoise_compute_start",
    "denoise_compute_end",
    "decode_start",
    "decode_end",
    "resize_start",
    "resize_end",
    "complete",
    "cancelled",
    "error",
]


@dataclass(frozen=True)
class PipelineProgressEvent:
    kind: PipelineProgressKind
    stage: str
    message: str
    progress: float | None = None
    step_index: int | None = None
    step_count: int | None = None
    sigma: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

# Raise BooguTurboGenerationCancelled from the callback to cooperatively cancel
# generation. Other callback exceptions are surfaced as generation errors.
PipelineProgressCallback = Callable[[PipelineProgressEvent], None]


@dataclass(frozen=True)
class BooguTurboGenerationResult:
    image: "Image.Image"
    prompt: str
    prompt_index: int
    image_index: int
    batch_index: int
    seed: int | None
    height: int
    width: int
    denoise_height: int
    denoise_width: int
    output_height: int
    output_width: int
    steps: int
    step_count: int
    sigmas: tuple[float, ...]
    timesteps: tuple[float, ...] | None
    max_sequence_length: int
    truncate_instruction_sequence: bool
    dmd_conditioning_sigma: float
    token_count: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "prompt_index": self.prompt_index,
            "image_index": self.image_index,
            "batch_index": self.batch_index,
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "denoise_height": self.denoise_height,
            "denoise_width": self.denoise_width,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "steps": self.steps,
            "step_count": self.step_count,
            "sigmas": list(self.sigmas),
            "timesteps": None if self.timesteps is None else list(self.timesteps),
            "max_sequence_length": self.max_sequence_length,
            "truncate_instruction_sequence": self.truncate_instruction_sequence,
            "dmd_conditioning_sigma": self.dmd_conditioning_sigma,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class BooguTurboGenerationBatch:
    items: tuple[BooguTurboGenerationResult, ...]

    @property
    def images(self) -> tuple["Image.Image", ...]:
        return tuple(item.image for item in self.items)

    def to_metadata(self) -> list[dict[str, Any]]:
        return [item.to_metadata() for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[BooguTurboGenerationResult]:
        return iter(self.items)

    def __getitem__(self, index: int) -> BooguTurboGenerationResult:
        return self.items[index]


@dataclass(frozen=True)
class _PipelineComponents:
    encoder: Qwen3VLInstructionEncoder
    transformer: BooguImageTransformer
    vae: AutoencoderKLDecoder
    freqs_cis: Any


@dataclass(frozen=True)
class _TransformerRuntime:
    transformer: BooguImageTransformer
    freqs_cis: Any


@dataclass
class _StageMetric:
    wall_time_seconds: float = 0.0
    active_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    cache_memory_bytes: int | None = None
    calls: int = 0

    def record(
        self,
        *,
        wall_time_seconds: float,
        active_memory_bytes: int | None,
        peak_memory_bytes: int | None,
        cache_memory_bytes: int | None,
    ) -> None:
        self.wall_time_seconds += wall_time_seconds
        self.active_memory_bytes = _max_optional_int(
            self.active_memory_bytes,
            active_memory_bytes,
        )
        self.peak_memory_bytes = _max_optional_int(
            self.peak_memory_bytes,
            peak_memory_bytes,
        )
        self.cache_memory_bytes = _max_optional_int(
            self.cache_memory_bytes,
            cache_memory_bytes,
        )
        self.calls += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_time_seconds": self.wall_time_seconds,
            "active_memory_bytes": self.active_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "cache_memory_bytes": self.cache_memory_bytes,
            "calls": self.calls,
        }


class _BenchmarkRecorder:
    def __init__(self) -> None:
        self._metrics: dict[str, _StageMetric] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        mx = _load_mlx()
        _reset_peak_memory(mx)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            active, peak, cache = _memory_snapshot(mx)
            self._metrics.setdefault(name, _StageMetric()).record(
                wall_time_seconds=elapsed,
                active_memory_bytes=active,
                peak_memory_bytes=peak,
                cache_memory_bytes=cache,
            )

    def run_peak_memory_bytes(self) -> int | None:
        """Run-wide peak as the max of per-stage peaks.

        ``stage()`` resets MLX's peak counter on entry to measure each stage in
        isolation, so the global ``get_peak_memory()`` only holds the final
        stage's peak once a run finishes. The per-stage peaks were captured
        before each reset, so their max is the true run-wide peak (inter-stage
        gaps only free memory and cannot exceed an adjacent stage's peak).
        """
        peak: int | None = None
        for metric in self._metrics.values():
            peak = _max_optional_int(peak, metric.peak_memory_bytes)
        return peak

    def to_dict(self) -> dict[str, Any]:
        return {
            name: metric.to_dict()
            for name, metric in sorted(self._metrics.items())
        }


@dataclass
class BooguTurboPipeline:
    model_path: Path
    memory_mode: str = MEMORY_MODE_RESIDENT
    _components: _PipelineComponents | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser()
        self.memory_mode = _validate_memory_mode(self.memory_mode)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        memory_mode: str = MEMORY_MODE_RESIDENT,
    ) -> "BooguTurboPipeline":
        return cls(
            Path(path).expanduser(),
            memory_mode=memory_mode,
        )

    @property
    def is_loaded(self) -> bool:
        return self.memory_mode == MEMORY_MODE_RESIDENT and self._components is not None

    def load(
        self,
        *,
        progress_callback: PipelineProgressCallback | None = None,
    ) -> "BooguTurboPipeline":
        if self._components is not None:
            return self

        try:
            root = self.model_path
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="load_start",
                    stage="model_load",
                    message="Loading Boogu Turbo MLX artifact",
                    progress=0.0,
                    details={"model_path": str(root)},
                ),
            )
            _validate_artifact_root(root)
            if self.memory_mode == MEMORY_MODE_LOW_MEMORY:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        kind="load_complete",
                        stage="model_load",
                        message="Model artifact validated",
                        progress=1.0,
                        details={
                            "model_path": str(root),
                            "memory_mode": self.memory_mode,
                        },
                    ),
                )
                return self

            encoder = self._load_instruction_encoder(
                progress_callback=progress_callback,
                start_progress=0.1,
                end_progress=0.3,
            )
            transformer_runtime = self._load_transformer_runtime(
                progress_callback=progress_callback,
                start_progress=0.35,
                end_progress=0.7,
            )
            vae = self._load_vae_decoder(
                progress_callback=progress_callback,
                start_progress=0.75,
                end_progress=0.95,
            )

            self._components = _PipelineComponents(
                encoder=encoder,
                transformer=transformer_runtime.transformer,
                vae=vae,
                freqs_cis=transformer_runtime.freqs_cis,
            )
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="load_complete",
                    stage="model_load",
                    message="Model ready",
                    progress=1.0,
                    details={"model_path": str(root)},
                ),
            )
            return self
        except Exception as exc:
            _emit_terminal_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="error",
                    stage="model_load",
                    message=str(exc),
                    details={"error_type": type(exc).__name__},
                ),
            )
            raise

    def _load_instruction_encoder(
        self,
        *,
        progress_callback: PipelineProgressCallback | None = None,
        start_progress: float | None = None,
        end_progress: float | None = None,
    ) -> Qwen3VLInstructionEncoder:
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_start",
                stage="model_load",
                message="Loading instruction encoder",
                progress=start_progress,
                details={"component": "encoder"},
            ),
        )
        encoder = Qwen3VLInstructionEncoder.from_pretrained(
            self.model_path,
        )
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_end",
                stage="model_load",
                message="Instruction encoder loaded",
                progress=end_progress,
                details={"component": "encoder"},
            ),
        )
        return encoder

    def _load_transformer_runtime(
        self,
        *,
        progress_callback: PipelineProgressCallback | None = None,
        start_progress: float | None = None,
        end_progress: float | None = None,
    ) -> _TransformerRuntime:
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_start",
                stage="model_load",
                message="Loading image transformer",
                progress=start_progress,
                details={"component": "transformer"},
            ),
        )
        transformer = BooguImageTransformer.from_pretrained(
            self.model_path,
        )
        freqs_cis = build_transformer_freqs_cis(transformer.config)
        _eval_mlx_values(freqs_cis)
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_end",
                stage="model_load",
                message="Image transformer loaded",
                progress=end_progress,
                details={"component": "transformer"},
            ),
        )
        return _TransformerRuntime(transformer=transformer, freqs_cis=freqs_cis)

    def _load_vae_decoder(
        self,
        *,
        progress_callback: PipelineProgressCallback | None = None,
        start_progress: float | None = None,
        end_progress: float | None = None,
    ) -> AutoencoderKLDecoder:
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_start",
                stage="model_load",
                message="Loading VAE decoder",
                progress=start_progress,
                details={"component": "vae"},
            ),
        )
        vae = AutoencoderKLDecoder.from_pretrained(self.model_path)
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="load_component_end",
                stage="model_load",
                message="VAE decoder loaded",
                progress=end_progress,
                details={"component": "vae"},
            ),
        )
        return vae

    def unload(
        self,
        *,
        progress_callback: PipelineProgressCallback | None = None,
    ) -> None:
        was_loaded = self._components is not None
        self._components = None
        _clear_runtime_cache()
        if was_loaded:
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="unload",
                    stage="model_load",
                    message="Model unloaded",
                    progress=0.0,
                ),
            )

    def generate(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        steps: int = DEFAULT_STEPS,
        seed: int | None = None,
        timesteps: Sequence[float] | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        truncate_instruction_sequence: bool = False,
        dmd_conditioning_sigma: float = DEFAULT_DMD_CONDITIONING_SIGMA,
        denoise_batch_size: int | None = None,
        decode_batch_size: int | None = None,
        progress_callback: PipelineProgressCallback | None = None,
    ) -> "Image.Image":
        """Generate one image from a prompt and converted MLX artifact.

        A progress callback may raise BooguTurboGenerationCancelled to request
        cooperative cancellation between load, encode, denoise, decode, resize,
        and completion events.
        """

        return self.generate_batch(
            prompt,
            height=height,
            width=width,
            steps=steps,
            seed=seed,
            timesteps=timesteps,
            max_sequence_length=max_sequence_length,
            truncate_instruction_sequence=truncate_instruction_sequence,
            dmd_conditioning_sigma=dmd_conditioning_sigma,
            denoise_batch_size=denoise_batch_size,
            decode_batch_size=decode_batch_size,
            progress_callback=progress_callback,
        ).items[0].image

    def generate_batch(
        self,
        prompt_or_prompts: str | Sequence[str],
        *,
        height: int,
        width: int,
        steps: int = DEFAULT_STEPS,
        seed: int | None = None,
        seeds: Sequence[int] | None = None,
        num_images_per_prompt: int = 1,
        timesteps: Sequence[float] | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        truncate_instruction_sequence: bool = False,
        dmd_conditioning_sigma: float = DEFAULT_DMD_CONDITIONING_SIGMA,
        denoise_batch_size: int | None = None,
        decode_batch_size: int | None = None,
        progress_callback: PipelineProgressCallback | None = None,
        _benchmark_recorder: _BenchmarkRecorder | None = None,
    ) -> BooguTurboGenerationBatch:
        """Generate an ordered batch of images for one or more prompts."""

        try:
            prompts, _ = _normalize_prompts(prompt_or_prompts)
            if num_images_per_prompt <= 0:
                raise ValueError("num_images_per_prompt must be positive")
            if timesteps is None and steps <= 0:
                raise ValueError("steps must be positive")
            if max_sequence_length <= 0:
                raise ValueError("max_sequence_length must be positive")
            denoise_batch_size = _validate_optional_batch_size(
                denoise_batch_size,
                "denoise_batch_size",
            )
            decode_batch_size = _validate_optional_batch_size(
                decode_batch_size,
                "decode_batch_size",
            )

            height, width = _validate_generation_dimensions(height, width)
            expanded_prompts = _expand_prompt_batch(prompts, num_images_per_prompt)
            batch_size = len(expanded_prompts)
            generation_seeds = _normalize_generation_seeds(
                seed=seed,
                seeds=seeds,
                batch_size=batch_size,
            )
            denoise_height, denoise_width = _denoise_size_for_request(height, width)
            normalized_timesteps = _normalize_timesteps(timesteps)
            sigmas = sigma_schedule(
                steps,
                dmd_conditioning_sigma,
                timesteps=normalized_timesteps,
            )
            if self.memory_mode == MEMORY_MODE_LOW_MEMORY and self._components is None:
                (
                    encoding,
                    token_counts,
                    images,
                ) = self._generate_batch_low_memory(
                    prompts=prompts,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    sigmas=sigmas,
                    denoise_height=denoise_height,
                    denoise_width=denoise_width,
                    output_height=height,
                    output_width=width,
                    max_sequence_length=max_sequence_length,
                    truncate_instruction_sequence=truncate_instruction_sequence,
                    generation_seeds=generation_seeds,
                    denoise_batch_size=denoise_batch_size,
                    decode_batch_size=decode_batch_size,
                    progress_callback=progress_callback,
                    benchmark_recorder=_benchmark_recorder,
                )
            else:
                with _maybe_benchmark_stage(_benchmark_recorder, "load"):
                    components = self.load(progress_callback=progress_callback)._components
                if components is None:  # pragma: no cover - defensive invariant.
                    raise BooguTurboMlxError("Pipeline failed to load model components")

                (
                    encoding,
                    token_counts,
                    images,
                ) = self._generate_batch_resident(
                    components=components,
                    prompts=prompts,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    sigmas=sigmas,
                    denoise_height=denoise_height,
                    denoise_width=denoise_width,
                    output_height=height,
                    output_width=width,
                    max_sequence_length=max_sequence_length,
                    truncate_instruction_sequence=truncate_instruction_sequence,
                    generation_seeds=generation_seeds,
                    denoise_batch_size=denoise_batch_size,
                    decode_batch_size=decode_batch_size,
                    progress_callback=progress_callback,
                    benchmark_recorder=_benchmark_recorder,
                )
            if len(images) != batch_size:  # pragma: no cover - defensive invariant.
                raise BooguTurboMlxError(
                    "Generated image count does not match requested batch size"
                )
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="complete",
                    stage="complete",
                    message="Generation complete",
                    progress=1.0,
                    details={
                        "height": images[0].height,
                        "width": images[0].width,
                        "batch_size": len(images),
                    },
                ),
            )
            return BooguTurboGenerationBatch(
                tuple(
                    BooguTurboGenerationResult(
                        image=image,
                        prompt=prompt,
                        prompt_index=prompt_index,
                        image_index=image_index,
                        batch_index=batch_index,
                        seed=(
                            None
                            if generation_seeds is None
                            else int(generation_seeds[batch_index])
                        ),
                        height=int(height),
                        width=int(width),
                        denoise_height=int(denoise_height),
                        denoise_width=int(denoise_width),
                        output_height=int(image.height),
                        output_width=int(image.width),
                        steps=int(steps),
                        step_count=len(sigmas),
                        sigmas=tuple(float(sigma) for sigma in sigmas),
                        timesteps=(
                            None
                            if normalized_timesteps is None
                            else tuple(float(timestep) for timestep in normalized_timesteps)
                        ),
                        max_sequence_length=int(max_sequence_length),
                        truncate_instruction_sequence=bool(
                            truncate_instruction_sequence
                        ),
                        dmd_conditioning_sigma=float(dmd_conditioning_sigma),
                        token_count=int(token_counts[batch_index]),
                    )
                    for batch_index, (image, (prompt, prompt_index, image_index)) in enumerate(
                        zip(images, expanded_prompts)
                    )
                )
            )
        except BooguTurboGenerationCancelled:
            _emit_terminal_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="cancelled",
                    stage="cancelled",
                    message="Generation cancelled",
                ),
            )
            raise
        except Exception as exc:
            _emit_terminal_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="error",
                    stage="error",
                    message=str(exc),
                    details={"error_type": type(exc).__name__},
                ),
            )
            raise
        finally:
            _clear_runtime_cache()

    def _load_components(self) -> _PipelineComponents:
        self.load()
        if self._components is None:  # pragma: no cover - defensive invariant.
            raise BooguTurboMlxError("Pipeline failed to load model components")
        return self._components

    def _generate_batch_resident(
        self,
        *,
        components: _PipelineComponents,
        prompts: Sequence[str],
        batch_size: int,
        num_images_per_prompt: int,
        sigmas: Sequence[float],
        denoise_height: int,
        denoise_width: int,
        output_height: int,
        output_width: int,
        max_sequence_length: int,
        truncate_instruction_sequence: bool,
        generation_seeds: Sequence[int] | None,
        denoise_batch_size: int | None,
        decode_batch_size: int | None,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> tuple[InstructionEncoding, list[int], list["Image.Image"]]:
        encoding, token_counts = self._encode_prompt_batch(
            encoder=components.encoder,
            prompts=prompts,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            truncate_instruction_sequence=truncate_instruction_sequence,
            progress_callback=progress_callback,
            benchmark_recorder=benchmark_recorder,
        )
        transformer_runtime = _TransformerRuntime(
            transformer=components.transformer,
            freqs_cis=components.freqs_cis,
        )
        images = self._generate_images_from_encoding(
            encoding=encoding,
            transformer_runtime=transformer_runtime,
            vae=components.vae,
            sigmas=sigmas,
            denoise_height=denoise_height,
            denoise_width=denoise_width,
            output_height=output_height,
            output_width=output_width,
            generation_seeds=generation_seeds,
            denoise_batch_size=denoise_batch_size,
            decode_batch_size=decode_batch_size,
            progress_callback=progress_callback,
            benchmark_recorder=benchmark_recorder,
        )
        return encoding, token_counts, images

    def _generate_batch_low_memory(
        self,
        *,
        prompts: Sequence[str],
        batch_size: int,
        num_images_per_prompt: int,
        sigmas: Sequence[float],
        denoise_height: int,
        denoise_width: int,
        output_height: int,
        output_width: int,
        max_sequence_length: int,
        truncate_instruction_sequence: bool,
        generation_seeds: Sequence[int] | None,
        denoise_batch_size: int | None,
        decode_batch_size: int | None,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> tuple[InstructionEncoding, list[int], list["Image.Image"]]:
        with _maybe_benchmark_stage(benchmark_recorder, "load"):
            self.load(progress_callback=progress_callback)
        with _maybe_benchmark_stage(benchmark_recorder, "load"):
            encoder = self._load_instruction_encoder(
                progress_callback=progress_callback,
                start_progress=0.1,
                end_progress=0.2,
            )
        try:
            encoding, token_counts = self._encode_prompt_batch(
                encoder=encoder,
                prompts=prompts,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                truncate_instruction_sequence=truncate_instruction_sequence,
                progress_callback=progress_callback,
                benchmark_recorder=benchmark_recorder,
            )
        finally:
            encoder = None
            _clear_runtime_cache()

        if denoise_batch_size is None:
            with _maybe_benchmark_stage(benchmark_recorder, "load"):
                transformer_runtime = self._load_transformer_runtime(
                    progress_callback=progress_callback,
                    start_progress=0.25,
                    end_progress=0.35,
                )
            try:
                latents = self._generate_latents(
                    encoding=encoding,
                    sigmas=sigmas,
                    denoise_height=denoise_height,
                    denoise_width=denoise_width,
                    seed=None,
                    seeds=generation_seeds,
                    transformer_runtime=transformer_runtime,
                    progress_callback=progress_callback,
                    benchmark_recorder=benchmark_recorder,
                )
            finally:
                transformer_runtime = None
                _clear_runtime_cache()

            with _maybe_benchmark_stage(benchmark_recorder, "load"):
                vae = self._load_vae_decoder(
                    progress_callback=progress_callback,
                    start_progress=0.75,
                    end_progress=0.78,
                )
            try:
                return (
                    encoding,
                    token_counts,
                    self._decode_latents_to_images(
                        vae=vae,
                        latents=latents,
                        output_height=output_height,
                        output_width=output_width,
                        decode_batch_size=decode_batch_size,
                        progress_callback=progress_callback,
                        benchmark_recorder=benchmark_recorder,
                    ),
                )
            finally:
                latents = None
                vae = None
                _clear_runtime_cache()

        images: list["Image.Image"] = []
        for start, end in _chunk_ranges(batch_size, denoise_batch_size):
            chunk_encoding = _slice_encoding(encoding, start, end)
            chunk_seeds = _slice_optional_sequence(generation_seeds, start, end)
            with _maybe_benchmark_stage(benchmark_recorder, "load"):
                transformer_runtime = self._load_transformer_runtime(
                    progress_callback=progress_callback,
                    start_progress=0.25,
                    end_progress=0.35,
                )
            try:
                latents = self._generate_latents(
                    encoding=chunk_encoding,
                    sigmas=sigmas,
                    denoise_height=denoise_height,
                    denoise_width=denoise_width,
                    seed=None,
                    seeds=chunk_seeds,
                    transformer_runtime=transformer_runtime,
                    progress_callback=progress_callback,
                    benchmark_recorder=benchmark_recorder,
                )
            finally:
                transformer_runtime = None
                _clear_runtime_cache()

            with _maybe_benchmark_stage(benchmark_recorder, "load"):
                vae = self._load_vae_decoder(
                    progress_callback=progress_callback,
                    start_progress=0.75,
                    end_progress=0.78,
                )
            try:
                images.extend(
                    self._decode_latents_to_images(
                        vae=vae,
                        latents=latents,
                        output_height=output_height,
                        output_width=output_width,
                        decode_batch_size=decode_batch_size,
                        progress_callback=progress_callback,
                        benchmark_recorder=benchmark_recorder,
                    )
                )
            finally:
                latents = None
                vae = None
                _clear_runtime_cache()
        return encoding, token_counts, images

    def _encode_prompt_batch(
        self,
        *,
        encoder: Qwen3VLInstructionEncoder,
        prompts: Sequence[str],
        batch_size: int,
        num_images_per_prompt: int,
        max_sequence_length: int,
        truncate_instruction_sequence: bool,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> tuple[InstructionEncoding, list[int]]:
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="encode_start",
                stage="encode",
                message="Encoding prompt",
                progress=0.05,
                details={
                    "prompt_count": len(prompts),
                    "batch_size": batch_size,
                    "num_images_per_prompt": num_images_per_prompt,
                    "max_sequence_length": max_sequence_length,
                    "truncate_instruction_sequence": truncate_instruction_sequence,
                },
            ),
        )
        with _maybe_benchmark_stage(benchmark_recorder, "encode"):
            base_encoding = encoder.encode(
                prompts[0] if len(prompts) == 1 else prompts,
                max_sequence_length=max_sequence_length,
                truncate=truncate_instruction_sequence,
            )
            encoding = _repeat_encoding(base_encoding, num_images_per_prompt)
            _eval_mlx_values(
                (
                    encoding.hidden_states,
                    encoding.attention_mask,
                    encoding.input_ids,
                )
            )
            token_counts = _token_counts_by_batch_item(encoding.attention_mask)
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="encode_end",
                stage="encode",
                message="Prompt encoded",
                progress=0.2,
                details={
                    "token_count": int(token_counts[0]),
                    "token_counts": tuple(token_counts),
                },
            ),
        )
        return encoding, token_counts

    def _generate_images_from_encoding(
        self,
        *,
        encoding: InstructionEncoding,
        transformer_runtime: _TransformerRuntime,
        vae: AutoencoderKLDecoder,
        sigmas: Sequence[float],
        denoise_height: int,
        denoise_width: int,
        output_height: int,
        output_width: int,
        generation_seeds: Sequence[int] | None,
        denoise_batch_size: int | None,
        decode_batch_size: int | None,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> list["Image.Image"]:
        batch_size = int(encoding.hidden_states.shape[0])
        if denoise_batch_size is None:
            latents = self._generate_latents(
                encoding=encoding,
                sigmas=sigmas,
                denoise_height=denoise_height,
                denoise_width=denoise_width,
                seed=None,
                seeds=generation_seeds,
                transformer_runtime=transformer_runtime,
                progress_callback=progress_callback,
                benchmark_recorder=benchmark_recorder,
            )
            return self._decode_latents_to_images(
                vae=vae,
                latents=latents,
                output_height=output_height,
                output_width=output_width,
                decode_batch_size=decode_batch_size,
                progress_callback=progress_callback,
                benchmark_recorder=benchmark_recorder,
            )

        images: list["Image.Image"] = []
        for start, end in _chunk_ranges(batch_size, denoise_batch_size):
            chunk_encoding = _slice_encoding(encoding, start, end)
            chunk_seeds = _slice_optional_sequence(generation_seeds, start, end)
            latents = self._generate_latents(
                encoding=chunk_encoding,
                sigmas=sigmas,
                denoise_height=denoise_height,
                denoise_width=denoise_width,
                seed=None,
                seeds=chunk_seeds,
                transformer_runtime=transformer_runtime,
                progress_callback=progress_callback,
                benchmark_recorder=benchmark_recorder,
            )
            images.extend(
                self._decode_latents_to_images(
                    vae=vae,
                    latents=latents,
                    output_height=output_height,
                    output_width=output_width,
                    decode_batch_size=decode_batch_size,
                    progress_callback=progress_callback,
                    benchmark_recorder=benchmark_recorder,
                )
            )
        return images

    def _decode_latents_to_images(
        self,
        *,
        vae: AutoencoderKLDecoder,
        latents: Any,
        output_height: int,
        output_width: int,
        decode_batch_size: int | None,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> list["Image.Image"]:
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="decode_start",
                stage="decode",
                message="Decoding latents",
                progress=0.78,
            ),
        )
        batch_size = int(latents.shape[0])
        if decode_batch_size is None:
            with _maybe_benchmark_stage(benchmark_recorder, "decode"):
                decoded = vae.decode(latents)
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="decode_end",
                    stage="decode",
                    message="Latents decoded",
                    progress=0.88,
                    details={
                        "decoded_shape": tuple(int(item) for item in decoded.shape)
                    },
                ),
            )
            return self._postprocess_decoded_batch(
                decoded=decoded,
                output_height=output_height,
                output_width=output_width,
                progress_callback=progress_callback,
                benchmark_recorder=benchmark_recorder,
            )

        images: list["Image.Image"] = []
        for start, end in _chunk_ranges(batch_size, decode_batch_size):
            with _maybe_benchmark_stage(benchmark_recorder, "decode"):
                decoded = vae.decode(latents[start:end])
            images.extend(
                self._postprocess_decoded_batch(
                    decoded=decoded,
                    output_height=output_height,
                    output_width=output_width,
                    progress_callback=progress_callback,
                    benchmark_recorder=benchmark_recorder,
                )
            )
        _emit_progress(
            progress_callback,
            PipelineProgressEvent(
                kind="decode_end",
                stage="decode",
                message="Latents decoded",
                progress=0.88,
                details={"batch_size": len(images)},
            ),
        )
        return images

    def _postprocess_decoded_batch(
        self,
        *,
        decoded: Any,
        output_height: int,
        output_width: int,
        progress_callback: PipelineProgressCallback | None,
        benchmark_recorder: _BenchmarkRecorder | None,
    ) -> list["Image.Image"]:
        with _maybe_benchmark_stage(benchmark_recorder, "postprocess"):
            needs_resize = (
                decoded.shape[2] != output_height or decoded.shape[3] != output_width
            )
            if needs_resize:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        kind="resize_start",
                        stage="resize",
                        message="Resizing decoded image",
                        progress=0.9,
                        details={
                            "source_height": int(decoded.shape[2]),
                            "source_width": int(decoded.shape[3]),
                            "target_height": output_height,
                            "target_width": output_width,
                        },
                    ),
                )
            decoded = _resize_decoded_to_request(
                decoded,
                output_height,
                output_width,
            )
            if needs_resize:
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        kind="resize_end",
                        stage="resize",
                        message="Image sized for request",
                        progress=0.95,
                        details={"height": output_height, "width": output_width},
                    ),
                )
            return _postprocess_batch_to_pil(decoded)

    def _generate_latents(
        self,
        *,
        encoding: InstructionEncoding,
        sigmas: Sequence[float],
        denoise_height: int,
        denoise_width: int,
        seed: int | None,
        seeds: Sequence[int] | None = None,
        initial_noise: Any | None = None,
        renoise_noises: Sequence[Any] | None = None,
        runtime_dtype: Any | None = None,
        progress_callback: PipelineProgressCallback | None = None,
        transformer_runtime: _TransformerRuntime | None = None,
        benchmark_recorder: _BenchmarkRecorder | None = None,
    ) -> Any:
        if transformer_runtime is None:
            components = self._load_components()
            transformer_runtime = _TransformerRuntime(
                transformer=components.transformer,
                freqs_cis=components.freqs_cis,
            )
        mx = _load_mlx()
        if not sigmas:
            raise ValueError("sigmas must not be empty")

        latent_height = denoise_height // VAE_SCALE_FACTOR
        latent_width = denoise_width // VAE_SCALE_FACTOR
        transformer = transformer_runtime.transformer
        channels = int(transformer.config.in_channels)
        batch_size = int(encoding.hidden_states.shape[0])
        latent_shape = (batch_size, channels, latent_height, latent_width)
        dtype = runtime_dtype or getattr(encoding.hidden_states, "dtype", mx.float32)
        generation_seeds = _normalize_generation_seeds(
            seed=seed,
            seeds=seeds,
            batch_size=batch_size,
        )

        if initial_noise is not None and tuple(initial_noise.shape) != latent_shape:
            raise ValueError(
                f"initial_noise must have shape {latent_shape}, got {tuple(initial_noise.shape)}"
            )
        expected_renoise_count = max(len(sigmas) - 1, 0)
        if renoise_noises is not None and len(renoise_noises) != expected_renoise_count:
            raise ValueError(
                "renoise_noises length must equal the number of non-final steps"
            )

        keys_by_batch_item = _split_noise_keys_for_batch(
            generation_seeds,
            1 + expected_renoise_count,
        )
        if initial_noise is None:
            latents = _normal_batch(
                latent_shape,
                dtype=dtype,
                keys=(
                    None
                    if keys_by_batch_item is None
                    else [item_keys[0] for item_keys in keys_by_batch_item]
                ),
            )
        else:
            latents = initial_noise.astype(dtype)

        instruction_hidden_states = encoding.hidden_states.astype(dtype)
        instruction_attention_mask = encoding.attention_mask
        prepared_inputs = None
        prepare_forward_inputs = getattr(transformer, "prepare_forward_inputs", None)
        if callable(prepare_forward_inputs):
            prepared_inputs = prepare_forward_inputs(
                transformer_runtime.freqs_cis,
                instruction_attention_mask,
                latent_height,
                latent_width,
            )
            mlx_values = getattr(prepared_inputs, "mlx_values", None)
            if callable(mlx_values):
                _eval_mlx_values(mlx_values())

        def build_denoise_graph() -> None:
            nonlocal latents
            for index, sigma in enumerate(sigmas):
                step_progress_start = 0.25 + 0.5 * (index / len(sigmas))
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        kind="denoise_step_start",
                        stage="denoise",
                        message=f"Denoising step {index + 1} of {len(sigmas)}",
                        progress=step_progress_start,
                        step_index=index,
                        step_count=len(sigmas),
                        sigma=float(sigma),
                        details={
                            "latent_shape": tuple(int(item) for item in latent_shape),
                            "batch_size": batch_size,
                            "denoise_height": denoise_height,
                            "denoise_width": denoise_width,
                        },
                    ),
                )
                timestep = mx.full((batch_size,), float(sigma), dtype=dtype)
                transformer_kwargs = (
                    {}
                    if prepared_inputs is None
                    else {"prepared_inputs": prepared_inputs}
                )
                model_prediction = transformer(
                    latents,
                    timestep,
                    instruction_hidden_states,
                    transformer_runtime.freqs_cis,
                    instruction_attention_mask,
                    **transformer_kwargs,
                )
                latents = predict_dmd_update(
                    latents,
                    float(sigma),
                    model_prediction,
                ).astype(dtype)

                if index < len(sigmas) - 1:
                    if renoise_noises is None:
                        noise = _normal_batch(
                            latent_shape,
                            dtype=dtype,
                            keys=(
                                None
                                if keys_by_batch_item is None
                                else [
                                    item_keys[index + 1]
                                    for item_keys in keys_by_batch_item
                                ]
                            ),
                        )
                    else:
                        noise = renoise_noises[index]
                        if tuple(noise.shape) != latent_shape:
                            raise ValueError(
                                f"renoise noise {index} must have shape {latent_shape}, "
                                f"got {tuple(noise.shape)}"
                            )
                        noise = noise.astype(dtype)
                    latents = renoise_dmd_latents(
                        latents,
                        float(sigmas[index + 1]),
                        noise,
                    ).astype(dtype)
                _emit_progress(
                    progress_callback,
                    PipelineProgressEvent(
                        kind="denoise_step_end",
                        stage="denoise",
                        message=f"Denoising step {index + 1} prepared",
                        progress=0.25 + 0.5 * ((index + 1) / len(sigmas)),
                        step_index=index,
                        step_count=len(sigmas),
                        sigma=float(sigma),
                    ),
                )

        def run_denoise_compute() -> None:
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="denoise_compute_start",
                    stage="denoise",
                    message="Running denoising",
                    progress=0.75,
                    step_count=len(sigmas),
                    details={
                        "latent_shape": tuple(int(item) for item in latent_shape),
                        "batch_size": batch_size,
                        "denoise_height": denoise_height,
                        "denoise_width": denoise_width,
                    },
                ),
            )
            mx.eval(latents)
            _emit_progress(
                progress_callback,
                PipelineProgressEvent(
                    kind="denoise_compute_end",
                    stage="denoise",
                    message="Denoising complete",
                    progress=0.77,
                    step_count=len(sigmas),
                ),
            )

        if benchmark_recorder is None:
            build_denoise_graph()
            run_denoise_compute()
        else:
            with benchmark_recorder.stage("denoise_graph_build"):
                build_denoise_graph()
            with benchmark_recorder.stage("denoise_eval"):
                run_denoise_compute()
        return latents


def _expand_prompt_batch(
    prompts: Sequence[str],
    num_images_per_prompt: int,
) -> list[tuple[str, int, int]]:
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    expanded = []
    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(num_images_per_prompt):
            expanded.append((prompt, prompt_index, image_index))
    return expanded


def _repeat_encoding(
    encoding: InstructionEncoding,
    num_images_per_prompt: int,
) -> InstructionEncoding:
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be positive")
    if num_images_per_prompt == 1:
        return encoding
    mx = _load_mlx()
    return InstructionEncoding(
        hidden_states=mx.repeat(
            encoding.hidden_states,
            num_images_per_prompt,
            axis=0,
        ),
        attention_mask=mx.repeat(
            encoding.attention_mask,
            num_images_per_prompt,
            axis=0,
        ),
        input_ids=mx.repeat(
            encoding.input_ids,
            num_images_per_prompt,
            axis=0,
        ),
    )


def _slice_encoding(
    encoding: InstructionEncoding,
    start: int,
    end: int,
) -> InstructionEncoding:
    return InstructionEncoding(
        hidden_states=encoding.hidden_states[start:end],
        attention_mask=encoding.attention_mask[start:end],
        input_ids=encoding.input_ids[start:end],
    )


def _slice_optional_sequence(
    values: Sequence[int] | None,
    start: int,
    end: int,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    return tuple(int(value) for value in values[start:end])


def _chunk_ranges(batch_size: int, chunk_size: int | None) -> Iterator[tuple[int, int]]:
    if chunk_size is None:
        yield 0, batch_size
        return
    for start in range(0, batch_size, chunk_size):
        yield start, min(start + chunk_size, batch_size)


def _validate_optional_batch_size(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def _token_counts_by_batch_item(attention_mask: Any) -> list[int]:
    mx = _load_mlx()
    return [int(value) for value in mx.sum(attention_mask, axis=1).tolist()]


def _normalize_generation_seeds(
    *,
    seed: int | None,
    seeds: Sequence[int] | None,
    batch_size: int,
) -> tuple[int, ...] | None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seed is not None and seeds is not None:
        raise ValueError("provide either seed or seeds, not both")
    if seeds is not None:
        if isinstance(seeds, (str, bytes)):
            raise ValueError("seeds must be a sequence of integers, not a string")
        values = tuple(
            _validate_generation_seed(value, label="seed")
            for value in seeds
        )
        if len(values) != batch_size:
            raise ValueError(
                f"seeds length must match batch size {batch_size}, got {len(values)}"
            )
        return values
    if seed is None:
        return None
    base_seed = _validate_generation_seed(seed, label="seed")
    return tuple(
        _validate_generation_seed(base_seed + index, label="expanded seed")
        for index in range(batch_size)
    )


def _validate_generation_dimensions(height: int, width: int) -> tuple[int, int]:
    return (
        _validate_generation_dimension(height, "height"),
        _validate_generation_dimension(width, "width"),
    )


def _validate_generation_dimension(value: int, label: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if integer <= 0:
        raise ValueError(f"{label} must be positive")
    if integer > MAX_GENERATION_SIZE:
        raise ValueError(f"{label} must be {MAX_GENERATION_SIZE} or smaller")
    if integer % OUTPUT_ALIGNMENT:
        raise ValueError(f"{label} must be a multiple of {OUTPUT_ALIGNMENT}")
    return integer


def _validate_generation_seed(value: int, *, label: str) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if integer < 0 or integer > MAX_GENERATION_SEED:
        raise ValueError(
            f"{label} must be an integer from 0 to {MAX_GENERATION_SEED}"
        )
    return integer


def _denoise_size_for_request(
    height: int,
    width: int,
    *,
    max_pixels: int = MAX_DENOISE_PIXELS,
    alignment: int = OUTPUT_ALIGNMENT,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    ratio = min((max_pixels / (height * width)) ** 0.5, 1.0)
    denoise_height = int(height * ratio) // alignment * alignment
    denoise_width = int(width * ratio) // alignment * alignment
    if denoise_height == 0 or denoise_width == 0:
        raise ValueError(
            f"height and width must be at least {alignment} after alignment"
        )
    return denoise_height, denoise_width


def _resize_decoded_to_request(decoded: Any, height: int, width: int) -> Any:
    if decoded.ndim != 4:
        raise ValueError("decoded image tensor must be shaped [B, C, H, W]")
    if decoded.shape[2] == height and decoded.shape[3] == width:
        return decoded
    return _bilinear_resize_nchw(decoded, height, width)


def _bilinear_resize_nchw(image: Any, height: int, width: int) -> Any:
    if height <= 0 or width <= 0:
        raise ValueError("resize height and width must be positive")
    if image.ndim != 4:
        raise ValueError("image must be shaped [B, C, H, W]")
    nhwc = image.transpose(0, 2, 3, 1)
    nhwc = _resize_axis_nhwc(nhwc, height, axis=1)
    nhwc = _resize_axis_nhwc(nhwc, width, axis=2)
    return nhwc.transpose(0, 3, 1, 2)


def _resize_axis_nhwc(image: Any, out_size: int, *, axis: int) -> Any:
    mx = _load_mlx()
    in_size = int(image.shape[axis])
    if out_size == in_size:
        return image

    positions = (
        (mx.arange(out_size, dtype=mx.float32) + 0.5)
        * (float(in_size) / float(out_size))
        - 0.5
    )
    positions = mx.clip(positions, 0.0, float(in_size - 1))
    lower = mx.floor(positions).astype(mx.int32)
    upper = mx.minimum(lower + 1, in_size - 1).astype(mx.int32)
    weight = positions - lower.astype(mx.float32)

    lower_values = mx.take(image, lower, axis=axis)
    upper_values = mx.take(image, upper, axis=axis)
    if axis == 1:
        weight = weight.reshape(1, out_size, 1, 1)
    elif axis == 2:
        weight = weight.reshape(1, 1, out_size, 1)
    else:
        raise ValueError("axis must be 1 or 2 for NHWC spatial resize")
    return lower_values * (1.0 - weight) + upper_values * weight


def _postprocess_to_pil(image: Any) -> "Image.Image":
    return _postprocess_batch_to_pil(image)[0]


def _postprocess_batch_to_pil(image: Any) -> list["Image.Image"]:
    mx = _load_mlx()
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise BooguTurboMlxError(
            "Image generation requires numpy>=1.26 and pillow>=10. Install "
            "`boogu-turbo-mlx[runtime]`."
        ) from exc

    if image.ndim != 4:
        raise ValueError("image must be shaped [B, C, H, W]")
    if image.shape[0] < 1:
        raise ValueError("image batch must not be empty")
    if image.shape[1] not in {1, 3, 4}:
        raise ValueError("image channel count must be 1, 3, or 4 for PIL output")

    image = image.astype(mx.float32)
    mx.eval(image)
    array = np.array(image)
    array = np.clip(array / 2.0 + 0.5, 0.0, 1.0)
    array = np.transpose(array, (0, 2, 3, 1))
    array = (array * 255).round().astype("uint8")
    images = []
    for item in array:
        if item.shape[-1] == 1:
            images.append(Image.fromarray(item.squeeze(-1), mode="L"))
        else:
            images.append(Image.fromarray(item))
    return images


def _validate_artifact_root(root: Path) -> None:
    if not root.exists():
        raise BooguTurboMlxError(f"Model artifact directory does not exist: {root}")
    if not root.is_dir():
        raise BooguTurboMlxError(f"Model artifact path must be a directory: {root}")

    artifact_path = root / "artifact.json"
    if not artifact_path.exists():
        raise BooguTurboMlxError(
            f"Missing required model artifact file: {artifact_path}"
        )
    artifact = read_artifact(root, "model")

    required = [
        component_config_path(root, artifact, "mllm"),
        processor_path(root, artifact),
        component_config_path(root, artifact, "transformer"),
        component_config_path(root, artifact, "vae"),
        component_weights_index_path(root, artifact, "mllm"),
        component_weights_index_path(root, artifact, "transformer"),
        component_weights_index_path(root, artifact, "vae"),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise BooguTurboMlxError(
            "Model artifact is missing required files: "
            + ", ".join(str(path) for path in missing[:8])
        )


def _normalize_timesteps(timesteps: Sequence[float] | None) -> list[float] | None:
    if timesteps is None:
        return None
    if isinstance(timesteps, (str, bytes)):
        raise ValueError("timesteps must be a sequence of floats, not a string")
    return [float(timestep) for timestep in timesteps]


def _split_noise_keys(seed: int | None, count: int) -> Any | None:
    if seed is None:
        return None
    mx = _load_mlx()
    return mx.random.split(mx.random.key(int(seed)), count)


def _split_noise_keys_for_batch(
    seeds: Sequence[int] | None,
    count: int,
) -> list[Any] | None:
    if seeds is None:
        return None
    return [_split_noise_keys(seed, count) for seed in seeds]


def _normal(shape: tuple[int, ...], *, dtype: Any, key: Any | None) -> Any:
    mx = _load_mlx()
    if key is None:
        return mx.random.normal(shape, dtype=dtype)
    return mx.random.normal(shape, dtype=dtype, key=key)


def _normal_batch(
    shape: tuple[int, ...],
    *,
    dtype: Any,
    keys: Sequence[Any] | None,
) -> Any:
    if keys is None:
        return _normal(shape, dtype=dtype, key=None)
    if len(keys) != shape[0]:
        raise ValueError(f"keys length must match batch size {shape[0]}")
    mx = _load_mlx()
    item_shape = (1, *shape[1:])
    return mx.concatenate(
        [_normal(item_shape, dtype=dtype, key=key) for key in keys],
        axis=0,
    )


def _eval_mlx_values(values: Any) -> None:
    mx = _load_mlx()
    if isinstance(values, (list, tuple)):
        mx.eval(*values)
    else:
        mx.eval(values)


def _clear_runtime_cache() -> None:
    gc.collect()
    try:
        mx = _load_mlx()
    except BooguTurboMlxError:
        return
    clear_cache = getattr(mx, "clear_cache", None)
    if clear_cache is None:
        clear_cache = getattr(getattr(mx, "metal", None), "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _validate_memory_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in VALID_MEMORY_MODES:
        allowed = ", ".join(sorted(item.replace("_", "-") for item in VALID_MEMORY_MODES))
        raise BooguTurboMlxError(
            f"memory_mode must be one of {allowed}; got {mode!r}"
        )
    return normalized


@contextmanager
def _maybe_benchmark_stage(
    recorder: _BenchmarkRecorder | None,
    name: str,
) -> Iterator[None]:
    if recorder is None:
        yield
        return
    with recorder.stage(name):
        yield


def _reset_peak_memory(mx: Any) -> None:
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()


def _memory_snapshot(mx: Any) -> tuple[int | None, int | None, int | None]:
    return (
        _call_optional_int(getattr(mx, "get_active_memory", None)),
        _call_optional_int(getattr(mx, "get_peak_memory", None)),
        _call_optional_int(getattr(mx, "get_cache_memory", None)),
    )


def _call_optional_int(function: Any) -> int | None:
    if not callable(function):
        return None
    return int(function())


def _max_optional_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _emit_progress(
    progress_callback: PipelineProgressCallback | None,
    event: PipelineProgressEvent,
) -> None:
    if progress_callback is None:
        return
    progress_callback(event)


def _emit_terminal_progress(
    progress_callback: PipelineProgressCallback | None,
    event: PipelineProgressEvent,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception:
        pass


def _load_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise BooguTurboMlxError(
            "The Boogu Turbo pipeline requires MLX. Install "
            "`boogu-turbo-mlx[runtime]` on an MLX-supported machine."
        ) from exc
    return mx
