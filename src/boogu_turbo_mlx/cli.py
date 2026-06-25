from __future__ import annotations

import argparse
import json
import secrets
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

from . import __version__
from .constants import (
    DEFAULT_DMD_CONDITIONING_SIGMA,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_STEPS,
    MAX_GENERATION_SEED,
    OFFICIAL_TURBO_HF_ID,
)
from .conversion import convert_weights
from .dmd import sigma_schedule
from .doctor import format_doctor_report, run_doctor
from .errors import BooguTurboMlxError
from .hf import download_source
from .manifest import generate_manifest
from .pipeline import (
    BooguTurboGenerationResult,
    BooguTurboPipeline,
    PipelineProgressEvent,
    _BenchmarkRecorder,
    _validate_generation_dimensions,
)
from .png import (
    PNG_METADATA_KEY as _PNG_METADATA_KEY,
    PNG_PARAMETERS_KEY as _PNG_PARAMETERS_KEY,
    save_generation_png,
)
from .quantization import Q8_BITS, Q8_GROUP_SIZE, Q8_MODE, quantize_artifact

PNG_METADATA_KEY = _PNG_METADATA_KEY
PNG_PARAMETERS_KEY = _PNG_PARAMETERS_KEY
RANDOM_SEED_BITS = 32
DEFAULT_OUTPUT_TEMPLATE = (
    "image-{batch_index:04d}-p{prompt_index:02d}-i{image_index:02d}-seed{seed}.png"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boogu-turbo-mlx",
        description="MLX-native Boogu Image 0.1 Turbo tooling.",
    )
    parser.add_argument("--version", action="store_true", help="Print the package version.")

    subparsers = parser.add_subparsers(dest="command")

    setup = subparsers.add_parser(
        "setup",
        help="Configure, download, convert, and validate a local Boogu Turbo artifact.",
    )
    setup.add_argument(
        "--config",
        type=Path,
        default=Path(".boogu-turbo-mlx") / "config.json",
        help="Project-local setup config path.",
    )
    setup.add_argument(
        "--source-repo",
        help="Hugging Face source repo to download when the source folder is absent.",
    )
    setup.add_argument(
        "--source-revision",
        help=(
            "Pinned Hugging Face revision for --source-repo. The official source "
            "uses the audited revision by default."
        ),
    )
    setup.add_argument(
        "--source-dir",
        type=Path,
        help="Local official source folder.",
    )
    setup.add_argument(
        "--artifact",
        choices=("bf16", "q8"),
        help="Artifact to prepare; defaults to bf16.",
    )
    setup.add_argument(
        "--bf16-output",
        type=Path,
        help="Converted bf16 MLX artifact folder.",
    )
    setup.add_argument(
        "--q8-output",
        type=Path,
        help="Optional q8 MLX artifact folder.",
    )
    setup.add_argument(
        "--q8-intermediate",
        type=Path,
        help="Temporary bf16 artifact folder used while preparing q8.",
    )
    setup.add_argument(
        "--output-dir",
        type=Path,
        help="Default generated image folder stored in setup config.",
    )
    setup.add_argument(
        "--memory-mode",
        choices=("resident", "low-memory"),
        help="Default generation memory mode stored in setup config.",
    )
    cleanup = setup.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--cleanup-intermediates",
        dest="cleanup_intermediates",
        action="store_true",
        default=None,
        help="Remove temporary conversion intermediates after validation.",
    )
    cleanup.add_argument(
        "--keep-intermediates",
        dest="cleanup_intermediates",
        action="store_false",
        help="Keep temporary conversion intermediates.",
    )
    source_cleanup = setup.add_mutually_exclusive_group()
    source_cleanup.add_argument(
        "--cleanup-source",
        dest="cleanup_source",
        action="store_true",
        default=None,
        help=(
            "Remove a project-local source folder after artifact validation. "
            "Future new artifacts will need a re-download or an explicit source path."
        ),
    )
    source_cleanup.add_argument(
        "--keep-source",
        dest="cleanup_source",
        action="store_false",
        help="Keep the downloaded source folder after setup.",
    )
    setup.add_argument(
        "--accept-defaults",
        action="store_true",
        help="Use saved/default choices without opening the setup page.",
    )
    setup.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the setup URL without opening a browser.",
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the setup plan without downloading or converting.",
    )
    setup.set_defaults(handler=_run_setup)

    gui = subparsers.add_parser(
        "gui",
        help="Launch the local browser GUI for the configured model.",
    )
    gui.add_argument(
        "--config",
        type=Path,
        default=Path(".boogu-turbo-mlx") / "config.json",
        help="Project-local setup config path.",
    )
    gui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local interface for the GUI server.",
    )
    gui.add_argument(
        "--unsafe-host",
        action="store_true",
        help=(
            "Allow binding the GUI to a non-loopback host. Only use this on "
            "a trusted network."
        ),
    )
    gui.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local port for the GUI server; 0 chooses a free port.",
    )
    gui.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the GUI URL without opening a browser.",
    )
    gui.add_argument(
        "--no-preload",
        action="store_true",
        help="Start the GUI without immediately loading the model into memory.",
    )
    gui.set_defaults(handler=_run_gui)

    manifest = subparsers.add_parser(
        "manifest",
        help="Inspect an official model source without loading large tensors.",
    )
    manifest.add_argument("--source", required=True, help="Official model directory or HF model id.")
    manifest.add_argument(
        "--revision",
        help=(
            "Pinned Hugging Face revision. Defaults to the audited official "
            "Boogu Turbo revision for the official source."
        ),
    )
    manifest.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    manifest.set_defaults(handler=_run_manifest)

    download = subparsers.add_parser(
        "download",
        help="Download the official Boogu Turbo source into models/.",
    )
    download.add_argument(
        "--source",
        default=OFFICIAL_TURBO_HF_ID,
        help="Hugging Face model id or existing local directory.",
    )
    download.add_argument(
        "--revision",
        help=(
            "Pinned Hugging Face revision. Defaults to the audited official "
            "Boogu Turbo revision for the official source."
        ),
    )
    download.add_argument(
        "--dest",
        type=Path,
        default=Path("models"),
        help="Destination root for Hugging Face snapshots.",
    )
    download.set_defaults(handler=_run_download)

    convert = subparsers.add_parser(
        "convert",
        help="Convert official weights into the MLX artifact layout.",
    )
    convert.add_argument(
        "--source",
        required=True,
        help="Official model directory or Hugging Face model id.",
    )
    convert.add_argument(
        "--revision",
        help=(
            "Pinned Hugging Face revision when --source is a model id. Defaults "
            "to the audited official Boogu Turbo revision for the official source."
        ),
    )
    convert.add_argument("--output", required=True, help="MLX artifact output directory.")
    convert.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
        help="Target tensor dtype for converted runtime weights; auto preserves source dtype.",
    )
    convert.set_defaults(handler=_run_convert)

    quantize = subparsers.add_parser(
        "quantize",
        help="Create a persistent MLX-native q8 artifact from a converted artifact.",
    )
    quantize.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Converted MLX artifact directory.",
    )
    quantize.add_argument(
        "--output",
        required=True,
        type=Path,
        help="q8 artifact output directory.",
    )
    quantize.add_argument(
        "--bits",
        type=_parse_q8_bits,
        default=Q8_BITS,
        help="Quantization bits. This release supports only 8.",
    )
    quantize.add_argument(
        "--group-size",
        type=_parse_q8_group_size,
        default=Q8_GROUP_SIZE,
        help="Quantization group size. This release supports only 32.",
    )
    quantize.add_argument(
        "--mode",
        choices=(Q8_MODE,),
        default=Q8_MODE,
        help="Quantization mode. This release supports only affine.",
    )
    quantize.set_defaults(handler=_run_quantize)

    generate = subparsers.add_parser(
        "generate",
        help="Generate an image with a converted MLX artifact.",
    )
    generate.add_argument("--model", required=True, help="Converted MLX artifact directory.")
    generate.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text instruction. Repeat to generate a prompt batch.",
    )
    generate.add_argument("--height", type=_parse_positive_int)
    generate.add_argument("--width", type=_parse_positive_int)
    generate.add_argument(
        "--size",
        help="Output size as WIDTHxHEIGHT, e.g. 512x512.",
    )
    generate.add_argument(
        "--output",
        type=Path,
        help="PNG output path for single-image generation.",
    )
    generate.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch generation.",
    )
    generate.add_argument(
        "--output-template",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help="Batch filename template.",
    )
    generate.add_argument("--steps", type=_parse_positive_int, default=DEFAULT_STEPS)
    generate.add_argument("--seed", type=_parse_seed)
    generate.add_argument(
        "--seeds",
        help="Comma-separated per-image seeds for batch generation.",
    )
    generate.add_argument(
        "--timesteps",
        type=_parse_timesteps,
        help="Optional comma-separated DMD sigmas/timesteps, e.g. 1,250,500.",
    )
    generate.add_argument(
        "--max-sequence-length",
        type=_parse_positive_int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    generate.add_argument(
        "--truncate-instruction-sequence",
        action="store_true",
        help="Allow tokenizer truncation at --max-sequence-length.",
    )
    generate.add_argument(
        "--dmd-conditioning-sigma",
        type=_parse_unit_interval_float,
        default=DEFAULT_DMD_CONDITIONING_SIGMA,
    )
    generate.add_argument(
        "--num-images-per-prompt",
        type=_parse_positive_int,
        default=1,
    )
    generate.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
        help="Progress reporting mode; progress is written to stderr.",
    )
    _add_runtime_generation_arguments(generate)
    generate.set_defaults(handler=_run_generate)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check the local environment and Boogu Turbo artifacts.",
    )
    doctor.add_argument("--model", type=Path, help="Converted or q8 MLX artifact to validate.")
    doctor.add_argument("--source", type=Path, help="Official local source directory to validate.")
    doctor.add_argument("--json", action="store_true", help="Write machine-readable JSON.")
    doctor.set_defaults(handler=_run_doctor)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Measure generation latency and MLX memory for a converted artifact.",
    )
    benchmark.add_argument("--model", required=True, help="Converted MLX artifact directory.")
    benchmark.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Text instruction. Repeat to benchmark prompt-list batching.",
    )
    benchmark.add_argument("--height", type=_parse_positive_int, required=True)
    benchmark.add_argument("--width", type=_parse_positive_int, required=True)
    benchmark.add_argument("--steps", type=_parse_positive_int, default=DEFAULT_STEPS)
    benchmark.add_argument("--seed", type=_parse_seed)
    benchmark.add_argument(
        "--timesteps",
        type=_parse_timesteps,
        help="Optional comma-separated DMD sigmas/timesteps, e.g. 1,250,500.",
    )
    benchmark.add_argument(
        "--max-sequence-length",
        type=_parse_positive_int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    benchmark.add_argument(
        "--truncate-instruction-sequence",
        action="store_true",
        help="Allow tokenizer truncation at --max-sequence-length.",
    )
    benchmark.add_argument(
        "--dmd-conditioning-sigma",
        type=_parse_unit_interval_float,
        default=DEFAULT_DMD_CONDITIONING_SIGMA,
    )
    benchmark.add_argument(
        "--num-images-per-prompt",
        type=_parse_positive_int,
        default=1,
    )
    benchmark.add_argument("--runs", type=_parse_positive_int, default=3)
    benchmark.add_argument("--warmup-runs", type=_parse_non_negative_int, default=1)
    benchmark.add_argument("--json-output", type=Path)
    _add_runtime_generation_arguments(benchmark)
    benchmark.set_defaults(handler=_run_benchmark)

    return parser


def _add_runtime_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-mode",
        choices=("resident", "low-memory"),
        default="resident",
        help="Runtime loading mode.",
    )
    parser.add_argument(
        "--denoise-batch-size",
        type=_parse_positive_int,
        help="Optional batch chunk size for denoising.",
    )
    parser.add_argument(
        "--decode-batch-size",
        type=_parse_positive_int,
        help="Optional batch chunk size for VAE decode/postprocess.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    try:
        return int(args.handler(args) or 0)
    except BooguTurboMlxError as exc:
        print(f"boogu-turbo-mlx: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"boogu-turbo-mlx: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"boogu-turbo-mlx: {exc}", file=sys.stderr)
        return 1


def _run_manifest(args: argparse.Namespace) -> int:
    manifest = generate_manifest(args.source, revision=args.revision)
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _run_setup(args: argparse.Namespace) -> int:
    from .setup_flow import run_setup_cli

    return run_setup_cli(args, stream=sys.stderr)


def _run_gui(args: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(
        config_path=args.config,
        host=args.host,
        port=args.port,
        allow_unsafe_host=args.unsafe_host,
        open_browser=not args.no_browser,
        preload=not args.no_preload,
        stream=sys.stdout,
    )


def _run_download(args: argparse.Namespace) -> int:
    local_path = download_source(
        args.source,
        revision=args.revision,
        dest_root=args.dest,
    )
    print(local_path)
    return 0


def _run_convert(args: argparse.Namespace) -> int:
    source_path = Path(str(args.source)).expanduser()
    source = (
        source_path
        if source_path.exists()
        else download_source(args.source, revision=args.revision)
    )
    convert_weights(source, args.output, dtype=args.dtype)
    return 0


def _run_quantize(args: argparse.Namespace) -> int:
    report = quantize_artifact(
        args.source,
        args.output,
        mode=args.mode,
        bits=args.bits,
        group_size=args.group_size,
    )
    print(args.output)
    for component in ("mllm", "transformer"):
        component_report = report["components"][component]
        source_size = int(component_report["source_disk_size"])
        output_size = int(component_report["output_disk_size"])
        ratio = component_report["disk_size_ratio"]
        ratio_text = "unknown" if ratio is None else f"{ratio:.3f}x"
        print(f"{component}: {source_size} -> {output_size} bytes ({ratio_text})")
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    prompts = _validate_prompts(args.prompt)
    height, width = _resolve_output_size(args)
    seeds = _parse_seed_list(args.seeds) if args.seeds is not None else None
    batch_size = len(prompts) * int(args.num_images_per_prompt)
    if args.seed is not None and seeds is not None:
        raise ValueError("provide either --seed or --seeds, not both")
    if seeds is not None and len(seeds) != batch_size:
        raise ValueError(f"--seeds must provide {batch_size} seed(s), got {len(seeds)}")
    seed = None if seeds is not None else _resolve_generation_seed(
        args.seed,
        batch_size=batch_size,
    )
    single_mode = len(prompts) == 1 and int(args.num_images_per_prompt) == 1
    _validate_generation_outputs(args, single_mode=single_mode)
    progress_callback = _progress_callback(args.progress)

    pipeline = BooguTurboPipeline.from_pretrained(
        args.model,
        memory_mode=args.memory_mode,
    )
    generation_kwargs = {
        "height": height,
        "width": width,
        "steps": args.steps,
        "seed": seed,
        "seeds": seeds,
        "num_images_per_prompt": args.num_images_per_prompt,
        "timesteps": args.timesteps,
        "max_sequence_length": args.max_sequence_length,
        "truncate_instruction_sequence": args.truncate_instruction_sequence,
        "dmd_conditioning_sigma": args.dmd_conditioning_sigma,
        "progress_callback": progress_callback,
    }
    if args.denoise_batch_size is not None:
        generation_kwargs["denoise_batch_size"] = args.denoise_batch_size
    if args.decode_batch_size is not None:
        generation_kwargs["decode_batch_size"] = args.decode_batch_size

    prompt_or_prompts = prompts[0] if len(prompts) == 1 else prompts
    batch = pipeline.generate_batch(
        prompt_or_prompts,
        **generation_kwargs,
    )
    warned_sizes: set[tuple[int, int, int, int]] = set()
    output_paths: list[Path] = []
    for result in batch:
        output_path = (
            args.output
            if single_mode
            else _batch_output_path(args.output_dir, args.output_template, result)
        )
        save_generation_png(
            result,
            output_path,
            model=args.model,
            memory_mode=args.memory_mode,
            denoise_batch_size=args.denoise_batch_size,
            decode_batch_size=args.decode_batch_size,
        )
        _warn_if_denoise_size_changed(result, warned_sizes)
        output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path)
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(model=args.model, source=args.source)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_report(report), end="")
    return 1 if report["error_count"] else 0


def _run_benchmark(args: argparse.Namespace) -> int:
    args.seed = _resolve_generation_seed(
        args.seed,
        batch_size=len(args.prompt) * int(args.num_images_per_prompt),
    )
    args.height, args.width = _validate_cli_size(int(args.height), int(args.width))
    pipeline = BooguTurboPipeline.from_pretrained(
        args.model,
        memory_mode=args.memory_mode,
    )
    prompts = args.prompt[0] if len(args.prompt) == 1 else list(args.prompt)
    measured_runs: list[dict[str, Any]] = []
    for run_index in range(args.warmup_runs + args.runs):
        recorder = _BenchmarkRecorder()
        mx_module = _try_load_mlx_core()
        _reset_peak_memory(mx_module)
        start = time.perf_counter()
        batch = pipeline.generate_batch(
            prompts,
            height=args.height,
            width=args.width,
            steps=args.steps,
            seed=args.seed,
            num_images_per_prompt=args.num_images_per_prompt,
            timesteps=args.timesteps,
            max_sequence_length=args.max_sequence_length,
            truncate_instruction_sequence=args.truncate_instruction_sequence,
            dmd_conditioning_sigma=args.dmd_conditioning_sigma,
            denoise_batch_size=args.denoise_batch_size,
            decode_batch_size=args.decode_batch_size,
            _benchmark_recorder=recorder,
        )
        total_wall_time = time.perf_counter() - start
        active_memory, peak_memory, cache_memory = _memory_snapshot(mx_module)
        # Each benchmark stage resets MLX's peak counter on entry, so the global
        # peak read above only reflects the final (often lightweight) stage. Use
        # the max of the per-stage peaks for the true run-wide peak.
        run_peak_memory = recorder.run_peak_memory_bytes()
        if run_peak_memory is not None:
            peak_memory = (
                run_peak_memory
                if peak_memory is None
                else max(peak_memory, run_peak_memory)
            )
        if run_index < args.warmup_runs:
            continue
        measured_runs.append(
            {
                "run_index": run_index - args.warmup_runs,
                "image_count": len(batch),
                "total": {
                    "wall_time_seconds": total_wall_time,
                    "active_memory_bytes": active_memory,
                    "peak_memory_bytes": peak_memory,
                    "cache_memory_bytes": cache_memory,
                    "calls": 1,
                },
                "stages": recorder.to_dict(),
            }
        )
    payload = _benchmark_payload(args, measured_runs)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
        print(args.json_output)
    else:
        print(text)
    return 0


def _resolve_generation_seed(seed: int | None, *, batch_size: int = 1) -> int:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if seed is not None:
        base_seed = _seed_value(str(seed))
        if base_seed + batch_size - 1 > MAX_GENERATION_SEED:
            raise ValueError(
                "seed plus batch expansion must stay within 0 to "
                f"{MAX_GENERATION_SEED}"
            )
        return base_seed
    if batch_size == 1:
        return secrets.randbits(RANDOM_SEED_BITS)
    return secrets.randbelow(MAX_GENERATION_SEED - batch_size + 2)


def _validate_prompts(values: Sequence[str]) -> list[str]:
    prompts = [str(value) for value in values]
    if any(not prompt for prompt in prompts):
        raise ValueError("prompt must be a non-empty string")
    return prompts


def _resolve_output_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.size is not None:
        if args.height is not None or args.width is not None:
            raise ValueError("provide either --size or --height/--width, not both")
        return _validate_cli_size(*_parse_size(args.size))
    if args.height is None or args.width is None:
        raise ValueError("provide --size WIDTHxHEIGHT or both --height and --width")
    return _validate_cli_size(int(args.height), int(args.width))


def _parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--size must be WIDTHxHEIGHT, e.g. 512x512")
    try:
        width = _parse_positive_int(parts[0])
        height = _parse_positive_int(parts[1])
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"--size values {exc}") from exc
    return height, width


def _parse_seed_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        raise ValueError("--seeds must not be empty")
    try:
        return tuple(_seed_value(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            f"--seeds must be a comma-separated list of integers from 0 to {MAX_GENERATION_SEED}"
        ) from exc


def _validate_cli_size(height: int, width: int) -> tuple[int, int]:
    try:
        return _validate_generation_dimensions(height, width)
    except ValueError as exc:
        raise ValueError(f"output size {exc}") from exc


def _validate_generation_outputs(args: argparse.Namespace, *, single_mode: bool) -> None:
    if single_mode:
        if args.output is None:
            raise ValueError("--output is required for single-image generation")
        if args.output_dir is not None:
            raise ValueError("--output-dir is only valid for batch generation")
        return
    if args.output is not None:
        raise ValueError("--output is only valid for single-image generation")
    if args.output_dir is None:
        raise ValueError("--output-dir is required for batch generation")
    _validate_output_template(args.output_template)


def _validate_output_template(template: str) -> None:
    path = Path(template)
    if not template or path.is_absolute() or "/" in template or "\\" in template:
        raise ValueError("--output-template must be a filename template, not a path")


def _batch_output_path(
    output_dir: Path,
    template: str,
    result: BooguTurboGenerationResult,
) -> Path:
    try:
        filename = template.format(
            batch_index=result.batch_index,
            prompt_index=result.prompt_index,
            image_index=result.image_index,
            seed=result.seed if result.seed is not None else "none",
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid --output-template: {exc}") from exc
    _validate_output_template(filename)
    return output_dir / filename


def _progress_callback(mode: str) -> _CliProgressReporter | None:
    if mode == "never":
        return None
    if mode == "auto" and not sys.stderr.isatty():
        return None
    return _CliProgressReporter(sys.stderr)


def _warn_if_denoise_size_changed(
    result: BooguTurboGenerationResult,
    warned_sizes: set[tuple[int, int, int, int]],
) -> None:
    key = (
        int(result.denoise_width),
        int(result.denoise_height),
        int(result.output_width),
        int(result.output_height),
    )
    if key in warned_sizes:
        return
    warned_sizes.add(key)
    if (result.denoise_height, result.denoise_width) == (
        result.output_height,
        result.output_width,
    ):
        return
    print(
        "boogu-turbo-mlx: denoising at "
        f"{result.denoise_width}x{result.denoise_height} and resizing to "
        f"{result.output_width}x{result.output_height}",
        file=sys.stderr,
    )


class _CliProgressReporter:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.is_tty = stream.isatty()
        self._last_width = 0

    def __call__(self, event: PipelineProgressEvent) -> None:
        line = self._format_event(event)
        if not line:
            return
        terminal = event.kind in {"complete", "cancelled", "error"}
        if self.is_tty:
            padding = " " * max(self._last_width - len(line), 0)
            end = "\n" if terminal else ""
            self.stream.write(f"\r{line}{padding}{end}")
            self._last_width = 0 if terminal else len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def _format_event(self, event: PipelineProgressEvent) -> str:
        component = event.details.get("component")
        if event.kind == "load_start":
            return "[load] validating artifact"
        if event.kind == "load_component_start":
            return f"[load] {component or event.message}"
        if event.kind == "load_component_end":
            return f"[load] {component or event.message} ready"
        if event.kind == "load_complete":
            return "[load] ready"
        if event.kind == "encode_start":
            return "[encode] prompts"
        if event.kind == "encode_end":
            count = event.details.get("token_count")
            return "[encode] ready" if count is None else f"[encode] {count} tokens"
        if event.kind == "denoise_step_start":
            step = 0 if event.step_index is None else event.step_index + 1
            total = event.step_count or "?"
            sigma = "" if event.sigma is None else f" sigma={event.sigma:.6g}"
            return f"[denoise {step}/{total}]{sigma}"
        if event.kind == "denoise_compute_start":
            return "[denoise] running"
        if event.kind == "denoise_compute_end":
            return "[denoise] ready"
        if event.kind == "decode_start":
            return "[decode] latents"
        if event.kind == "decode_end":
            return "[decode] ready"
        if event.kind == "resize_start":
            return "[resize] output"
        if event.kind == "resize_end":
            return "[resize] ready"
        if event.kind == "complete":
            return "[complete] generation finished"
        if event.kind == "cancelled":
            return "[cancelled] generation cancelled"
        if event.kind == "error":
            return f"[error] {event.message}"
        return ""


def _benchmark_payload(
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    timesteps = (
        args.timesteps if args.timesteps is None else [float(t) for t in args.timesteps]
    )
    return {
        "schema_version": 1,
        "generator": "boogu-turbo-mlx",
        "generator_version": __version__,
        "model": str(args.model),
        "prompts": list(args.prompt),
        "height": int(args.height),
        "width": int(args.width),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "timesteps": timesteps,
        "sigmas": sigma_schedule(
            int(args.steps),
            float(args.dmd_conditioning_sigma),
            timesteps=timesteps,
        ),
        "max_sequence_length": int(args.max_sequence_length),
        "truncate_instruction_sequence": bool(args.truncate_instruction_sequence),
        "dmd_conditioning_sigma": float(args.dmd_conditioning_sigma),
        "num_images_per_prompt": int(args.num_images_per_prompt),
        "memory_mode": args.memory_mode,
        "denoise_batch_size": args.denoise_batch_size,
        "decode_batch_size": args.decode_batch_size,
        "warmup_runs": int(args.warmup_runs),
        "runs": runs,
        "summary": _benchmark_summary(runs),
    }


def _benchmark_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"run_count": 0, "median_wall_time_seconds": {}, "stage_calls": {}}
    stage_names = sorted(
        {
            stage_name
            for run in runs
            for stage_name in run.get("stages", {})
        }
    )
    medians = {
        "total": statistics.median(
            float(run["total"]["wall_time_seconds"]) for run in runs
        )
    }
    stage_calls: dict[str, int] = {}
    for stage_name in stage_names:
        values = [
            float(run["stages"][stage_name]["wall_time_seconds"])
            for run in runs
            if stage_name in run["stages"]
        ]
        if values:
            medians[stage_name] = statistics.median(values)
            stage_calls[stage_name] = max(
                int(run["stages"][stage_name]["calls"])
                for run in runs
                if stage_name in run["stages"]
            )
    return {
        "run_count": len(runs),
        "median_wall_time_seconds": medians,
        "stage_calls": stage_calls,
    }


def _try_load_mlx_core() -> Any | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx


def _reset_peak_memory(mx_module: Any | None) -> None:
    if mx_module is None:
        return
    reset_peak_memory = getattr(mx_module, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()


def _memory_snapshot(mx_module: Any | None) -> tuple[int | None, int | None, int | None]:
    if mx_module is None:
        return None, None, None
    return (
        _call_optional_int(getattr(mx_module, "get_active_memory", None)),
        _call_optional_int(getattr(mx_module, "get_peak_memory", None)),
        _call_optional_int(getattr(mx_module, "get_cache_memory", None)),
    )


def _call_optional_int(function: Any) -> int | None:
    if not callable(function):
        return None
    return int(function())


def _parse_timesteps(value: str) -> list[float]:
    if not value.strip():
        raise argparse.ArgumentTypeError("--timesteps must not be empty")
    try:
        timesteps = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--timesteps must be a comma-separated list of numbers"
        ) from exc
    if not timesteps:
        raise argparse.ArgumentTypeError("--timesteps must not be empty")
    return timesteps


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_seed(value: str) -> int:
    try:
        return _seed_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _seed_value(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("must be an integer") from exc
    if parsed < 0 or parsed > MAX_GENERATION_SEED:
        raise ValueError(f"must be from 0 to {MAX_GENERATION_SEED}")
    return parsed


def _parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parse_unit_interval_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _parse_q8_bits(value: str) -> int:
    parsed = _parse_positive_int(value)
    if parsed != Q8_BITS:
        raise argparse.ArgumentTypeError(f"must be {Q8_BITS}")
    return parsed


def _parse_q8_group_size(value: str) -> int:
    parsed = _parse_positive_int(value)
    if parsed != Q8_GROUP_SIZE:
        raise argparse.ArgumentTypeError(f"must be {Q8_GROUP_SIZE}")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
