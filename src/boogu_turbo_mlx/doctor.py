from __future__ import annotations

import importlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._version import __version__
from .artifacts import (
    component_config_path,
    component_weights_index_path,
    processor_path,
    read_artifact,
)
from .errors import BooguTurboMlxError
from .manifest import _summarize_safetensors_index, generate_manifest
from .safetensors_header import read_safetensors_header


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def run_doctor(
    *,
    model: str | Path | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    checks: list[DoctorCheck] = []
    checks.extend(_environment_checks())
    if model is not None:
        checks.extend(_model_checks(Path(model).expanduser()))
    if source is not None:
        checks.extend(_source_checks(Path(source).expanduser()))
    error_count = sum(1 for check in checks if check.status == "error")
    warning_count = sum(1 for check in checks if check.status == "warning")
    return {
        "schema_version": 1,
        "generator": "boogu-turbo-mlx",
        "generator_version": __version__,
        "status": "error" if error_count else "ok",
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": [check.to_dict() for check in checks],
    }


def format_doctor_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"boogu-turbo-mlx doctor: {report.get('status', 'unknown')}",
        f"errors: {report.get('error_count', 0)}, warnings: {report.get('warning_count', 0)}",
    ]
    for check in report.get("checks", []):
        status = str(check.get("status", "unknown")).upper()
        lines.append(f"{status}: {check.get('name')}: {check.get('message')}")
    return "\n".join(lines) + "\n"


def _environment_checks() -> list[DoctorCheck]:
    checks = [
        _ok(
            "python",
            f"{platform.python_implementation()} {platform.python_version()}",
            minimum="3.10",
        )
        if sys.version_info >= (3, 10)
        else _error(
            "python",
            f"Python 3.10 or newer is required; found {platform.python_version()}",
        ),
        _ok("package", f"boogu-turbo-mlx {__version__}"),
    ]

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        checks.append(_ok("platform", "macOS on Apple Silicon"))
    else:
        checks.append(
            _warning(
                "platform",
                f"MLX generation is intended for macOS on Apple Silicon; found {system} {machine}",
            )
        )

    checks.append(_module_check("mlx.core", "mlx", required_for="runtime"))
    for module_name, label, required_for, missing_status in (
        ("PIL", "pillow", "optional path", "warning"),
        ("numpy", "numpy", "optional path", "warning"),
        ("transformers", "transformers", "optional path", "warning"),
        ("jinja2", "jinja2", "generation", "error"),
        ("safetensors", "safetensors", "optional path", "warning"),
        ("huggingface_hub", "huggingface-hub", "optional path", "warning"),
    ):
        checks.append(
            _module_check(
                module_name,
                label,
                required_for=required_for,
                missing_status=missing_status,
            )
        )
    return checks


def _module_check(
    module_name: str,
    label: str,
    *,
    required_for: str,
    missing_status: str = "warning",
) -> DoctorCheck:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        message = f"{label} is not importable ({required_for})"
        if missing_status == "error":
            return _error(label, message)
        return _warning(label, message)
    except Exception as exc:  # pragma: no cover - depends on third-party import side effects.
        return _warning(label, f"{label} import failed ({required_for}): {exc}")
    version = getattr(module, "__version__", None)
    message = label if version is None else f"{label} {version}"
    return _ok(label, message)


def _model_checks(root: Path) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if not root.exists():
        return [_error("model", f"Model artifact directory does not exist: {root}")]
    if not root.is_dir():
        return [_error("model", f"Model artifact path must be a directory: {root}")]

    try:
        artifact = read_artifact(root, "model")
    except BooguTurboMlxError as exc:
        return [_error("model", str(exc))]

    checks.append(_ok("model.artifact", f"Artifact metadata found at {root / 'artifact.json'}"))
    required_paths = [
        ("processor", processor_path(root, artifact)),
        (
            "scheduler",
            root / str(artifact.get("scheduler_config", "scheduler/scheduler_config.json")),
        ),
    ]
    for component in ("mllm", "transformer", "vae"):
        required_paths.append((f"{component}.config", component_config_path(root, artifact, component)))
        required_paths.append((f"{component}.weights_index", component_weights_index_path(root, artifact, component)))

    missing = [str(path) for _, path in required_paths if not path.exists()]
    if missing:
        checks.append(_error("model.required_files", "Missing required artifact files", missing=missing))
    else:
        checks.append(_ok("model.required_files", "Required artifact files are present"))

    for component in ("mllm", "transformer", "vae"):
        index_path = component_weights_index_path(root, artifact, component)
        checks.extend(_artifact_index_checks(root, component, index_path))

    quantization = artifact.get("quantization")
    if isinstance(quantization, Mapping):
        checks.append(
            _ok(
                "model.quantization",
                "q8 quantization metadata detected",
                mode=quantization.get("mode"),
                bits=quantization.get("bits"),
                group_size=quantization.get("group_size"),
            )
        )
    else:
        checks.append(_ok("model.quantization", "No persistent quantization metadata"))
    return checks


def _artifact_index_checks(root: Path, component: str, index_path: Path) -> list[DoctorCheck]:
    if not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [_error(f"model.{component}.index", f"Invalid weight index: {exc}")]
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return [_error(f"model.{component}.index", "Weight index has no object weight_map")]

    headers = _headers_for_weight_map(root, index_path, weight_map)
    header_errors = [f"{item.get('path')}: {item.get('error')}" for item in headers if item.get("error")]
    if header_errors:
        return [
            _error(
                f"model.{component}.weights",
                "Weight shards have invalid safetensors headers",
                errors=header_errors[:8],
            )
        ]

    summary = _summarize_safetensors_index(index_path, root, headers)
    validation = summary.get("validation", {})
    if validation.get("status") != "ok":
        return [
            _error(
                f"model.{component}.index",
                "Weight index does not match shard headers",
                component=component,
                **validation,
            )
        ]
    return [
        _ok(
            f"model.{component}.index",
            f"{summary.get('tensor_count', len(weight_map))} tensors across "
            f"{summary.get('shard_count', len(headers))} shard(s)",
        )
    ]


def _headers_for_weight_map(
    root: Path,
    index_path: Path,
    weight_map: Mapping[str, Any],
) -> list[dict[str, Any]]:
    headers = []
    for shard in sorted({str(value) for value in weight_map.values()}):
        path = index_path.parent / shard
        rel = _relative(path, root)
        summary: dict[str, Any] = {"path": rel, "shard": shard}
        try:
            header = read_safetensors_header(path)
        except BooguTurboMlxError as exc:
            summary["error"] = str(exc)
            headers.append(summary)
            continue
        summary.update(
            {
                "metadata": header.metadata,
                "tensor_count": len(header.tensors),
                "tensors": sorted(header.tensors),
                "total_tensor_bytes": sum(tensor.byte_count for tensor in header.tensors.values()),
            }
        )
        headers.append(summary)
    return headers


def _source_checks(source: Path) -> list[DoctorCheck]:
    if not source.exists():
        return [_error("source", f"Source path does not exist: {source}; doctor does not download")]
    if not source.is_dir():
        return [_error("source", f"Source path must be a directory: {source}")]
    try:
        manifest = generate_manifest(source)
    except BooguTurboMlxError as exc:
        return [_error("source", str(exc))]

    checks: list[DoctorCheck] = [_ok("source.manifest", "Local source manifest generated")]
    required = [
        source / "processor",
        source / "mllm" / "config.json",
        source / "transformer" / "config.json",
        source / "vae" / "config.json",
        source / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        checks.append(_error("source.required_files", "Source is missing runtime metadata", missing=missing))
    else:
        checks.append(_ok("source.required_files", "Required source metadata is present"))

    header_errors = [
        f"{header.get('path')}: {header.get('error')}"
        for header in manifest.get("safetensors_headers", [])
        if header.get("error")
    ]
    if header_errors:
        checks.append(_error("source.safetensors", "Invalid safetensors headers", errors=header_errors[:8]))
    else:
        checks.append(_ok("source.safetensors", "Safetensors headers are readable"))

    index_errors = [
        str(index.get("path"))
        for index in manifest.get("safetensors_indexes", [])
        if index.get("validation", {}).get("status") == "error"
    ]
    if index_errors:
        checks.append(_error("source.indexes", "Safetensors indexes do not match headers", indexes=index_errors))
    else:
        checks.append(_ok("source.indexes", "Safetensors indexes match readable headers"))
    return checks


def _ok(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "ok", message, details or None)


def _warning(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "warning", message, details or None)


def _error(name: str, message: str, **details: Any) -> DoctorCheck:
    return DoctorCheck(name, "error", message, details or None)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
