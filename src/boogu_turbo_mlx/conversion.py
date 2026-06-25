from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_write import (
    atomic_output_dir,
    ensure_empty_or_missing_output_dir,
    preflight_free_space,
)
from .errors import BooguTurboMlxError
from .manifest import SCHEMA_VERSION, generate_manifest
from .selection import RUNTIME_WEIGHT_COMPONENTS
from .safetensors_header import DTYPE_BYTE_SIZES


SUPPORTED_DTYPES = ("auto", "bfloat16", "float16", "float32")


@dataclass(frozen=True)
class ConversionRequest:
    source: Path
    output: Path
    dtype: str = "auto"


def convert_weights(
    source: str | Path,
    output: str | Path,
    *,
    dtype: str = "auto",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    request = ConversionRequest(Path(source).expanduser(), Path(output).expanduser(), dtype)
    _emit_progress(progress_callback, "validating source")
    _validate_request(request)

    _emit_progress(progress_callback, "reading manifest")
    manifest = generate_manifest(request.source)
    plan = plan_conversion(manifest, dtype=request.dtype)
    _validate_manifest_for_conversion(manifest, plan)
    _validate_runtime_metadata(request.source)
    preflight_free_space(
        request.output,
        required_bytes=_estimate_conversion_output_bytes(plan, dtype=request.dtype),
        label="conversion",
    )

    mx = _load_mlx()
    with atomic_output_dir(request.output, label="conversion") as temp_output:
        _emit_progress(progress_callback, "copying runtime metadata")
        _copy_runtime_metadata(request.source, temp_output)
        _write_json(temp_output / "manifest.json", manifest)

        weight_map = _write_selected_weight_shards(
            mx,
            request.source,
            temp_output,
            plan,
            dtype=request.dtype,
            progress_callback=progress_callback,
        )
        _write_json(temp_output / "weight_map.json", weight_map)

        artifact = _artifact_metadata(request, plan)
        _write_json(temp_output / "artifact.json", artifact)

        report = _conversion_report(request, manifest, plan, weight_map)
        _write_json(temp_output / "conversion_report.json", report)
    return report


def plan_conversion(manifest: dict[str, Any], *, dtype: str = "auto") -> dict[str, Any]:
    if dtype not in SUPPORTED_DTYPES:
        raise BooguTurboMlxError(
            f"Unsupported dtype {dtype!r}; expected one of {', '.join(SUPPORTED_DTYPES)}."
        )

    source = str(manifest.get("source", ""))
    tensors = [
        dict(entry)
        for entry in manifest.get("tensor_inventory", [])
        if str(entry.get("component")) in RUNTIME_WEIGHT_COMPONENTS
    ]

    components: dict[str, dict[str, Any]] = {}
    for component in RUNTIME_WEIGHT_COMPONENTS:
        component_tensors = [
            entry for entry in tensors if entry.get("component") == component
        ]
        selected = [entry for entry in component_tensors if entry.get("selected")]
        excluded = [entry for entry in component_tensors if not entry.get("selected")]
        shard_groups = _selected_shard_groups(component, selected)
        components[component] = {
            "selected": len(selected),
            "excluded": len(excluded),
            "total": len(component_tensors),
            "index": _component_output_index(manifest, component, shard_groups),
            "shards": shard_groups,
            "exclusion_reasons": _count_exclusion_reasons(excluded),
        }

    planned_tensors = []
    for entry in sorted(
        tensors,
        key=lambda item: (
            str(item.get("component")),
            str(item.get("path")),
            str(item.get("key")),
        ),
    ):
        component = str(entry["component"])
        source_path = str(entry["path"])
        output_shard = None
        if entry.get("selected"):
            output_shard = _output_shard_path(component, source_path)
        planned_tensors.append(
            {
                "key": entry["key"],
                "component": component,
                "source_shard": source_path,
                "output_shard": output_shard,
                "dtype": entry.get("dtype"),
                "shape": entry.get("shape"),
                "source_byte_count": entry.get("byte_count"),
                "selected": bool(entry.get("selected")),
                "exclusion_reason": entry.get("exclusion_reason"),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "dtype": dtype,
        "components": components,
        "tensors": planned_tensors,
    }


def _validate_request(request: ConversionRequest) -> None:
    if request.dtype not in SUPPORTED_DTYPES:
        raise BooguTurboMlxError(
            f"Unsupported dtype {request.dtype!r}; expected one of {', '.join(SUPPORTED_DTYPES)}."
        )
    if not request.source.exists():
        raise BooguTurboMlxError(f"Source directory does not exist: {request.source}")
    if not request.source.is_dir():
        raise BooguTurboMlxError(f"Conversion source must be a directory: {request.source}")
    ensure_empty_or_missing_output_dir(request.output, label="conversion")


def _validate_runtime_metadata(source: Path) -> None:
    required = [
        source / "processor",
        source / "mllm" / "config.json",
        source / "transformer" / "config.json",
        source / "vae" / "config.json",
        source / "scheduler" / "scheduler_config.json",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise BooguTurboMlxError(
            "Source is missing required runtime metadata: " + ", ".join(missing)
        )


def _validate_manifest_for_conversion(
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    header_errors = [
        f"{header.get('path')}: {header.get('error')}"
        for header in manifest.get("safetensors_headers", [])
        if header.get("error")
    ]
    if header_errors:
        raise BooguTurboMlxError(
            "Cannot convert source with invalid safetensors headers: "
            + "; ".join(header_errors[:3])
        )

    index_errors = [
        str(index.get("path"))
        for index in manifest.get("safetensors_indexes", [])
        if index.get("validation", {}).get("status") == "error"
    ]
    if index_errors:
        raise BooguTurboMlxError(
            "Cannot convert source with safetensors index mismatches: "
            + ", ".join(index_errors)
        )

    missing_components = [
        component
        for component in RUNTIME_WEIGHT_COMPONENTS
        if plan["components"][component]["selected"] == 0
    ]
    if missing_components:
        raise BooguTurboMlxError(
            "Cannot convert source without selected runtime tensors for: "
            + ", ".join(missing_components)
        )


def _estimate_conversion_output_bytes(plan: dict[str, Any], *, dtype: str) -> int:
    total = 0
    for tensor in plan.get("tensors", []):
        if not tensor.get("selected"):
            continue
        if dtype == "auto":
            total += int(tensor.get("source_byte_count") or 0)
            continue
        shape = tensor.get("shape")
        dtype_size = DTYPE_BYTE_SIZES.get(_safetensors_dtype(dtype), 0)
        if isinstance(shape, list) and dtype_size:
            count = 1
            for dim in shape:
                count *= int(dim)
            total += count * dtype_size
        else:
            total += int(tensor.get("source_byte_count") or 0)
    return total


def _safetensors_dtype(dtype: str) -> str:
    return {
        "bfloat16": "BF16",
        "float16": "F16",
        "float32": "F32",
    }.get(dtype, "")


def _copy_runtime_metadata(source: Path, output: Path) -> None:
    processor_source = source / "processor"
    processor_output = output / "processor"
    if processor_output.exists():
        shutil.rmtree(processor_output)
    shutil.copytree(
        processor_source,
        processor_output,
        ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.safetensors"),
    )

    for component in ("mllm", "transformer", "vae"):
        target_dir = output / component
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / component / "config.json", target_dir / "config.json")

    scheduler_dir = output / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source / "scheduler" / "scheduler_config.json",
        scheduler_dir / "scheduler_config.json",
    )


def _write_selected_weight_shards(
    mx: Any,
    source: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    dtype: str,
    progress_callback: Callable[[str], None] | None,
) -> dict[str, Any]:
    weight_map = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "dtype": dtype,
        "components": {},
        "tensors": plan["tensors"],
    }

    for component, component_plan in plan["components"].items():
        _emit_progress(progress_callback, f"{component}: starting")
        component_dir = output / "weights" / component
        component_dir.mkdir(parents=True, exist_ok=True)
        index_weight_map: dict[str, str] = {}
        total_size = 0

        shards = component_plan["shards"]
        for shard_index, shard in enumerate(shards, start=1):
            _emit_progress(
                progress_callback,
                f"{component}: shard {shard_index}/{len(shards)}",
            )
            source_shard = source / shard["source"]
            output_shard = output / shard["output"]
            arrays = mx.load(source_shard)
            selected_keys = set(shard["tensors"])
            arrays_to_save = {
                key: arrays[key]
                for key in sorted(selected_keys)
                if key in arrays
            }
            missing = sorted(selected_keys - set(arrays_to_save))
            if missing:
                raise BooguTurboMlxError(
                    f"{source_shard} is missing selected tensors: {', '.join(missing[:5])}"
                )

            if dtype != "auto":
                target_dtype = _mlx_dtype(mx, dtype)
                arrays_to_save = {
                    key: value.astype(target_dtype)
                    for key, value in arrays_to_save.items()
                }

            values = list(arrays_to_save.values())
            if values:
                mx.eval(*values)
            mx.save_safetensors(
                output_shard,
                arrays_to_save,
                metadata={"format": "mlx", "source": "boogu-turbo-mlx"},
            )

            shard_total_size = sum(int(array.nbytes) for array in arrays_to_save.values())
            total_size += shard_total_size
            shard["output_size"] = shard_total_size
            for key in sorted(arrays_to_save):
                index_weight_map[key] = Path(shard["output"]).name

        index_payload = {
            "metadata": {"total_size": total_size},
            "weight_map": index_weight_map,
        }
        index_path = output / component_plan["index"]
        _write_json(index_path, index_payload)

        weight_map["components"][component] = {
            "index": component_plan["index"],
            "selected": component_plan["selected"],
            "excluded": component_plan["excluded"],
            "shards": [
                {
                    "source": shard["source"],
                    "output": shard["output"],
                    "tensor_count": shard["tensor_count"],
                    "output_size": shard.get("output_size", 0),
                }
                for shard in component_plan["shards"]
            ],
        }
        _emit_progress(progress_callback, f"{component}: complete")

    return weight_map


def _selected_shard_groups(
    component: str,
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in selected:
        by_source.setdefault(str(entry["path"]), []).append(entry)

    groups = []
    for source_path, entries in sorted(by_source.items()):
        groups.append(
            {
                "source": source_path,
                "output": _output_shard_path(component, source_path),
                "tensor_count": len(entries),
                "source_bytes": sum(int(entry.get("byte_count") or 0) for entry in entries),
                "tensors": sorted(str(entry["key"]) for entry in entries),
            }
        )
    return groups


def _component_output_index(
    manifest: dict[str, Any],
    component: str,
    shard_groups: list[dict[str, Any]],
) -> str:
    for index in manifest.get("safetensors_indexes", []):
        if index.get("component") == component:
            return f"weights/{component}/{Path(str(index['path'])).name}"

    if len(shard_groups) == 1:
        source_name = Path(str(shard_groups[0]["source"])).name
        return f"weights/{component}/{source_name}.index.json"
    return f"weights/{component}/model.safetensors.index.json"


def _output_shard_path(component: str, source_path: str) -> str:
    return f"weights/{component}/{Path(source_path).name}"


def _count_exclusion_reasons(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        reason = str(entry.get("exclusion_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _artifact_metadata(
    request: ConversionRequest,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "boogu-turbo-mlx-artifact",
        "source": str(request.source),
        "dtype": request.dtype,
        "components": {
            component: {
                "config": f"{component}/config.json" if component != "scheduler" else None,
                "weights_index": component_plan["index"],
            }
            for component, component_plan in plan["components"].items()
        },
        "processor": "processor",
        "scheduler_config": "scheduler/scheduler_config.json",
    }


def _conversion_report(
    request: ConversionRequest,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    weight_map: dict[str, Any],
) -> dict[str, Any]:
    selected = sum(component["selected"] for component in plan["components"].values())
    excluded = sum(component["excluded"] for component in plan["components"].values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "converted",
        "source": str(request.source),
        "output": str(request.output),
        "dtype": request.dtype,
        "manifest_status": manifest.get("status"),
        "selected_tensor_count": selected,
        "excluded_tensor_count": excluded,
        "components": plan["components"],
        "weight_map": "weight_map.json",
        "artifact": "artifact.json",
        "manifest": "manifest.json",
        "component_weight_maps": weight_map["components"],
    }


def _load_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BooguTurboMlxError(
            "Weight conversion requires MLX. Install `boogu-turbo-mlx[conversion]` "
            "on an MLX-supported machine."
        ) from exc
    return mx


def _mlx_dtype(mx: Any, dtype: str) -> Any:
    return {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }[dtype]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)
