from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_write import (
    atomic_output_dir,
    ensure_empty_or_missing_output_dir,
    preflight_free_space,
)
from .artifacts import (
    DEFAULT_COMPONENT_CONFIG_PATHS,
    component_weights_index_path,
    flatten_parameters,
    read_artifact,
    read_json,
)
from .constants import OFFICIAL_TEXT_WEIGHT_PREFIX
from .errors import BooguTurboMlxError

try:  # Keep CLI/parser imports lightweight on non-MLX systems.
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised in non-MLX environments.
    nn = None


Q8_FORMAT = "mlx-native-weight-only"
Q8_MODE = "affine"
Q8_BITS = 8
Q8_GROUP_SIZE = 32
Q8_COMPONENTS = ("mllm", "transformer")


@dataclass(frozen=True)
class QuantizationSettings:
    mode: str = Q8_MODE
    bits: int = Q8_BITS
    group_size: int = Q8_GROUP_SIZE


def quantize_artifact(
    source: str | Path,
    output: str | Path,
    *,
    mode: str = Q8_MODE,
    bits: int = Q8_BITS,
    group_size: int = Q8_GROUP_SIZE,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write a persistent MLX-native q8 artifact from an existing MLX artifact."""

    _emit_progress(progress_callback, "validating source artifact")
    settings = validate_quantization_settings(
        mode=mode,
        bits=bits,
        group_size=group_size,
    )
    source_path = Path(source).expanduser()
    output_path = Path(output).expanduser()
    _validate_quantization_request(source_path, output_path)

    artifact = read_artifact(source_path, "quantization source")
    _reject_quantized_source(source_path, artifact)
    preflight_free_space(
        output_path,
        required_bytes=_estimate_quantized_output_bytes(source_path, artifact),
        label="quantization",
    )

    with atomic_output_dir(output_path, label="quantization") as temp_output:
        _emit_progress(progress_callback, "copying runtime metadata and VAE")
        _copy_runtime_metadata_and_vae(source_path, temp_output, artifact)

        component_reports: dict[str, Any] = {}
        for component in Q8_COMPONENTS:
            _emit_progress(progress_callback, f"{component}: starting")
            component_reports[component] = _quantize_component(
                source_path,
                temp_output,
                artifact,
                component,
                settings,
            )
            _emit_progress(progress_callback, f"{component}: complete")
            gc.collect()

        output_artifact = dict(artifact)
        output_artifact["quantization"] = _quantization_metadata(source_path, settings)
        _write_json(temp_output / "artifact.json", output_artifact)

        component_reports["vae"] = _component_copy_report(
            source_path,
            temp_output,
            artifact,
            "vae",
        )
        validation = _verify_quantized_artifact(temp_output, settings)
        report = {
            "schema_version": int(artifact.get("schema_version", 2)),
            "status": "quantized",
            "source": str(source_path),
            "output": str(output_path),
            "artifact": "artifact.json",
            "quantization": output_artifact["quantization"],
            "components": component_reports,
            "validation": validation,
        }
        _write_json(temp_output / "quantization_report.json", report)
    _emit_progress(progress_callback, "complete")
    return report


def validate_quantization_settings(
    *,
    mode: str,
    bits: int,
    group_size: int,
) -> QuantizationSettings:
    if mode != Q8_MODE or int(bits) != Q8_BITS or int(group_size) != Q8_GROUP_SIZE:
        raise BooguTurboMlxError(
            "Unsupported quantization settings for this release; expected "
            f"mode={Q8_MODE!r}, bits={Q8_BITS}, group_size={Q8_GROUP_SIZE}."
        )
    return QuantizationSettings(mode=mode, bits=int(bits), group_size=int(group_size))


def linear_only_predicate(path: str, module: Any) -> bool:
    del path
    return nn is not None and isinstance(module, nn.Linear)


def quantized_linear_paths_for_component(
    root: Path,
    artifact: Mapping[str, Any],
    component: str,
    *,
    key_transform: Callable[[str], str | None] | None = None,
) -> tuple[str, ...]:
    index_path = component_weights_index_path(root, artifact, component)
    index = read_json(index_path, component)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise BooguTurboMlxError(f"Invalid {component} weight index: {index_path}")

    paths = set()
    for key in weight_map:
        local_key = key_transform(str(key)) if key_transform is not None else str(key)
        if local_key is not None and local_key.endswith(".scales"):
            paths.add(local_key.removesuffix(".scales"))
    return tuple(sorted(paths))


def apply_quantization_to_model(
    model: Any,
    paths: tuple[str, ...],
    settings: QuantizationSettings,
) -> None:
    if not paths:
        return
    _require_mlx()
    path_set = set(paths)
    linear_paths = {path for path, _ in _linear_leaf_modules(model)}
    unknown = sorted(path_set - linear_paths)
    if unknown:
        raise BooguTurboMlxError(
            "Quantized artifact references unknown or non-linear modules: "
            + ", ".join(unknown[:10])
        )
    preflight_linear_group_size(model, group_size=settings.group_size, paths=paths)

    nn.quantize(
        model,
        group_size=settings.group_size,
        bits=settings.bits,
        mode=settings.mode,
        class_predicate=lambda path, module: path in path_set
        and linear_only_predicate(path, module),
    )


def preflight_linear_group_size(
    model: Any,
    *,
    group_size: int,
    paths: tuple[str, ...] | None = None,
) -> None:
    path_filter = set(paths) if paths is not None else None
    mismatched = []
    for path, module in _linear_leaf_modules(model):
        if path_filter is not None and path not in path_filter:
            continue
        input_dim = int(module.weight.shape[-1])
        if input_dim % group_size:
            mismatched.append(f"{path}.weight: input_dim={input_dim}")
    if mismatched:
        raise BooguTurboMlxError(
            f"Cannot quantize Linear weights with group_size={group_size}; "
            "input dimensions must be divisible by the group size: "
            + "; ".join(mismatched[:10])
        )


def quantization_settings_for_component(
    artifact: Mapping[str, Any],
    component: str,
    quantized_paths: tuple[str, ...],
) -> QuantizationSettings | None:
    if not quantized_paths:
        return None

    payload = artifact.get("quantization")
    if not isinstance(payload, Mapping):
        raise BooguTurboMlxError(
            f"{component} weights contain q8 tensors but artifact.json has no "
            "quantization metadata."
        )
    if payload.get("format") != Q8_FORMAT:
        raise BooguTurboMlxError(
            f"Unsupported quantization format {payload.get('format')!r}; "
            f"expected {Q8_FORMAT!r}."
        )
    components = payload.get("components")
    if not isinstance(components, list) or component not in components:
        raise BooguTurboMlxError(
            f"Quantization metadata does not list component {component!r}."
        )
    if payload.get("layers") not in (None, "linear"):
        raise BooguTurboMlxError(
            f"Unsupported quantized layer set {payload.get('layers')!r}; "
            "expected 'linear'."
        )
    return validate_quantization_settings(
        mode=str(payload.get("mode")),
        bits=int(payload.get("bits")),
        group_size=int(payload.get("group_size")),
    )


def _quantize_component(
    source: Path,
    output: Path,
    artifact: Mapping[str, Any],
    component: str,
    settings: QuantizationSettings,
) -> dict[str, Any]:
    mx = _load_mlx_core()
    model = _load_quantizable_component(source, component)
    paths = tuple(path for path, _ in _linear_leaf_modules(model))
    preflight_linear_group_size(model, group_size=settings.group_size, paths=paths)
    nn.quantize(
        model,
        group_size=settings.group_size,
        bits=settings.bits,
        mode=settings.mode,
        class_predicate=linear_only_predicate,
    )
    report = _write_quantized_component_weights(
        mx,
        source,
        output,
        artifact,
        component,
        model,
        paths,
    )
    del model
    return report


def _write_quantized_component_weights(
    mx: Any,
    source: Path,
    output: Path,
    artifact: Mapping[str, Any],
    component: str,
    model: Any,
    quantized_paths: tuple[str, ...],
) -> dict[str, Any]:
    source_index_path = component_weights_index_path(source, artifact, component)
    output_index_path = component_weights_index_path(output, artifact, component)
    source_index = read_json(source_index_path, component)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, Mapping):
        raise BooguTurboMlxError(f"Invalid {component} weight index: {source_index_path}")

    local_to_shard = _local_source_weight_map(component, source_weight_map)
    arrays_by_shard: dict[str, dict[str, Any]] = {}
    output_weight_map: dict[str, str] = {}
    total_tensor_bytes = 0
    for local_key, array in sorted(flatten_parameters(model.parameters())):
        shard = _source_shard_for_output_key(local_key, local_to_shard)
        index_key = _local_to_index_key(component, local_key)
        arrays_by_shard.setdefault(shard, {})[index_key] = array
        output_weight_map[index_key] = shard
        total_tensor_bytes += int(array.nbytes)

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    for shard, arrays in sorted(arrays_by_shard.items()):
        values = list(arrays.values())
        if values:
            mx.eval(*values)
        mx.save_safetensors(
            output_index_path.parent / shard,
            arrays,
            metadata={"format": "mlx", "source": "boogu-turbo-mlx-q8"},
        )

    index_payload = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": output_weight_map,
    }
    _write_json(output_index_path, index_payload)

    source_disk_size = _component_index_disk_size(source_index_path, source_index)
    output_disk_size = _component_index_disk_size(output_index_path, index_payload)
    return {
        "status": "quantized",
        "linear_count": len(quantized_paths),
        "tensor_count": len(output_weight_map),
        "source_disk_size": source_disk_size,
        "output_disk_size": output_disk_size,
        "disk_size_ratio": (
            output_disk_size / source_disk_size if source_disk_size else None
        ),
        "source_index": str(source_index_path.relative_to(source)),
        "output_index": str(output_index_path.relative_to(output)),
    }


def _copy_runtime_metadata_and_vae(
    source: Path,
    output: Path,
    artifact: Mapping[str, Any],
) -> None:
    _copy_required_path(
        source / _processor_rel_path(artifact),
        output / _processor_rel_path(artifact),
        ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.safetensors"),
    )
    for component in ("mllm", "transformer", "vae"):
        rel_path = _component_rel_path(
            artifact,
            component,
            "config",
            DEFAULT_COMPONENT_CONFIG_PATHS[component],
        )
        _copy_required_path(source / rel_path, output / rel_path)

    scheduler = artifact.get("scheduler_config")
    if isinstance(scheduler, str):
        scheduler_path = Path(scheduler)
        if (source / scheduler_path).exists():
            _copy_required_path(source / scheduler_path, output / scheduler_path)

    _copy_component_weights(source, output, artifact, "vae")


def _copy_component_weights(
    source: Path,
    output: Path,
    artifact: Mapping[str, Any],
    component: str,
) -> None:
    source_index_path = component_weights_index_path(source, artifact, component)
    output_index_path = component_weights_index_path(output, artifact, component)
    index = read_json(source_index_path, component)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise BooguTurboMlxError(f"Invalid {component} weight index: {source_index_path}")
    _copy_required_path(source_index_path, output_index_path)
    for shard in sorted({str(value) for value in weight_map.values()}):
        _copy_required_path(
            source_index_path.parent / shard,
            output_index_path.parent / shard,
        )


def _component_copy_report(
    source: Path,
    output: Path,
    artifact: Mapping[str, Any],
    component: str,
) -> dict[str, Any]:
    source_index_path = component_weights_index_path(source, artifact, component)
    output_index_path = component_weights_index_path(output, artifact, component)
    source_index = read_json(source_index_path, component)
    output_index = read_json(output_index_path, component)
    return {
        "status": "copied",
        "source_disk_size": _component_index_disk_size(source_index_path, source_index),
        "output_disk_size": _component_index_disk_size(output_index_path, output_index),
        "source_index": str(source_index_path.relative_to(source)),
        "output_index": str(output_index_path.relative_to(output)),
    }


def _validate_quantization_request(source: Path, output: Path) -> None:
    if not source.exists():
        raise BooguTurboMlxError(f"Source artifact directory does not exist: {source}")
    if not source.is_dir():
        raise BooguTurboMlxError(f"Quantization source must be a directory: {source}")
    ensure_empty_or_missing_output_dir(output, label="quantization")


def _estimate_quantized_output_bytes(source: Path, artifact: Mapping[str, Any]) -> int:
    total = 0
    for component in Q8_COMPONENTS:
        index_path = component_weights_index_path(source, artifact, component)
        index = read_json(index_path, component)
        total += int(_component_index_disk_size(index_path, index) * 0.75)
    vae_index_path = component_weights_index_path(source, artifact, "vae")
    vae_index = read_json(vae_index_path, "vae")
    total += _component_index_disk_size(vae_index_path, vae_index)
    return total


def _verify_quantized_artifact(
    output: Path,
    expected_settings: QuantizationSettings,
) -> dict[str, Any]:
    artifact = read_artifact(output, "quantized artifact")
    components: dict[str, Any] = {}
    for component in Q8_COMPONENTS:
        paths = quantized_linear_paths_for_component(
            output,
            artifact,
            component,
            key_transform=_mllm_index_key_to_local if component == "mllm" else None,
        )
        if not paths:
            raise BooguTurboMlxError(
                f"Quantized artifact has no q8 linear paths for {component}."
            )
        settings = quantization_settings_for_component(artifact, component, paths)
        if settings != expected_settings:
            raise BooguTurboMlxError(
                f"Quantized artifact settings changed while writing {component}."
            )
        components[component] = {"quantized_linear_count": len(paths)}

    vae_index = component_weights_index_path(output, artifact, "vae")
    read_json(vae_index, "vae")
    components["vae"] = {"status": "copied"}
    return {"status": "ok", "components": components}


def _reject_quantized_source(source: Path, artifact: Mapping[str, Any]) -> None:
    if "quantization" in artifact:
        raise BooguTurboMlxError("Source artifact is already quantized.")
    for component in Q8_COMPONENTS:
        paths = quantized_linear_paths_for_component(
            source,
            artifact,
            component,
            key_transform=_mllm_index_key_to_local if component == "mllm" else None,
        )
        if paths:
            raise BooguTurboMlxError(
                f"Source artifact already contains quantized {component} tensors."
            )


def _load_quantizable_component(source: Path, component: str) -> Any:
    if component == "mllm":
        from .encoding import Qwen3VLInstructionEncoder

        return Qwen3VLInstructionEncoder.from_pretrained(source)
    if component == "transformer":
        from .transformer import BooguImageTransformer

        return BooguImageTransformer.from_pretrained(source)
    raise BooguTurboMlxError(f"Unsupported quantized component: {component}")


def _linear_leaf_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (path, module)
        for path, module in _flatten_leaf_module_tree(model.leaf_modules())
        if linear_only_predicate(path, module)
    )


def _flatten_leaf_module_tree(value: Any, prefix: str = "") -> tuple[tuple[str, Any], ...]:
    if nn is not None and isinstance(value, nn.Module):
        return ((prefix, value),)
    if isinstance(value, dict):
        items = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_flatten_leaf_module_tree(child, child_prefix))
        return tuple(items)
    if isinstance(value, list):
        items = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            items.extend(_flatten_leaf_module_tree(child, child_prefix))
        return tuple(items)
    return ()


def _local_source_weight_map(
    component: str,
    source_weight_map: Mapping[str, Any],
) -> dict[str, str]:
    local_to_shard: dict[str, str] = {}
    duplicates = []
    unexpected = []
    for index_key, shard in source_weight_map.items():
        local_key = (
            _mllm_index_key_to_local(str(index_key))
            if component == "mllm"
            else str(index_key)
        )
        if local_key is None:
            unexpected.append(str(index_key))
            continue
        if local_key in local_to_shard:
            duplicates.append(str(index_key))
        local_to_shard[local_key] = str(shard)
    if unexpected or duplicates:
        raise BooguTurboMlxError(
            f"{component} source index has unsupported tensor keys: "
            + ", ".join((unexpected + duplicates)[:10])
        )
    return local_to_shard


def _source_shard_for_output_key(local_key: str, local_to_shard: Mapping[str, str]) -> str:
    if local_key in local_to_shard:
        return local_to_shard[local_key]
    if local_key.endswith((".scales", ".biases")):
        weight_key = local_key.rsplit(".", 1)[0] + ".weight"
        if weight_key in local_to_shard:
            return local_to_shard[weight_key]
    raise BooguTurboMlxError(
        f"Cannot preserve shard locality for generated tensor {local_key!r}; "
        "source weight shard was not found."
    )


def _component_index_disk_size(index_path: Path, index: Mapping[str, Any]) -> int:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return 0
    total = index_path.stat().st_size if index_path.exists() else 0
    for shard in sorted({str(value) for value in weight_map.values()}):
        shard_path = index_path.parent / shard
        if shard_path.exists():
            total += shard_path.stat().st_size
    return total


def _copy_required_path(
    source: Path,
    target: Path,
    *,
    ignore: Callable[[str, list[str]], set[str]] | None = None,
) -> None:
    if not source.exists():
        raise BooguTurboMlxError(f"Missing required artifact path: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=ignore)
    else:
        shutil.copy2(source, target)


def _quantization_metadata(
    source: Path,
    settings: QuantizationSettings,
) -> dict[str, Any]:
    return {
        "format": Q8_FORMAT,
        "mode": settings.mode,
        "bits": settings.bits,
        "group_size": settings.group_size,
        "components": list(Q8_COMPONENTS),
        "layers": "linear",
        "source_artifact": str(source),
    }


def _mllm_index_key_to_local(key: str) -> str | None:
    if not key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX):
        return None
    return key.removeprefix(OFFICIAL_TEXT_WEIGHT_PREFIX)


def _local_to_index_key(component: str, local_key: str) -> str:
    if component == "mllm":
        return OFFICIAL_TEXT_WEIGHT_PREFIX + local_key
    return local_key


def _component_rel_path(
    artifact: Mapping[str, Any],
    component: str,
    field: str,
    default: str,
) -> Path:
    components = artifact.get("components")
    component_payload: Any = {}
    if isinstance(components, Mapping):
        component_payload = components.get(component) or {}
    if not isinstance(component_payload, Mapping):
        component_payload = {}
    return Path(str(component_payload.get(field, default)))


def _processor_rel_path(artifact: Mapping[str, Any]) -> Path:
    return Path(str(artifact.get("processor", "processor")))


def _load_mlx_core() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BooguTurboMlxError(
            "Artifact quantization requires MLX. Install `boogu-turbo-mlx[conversion]` "
            "on an MLX-supported machine."
        ) from exc
    return mx


def _require_mlx() -> None:
    if nn is None:
        raise BooguTurboMlxError(
            "Artifact quantization requires MLX. Install `boogu-turbo-mlx[conversion]` "
            "on an MLX-supported machine."
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)
