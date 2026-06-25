from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .errors import BooguTurboMlxError


DEFAULT_COMPONENT_CONFIG_PATHS = {
    "mllm": "mllm/config.json",
    "transformer": "transformer/config.json",
    "vae": "vae/config.json",
}

DEFAULT_COMPONENT_WEIGHT_INDEXES = {
    "mllm": "weights/mllm/model.safetensors.index.json",
    "transformer": "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae": "weights/vae/diffusion_pytorch_model.safetensors.index.json",
}

KeyTransform = Callable[[str], str | None]
WeightTransform = Callable[[str, Any], Any]


def read_json(path: Path, component: str) -> dict[str, Any]:
    if not path.exists():
        raise BooguTurboMlxError(f"Missing required {component} artifact file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BooguTurboMlxError(
            f"Invalid {component} artifact JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BooguTurboMlxError(f"Invalid {component} artifact JSON: {path}")
    return payload


def read_artifact(root: Path, component: str) -> dict[str, Any]:
    artifact = read_json(root / "artifact.json", component)
    if artifact.get("format") != "boogu-turbo-mlx-artifact":
        raise BooguTurboMlxError(f"Expected a boogu-turbo-mlx artifact at {root}")
    return artifact


def component_config_path(
    root: Path,
    artifact: Mapping[str, Any],
    component: str,
) -> Path:
    return _component_path(
        root,
        artifact,
        component,
        "config",
        DEFAULT_COMPONENT_CONFIG_PATHS[component],
    )


def component_weights_index_path(
    root: Path,
    artifact: Mapping[str, Any],
    component: str,
) -> Path:
    return _component_path(
        root,
        artifact,
        component,
        "weights_index",
        DEFAULT_COMPONENT_WEIGHT_INDEXES[component],
    )


def processor_path(root: Path, artifact: Mapping[str, Any]) -> Path:
    return root / str(artifact.get("processor", "processor"))


def load_indexed_component_weights(
    *,
    root: Path,
    artifact: Mapping[str, Any],
    component: str,
    expected_shapes: Mapping[str, tuple[int, ...]],
    mx_module: Any,
    artifact_label: str,
    weight_label: str,
    key_transform: KeyTransform | None = None,
    weight_transform: WeightTransform | None = None,
) -> dict[str, Any]:
    index_path = component_weights_index_path(root, artifact, component)
    index = read_json(index_path, artifact_label)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise BooguTurboMlxError(f"Invalid {weight_label} weight index: {index_path}")

    local_to_index: dict[str, tuple[str, str]] = {}
    unexpected_index_keys: list[str] = []
    duplicate_index_keys: list[str] = []
    for key, shard in weight_map.items():
        index_key = str(key)
        local_key = key_transform(index_key) if key_transform is not None else index_key
        if local_key is None:
            unexpected_index_keys.append(index_key)
            continue
        if local_key in local_to_index:
            duplicate_index_keys.append(index_key)
            continue
        local_to_index[local_key] = (index_key, str(shard))

    index_keys = set(local_to_index)
    expected_keys = set(expected_shapes)
    missing = sorted(expected_keys - index_keys)
    extra = sorted(index_keys - expected_keys)
    if missing:
        raise BooguTurboMlxError(
            f"{artifact_label} artifact is missing selected tensors: "
            + ", ".join(missing[:10])
        )
    if unexpected_index_keys or duplicate_index_keys or extra:
        unexpected = unexpected_index_keys + duplicate_index_keys + extra
        raise BooguTurboMlxError(
            f"{artifact_label} artifact has unexpected selected tensors: "
            + ", ".join(unexpected[:10])
        )

    weights_dir = index_path.parent
    selected_weights: dict[str, Any] = {}
    keys_by_shard: dict[str, list[tuple[str, str]]] = {}
    for local_key, (index_key, shard) in local_to_index.items():
        keys_by_shard.setdefault(shard, []).append((index_key, local_key))

    for shard, keys in sorted(keys_by_shard.items()):
        shard_path = weights_dir / shard
        if not shard_path.exists():
            raise BooguTurboMlxError(f"Missing {weight_label} weight shard: {shard_path}")
        arrays = mx_module.load(shard_path)
        shard_index_keys = {index_key for index_key, _ in keys}
        loaded_keys = set(arrays)
        missing_in_shard = sorted(shard_index_keys - loaded_keys)
        extra_in_shard = sorted(loaded_keys - shard_index_keys)
        if missing_in_shard:
            raise BooguTurboMlxError(
                f"{shard_path} is missing selected tensors: "
                + ", ".join(missing_in_shard[:10])
            )
        if extra_in_shard:
            raise BooguTurboMlxError(
                f"{shard_path} contains tensors not listed in the artifact index: "
                + ", ".join(extra_in_shard[:10])
            )
        for index_key, local_key in keys:
            array = arrays[index_key]
            if weight_transform is not None:
                array = weight_transform(local_key, array)
            selected_weights[local_key] = array

    mismatched = [
        f"{key}: expected {expected_shapes[key]}, got {tuple(array.shape)}"
        for key, array in sorted(selected_weights.items())
        if tuple(array.shape) != expected_shapes[key]
    ]
    if mismatched:
        raise BooguTurboMlxError(
            f"{artifact_label} artifact has shape-mismatched tensors: "
            + "; ".join(mismatched[:10])
        )

    return selected_weights


def flatten_parameter_shapes(parameters: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in flatten_parameters(parameters)
    }


def flatten_parameters(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_parameters(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_parameters(child, child_prefix)
    else:
        yield prefix, value


def _component_path(
    root: Path,
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
    return root / str(component_payload.get(field, default))
