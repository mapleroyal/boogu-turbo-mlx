from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .constants import (
    OFFICIAL_BOOGU_GITHUB_REVISION,
    OFFICIAL_COMFYUI_BOOGU_REVISION,
    OFFICIAL_TURBO_HF_ID,
    OFFICIAL_TURBO_REVISION,
)
from .errors import BooguTurboMlxError
from .safetensors_header import read_safetensors_header
from .selection import EXPECTED_OFFICIAL_TENSOR_COUNTS, select_tensor

SCHEMA_VERSION = 2
REMOTE_COMPONENTS = ("root", "mllm", "processor", "transformer", "vae", "scheduler")
MAX_REMOTE_METADATA_BYTES = 16 * 1024 * 1024

_REMOTE_JSON_METADATA_FILES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model_index.json",
    "preprocessor_config.json",
    "scheduler_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
}
_REMOTE_TEXT_METADATA_FILES = {"chat_template.jinja"}
_LOCAL_JSON_METADATA_FILES = _REMOTE_JSON_METADATA_FILES | {
    "chat_template.json",
    "video_preprocessor_config.json",
}
_LOCAL_TEXT_METADATA_FILES = _REMOTE_TEXT_METADATA_FILES


def generate_manifest(source: str | Path, revision: str | None = None) -> dict[str, Any]:
    source_text = str(source)
    source_path = Path(source_text).expanduser()
    if source_path.exists():
        return _generate_local_manifest(source_path)
    return _generate_remote_manifest(source_text, revision)


def _generate_local_manifest(source_path: Path) -> dict[str, Any]:
    root = source_path if source_path.is_dir() else source_path.parent
    safetensors_paths = _local_safetensors_paths(source_path)
    tensor_inventory, safetensors_headers = _build_local_tensor_inventory(
        safetensors_paths,
        root,
    )

    configs = [
        _summarize_config(path, root)
        for path in _local_config_paths(source_path)
    ]
    chat_templates = [
        _summarize_chat_template(path.read_text(encoding="utf-8"), _relative(path, root))
        for path in _local_text_metadata_paths(source_path)
    ]
    indexes = [
        _summarize_safetensors_index(path, root, safetensors_headers)
        for path in _local_safetensors_index_paths(source_path)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_path),
        "source_kind": "local_directory" if source_path.is_dir() else "local_file",
        "official_revisions": _official_revisions(),
        "configs": configs,
        "chat_templates": chat_templates,
        "component_configs": _group_component_configs(configs, chat_templates),
        "safetensors_indexes": indexes,
        "safetensors_files": [_relative(path, root) for path in safetensors_paths],
        "safetensors_headers": safetensors_headers,
        "tensor_inventory": tensor_inventory,
        "tensor_summary": _summarize_tensor_inventory(tensor_inventory),
        "status": "local_manifest_generated",
    }


def _generate_remote_manifest(source: str, revision: str | None) -> dict[str, Any]:
    effective_revision = _resolve_remote_revision(source, revision)
    HfApi, hf_hub_download = _load_hf_tools()
    info = HfApi().model_info(
        repo_id=source,
        revision=effective_revision,
        files_metadata=True,
    )

    inventory = [_summarize_remote_file(sibling) for sibling in _siblings(info)]
    inventory = sorted(
        (item for item in inventory if item.get("path")),
        key=lambda item: str(item["path"]),
    )

    configs: list[dict[str, Any]] = []
    chat_templates: list[dict[str, Any]] = []
    safetensors_indexes: list[dict[str, Any]] = []
    remote_files_read: list[str] = []

    for item in inventory:
        path = str(item["path"])
        if not _should_download_remote_metadata(path, item.get("size")):
            continue

        remote_files_read.append(path)
        content = _download_remote_text_file(
            hf_hub_download,
            repo_id=source,
            revision=effective_revision,
            filename=path,
        )

        if path.endswith(".safetensors.index.json"):
            safetensors_indexes.append(_summarize_safetensors_index_text(content, path))
        elif path.endswith(".jinja"):
            chat_templates.append(_summarize_chat_template(content, path))
        elif path.endswith(".json"):
            configs.append(_summarize_config_text(content, path))

    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_kind": "huggingface_model",
        "repo_id": _repo_id(info, source),
        "requested_revision": revision,
        "resolved_revision": _resolved_revision(info, effective_revision),
        "last_modified": _isoformat(_get_attr(info, "last_modified")),
        "license": _license(info),
        "tags": sorted(str(tag) for tag in (_get_attr(info, "tags") or [])),
        "library_name": _get_attr(info, "library_name"),
        "pipeline_tag": _get_attr(info, "pipeline_tag"),
        "official_revisions": _official_revisions(),
        "configs": configs,
        "chat_templates": chat_templates,
        "component_configs": _group_component_configs(configs, chat_templates),
        "file_inventory": inventory,
        "remote_files_read": remote_files_read,
        "safetensors_indexes": safetensors_indexes,
        "safetensors_files": [
            str(item["path"])
            for item in inventory
            if str(item["path"]).endswith(".safetensors")
        ],
        "status": "remote_manifest_generated",
    }


def _local_config_paths(source_path: Path) -> list[Path]:
    root = source_path if source_path.is_dir() else source_path.parent
    if not source_path.is_dir():
        return []
    return [
        path
        for path in sorted(root.rglob("*.json"))
        if _is_local_json_metadata(path, root)
    ]


def _local_text_metadata_paths(source_path: Path) -> list[Path]:
    root = source_path if source_path.is_dir() else source_path.parent
    if not source_path.is_dir():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_local_text_metadata(path, root)
    ]


def _local_safetensors_index_paths(source_path: Path) -> list[Path]:
    root = source_path if source_path.is_dir() else source_path.parent
    if not source_path.is_dir():
        return []
    return sorted(root.rglob("*.safetensors.index.json"))


def _local_safetensors_paths(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path] if source_path.name.endswith(".safetensors") else []
    return [
        path
        for path in sorted(source_path.rglob("*.safetensors"))
        if not path.name.endswith(".index.json")
    ]


def _is_local_json_metadata(path: Path, root: Path) -> bool:
    if path.name.endswith(".safetensors.index.json"):
        return False

    rel = _relative(path, root)
    parts = rel.split("/")
    if len(parts) == 1:
        return path.name == "model_index.json"

    component = parts[0]
    if component in {"mllm", "transformer", "vae"}:
        return path.name in {"config.json", "generation_config.json"}
    if component == "processor":
        return path.name in _LOCAL_JSON_METADATA_FILES
    if component == "scheduler":
        return path.name == "scheduler_config.json"
    return False


def _is_local_text_metadata(path: Path, root: Path) -> bool:
    rel = _relative(path, root)
    parts = rel.split("/")
    if len(parts) == 1:
        return False
    return parts[0] in {"mllm", "processor"} and path.name in _LOCAL_TEXT_METADATA_FILES


def _build_local_tensor_inventory(
    safetensors_paths: Iterable[Path],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    header_summaries: list[dict[str, Any]] = []

    for path in safetensors_paths:
        rel = _relative(path, root)
        component = _component_for_path(rel)
        shard = _shard_for_component_path(rel, component)
        header_summary: dict[str, Any] = {
            "path": rel,
            "component": component,
            "shard": shard,
        }

        try:
            header = read_safetensors_header(path)
        except BooguTurboMlxError as exc:
            header_summary["error"] = str(exc)
            header_summaries.append(header_summary)
            continue

        header_summary.update(
            {
                "metadata": header.metadata,
                "tensor_count": len(header.tensors),
                "tensors": sorted(header.tensors),
                "total_tensor_bytes": sum(tensor.byte_count for tensor in header.tensors.values()),
            }
        )
        header_summaries.append(header_summary)

        for key, tensor in sorted(header.tensors.items()):
            selection = select_tensor(component, key)
            inventory.append(
                {
                    "key": key,
                    "component": component,
                    "shard": shard,
                    "path": rel,
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "byte_count": tensor.byte_count,
                    "selected": selection.selected,
                    "exclusion_reason": selection.exclusion_reason,
                }
            )

    return inventory, header_summaries


def _validate_index_against_headers(
    index_path: Path,
    root: Path,
    weight_map: dict[str, Any],
    safetensors_headers: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    index_component = _component_for_path(_relative(index_path, root))
    index_dir = index_path.parent
    header_by_path = {
        str(header["path"]): header
        for header in safetensors_headers
        if "path" in header and "error" not in header
    }

    normalized_weight_map = {
        str(key): _relative(index_dir / str(shard), root)
        for key, shard in weight_map.items()
    }
    expected_shard_paths = sorted(set(normalized_weight_map.values()))
    missing_shards = [
        shard for shard in expected_shard_paths if shard not in header_by_path
    ]

    missing_tensors: list[dict[str, str]] = []
    for key, shard in sorted(normalized_weight_map.items()):
        header = header_by_path.get(shard)
        if header is None:
            continue
        tensors = set(str(tensor) for tensor in header.get("tensors", []))
        if key not in tensors:
            missing_tensors.append({"key": key, "shard": _shard_for_component_path(shard, index_component)})

    indexed_pairs = set(normalized_weight_map.items())
    extra_tensors: list[dict[str, str]] = []
    for shard in expected_shard_paths:
        header = header_by_path.get(shard)
        if header is None:
            continue
        for key in sorted(str(tensor) for tensor in header.get("tensors", [])):
            if (key, shard) not in indexed_pairs:
                extra_tensors.append(
                    {"key": key, "shard": _shard_for_component_path(shard, index_component)}
                )

    component_shards = {
        str(header["path"])
        for header in safetensors_headers
        if header.get("component") == index_component and "error" not in header
    }
    extra_shards = sorted(component_shards - set(expected_shard_paths))

    status = "ok"
    if missing_shards or missing_tensors or extra_tensors or extra_shards:
        status = "error"

    return {
        "status": status,
        "missing_shards": [_shard_for_component_path(shard, index_component) for shard in missing_shards],
        "extra_shards": [_shard_for_component_path(shard, index_component) for shard in extra_shards],
        "missing_tensors": missing_tensors,
        "extra_tensors": extra_tensors,
    }


def _summarize_tensor_inventory(inventory: Iterable[dict[str, Any]]) -> dict[str, Any]:
    component_summaries: dict[str, dict[str, Any]] = {}
    total_count = 0
    selected_count = 0

    for entry in inventory:
        total_count += 1
        component = str(entry.get("component", "other"))
        summary = component_summaries.setdefault(
            component,
            {
                "total": 0,
                "selected": 0,
                "excluded": 0,
                "dtypes": {},
                "exclusion_reasons": {},
            },
        )
        summary["total"] += 1
        dtype = str(entry.get("dtype"))
        summary["dtypes"][dtype] = summary["dtypes"].get(dtype, 0) + 1

        if entry.get("selected"):
            selected_count += 1
            summary["selected"] += 1
        else:
            summary["excluded"] += 1
            reason = str(entry.get("exclusion_reason") or "unknown")
            summary["exclusion_reasons"][reason] = (
                summary["exclusion_reasons"].get(reason, 0) + 1
            )

    for component, expected in EXPECTED_OFFICIAL_TENSOR_COUNTS.items():
        if component not in component_summaries:
            continue
        summary = component_summaries[component]
        summary["expected_official"] = expected
        summary["matches_expected_official"] = all(
            summary.get(key) == expected[key] for key in ("total", "selected", "excluded")
        ) and summary.get("exclusion_reasons") == expected.get("exclusion_reasons")

    return {
        "total": total_count,
        "selected": selected_count,
        "excluded": total_count - selected_count,
        "components": component_summaries,
    }


def _summarize_config(path: Path, root: Path) -> dict[str, Any]:
    rel = _relative(path, root)
    summary: dict[str, Any] = {
        "path": rel,
        "component": _component_for_path(rel),
        "kind": "json",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary["error"] = str(exc)
        return summary

    summary["top_level_keys"] = sorted(data.keys())
    for key in ("_class_name", "architectures", "model_type", "torch_dtype"):
        if key in data:
            summary[key] = data[key]
    if path.name == "model_index.json":
        summary["component_classes"] = _component_classes(data)
    summary.update(_config_facts(data))
    return summary


def _summarize_safetensors_index(
    path: Path,
    root: Path,
    safetensors_headers: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rel = _relative(path, root)
    summary: dict[str, Any] = {"path": rel, "component": _component_for_path(rel)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary["error"] = str(exc)
        return summary

    weight_map = data.get("weight_map", {})
    metadata = data.get("metadata", {})
    if not isinstance(weight_map, dict):
        weight_map = {}
    if not isinstance(metadata, dict):
        metadata = {}

    shards = sorted({str(shard) for shard in weight_map.values()})
    summary.update(
        {
            "metadata": metadata,
            "tensor_count": len(weight_map),
            "shard_count": len(shards),
            "shards": shards,
            "total_size": metadata.get("total_size"),
        }
    )
    if safetensors_headers is not None:
        summary["validation"] = _validate_index_against_headers(
            path,
            root,
            weight_map,
            safetensors_headers,
        )
    return summary


def _summarize_config_text(content: str, path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": path,
        "component": _component_for_path(path),
        "kind": "json",
    }
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        summary["error"] = str(exc)
        return summary

    if not isinstance(data, dict):
        summary["error"] = "JSON metadata is not an object"
        return summary

    summary["top_level_keys"] = sorted(str(key) for key in data.keys())
    for key in ("_class_name", "architectures", "model_type", "torch_dtype"):
        if key in data:
            summary[key] = data[key]
    if path.endswith("model_index.json"):
        summary["component_classes"] = _component_classes(data)
    summary.update(_config_facts(data))
    return summary


def _summarize_chat_template(content: str, path: str) -> dict[str, Any]:
    return {
        "path": path,
        "component": _component_for_path(path),
        "kind": "chat_template",
        "char_count": len(content),
        "contains_system_role": "system" in content,
        "contains_user_role": "user" in content,
        "contains_generation_prompt": "add_generation_prompt" in content,
    }


def _summarize_safetensors_index_text(content: str, path: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": path, "component": _component_for_path(path)}
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        summary["error"] = str(exc)
        return summary

    weight_map = data.get("weight_map", {})
    metadata = data.get("metadata", {})
    if not isinstance(weight_map, dict):
        weight_map = {}
    if not isinstance(metadata, dict):
        metadata = {}

    shards = sorted({str(shard) for shard in weight_map.values()})
    summary.update(
        {
            "metadata": metadata,
            "tensor_count": len(weight_map),
            "shard_count": len(shards),
            "shards": shards,
            "total_size": metadata.get("total_size"),
        }
    )
    return summary


def _config_facts(data: dict[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}

    text_config = data.get("text_config")
    if isinstance(text_config, dict):
        facts["text_config"] = {
            key: text_config[key]
            for key in (
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "intermediate_size",
                "max_position_embeddings",
                "rope_theta",
                "rope_scaling",
                "vocab_size",
            )
            if key in text_config
        }

    vision_config = data.get("vision_config")
    if isinstance(vision_config, dict):
        facts["vision_config"] = {
            key: vision_config[key]
            for key in (
                "depth",
                "hidden_size",
                "num_heads",
                "out_hidden_size",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
            )
            if key in vision_config
        }

    for key in (
        "processor_class",
        "tokenizer_class",
        "padding_side",
        "pad_token",
        "pad_token_id",
        "eos_token",
        "eos_token_id",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "instruction_feat_dim",
        "num_instruction_feature_layers",
        "reduce_type",
    ):
        if key in data:
            facts[key] = data[key]

    return facts


def _component_classes(data: dict[str, Any]) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, str) for item in value)
        ):
            classes[key] = value
    return classes


def _group_component_configs(
    configs: Iterable[dict[str, Any]],
    chat_templates: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {component: [] for component in REMOTE_COMPONENTS}
    for summary in [*configs, *chat_templates]:
        component = str(summary.get("component", "root"))
        groups.setdefault(component, []).append(summary)
    return groups


def _resolve_remote_revision(source: str, revision: str | None) -> str:
    if revision:
        return revision
    if source == OFFICIAL_TURBO_HF_ID:
        return OFFICIAL_TURBO_REVISION
    raise BooguTurboMlxError(
        f"Remote manifest for non-official source {source!r} requires --revision "
        "so the audit is reproducible."
    )


def _load_hf_tools() -> tuple[type[Any], Callable[..., str]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise BooguTurboMlxError(
            "Remote Hugging Face manifests require huggingface-hub. "
            "Install `boogu-turbo-mlx[conversion]` or add `huggingface-hub`."
        ) from exc
    return HfApi, hf_hub_download


def _download_remote_text_file(
    hf_hub_download: Callable[..., str],
    *,
    repo_id: str,
    revision: str,
    filename: str,
) -> str:
    local_path = hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)
    return Path(local_path).read_text(encoding="utf-8")


def _should_download_remote_metadata(path: str, size: Any) -> bool:
    if path.endswith(".safetensors"):
        return False

    name = path.rsplit("/", 1)[-1]
    is_metadata = (
        path.endswith(".safetensors.index.json")
        or name in _REMOTE_JSON_METADATA_FILES
        or name in _REMOTE_TEXT_METADATA_FILES
    )
    if not is_metadata:
        return False

    if isinstance(size, int) and size > MAX_REMOTE_METADATA_BYTES:
        return False
    return True


def _summarize_remote_file(sibling: Any) -> dict[str, Any]:
    path = _get_attr(sibling, "rfilename") or _get_attr(sibling, "path")
    size = _get_attr(sibling, "size")
    lfs = _get_attr(sibling, "lfs")
    if not isinstance(lfs, dict):
        lfs = {}

    summary: dict[str, Any] = {
        "path": path,
        "size": size,
        "component": _component_for_path(str(path)),
    }
    lfs_sha256 = lfs.get("sha256") or lfs.get("oid")
    if lfs_sha256:
        summary["lfs_sha256"] = lfs_sha256
    return summary


def _component_for_path(path: str) -> str:
    if "/" not in path:
        return "root"
    component = path.split("/", 1)[0]
    if component in REMOTE_COMPONENTS:
        return component
    return "other"


def _shard_for_component_path(path: str, component: str) -> str:
    prefix = f"{component}/"
    if component != "root" and path.startswith(prefix):
        return path[len(prefix) :]
    return path


def _siblings(info: Any) -> Iterable[Any]:
    siblings = _get_attr(info, "siblings")
    if siblings is None:
        return []
    return siblings


def _repo_id(info: Any, fallback: str) -> str:
    return str(_get_attr(info, "id") or _get_attr(info, "repo_id") or fallback)


def _resolved_revision(info: Any, fallback: str) -> str:
    return str(_get_attr(info, "sha") or fallback)


def _license(info: Any) -> str | None:
    card_data = _get_attr(info, "cardData") or _get_attr(info, "card_data") or {}
    if isinstance(card_data, dict):
        value = card_data.get("license")
        if value:
            return str(value)

    for tag in _get_attr(info, "tags") or []:
        tag_text = str(tag)
        if tag_text.startswith("license:"):
            return tag_text.split(":", 1)[1]
    return None


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _get_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _official_revisions() -> dict[str, str]:
    return {
        "boogu_github": OFFICIAL_BOOGU_GITHUB_REVISION,
        "comfyui_boogu_github": OFFICIAL_COMFYUI_BOOGU_REVISION,
        "turbo_hf": OFFICIAL_TURBO_REVISION,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
