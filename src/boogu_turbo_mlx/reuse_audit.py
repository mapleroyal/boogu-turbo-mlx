from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .constants import INSTRUCTION_FEATURE_DIM, OFFICIAL_TEXT_WEIGHT_PREFIX
from .errors import BooguTurboMlxError

T2I_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates high-quality images based on "
    "user instructions. The instructions are as follows."
)

EXCLUDED_WEIGHT_PREFIXES = ("model.visual.", "lm_head.")

MLX_VLM_TEXT_CONFIG_REQUIRED_FIELDS = (
    "model_type",
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "rms_norm_eps",
    "vocab_size",
    "num_key_value_heads",
    "head_dim",
    "rope_theta",
    "max_position_embeddings",
)


def build_t2i_chat_messages(instruction: str) -> list[dict[str, Any]]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": T2I_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        },
    ]


def filter_text_encoder_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Select only the official Qwen3-VL language-model weights needed for T2I."""

    return {
        key: value
        for key, value in weights.items()
        if key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX)
    }


def select_text_encoder_weight_keys(weight_keys: Iterable[str]) -> list[str]:
    return sorted(key for key in weight_keys if key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX))


def audit_mlx_vlm_qwen3vl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    vision_config = config.get("vision_config")
    if not isinstance(text_config, Mapping):
        text_config = {}
    if not isinstance(vision_config, Mapping):
        vision_config = {}

    missing_text_fields = [
        field for field in MLX_VLM_TEXT_CONFIG_REQUIRED_FIELDS if field not in text_config
    ]
    compatible = (
        config.get("model_type") == "qwen3_vl"
        and not missing_text_fields
        and bool(vision_config)
    )

    return {
        "compatible": compatible,
        "model_type": config.get("model_type"),
        "missing_text_fields": missing_text_fields,
        "has_vision_config": bool(vision_config),
        "text_hidden_size": text_config.get("hidden_size"),
        "text_layers": text_config.get("num_hidden_layers"),
        "text_model_type": text_config.get("model_type"),
    }


@dataclass
class MlxVlmPreHeadTextAdapter:
    """Audit adapter that calls the mlx-vlm pre-head text core, not lm_head logits."""

    language_model: Any
    hidden_size: int = INSTRUCTION_FEATURE_DIM

    def encode_hidden_states(
        self,
        input_ids: Any,
        attention_mask: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        core = getattr(self.language_model, "model", None)
        if core is None:
            raise BooguTurboMlxError(
                "mlx-vlm LanguageModel does not expose a pre-head `.model` core."
            )

        position_ids = kwargs.pop("position_ids", None)
        if position_ids is None and hasattr(self.language_model, "get_rope_index"):
            position_ids, _ = self.language_model.get_rope_index(
                input_ids,
                attention_mask=attention_mask,
            )

        hidden_states = core(
            input_ids,
            mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        validate_hidden_state_shape(hidden_states, hidden_size=self.hidden_size)
        return hidden_states


def validate_hidden_state_shape(hidden_states: Any, *, hidden_size: int) -> tuple[int, int, int]:
    shape = getattr(hidden_states, "shape", None)
    if shape is None:
        raise BooguTurboMlxError("hidden states must expose a shape")

    shape_tuple = tuple(int(dim) for dim in shape)
    if len(shape_tuple) != 3 or shape_tuple[-1] != hidden_size:
        raise BooguTurboMlxError(
            "expected instruction hidden states shaped [B, T, "
            f"{hidden_size}], got {shape_tuple}"
        )
    return shape_tuple
