from __future__ import annotations

from dataclasses import dataclass

from .constants import OFFICIAL_TEXT_WEIGHT_PREFIX


RUNTIME_WEIGHT_COMPONENTS = ("mllm", "transformer", "vae")

EXPECTED_OFFICIAL_TENSOR_COUNTS = {
    "mllm": {
        "total": 750,
        "selected": 398,
        "excluded": 352,
        "exclusion_reasons": {
            "qwen_vision_excluded_m2": 351,
            "lm_head_excluded_m2": 1,
        },
    },
    "transformer": {
        "total": 942,
        "selected": 910,
        "excluded": 32,
        "exclusion_reasons": {
            "reference_image_excluded_m2": 32,
        },
    },
    "vae": {
        "total": 244,
        "selected": 138,
        "excluded": 106,
        "exclusion_reasons": {
            "vae_encoder_excluded_m2": 106,
        },
    },
}


@dataclass(frozen=True)
class TensorSelection:
    selected: bool
    exclusion_reason: str | None = None


def select_tensor(component: str, key: str) -> TensorSelection:
    if component == "mllm":
        if key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX):
            return TensorSelection(selected=True)
        if key.startswith("model.visual."):
            return TensorSelection(selected=False, exclusion_reason="qwen_vision_excluded_m2")
        if key.startswith("lm_head."):
            return TensorSelection(selected=False, exclusion_reason="lm_head_excluded_m2")
        return TensorSelection(selected=False, exclusion_reason="mllm_non_text_excluded_m2")

    if component == "transformer":
        if key.startswith(("ref_image_patch_embedder.", "ref_image_refiner.")):
            return TensorSelection(selected=False, exclusion_reason="reference_image_excluded_m2")
        return TensorSelection(selected=True)

    if component == "vae":
        if key.startswith("decoder."):
            return TensorSelection(selected=True)
        if key.startswith("encoder."):
            return TensorSelection(selected=False, exclusion_reason="vae_encoder_excluded_m2")
        return TensorSelection(selected=False, exclusion_reason="vae_non_decoder_excluded_m2")

    return TensorSelection(selected=False, exclusion_reason="component_not_converted_m2")
