from __future__ import annotations

import unittest

from boogu_turbo_mlx.constants import INSTRUCTION_FEATURE_DIM
from boogu_turbo_mlx.reuse_audit import (
    MlxVlmPreHeadTextAdapter,
    audit_mlx_vlm_qwen3vl_config,
    build_t2i_chat_messages,
    filter_text_encoder_weights,
    select_text_encoder_weight_keys,
)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeTextCore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, inputs: object, **kwargs: object) -> FakeArray:
        self.calls.append({"inputs": inputs, **kwargs})
        return FakeArray((2, 7, INSTRUCTION_FEATURE_DIM))


class FakeLanguageModel:
    def __init__(self) -> None:
        self.model = FakeTextCore()
        self.logit_calls = 0
        self.rope_calls: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> FakeArray:
        self.logit_calls += 1
        return FakeArray((2, 7, 151936))

    def get_rope_index(self, input_ids: object, **kwargs: object) -> tuple[str, str]:
        self.rope_calls.append({"input_ids": input_ids, **kwargs})
        return "position-ids", "rope-deltas"


class ReuseAuditTests(unittest.TestCase):
    def test_t2i_chat_messages_match_official_prompt_structure(self) -> None:
        messages = build_t2i_chat_messages("a glass library at sunrise")

        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertIn("generates high-quality images", messages[0]["content"][0]["text"])
        self.assertEqual(messages[1]["content"][0]["text"], "a glass library at sunrise")

    def test_text_weight_filter_excludes_vision_and_lm_head(self) -> None:
        weights = {
            "model.language_model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.patch_embed.proj.weight": object(),
            "lm_head.weight": object(),
            "transformer_blocks.0.weight": object(),
        }

        filtered = filter_text_encoder_weights(weights)
        self.assertEqual(
            list(filtered),
            ["model.language_model.layers.0.self_attn.q_proj.weight"],
        )
        self.assertEqual(
            select_text_encoder_weight_keys(weights),
            ["model.language_model.layers.0.self_attn.q_proj.weight"],
        )

    def test_config_audit_accepts_official_boogu_qwen3vl_shape(self) -> None:
        result = audit_mlx_vlm_qwen3vl_config(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "num_hidden_layers": 36,
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "num_attention_heads": 32,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 151936,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rope_theta": 5000000,
                    "max_position_embeddings": 262144,
                },
                "vision_config": {"hidden_size": 1152},
            }
        )

        self.assertTrue(result["compatible"])
        self.assertEqual(result["text_hidden_size"], 4096)
        self.assertEqual(result["text_layers"], 36)

    def test_adapter_candidate_uses_pre_head_core_not_logits(self) -> None:
        language_model = FakeLanguageModel()
        adapter = MlxVlmPreHeadTextAdapter(language_model)

        hidden_states = adapter.encode_hidden_states(
            FakeArray((2, 7)),
            attention_mask=FakeArray((2, 7)),
        )

        self.assertEqual(hidden_states.shape, (2, 7, INSTRUCTION_FEATURE_DIM))
        self.assertEqual(language_model.logit_calls, 0)
        self.assertEqual(len(language_model.model.calls), 1)
        self.assertEqual(language_model.model.calls[0]["position_ids"], "position-ids")


# NOTE: The slow M1 hidden-state oracle was removed. As written it called the
# top-level `Qwen3VLForConditionalGeneration` and read `output.last_hidden_state`,
# which that class never returns (it returns `logits`), so it could never pass and
# contradicted M1's own audit conclusion (use the pre-head text core, not the
# generation head). Hidden-state parity against the official text core is proven by
# `tests/test_encoding.py::ManualM5OracleTests::test_hidden_states_match_official_inner_model`,
# which calls the pre-head `model.model(...)` path and gates float32 parity at
# fail_count=0.


if __name__ == "__main__":
    unittest.main()
