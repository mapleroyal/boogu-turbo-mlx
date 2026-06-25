from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from boogu_turbo_mlx import (
    InstructionEncoding,
    Qwen3VLInstructionEncoder,
    encode_instruction,
)
from boogu_turbo_mlx.constants import DEFAULT_MAX_SEQUENCE_LENGTH
from boogu_turbo_mlx.encoding import (
    OFFICIAL_TEXT_WEIGHT_PREFIX,
    Qwen3VLTextAttention,
    Qwen3VLTextConfig,
    _apply_rotary_pos_emb,
    _causal_padding_attention_mask,
    _flatten_parameter_shapes,
    _load_tokenizer,
    build_text_position_ids,
    build_text_rotary_embeddings,
)
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.reuse_audit import build_t2i_chat_messages

try:
    import mlx.core as mx
    from mlx.utils import tree_map
except ImportError:  # pragma: no cover - environment dependent.
    mx = None
    tree_map = None


ROOT = Path(__file__).resolve().parents[1]
LOCAL_OFFICIAL = ROOT / "models" / "Boogu-Image-0.1-Turbo"


class InstructionConfigTests(unittest.TestCase):
    def test_official_qwen3vl_text_config_is_parsed(self) -> None:
        config = Qwen3VLTextConfig.from_mllm_config(_official_mllm_config())

        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 36)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.mrope_section, (24, 20, 20))

    def test_config_rejects_unsupported_variants(self) -> None:
        payload = _official_mllm_config()
        payload["text_config"]["hidden_act"] = "gelu"
        with self.assertRaises(ValueError):
            Qwen3VLTextConfig.from_mllm_config(payload)

        payload = _official_mllm_config()
        payload["text_config"]["rope_scaling"]["mrope_interleaved"] = False
        with self.assertRaises(ValueError):
            Qwen3VLTextConfig.from_mllm_config(payload)

    def test_encode_instruction_validates_public_inputs(self) -> None:
        with self.assertRaises(ValueError):
            encode_instruction("")
        with self.assertRaises(ValueError):
            encode_instruction("a quiet glass library", max_sequence_length=0)
        with self.assertRaises(ValueError) as ctx:
            encode_instruction("a quiet glass library")
        self.assertIn("model_path or encoder", str(ctx.exception))

    def test_encode_instruction_accepts_existing_encoder(self) -> None:
        fake = FakeEncoder()

        result = encode_instruction(
            ["a quiet glass library", "一座安静的海边图书馆"],
            encoder=fake,
            max_sequence_length=17,
            truncate=True,
        )

        self.assertEqual(result.hidden_states, "hidden")
        self.assertEqual(fake.calls[0]["max_sequence_length"], 17)
        self.assertTrue(fake.calls[0]["truncate"])


class InstructionTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        processor = LOCAL_OFFICIAL / "processor"
        if not processor.exists():
            self.skipTest(f"local official processor not found at {processor}")

    def test_tokenizer_path_reproduces_t2i_chat_template_and_right_padding(self) -> None:
        tokenizer = _load_tokenizer(LOCAL_OFFICIAL / "processor")
        prompts = [
            "a glass library at sunrise",
            "一座安静的海边图书馆 with warm window light",
        ]
        rendered = [
            tokenizer.apply_chat_template(
                build_t2i_chat_messages(prompt),
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt in prompts
        ]

        self.assertTrue(rendered[0].startswith("<|im_start|>system\n"))
        self.assertIn("generates high-quality images", rendered[0])
        # The official Turbo T2I pipeline does not append an assistant generation
        # prompt; the instruction ends after the user turn.
        self.assertTrue(rendered[0].endswith("<|im_end|>\n"))
        self.assertNotIn("<|im_start|>assistant", rendered[0])

        batch = tokenizer(
            rendered,
            padding="longest",
            max_length=1280,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
        )
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            first_pad = mask.index(0) if 0 in mask else len(mask)
            self.assertTrue(all(value == 1 for value in mask[:first_pad]))
            self.assertTrue(all(value == 0 for value in mask[first_pad:]))
            if first_pad < len(mask):
                self.assertEqual(ids[first_pad], tokenizer.pad_token_id)

    def test_official_language_weight_index_selects_398_text_tensors(self) -> None:
        index_path = LOCAL_OFFICIAL / "mllm" / "model.safetensors.index.json"
        if not index_path.exists():
            self.skipTest(f"local official MLLM index not found at {index_path}")

        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        selected = [
            key for key in weight_map if key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX)
        ]
        excluded = [
            key
            for key in weight_map
            if key.startswith(("model.visual.", "lm_head."))
        ]

        self.assertEqual(len(selected), 398)
        self.assertEqual(len(excluded), 352)
        self.assertIn(
            "model.language_model.layers.35.self_attn.q_proj.weight",
            selected,
        )
        self.assertNotIn("lm_head.weight", selected)


@unittest.skipIf(mx is None, "MLX is not installed")
class MlxInstructionEncoderTests(unittest.TestCase):
    def test_text_positions_use_cumulative_non_pad_tokens(self) -> None:
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.int32)

        position_ids = build_text_position_ids(attention_mask)

        expected = mx.array(
            [
                [[0, 1, 2, 1], [0, 1, 1, 1]],
                [[0, 1, 2, 1], [0, 1, 1, 1]],
                [[0, 1, 2, 1], [0, 1, 1, 1]],
            ],
            dtype=mx.int32,
        )
        self.assertTrue(bool(mx.all(position_ids == expected).item()))

    def test_tiny_encoder_forward_returns_hidden_states_and_handles_padding(self) -> None:
        config = _tiny_text_config()
        encoder = Qwen3VLInstructionEncoder(config)
        input_ids = mx.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=mx.int32)
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.int32)

        hidden_states = encoder(input_ids, attention_mask)

        mx.eval(hidden_states)
        self.assertEqual(hidden_states.shape, (2, 4, config.hidden_size))
        self.assertFalse(bool(mx.any(mx.isnan(hidden_states)).item()))

    def test_attention_matches_repeated_kv_reference(self) -> None:
        config = _tiny_text_config(num_hidden_layers=1)
        attention = Qwen3VLTextAttention(config)
        hidden_states = mx.arange(
            2 * 4 * config.hidden_size,
            dtype=mx.float32,
        ).reshape(2, 4, config.hidden_size) / 101
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=mx.int32)
        position_ids = build_text_position_ids(attention_mask)
        position_embeddings = build_text_rotary_embeddings(
            config,
            position_ids,
            dtype=hidden_states.dtype,
        )

        actual = attention(
            hidden_states,
            position_embeddings,
            _causal_padding_attention_mask(attention_mask),
        )
        expected = _manual_repeated_kv_attention(
            attention,
            hidden_states,
            position_embeddings,
            attention_mask,
        )

        # Production uses the fused mx.fast.scaled_dot_product_attention kernel;
        # the reference does explicit matmul/softmax/matmul. They are algebraically
        # identical but the fused kernel's parallel reductions are non-deterministic
        # in float32, so run-to-run max_abs swings up to ~3e-4. 1e-3 stays well above
        # that noise floor while still catching real GQA/RoPE/mask wiring bugs, which
        # diverge by orders of magnitude more.
        _assert_mx_allclose(self, actual, expected, atol=1e-3)

    def test_from_pretrained_loads_strict_m2_artifact(self) -> None:
        config = _tiny_text_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_encoder_artifact(root, config)

            with mock.patch("boogu_turbo_mlx.encoding._load_tokenizer", return_value=object()):
                loaded = Qwen3VLInstructionEncoder.from_pretrained(root)

        self.assertEqual(loaded.config.hidden_size, config.hidden_size)

    def test_from_pretrained_rejects_missing_extra_and_mismatched_tensors(self) -> None:
        config = _tiny_text_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_encoder_artifact(root, config, missing={"norm.weight"})
            with mock.patch("boogu_turbo_mlx.encoding._load_tokenizer", return_value=object()):
                with self.assertRaises(BooguTurboMlxError) as ctx:
                    Qwen3VLInstructionEncoder.from_pretrained(root)
            self.assertIn("missing selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_encoder_artifact(root, config, extra={"lm_head.weight"})
            with mock.patch("boogu_turbo_mlx.encoding._load_tokenizer", return_value=object()):
                with self.assertRaises(BooguTurboMlxError) as ctx:
                    Qwen3VLInstructionEncoder.from_pretrained(root)
            self.assertIn("unexpected selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_encoder_artifact(
                root,
                config,
                shape_overrides={"norm.weight": (config.hidden_size + 1,)},
            )
            with mock.patch("boogu_turbo_mlx.encoding._load_tokenizer", return_value=object()):
                with self.assertRaises(BooguTurboMlxError) as ctx:
                    Qwen3VLInstructionEncoder.from_pretrained(root)
            self.assertIn("shape-mismatched tensors", str(ctx.exception))


class ManualM5OracleTests(unittest.TestCase):
    ORACLE_ENV = "BOOGU_TURBO_MLX_ENCODER_ORACLE"
    ORACLE_ARTIFACT_ENV = "BOOGU_TURBO_MLX_ENCODER_ORACLE_ARTIFACT"
    ORACLE_SOURCE_ENV = "BOOGU_TURBO_MLX_ENCODER_ORACLE_SOURCE"

    @unittest.skipUnless(
        os.environ.get("BOOGU_TURBO_MLX_OFFICIAL_ENCODER_SMOKE"),
        "set BOOGU_TURBO_MLX_OFFICIAL_ENCODER_SMOKE=1 to load official M2 encoder weights",
    )
    def test_official_encoder_artifact_loads_and_encodes_prompts(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        artifact = Path(os.environ["BOOGU_TURBO_MLX_OFFICIAL_ENCODER_SMOKE"])
        encoder = Qwen3VLInstructionEncoder.from_pretrained(artifact)
        result = encoder.encode(
            [
                "a glass library at sunrise",
                "一座安静的海边图书馆",
            ]
        )
        mx.eval(result.hidden_states)
        self.assertEqual(result.hidden_states.shape[-1], 4096)
        self.assertEqual(result.attention_mask.dtype, mx.int32)

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M5 encoder oracle",
    )
    def test_official_processor_matches_runtime_tokenizer_path(self) -> None:
        source = self._oracle_source()
        try:
            import numpy as np
            from transformers import Qwen3VLProcessor
        except ImportError as exc:
            self.skipTest(f"M5 processor oracle dependencies are unavailable: {exc}")

        prompts = _m5_oracle_prompts()
        messages = [build_t2i_chat_messages(prompt) for prompt in prompts]
        processor = Qwen3VLProcessor.from_pretrained(
            source / "processor",
            local_files_only=True,
        )
        official = processor.apply_chat_template(
            messages,
            padding="longest",
            max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            truncation=False,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )

        tokenizer = _load_tokenizer(source / "processor")
        rendered = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False,
            )
            for message in messages
        ]
        runtime = tokenizer(
            rendered,
            padding="longest",
            max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        np.testing.assert_array_equal(
            official["input_ids"].cpu().numpy(),
            np.array(runtime["input_ids"]),
        )
        np.testing.assert_array_equal(
            official["attention_mask"].cpu().numpy(),
            np.array(runtime["attention_mask"]),
        )

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M5 encoder oracle",
    )
    def test_hidden_states_match_official_inner_model(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        artifact = self._oracle_artifact()
        source = self._oracle_source()
        try:
            import numpy as np
            import torch
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError as exc:
            self.skipTest(f"M5 hidden-state oracle dependencies are unavailable: {exc}")

        prompts = _m5_oracle_prompts()
        messages = [build_t2i_chat_messages(prompt) for prompt in prompts]
        encoder = Qwen3VLInstructionEncoder.from_pretrained(artifact)
        actual = encoder.encode(prompts)
        mx.eval(actual.hidden_states, actual.attention_mask, actual.input_ids)

        processor = Qwen3VLProcessor.from_pretrained(
            source / "processor",
            local_files_only=True,
        )
        official_inputs = processor.apply_chat_template(
            messages,
            padding="longest",
            max_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            truncation=False,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        np.testing.assert_array_equal(
            official_inputs["input_ids"].cpu().numpy(),
            np.array(actual.input_ids),
        )
        np.testing.assert_array_equal(
            official_inputs["attention_mask"].cpu().numpy(),
            np.array(actual.attention_mask),
        )

        input_ids_pt = official_inputs["input_ids"]
        attention_mask_pt = official_inputs["attention_mask"]
        mask = np.array(actual.attention_mask).astype(bool)

        def hf_hidden_states(dtype: "torch.dtype") -> "np.ndarray":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                source / "mllm",
                local_files_only=True,
                torch_dtype=dtype,
            )
            model.eval()
            with torch.no_grad():
                output = model.model(
                    input_ids=input_ids_pt,
                    attention_mask=attention_mask_pt,
                    use_cache=False,
                )
            return output.last_hidden_state.float().cpu().numpy()

        def mlx_hidden_states(dtype: "mx.Dtype") -> "np.ndarray":
            params = tree_map(lambda value: value.astype(dtype), encoder.parameters())
            encoder.update(params)
            mx.eval(encoder.parameters())
            hidden = encoder(actual.input_ids, actual.attention_mask)
            mx.eval(hidden)
            return np.array(hidden.astype(mx.float32))

        def report(label: str, actual_arr: "np.ndarray", expected_arr: "np.ndarray") -> dict:
            diff = np.abs(actual_arr[mask] - expected_arr[mask])
            tolerance = 5e-2 + 5e-2 * np.abs(expected_arr[mask])
            stats = {
                "max_abs": float(np.max(diff)),
                "mean_abs": float(np.mean(diff)),
                "fail_count": int(np.count_nonzero(diff > tolerance)),
                "total": int(diff.size),
            }
            print(
                f"M5 hidden-state oracle [{label}] "
                f"max_abs={stats['max_abs']:.6f} mean_abs={stats['mean_abs']:.6f} "
                f"fail_count={stats['fail_count']}/{stats['total']}"
            )
            return stats

        # Ground truth: the official text core evaluated in float32. bf16 weights
        # widen to float32 losslessly, so this is the exact reference both backends
        # are approximating.
        expected_fp32 = hf_hidden_states(torch.float32)
        # The official pipeline itself runs the encoder in bf16; measuring its own
        # deviation from the float32 truth gives the irreducible bf16 noise floor
        # that no cross-backend bf16 comparison can beat.
        expected_bf16 = hf_hidden_states(torch.bfloat16)
        floor = report("hf-bf16 vs hf-fp32 (noise floor)", expected_bf16, expected_fp32)

        # 1) Architectural parity: in float32 the MLX text core must reproduce the
        # official text core essentially exactly. A genuine architecture/weight bug
        # fails here; bf16 rounding cannot hide behind it.
        actual_fp32 = mlx_hidden_states(mx.float32)
        self.assertEqual(actual_fp32.shape, expected_fp32.shape)
        parity = report("mlx-fp32 vs hf-fp32 (parity)", actual_fp32, expected_fp32)
        self.assertEqual(
            parity["fail_count"],
            0,
            "float32 architectural parity failed (real implementation bug): "
            f"max_abs={parity['max_abs']:.6f}, mean_abs={parity['mean_abs']:.6f}, "
            f"fail_count={parity['fail_count']}/{parity['total']}",
        )

        # 2) bf16 runtime acceptability: the shipped bf16 forward must stay within
        # the bf16 noise floor (a small margin), i.e. it adds no more error than the
        # reference's own bf16 rounding does. This is the strongest bf16 bound that
        # is physically achievable across two independent backends.
        actual_bf16 = mlx_hidden_states(mx.bfloat16)
        runtime = report("mlx-bf16 vs hf-fp32 (runtime)", actual_bf16, expected_fp32)
        margin = 3.0
        self.assertLessEqual(
            runtime["mean_abs"],
            margin * floor["mean_abs"],
            "bf16 runtime mean error exceeds the bf16 noise floor: "
            f"runtime={runtime['mean_abs']:.6f}, floor={floor['mean_abs']:.6f}",
        )
        self.assertLessEqual(
            runtime["max_abs"],
            margin * floor["max_abs"],
            "bf16 runtime max error exceeds the bf16 noise floor: "
            f"runtime={runtime['max_abs']:.6f}, floor={floor['max_abs']:.6f}",
        )

    def _oracle_artifact(self) -> Path:
        artifact = os.environ.get(self.ORACLE_ARTIFACT_ENV)
        if not artifact:
            self.skipTest(f"set {self.ORACLE_ARTIFACT_ENV} to an M2 artifact root")
        path = Path(artifact)
        if not (path / "artifact.json").exists():
            self.skipTest(f"M2 artifact not found at {path}")
        return path

    def _oracle_source(self) -> Path:
        source = os.environ.get(
            self.ORACLE_SOURCE_ENV,
            "models/Boogu-Image-0.1-Turbo",
        )
        path = Path(source)
        if not (path / "mllm" / "config.json").exists():
            self.skipTest(f"official MLLM source not found at {path}")
        if not (path / "processor").exists():
            self.skipTest(f"official processor source not found at {path}")
        return path


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        prompt: object,
        *,
        max_sequence_length: int,
        truncate: bool,
    ) -> InstructionEncoding:
        self.calls.append(
            {
                "prompt": prompt,
                "max_sequence_length": max_sequence_length,
                "truncate": truncate,
            }
        )
        return InstructionEncoding("hidden", "mask", "ids")


def _official_mllm_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_vl",
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "max_position_embeddings": 262144,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
        },
    }


def _tiny_text_config(num_hidden_layers: int = 2) -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_mllm_config(
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": 32,
                "hidden_size": 12,
                "intermediate_size": 16,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 6,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "max_position_embeddings": 128,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_interleaved": True,
                    "mrope_section": [1, 1, 1],
                },
            },
        }
    )


def _m5_oracle_prompts() -> list[str]:
    return [
        "a glass library at sunrise",
        "一座安静的海边图书馆",
    ]


def _manual_repeated_kv_attention(
    attention: Qwen3VLTextAttention,
    hidden_states: object,
    position_embeddings: tuple[object, object],
    attention_mask: object,
) -> object:
    batch_size = hidden_states.shape[0]
    query = attention.q_proj(hidden_states)
    key = attention.k_proj(hidden_states)
    value = attention.v_proj(hidden_states)

    query = query.reshape(batch_size, -1, attention.heads, attention.head_dim)
    key = key.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)
    value = value.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)

    query = attention.q_norm(query)
    key = attention.k_norm(key)
    query, key = _apply_rotary_pos_emb(query, key, position_embeddings)

    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    repeat_factor = attention.heads // attention.kv_heads
    key = mx.repeat(key, repeat_factor, axis=1)
    value = mx.repeat(value, repeat_factor, axis=1)

    scores = (query @ key.transpose(0, 1, 3, 2)) * attention.scale
    mask = _causal_padding_attention_mask(attention_mask)
    scores = mx.where(
        mask,
        scores,
        mx.full(scores.shape, -1e9, dtype=scores.dtype),
    )
    weights = mx.softmax(scores, axis=-1)
    hidden_states = weights @ value
    hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
        batch_size,
        -1,
        attention.heads * attention.head_dim,
    )
    return attention.o_proj(hidden_states)


def _write_tiny_encoder_artifact(
    root: Path,
    config: Qwen3VLTextConfig,
    *,
    missing: set[str] | None = None,
    extra: set[str] | None = None,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
) -> None:
    missing = missing or set()
    extra = extra or set()
    shape_overrides = shape_overrides or {}

    encoder = Qwen3VLInstructionEncoder(config)
    shapes = _flatten_parameter_shapes(encoder.parameters())
    arrays = {
        OFFICIAL_TEXT_WEIGHT_PREFIX + key: mx.zeros(
            shape_overrides.get(key, shape),
            dtype=mx.float32,
        )
        for key, shape in shapes.items()
        if key not in missing
    }
    for key in extra:
        arrays[key] = mx.zeros((1,), dtype=mx.float32)

    (root / "mllm").mkdir(parents=True)
    (root / "mllm" / "config.json").write_text(
        json.dumps(config.to_mllm_config()),
        encoding="utf-8",
    )
    (root / "processor").mkdir()
    (root / "weights" / "mllm").mkdir(parents=True)
    shard_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(root / "weights" / "mllm" / shard_name, arrays)
    (root / "weights" / "mllm" / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: shard_name for key in arrays},
            }
        ),
        encoding="utf-8",
    )
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "boogu-turbo-mlx-artifact",
                "components": {
                    "mllm": {
                        "config": "mllm/config.json",
                        "weights_index": "weights/mllm/model.safetensors.index.json",
                    }
                },
                "processor": "processor",
            }
        ),
        encoding="utf-8",
    )


def _assert_mx_allclose(
    test_case: unittest.TestCase,
    actual: object,
    expected: object,
    *,
    atol: float = 1e-6,
) -> None:
    mx.eval(actual, expected)
    diff = mx.max(mx.abs(actual - expected))
    test_case.assertLessEqual(float(diff.item()), atol)


if __name__ == "__main__":
    unittest.main()
