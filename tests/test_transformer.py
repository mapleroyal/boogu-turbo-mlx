from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from boogu_turbo_mlx.constants import OFFICIAL_BOOGU_GITHUB_REVISION
from boogu_turbo_mlx.errors import BooguTurboMlxError, ComponentNotImplementedError
from boogu_turbo_mlx.transformer import (
    BooguAttention,
    BooguDoubleStreamSelfAttentionProcessor,
    BooguImageDoubleStreamTransformerBlock,
    BooguImageTransformer,
    BooguImageTransformerBlock,
    BooguImageTransformerConfig,
    _apply_rotary_emb,
    _attention_mask,
    _flatten_parameter_shapes,
    _flatten_parameters,
    build_rotary_embeddings_for_latents,
    build_transformer_freqs_cis,
    patchify_latents,
    unpatchify_latents,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - environment dependent.
    mx = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_BOOGU_SOURCE = (
    ROOT / "models" / f"Boogu-Image-source-{OFFICIAL_BOOGU_GITHUB_REVISION}"
)


class TransformerConfigTests(unittest.TestCase):
    def test_official_config_fields_are_parsed(self) -> None:
        config = BooguImageTransformerConfig.from_dict(
            {
                "patch_size": 2,
                "in_channels": 16,
                "out_channels": None,
                "hidden_size": 3360,
                "num_layers": 40,
                "num_double_stream_layers": 8,
                "num_refiner_layers": 2,
                "num_attention_heads": 28,
                "num_kv_heads": 7,
                "multiple_of": 256,
                "ffn_dim_multiplier": None,
                "norm_eps": 1e-5,
                "axes_dim_rope": [40, 40, 40],
                "axes_lens": [2048, 1664, 1664],
                "instruction_feature_configs": {
                    "instruction_feat_dim": 4096,
                    "num_instruction_feature_layers": 1,
                    "reduce_type": "mean",
                },
                "timestep_scale": 1000.0,
            }
        )

        self.assertEqual(config.out_channels_effective, 16)
        self.assertEqual(config.head_dim, 120)
        self.assertEqual(config.kv_dim, 840)
        self.assertEqual(config.num_single_stream_layers, 32)
        self.assertEqual(config.ffn_inner_dim, 13568)
        self.assertEqual(config.preprocessed_instruction_feat_dim, 4096)

    def test_config_accepts_legacy_kv_alias_and_validates_rope(self) -> None:
        with self.assertRaises(ValueError):
            BooguImageTransformerConfig.from_dict(
                {
                    "hidden_size": 12,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "axes_dim_rope": [2, 2, 4],
                }
            )

    def test_call_without_mlx_raises_runtime_dependency_error(self) -> None:
        script = f"""
import sys

class BlockMlxImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ImportError("blocked mlx for test")
        return None

sys.meta_path.insert(0, BlockMlxImport())

from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.transformer import BooguImageTransformer, BooguImageTransformerConfig

transformer = BooguImageTransformer(
    BooguImageTransformerConfig.from_dict({repr(_tiny_config_dict())})
)
try:
    transformer(None, None, None, None, None)
except BooguTurboMlxError:
    pass
else:
    raise SystemExit("expected BooguTurboMlxError when MLX is unavailable")
"""
        _run_python_subprocess(script)


@unittest.skipIf(mx is None, "MLX is not installed")
class MlxTransformerTests(unittest.TestCase):
    def test_patchify_unpatchify_round_trip_preserves_token_order(self) -> None:
        latents = mx.arange(16, dtype=mx.float32).reshape(1, 1, 4, 4)
        tokens = patchify_latents(latents, patch_size=2)
        round_trip = unpatchify_latents(tokens, patch_size=2, channels=1, height=4, width=4)

        self.assertEqual(tokens.shape, (1, 4, 4))
        self.assertTrue(bool(mx.all(latents == round_trip).item()))

    def test_rope_builder_uses_text_lengths_and_image_grid(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        attention_mask = mx.array([[1, 1, 0], [1, 1, 1]], dtype=mx.int32)
        freqs_cis = build_transformer_freqs_cis(config)

        rope = build_rotary_embeddings_for_latents(
            config, freqs_cis, attention_mask, height=4, width=4
        )

        self.assertEqual(rope.encoder_seq_lengths, [2, 3])
        self.assertEqual(rope.seq_lengths, [6, 7])
        self.assertEqual(rope.context_rotary_emb.shape, (2, 3, 3))
        self.assertEqual(rope.noise_rotary_emb.shape, (2, 4, 3))
        self.assertEqual(rope.rotary_emb.shape, (2, 7, 3))

    def test_rope_builder_matches_official_t2i_position_ids(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)
        freqs_cis = build_transformer_freqs_cis(config)

        rope = build_rotary_embeddings_for_latents(
            config, freqs_cis, attention_mask, height=4, width=4
        )

        expected_positions = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 0, 0),
            (2, 0, 1),
            (2, 1, 0),
            (2, 1, 1),
        ]
        expected = mx.stack(
            [
                mx.concatenate(
                    [
                        freqs_cis[0][text_pos],
                        freqs_cis[1][row_pos],
                        freqs_cis[2][col_pos],
                    ]
                )
                for text_pos, row_pos, col_pos in expected_positions
            ],
            axis=0,
        )

        _assert_mx_allclose(self, rope.rotary_emb[0], expected)
        _assert_mx_allclose(self, rope.context_rotary_emb[0, :2], expected[:2])
        _assert_mx_allclose(self, rope.noise_rotary_emb[0], expected[2:])
        _assert_mx_allclose(
            self,
            rope.context_rotary_emb[0, 2],
            mx.zeros((config.head_dim // 2,), dtype=expected.dtype),
        )

    def test_attention_mask_broadcasts_like_official_processor(self) -> None:
        mask_2d = mx.array([[1, 0, 1], [1, 1, 0]], dtype=mx.int32)
        broadcast_2d = _attention_mask(mask_2d, batch_size=2)

        self.assertEqual(broadcast_2d.shape, (2, 1, 1, 3))
        self.assertTrue(
            bool(
                mx.all(
                    broadcast_2d
                    == mx.array(
                        [[[[True, False, True]]], [[[True, True, False]]]],
                        dtype=mx.bool_,
                    )
                ).item()
            )
        )

        mask_3d = mx.array(
            [
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            ],
            dtype=mx.int32,
        )
        broadcast_3d = _attention_mask(mask_3d, batch_size=2)

        self.assertEqual(broadcast_3d.shape, (2, 1, 3, 3))
        self.assertTrue(
            bool(mx.all(broadcast_3d[:, 0] == mask_3d.astype(mx.bool_)).item())
        )
        with self.assertRaises(ValueError):
            _attention_mask(mx.zeros((1, 1, 1, 1, 1), dtype=mx.bool_), batch_size=1)

    def test_gqa_attention_matches_repeated_kv_reference_math(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        mx.random.seed(0)
        attention = BooguAttention(config)
        hidden_states = mx.arange(
            2 * 3 * config.hidden_size, dtype=mx.float32
        ).reshape(2, 3, config.hidden_size) / 1001
        attention_mask = mx.array([[1, 1, 0], [1, 1, 1]], dtype=mx.int32)
        rotary_emb = _rotary_from_angles(2, 3, config.head_dim // 2)

        actual = attention(hidden_states, hidden_states, attention_mask, rotary_emb)
        expected = _manual_repeated_kv_attention(
            attention, hidden_states, hidden_states, attention_mask, rotary_emb
        )

        _assert_mx_allclose(self, actual, expected, atol=1e-4)

    @unittest.skipUnless(
        os.environ.get("BOOGU_TURBO_MLX_TRANSFORMER_ORACLE"),
        "set BOOGU_TURBO_MLX_TRANSFORMER_ORACLE=1 to compare against PyTorch",
    )
    def test_oracle_gqa_attention_matches_pytorch_reference_math(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            self.skipTest(f"transformer oracle dependencies are unavailable: {exc}")

        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        mx.random.seed(0)
        attention = BooguAttention(config)
        hidden_states = mx.arange(
            2 * 3 * config.hidden_size, dtype=mx.float32
        ).reshape(2, 3, config.hidden_size) / 1001
        attention_mask = mx.array([[1, 1, 0], [1, 1, 1]], dtype=mx.int32)
        rotary_emb = _rotary_from_angles(2, 3, config.head_dim // 2)

        actual = attention(hidden_states, hidden_states, attention_mask, rotary_emb)
        expected = _torch_repeated_kv_attention(
            attention,
            np.array(hidden_states),
            np.array(attention_mask),
            np.array(rotary_emb),
        )
        mx.eval(actual)

        np.testing.assert_allclose(np.array(actual), expected, rtol=2e-4, atol=2e-4)

    def test_double_stream_concat_split_preserves_official_token_order(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        processor = BooguDoubleStreamSelfAttentionProcessor(config)
        img = mx.arange(2 * 4 * config.hidden_size, dtype=mx.float32).reshape(
            2, 4, config.hidden_size
        )
        instruct = (
            mx.arange(2 * 3 * config.hidden_size, dtype=mx.float32).reshape(
                2, 3, config.hidden_size
            )
            + 1000
        )

        joined = processor._concat_instruction_image_features(
            [img], [instruct], encoder_seq_lengths=[2, 3], seq_lengths=[6, 7]
        )[0]
        split_instruct, split_img = processor._split_instruction_image_features(
            joined, encoder_seq_lengths=[2, 3], seq_lengths=[6, 7]
        )

        _assert_mx_allclose(self, joined[0, :2], instruct[0, :2])
        _assert_mx_allclose(self, joined[0, 2:6], img[0])
        _assert_mx_allclose(
            self, joined[0, 6], mx.zeros((config.hidden_size,), dtype=joined.dtype)
        )
        _assert_mx_allclose(self, joined[1, :3], instruct[1])
        _assert_mx_allclose(self, joined[1, 3:7], img[1])
        _assert_mx_allclose(self, split_instruct[0, :2], instruct[0, :2])
        _assert_mx_allclose(
            self,
            split_instruct[0, 2],
            mx.zeros((config.hidden_size,), dtype=split_instruct.dtype),
        )
        _assert_mx_allclose(self, split_instruct[1], instruct[1])
        _assert_mx_allclose(self, split_img, img)

    def test_single_and_double_stream_blocks_run_directly(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        temb = mx.ones((1, config.conditioning_dim), dtype=mx.float32)

        single = BooguImageTransformerBlock(config, modulation=True)
        hidden_states = mx.arange(3 * config.hidden_size, dtype=mx.float32).reshape(
            1, 3, config.hidden_size
        ) / 13
        single_out = single(
            hidden_states,
            mx.array([[1, 1, 1]], dtype=mx.bool_),
            _rotary_from_angles(1, 3, config.head_dim // 2),
            temb,
        )

        double = BooguImageDoubleStreamTransformerBlock(config)
        img_hidden_states = mx.arange(
            4 * config.hidden_size, dtype=mx.float32
        ).reshape(1, 4, config.hidden_size) / 19
        instruct_hidden_states = mx.arange(
            3 * config.hidden_size, dtype=mx.float32
        ).reshape(1, 3, config.hidden_size) / 23
        img_out, instruct_out = double(
            img_hidden_states,
            instruct_hidden_states,
            mx.array([[1, 1, 1, 1]], dtype=mx.bool_),
            mx.array([[1, 1, 1, 1, 1, 1, 1]], dtype=mx.bool_),
            _rotary_from_angles(1, 4, config.head_dim // 2),
            _rotary_from_angles(1, 7, config.head_dim // 2),
            temb,
            encoder_seq_lengths=[3],
            seq_lengths=[7],
        )

        mx.eval(single_out, img_out, instruct_out)
        self.assertEqual(single_out.shape, hidden_states.shape)
        self.assertEqual(img_out.shape, img_hidden_states.shape)
        self.assertEqual(instruct_out.shape, instruct_hidden_states.shape)
        self.assertFalse(bool(mx.any(mx.isnan(single_out)).item()))
        self.assertFalse(bool(mx.any(mx.isnan(img_out)).item()))
        self.assertFalse(bool(mx.any(mx.isnan(instruct_out)).item()))

    def test_tiny_full_transformer_forward_shape(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        transformer = BooguImageTransformer(config)
        latents = mx.zeros((1, 1, 4, 4), dtype=mx.float32)
        timestep = mx.array([0.5], dtype=mx.float32)
        instruction = mx.zeros((1, 3, 5), dtype=mx.float32)
        mask = mx.array([[1, 1, 1]], dtype=mx.int32)

        out = transformer(
            latents,
            timestep,
            instruction,
            build_transformer_freqs_cis(config),
            mask,
        )

        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 4, 4))

    def test_prepared_transformer_inputs_match_public_forward(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        transformer = BooguImageTransformer(config)
        latents = mx.zeros((1, 1, 4, 4), dtype=mx.float32)
        timestep = mx.array([0.5], dtype=mx.float32)
        instruction = mx.zeros((1, 3, 5), dtype=mx.float32)
        mask = mx.array([[1, 1, 1]], dtype=mx.int32)
        freqs = build_transformer_freqs_cis(config)

        prepared = transformer.prepare_forward_inputs(freqs, mask, 4, 4)
        baseline = transformer(latents, timestep, instruction, freqs, mask)
        hoisted = transformer(
            latents,
            timestep,
            instruction,
            freqs,
            mask,
            prepared_inputs=prepared,
        )

        mx.eval(baseline, hoisted)
        self.assertLessEqual(float(mx.max(mx.abs(baseline - hoisted)).item()), 0.0)

    def test_ref_image_branch_is_milestone_gated(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        transformer = BooguImageTransformer(config)
        with self.assertRaises(ComponentNotImplementedError):
            transformer(
                mx.zeros((1, 1, 4, 4), dtype=mx.float32),
                mx.array([0.5], dtype=mx.float32),
                mx.zeros((1, 3, 5), dtype=mx.float32),
                build_transformer_freqs_cis(config),
                mx.array([[1, 1, 1]], dtype=mx.int32),
                ref_image_hidden_states=[[mx.zeros((1, 4, 4), dtype=mx.float32)]],
            )

    def test_from_pretrained_loads_strict_m2_artifact(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_transformer_artifact(root, config)

            loaded = BooguImageTransformer.from_pretrained(root)

        self.assertEqual(loaded.config.hidden_size, config.hidden_size)

    def test_from_pretrained_rejects_missing_extra_and_mismatched_tensors(self) -> None:
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = "x_embedder.bias"
            _write_tiny_transformer_artifact(root, config, missing={missing})
            with self.assertRaises(BooguTurboMlxError) as ctx:
                BooguImageTransformer.from_pretrained(root)
            self.assertIn("missing selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_transformer_artifact(root, config, extra={"ref_image_patch_embedder.weight"})
            with self.assertRaises(BooguTurboMlxError) as ctx:
                BooguImageTransformer.from_pretrained(root)
            self.assertIn("unexpected selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_transformer_artifact(
                root,
                config,
                shape_overrides={"x_embedder.bias": (config.hidden_size + 1,)},
            )
            with self.assertRaises(BooguTurboMlxError) as ctx:
                BooguImageTransformer.from_pretrained(root)
            self.assertIn("shape-mismatched tensors", str(ctx.exception))


class ManualM3OfficialOracleTests(unittest.TestCase):
    ORACLE_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_ORACLE"
    ORACLE_SOURCE_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_ORACLE_SOURCE"
    OFFICIAL_WEIGHTS_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_OFFICIAL_WEIGHTS_ORACLE"
    FULL_ORACLE_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_FULL_ORACLE"
    ORACLE_ARTIFACT_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_ORACLE_ARTIFACT"
    OFFICIAL_SOURCE_WEIGHTS_ENV = "BOOGU_TURBO_MLX_TRANSFORMER_OFFICIAL_SOURCE_WEIGHTS"

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M3 transformer oracle",
    )
    def test_official_rope_matches_mlx_t2i_positions(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"M3 official oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        attention_mask = mx.array([[1, 1, 0], [1, 1, 1]], dtype=mx.int32)
        mlx_rope = build_rotary_embeddings_for_latents(
            config,
            build_transformer_freqs_cis(config),
            attention_mask,
            height=4,
            width=4,
        )

        rope_embedder = official.Rope(
            theta=config.theta,
            axes_dim=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            patch_size=config.patch_size,
        )
        torch_rope = rope_embedder(
            rope_embedder.get_freqs_cis(
                config.axes_dim_rope, config.axes_lens, config.theta
            ),
            torch.tensor(np.array(attention_mask), dtype=torch.int32),
            [[0], [0]],
            [4, 4],
            [None, None],
            [(4, 4), (4, 4)],
            torch.device("cpu"),
        )

        for actual, expected in [
            (mlx_rope.context_rotary_emb, torch_rope[0]),
            (mlx_rope.ref_img_rotary_emb, torch_rope[1]),
            (mlx_rope.noise_rotary_emb, torch_rope[2]),
            (mlx_rope.rotary_emb, torch_rope[3]),
            (mlx_rope.combined_img_rotary_emb, torch_rope[6]),
        ]:
            np.testing.assert_allclose(
                np.array(actual),
                expected.detach().cpu().numpy(),
                rtol=1e-6,
                atol=1e-6,
            )
        self.assertEqual(mlx_rope.encoder_seq_lengths, torch_rope[4])
        self.assertEqual(mlx_rope.seq_lengths, torch_rope[5])
        self.assertEqual(mlx_rope.combined_img_seq_lengths, torch_rope[7])

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M3 transformer oracle",
    )
    def test_single_block_matches_official_module(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"M3 official oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        mx.random.seed(11)
        mlx_block = BooguImageTransformerBlock(config, modulation=True)
        torch_block = official.Block(
            dim=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            modulation=True,
        )
        _copy_mlx_parameters_to_torch(mlx_block, torch_block)
        torch_block.eval()

        hidden_states = mx.arange(3 * config.hidden_size, dtype=mx.float32).reshape(
            1, 3, config.hidden_size
        ) / 17
        attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)
        rotary_emb = _rotary_from_angles(1, 3, config.head_dim // 2)
        temb = mx.arange(config.conditioning_dim, dtype=mx.float32).reshape(
            1, config.conditioning_dim
        ) / 29

        actual = mlx_block(hidden_states, attention_mask, rotary_emb, temb)
        with torch.no_grad():
            expected = torch_block(
                _to_torch(hidden_states),
                _to_torch(attention_mask).bool(),
                _to_torch(rotary_emb),
                _to_torch(temb),
            )
        _assert_numpy_allclose(actual, expected, atol=3e-4)

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M3 transformer oracle",
    )
    def test_double_stream_block_matches_official_module(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"M3 official oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        mx.random.seed(13)
        mlx_block = BooguImageDoubleStreamTransformerBlock(config)
        torch_block = official.DoubleBlock(
            dim=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            modulation=True,
        )
        _copy_mlx_parameters_to_torch(mlx_block, torch_block)
        torch_block.eval()

        img_hidden_states = mx.arange(
            4 * config.hidden_size, dtype=mx.float32
        ).reshape(1, 4, config.hidden_size) / 31
        instruct_hidden_states = mx.arange(
            3 * config.hidden_size, dtype=mx.float32
        ).reshape(1, 3, config.hidden_size) / 37
        img_attention_mask = mx.array([[1, 1, 1, 1]], dtype=mx.bool_)
        joint_attention_mask = mx.array([[1, 1, 1, 1, 1, 1, 1]], dtype=mx.bool_)
        image_rotary_emb = _rotary_from_angles(1, 4, config.head_dim // 2)
        rotary_emb = _rotary_from_angles(1, 7, config.head_dim // 2)
        temb = mx.arange(config.conditioning_dim, dtype=mx.float32).reshape(
            1, config.conditioning_dim
        ) / 41

        actual_img, actual_instruct = mlx_block(
            img_hidden_states,
            instruct_hidden_states,
            img_attention_mask,
            joint_attention_mask,
            image_rotary_emb,
            rotary_emb,
            temb,
            encoder_seq_lengths=[3],
            seq_lengths=[7],
        )
        with torch.no_grad():
            expected_img, expected_instruct = torch_block(
                _to_torch(img_hidden_states),
                _to_torch(instruct_hidden_states),
                _to_torch(img_attention_mask).bool(),
                _to_torch(joint_attention_mask).bool(),
                _to_torch(image_rotary_emb),
                _to_torch(rotary_emb),
                _to_torch(temb),
                encoder_seq_lengths=[3],
                seq_lengths=[7],
            )

        _assert_numpy_allclose(actual_img, expected_img, atol=1e-3)
        _assert_numpy_allclose(actual_instruct, expected_instruct, atol=1e-3)

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M3 transformer oracle",
    )
    def test_tiny_full_transformer_matches_official_module(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest(f"M3 official oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        config = BooguImageTransformerConfig.from_dict(_tiny_config_dict())
        mx.random.seed(17)
        mlx_model = BooguImageTransformer(config)
        torch_model = official.Model(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_double_stream_layers=config.num_double_stream_layers,
            num_refiner_layers=config.num_refiner_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            axes_dim_rope=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            instruction_feature_configs={
                "instruction_feat_dim": config.instruction_feat_dim,
                "num_instruction_feature_layers": config.num_instruction_feature_layers,
                "reduce_type": config.reduce_type,
            },
            timestep_scale=config.timestep_scale,
        )
        _copy_mlx_parameters_to_torch(mlx_model, torch_model)
        torch_model.eval()

        latents = mx.arange(16, dtype=mx.float32).reshape(1, 1, 4, 4) / 43
        timestep = mx.array([0.5], dtype=mx.float32)
        instruction = mx.arange(3 * 5, dtype=mx.float32).reshape(1, 3, 5) / 47
        mask = mx.array([[1, 1, 0]], dtype=mx.int32)

        actual = mlx_model(
            latents,
            timestep,
            instruction,
            build_transformer_freqs_cis(config),
            mask,
        )

        rope_embedder = official.Rope(
            theta=config.theta,
            axes_dim=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            patch_size=config.patch_size,
        )
        with torch.no_grad():
            expected = torch_model(
                _to_torch(latents),
                _to_torch(timestep),
                _to_torch(instruction),
                rope_embedder.get_freqs_cis(
                    config.axes_dim_rope, config.axes_lens, config.theta
                ),
                torch.tensor(np.array(mask), dtype=torch.int64),
                ref_image_hidden_states=None,
                return_dict=False,
            )

        _assert_numpy_allclose(actual, expected, atol=8e-4)

    @unittest.skipUnless(
        os.environ.get(OFFICIAL_WEIGHTS_ENV),
        f"set {OFFICIAL_WEIGHTS_ENV}=1 to run the official-weight M3 oracle",
    )
    def test_official_weight_noise_refiner_block_matches_official_module(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"M3 official-weight oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        artifact = self._oracle_artifact()
        config = BooguImageTransformerConfig.from_dict(
            json.loads((artifact / "transformer" / "config.json").read_text(encoding="utf-8"))
        )
        prefix = "noise_refiner.0."
        block_weights = _load_torch_safetensor_prefix(
            artifact / "weights" / "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
            prefix,
        )

        mlx_block = BooguImageTransformerBlock(config, modulation=True)
        mlx_block.load_weights(
            [
                (key.removeprefix(prefix), mx.array(value.numpy(), dtype=mx.float32))
                for key, value in sorted(block_weights.items())
            ],
            strict=True,
        )
        torch_block = official.Block(
            dim=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            modulation=True,
        )
        torch_block.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in block_weights.items()
            },
            strict=True,
        )
        torch_block.eval()

        hidden_states = mx.arange(2 * config.hidden_size, dtype=mx.float32).reshape(
            1, 2, config.hidden_size
        ) / 1009
        attention_mask = mx.array([[1, 1]], dtype=mx.bool_)
        rotary_emb = _rotary_from_angles(1, 2, config.head_dim // 2)
        temb = mx.arange(config.conditioning_dim, dtype=mx.float32).reshape(
            1, config.conditioning_dim
        ) / 1013

        actual = mlx_block(hidden_states, attention_mask, rotary_emb, temb)
        with torch.no_grad():
            expected = torch_block(
                _to_torch(hidden_states),
                _to_torch(attention_mask).bool(),
                _to_torch(rotary_emb),
                _to_torch(temb),
            )

        _assert_numpy_allclose(actual, expected, atol=2e-3)

    @unittest.skipUnless(
        os.environ.get(FULL_ORACLE_ENV),
        f"set {FULL_ORACLE_ENV}=1 to run the full real-weight M3 oracle "
        "(loads the whole transformer on both backends; ~2-3 min)",
    )
    def test_official_weight_full_forward_matches_official_module(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import numpy as np
            import torch
            from mlx.utils import tree_map
        except ImportError as exc:
            self.skipTest(f"M3 full-weight oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        artifact = self._oracle_artifact()
        source_weights = self._official_source_weights()
        config = BooguImageTransformerConfig.from_dict(
            json.loads((artifact / "transformer" / "config.json").read_text(encoding="utf-8"))
        )

        # MLX side: exercise the real M2 loader, then upcast to float32 so the
        # comparison isolates forward math from bf16 storage noise (matching the
        # per-block oracles, which compare in float32).
        mlx_model = BooguImageTransformer.from_pretrained(artifact)
        mlx_model.update(
            tree_map(lambda p: p.astype(mx.float32), mlx_model.parameters())
        )
        mx.eval(mlx_model.parameters())

        # Official side: load the full original safetensors (including the unused
        # ref-image tensors the official module constructs) at float32.
        official_weights = _load_torch_safetensor_prefix(
            source_weights,
            "diffusion_pytorch_model.safetensors.index.json",
            "",
        )
        torch_model = official.Model(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_double_stream_layers=config.num_double_stream_layers,
            num_refiner_layers=config.num_refiner_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            axes_dim_rope=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            instruction_feature_configs={
                "instruction_feat_dim": config.instruction_feat_dim,
                "num_instruction_feature_layers": config.num_instruction_feature_layers,
                "reduce_type": config.reduce_type,
            },
            timestep_scale=config.timestep_scale,
        )
        torch_model.load_state_dict(official_weights, strict=True)
        torch_model.eval()

        # Small but path-complete input: 16x16 latents -> 8x8 image tokens, a
        # short instruction sequence, exercising every real weight and code path.
        height = width = 16
        instr_len = 8
        latents = mx.arange(
            config.in_channels * height * width, dtype=mx.float32
        ).reshape(1, config.in_channels, height, width) / 9973
        timestep = mx.array([0.5], dtype=mx.float32)
        instruction = mx.arange(
            instr_len * config.instruction_feat_dim, dtype=mx.float32
        ).reshape(1, instr_len, config.instruction_feat_dim) / 99991
        mask = mx.ones((1, instr_len), dtype=mx.int32)

        actual = mlx_model(
            latents,
            timestep,
            instruction,
            build_transformer_freqs_cis(config),
            mask,
        )

        rope_embedder = official.Rope(
            theta=config.theta,
            axes_dim=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            patch_size=config.patch_size,
        )
        with torch.no_grad():
            expected = torch_model(
                _to_torch(latents),
                _to_torch(timestep),
                _to_torch(instruction),
                rope_embedder.get_freqs_cis(
                    config.axes_dim_rope, config.axes_lens, config.theta
                ),
                torch.tensor(np.array(mask), dtype=torch.int64),
                ref_image_hidden_states=None,
                return_dict=False,
            )

        # Observed max abs diff is ~2.8e-3 across the full 40-layer real-weight
        # forward; 6e-3 leaves cross-machine BLAS margin.
        _assert_numpy_allclose(actual, expected, atol=6e-3)

    def _oracle_source(self) -> Path:
        source = os.environ.get(
            self.ORACLE_SOURCE_ENV,
            str(DEFAULT_OFFICIAL_BOOGU_SOURCE),
        )
        path = Path(source)
        expected = path / "boogu" / "models" / "transformers" / "transformer_boogu.py"
        if not expected.exists():
            self.skipTest(
                f"official Boogu source not found at {path}; set {self.ORACLE_SOURCE_ENV}"
            )
        return path

    def _oracle_artifact(self) -> Path:
        artifact = os.environ.get(
            self.ORACLE_ARTIFACT_ENV,
            str(ROOT / "artifacts" / "boogu-mlx"),
        )
        path = Path(artifact)
        if not (path / "artifact.json").exists():
            self.skipTest(
                f"M2 artifact not found at {path}; set {self.ORACLE_ARTIFACT_ENV}"
            )
        return path

    def _official_source_weights(self) -> Path:
        weights = os.environ.get(
            self.OFFICIAL_SOURCE_WEIGHTS_ENV,
            str(ROOT / "models" / "Boogu-Image-0.1-Turbo" / "transformer"),
        )
        path = Path(weights)
        if not (path / "diffusion_pytorch_model.safetensors.index.json").exists():
            self.skipTest(
                f"official source transformer weights not found at {path}; "
                f"set {self.OFFICIAL_SOURCE_WEIGHTS_ENV}"
            )
        return path


def _tiny_config_dict() -> dict[str, object]:
    return {
        "patch_size": 2,
        "in_channels": 1,
        "out_channels": None,
        "hidden_size": 12,
        "num_layers": 2,
        "num_double_stream_layers": 1,
        "num_refiner_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "multiple_of": 4,
        "ffn_dim_multiplier": None,
        "norm_eps": 1e-5,
        "axes_dim_rope": [2, 2, 2],
        "axes_lens": [32, 32, 32],
        "instruction_feature_configs": {
            "instruction_feat_dim": 5,
            "num_instruction_feature_layers": 1,
            "reduce_type": "mean",
        },
        "timestep_scale": 1.0,
    }


def _rotary_from_angles(batch_size: int, length: int, pairs: int) -> object:
    angles = mx.arange(batch_size * length * pairs, dtype=mx.float32).reshape(
        batch_size, length, pairs
    )
    angles = angles / 11
    return (mx.cos(angles) + 1j * mx.sin(angles)).astype(mx.complex64)


def _manual_repeated_kv_attention(
    attention: BooguAttention,
    hidden_states: object,
    encoder_hidden_states: object,
    attention_mask: object,
    rotary_emb: object,
) -> object:
    batch_size = hidden_states.shape[0]
    query = attention.to_q(hidden_states)
    key = attention.to_k(encoder_hidden_states)
    value = attention.to_v(encoder_hidden_states)

    query = query.reshape(batch_size, -1, attention.heads, attention.head_dim)
    key = key.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)
    value = value.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)

    query = attention.norm_q(query)
    key = attention.norm_k(key)
    query = _apply_rotary_emb(query, rotary_emb)
    key = _apply_rotary_emb(key, rotary_emb)

    query = query.transpose(0, 2, 1, 3)
    key = key.transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    repeat_factor = attention.heads // attention.kv_heads
    key = mx.repeat(key, repeat_factor, axis=1)
    value = mx.repeat(value, repeat_factor, axis=1)

    scores = (query @ key.transpose(0, 1, 3, 2)) * attention.scale
    mask = _attention_mask(attention_mask, batch_size)
    if mask is not None:
        scores = mx.where(
            mask,
            scores,
            mx.full(scores.shape, -1e9, dtype=scores.dtype),
        )
    weights = mx.softmax(scores, axis=-1)
    hidden_states = weights @ value
    hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
        batch_size, -1, attention.heads * attention.head_dim
    )
    return attention.to_out[0](hidden_states)


def _torch_repeated_kv_attention(
    attention: BooguAttention,
    hidden_states: object,
    attention_mask: object,
    rotary_emb: object,
) -> object:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as torch_functional
    except ImportError as exc:  # pragma: no cover - optional oracle dependency.
        raise unittest.SkipTest(
            f"transformer oracle dependencies are unavailable: {exc}"
        ) from exc

    def to_torch(value: object) -> object:
        return torch.from_numpy(np.array(value)).to(torch.float32)

    params = attention.parameters()
    hidden = torch.from_numpy(hidden_states).to(torch.float32)
    mask = torch.from_numpy(attention_mask.astype(bool))
    rotary = torch.from_numpy(rotary_emb)

    query = torch_functional.linear(hidden, to_torch(params["to_q"]["weight"]))
    key = torch_functional.linear(hidden, to_torch(params["to_k"]["weight"]))
    value = torch_functional.linear(hidden, to_torch(params["to_v"]["weight"]))

    batch_size = hidden.shape[0]
    query = query.reshape(batch_size, -1, attention.heads, attention.head_dim)
    key = key.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)
    value = value.reshape(batch_size, -1, attention.kv_heads, attention.head_dim)

    query = _torch_rms_norm(query, to_torch(params["norm_q"]["weight"]), eps=1e-5)
    key = _torch_rms_norm(key, to_torch(params["norm_k"]["weight"]), eps=1e-5)
    query = _torch_apply_rotary_emb(query, rotary)
    key = _torch_apply_rotary_emb(key, rotary)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    repeat_factor = attention.heads // attention.kv_heads
    key = key.repeat_interleave(repeat_factor, dim=1)
    value = value.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(query, key.transpose(-2, -1)) * attention.scale
    scores = scores.masked_fill(~mask[:, None, None, :], -1e9)
    weights = torch.softmax(scores, dim=-1)
    hidden = torch.matmul(weights, value)
    hidden = hidden.transpose(1, 2).reshape(
        batch_size, -1, attention.heads * attention.head_dim
    )
    out = torch_functional.linear(hidden, to_torch(params["to_out"][0]["weight"]))
    return out.detach().cpu().numpy()


def _torch_rms_norm(x: object, weight: object, *, eps: float) -> object:
    import torch

    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight


def _torch_apply_rotary_emb(x: object, freqs_cis: object) -> object:
    import torch

    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    )
    out = torch.view_as_real(x_complex * freqs_cis.unsqueeze(2)).flatten(3)
    return out.type_as(x)


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


def _assert_numpy_allclose(actual: object, expected: object, *, atol: float) -> None:
    import numpy as np

    if mx is not None:
        mx.eval(actual)
    if hasattr(expected, "detach"):
        expected = expected.detach().cpu().numpy()
    np.testing.assert_allclose(
        np.array(actual),
        expected,
        rtol=atol,
        atol=atol,
    )


def _to_torch(value: object) -> object:
    import numpy as np
    import torch

    array = np.array(value)
    if array.dtype == np.bool_:
        return torch.from_numpy(array)
    if np.issubdtype(array.dtype, np.complexfloating):
        return torch.from_numpy(array)
    return torch.from_numpy(array).to(torch.float32)


class _OfficialBooguModules:
    def __init__(self, source: Path) -> None:
        os.environ.setdefault("device", "cpu")
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
        from boogu.models.transformers.transformer_boogu import (
            BooguImageDoubleStreamTransformerBlock,
            BooguImageTransformer2DModel,
            BooguImageTransformerBlock,
        )

        self.Block = BooguImageTransformerBlock
        self.DoubleBlock = BooguImageDoubleStreamTransformerBlock
        self.Model = BooguImageTransformer2DModel
        self.Rope = BooguImageDoubleStreamRotaryPosEmbed


def _official_boogu_modules(source: Path) -> _OfficialBooguModules:
    try:
        return _OfficialBooguModules(source)
    except ImportError as exc:
        raise unittest.SkipTest(
            "official Boogu source dependencies are unavailable: " f"{exc}"
        ) from exc


def _copy_mlx_parameters_to_torch(mlx_module: object, torch_module: object) -> None:
    import numpy as np
    import torch

    state = torch_module.state_dict()
    for name, value in _flatten_parameters(mlx_module.parameters()):
        if name not in state:
            continue
        tensor = torch.from_numpy(np.array(value))
        expected_shape = tuple(state[name].shape)
        if tuple(tensor.shape) != expected_shape:
            raise AssertionError(
                f"Shape mismatch for {name}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        state[name] = tensor.to(dtype=state[name].dtype)
    torch_module.load_state_dict(state, strict=True)


def _load_torch_safetensor_prefix(
    weights_dir: Path,
    index_name: str,
    prefix: str,
) -> dict[str, object]:
    import torch
    from safetensors import safe_open

    index = json.loads((weights_dir / index_name).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not keys:
        raise AssertionError(f"No tensors found with prefix {prefix!r}")

    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(weights_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key).to(torch.float32)
    return tensors


def _write_tiny_transformer_artifact(
    root: Path,
    config: BooguImageTransformerConfig,
    *,
    missing: set[str] | None = None,
    extra: set[str] | None = None,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
) -> None:
    missing = missing or set()
    extra = extra or set()
    shape_overrides = shape_overrides or {}

    transformer = BooguImageTransformer(config)
    shapes = _flatten_parameter_shapes(transformer.parameters())
    arrays = {
        key: mx.zeros(shape_overrides.get(key, shape), dtype=mx.float32)
        for key, shape in shapes.items()
        if key not in missing
    }
    for key in extra:
        arrays[key] = mx.zeros((1,), dtype=mx.float32)

    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    (root / "weights" / "transformer").mkdir(parents=True)
    shard_name = "diffusion_pytorch_model-00001-of-00001.safetensors"
    mx.save_safetensors(root / "weights" / "transformer" / shard_name, arrays)
    (root / "weights" / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {key: shard_name for key in arrays}}),
        encoding="utf-8",
    )
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "boogu-turbo-mlx-artifact",
                "components": {
                    "transformer": {
                        "config": "transformer/config.json",
                        "weights_index": "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _run_python_subprocess(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath = [str(root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise AssertionError(
            "Python subprocess failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    unittest.main()
