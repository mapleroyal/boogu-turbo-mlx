from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from boogu_turbo_mlx import cli, quantize_artifact
from boogu_turbo_mlx.artifacts import flatten_parameter_shapes, flatten_parameters
from boogu_turbo_mlx.constants import OFFICIAL_TEXT_WEIGHT_PREFIX, VAE_SCALE_FACTOR
from boogu_turbo_mlx.encoding import Qwen3VLInstructionEncoder, Qwen3VLTextConfig
from boogu_turbo_mlx.pipeline import BooguTurboPipeline
from boogu_turbo_mlx.quantization import (
    Q8_BITS,
    Q8_GROUP_SIZE,
    Q8_MODE,
    linear_only_predicate,
    quantized_linear_paths_for_component,
)
from boogu_turbo_mlx.dmd import sigma_schedule
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.transformer import (
    BooguImageTransformer,
    BooguImageTransformerConfig,
)
from boogu_turbo_mlx.vae import AutoencoderKLDecoder, AutoencoderKLDecoderConfig

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - environment dependent.
    mx = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]


class QuantizationMetadataTests(unittest.TestCase):
    def test_quantized_path_detection_handles_transformer_and_prefixed_mllm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_minimal_artifact_indexes(root)
            artifact = json.loads((root / "artifact.json").read_text(encoding="utf-8"))

            transformer_paths = quantized_linear_paths_for_component(
                root,
                artifact,
                "transformer",
            )
            mllm_paths = quantized_linear_paths_for_component(
                root,
                artifact,
                "mllm",
                key_transform=_strip_official_text_weight_prefix,
            )

        self.assertEqual(transformer_paths, ("block.to_q",))
        self.assertEqual(mllm_paths, ("layers.0.self_attn.q_proj",))

    def test_cli_help_lists_quantize_and_parser_rejects_unsupported_q8_settings(self) -> None:
        parser = cli.build_parser()
        stream = StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stream):
            parser.parse_args(
                [
                    "quantize",
                    "--source",
                    "artifact",
                    "--output",
                    "artifact-q4",
                    "--bits",
                    "4",
                ]
            )
        self.assertIn("argument --bits: must be 8", stream.getvalue())

        stream = StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stream):
            parser.parse_args(
                [
                    "quantize",
                    "--source",
                    "artifact",
                    "--output",
                    "artifact-q8",
                    "--group-size",
                    "64",
                ]
            )
        self.assertIn("argument --group-size: must be 32", stream.getvalue())

        stream = StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stream):
            parser.parse_args(
                [
                    "quantize",
                    "--source",
                    "artifact",
                    "--output",
                    "artifact-q8",
                    "--mode",
                    "mxfp8",
                ]
            )
        self.assertIn("invalid choice: 'mxfp8'", stream.getvalue())

    def test_quantize_preflights_disk_space_before_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "q8"
            _write_quantization_preflight_source(source)

            with mock.patch(
                "boogu_turbo_mlx.artifact_write.shutil.disk_usage",
                return_value=SimpleNamespace(free=1),
            ), mock.patch(
                "boogu_turbo_mlx.quantization._load_quantizable_component",
            ) as load_component:
                with self.assertRaisesRegex(BooguTurboMlxError, "free disk space"):
                    quantize_artifact(source, output)

            load_component.assert_not_called()


@unittest.skipIf(mx is None or nn is None, "MLX is not installed")
class MlxQuantizationTests(unittest.TestCase):
    def test_linear_only_predicate_rejects_embedding_and_norms(self) -> None:
        self.assertTrue(linear_only_predicate("linear", nn.Linear(32, 64)))
        self.assertFalse(linear_only_predicate("embed", nn.Embedding(32, 64)))
        self.assertFalse(linear_only_predicate("norm", nn.RMSNorm(32)))

    def test_tiny_q8_artifact_strict_loads_quantized_linears(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            output = Path(temp_dir) / "q8"
            _write_q8_source_artifact(source)
            source_vae_bytes = _component_file_bytes(source, "vae")

            with mock.patch(
                "boogu_turbo_mlx.encoding._load_tokenizer",
                return_value=object(),
            ):
                report = quantize_artifact(source, output)

            self.assertEqual(report["status"], "quantized")
            self.assertEqual(report["quantization"]["mode"], Q8_MODE)
            self.assertEqual(report["quantization"]["bits"], Q8_BITS)
            self.assertEqual(report["quantization"]["group_size"], Q8_GROUP_SIZE)
            self.assertEqual(report["validation"]["status"], "ok")
            self.assertFalse((output / "weight_map.json").exists())
            self.assertEqual(_component_file_bytes(output, "vae"), source_vae_bytes)

            with mock.patch(
                "boogu_turbo_mlx.encoding._load_tokenizer",
                return_value=object(),
            ):
                encoder = Qwen3VLInstructionEncoder.from_pretrained(output)
            transformer = BooguImageTransformer.from_pretrained(output)

        self.assertIsInstance(encoder.embed_tokens, nn.Embedding)
        self.assertIsInstance(encoder.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        self.assertIsInstance(encoder.layers[0].mlp.down_proj, nn.QuantizedLinear)
        self.assertIsInstance(transformer.x_embedder, nn.QuantizedLinear)
        self.assertIsInstance(
            transformer.time_caption_embed.caption_embedder[0],
            nn.RMSNorm,
        )


@unittest.skipUnless(
    os.environ.get("BOOGU_TURBO_MLX_QUANT_ORACLE"),
    "set BOOGU_TURBO_MLX_QUANT_ORACLE=1 to run the M7 q8 oracle",
)
class QuantOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        if mx is None or nn is None:
            self.skipTest("MLX is not installed")

    def test_q8_component_disk_size_is_below_plan_threshold(self) -> None:
        source, q8 = self._artifact_pair()
        report = json.loads((q8 / "quantization_report.json").read_text(encoding="utf-8"))
        source_size = sum(
            int(report["components"][component]["source_disk_size"])
            for component in ("mllm", "transformer")
        )
        output_size = sum(
            int(report["components"][component]["output_disk_size"])
            for component in ("mllm", "transformer")
        )
        ratio = output_size / source_size
        print(f"M7 q8 disk ratio: {ratio:.6f} ({output_size}/{source_size})")
        self.assertLess(ratio, 0.65)

    def test_full_pipeline_latent_drift_stays_within_q8_gate(self) -> None:
        source, q8 = self._artifact_pair()
        prompt = "a glass library at sunrise"
        sigmas = sigma_schedule(timesteps=[1.0, 250.0])
        initial_noise, renoise_noises = _oracle_noises(
            source,
            denoise_height=512,
            denoise_width=512,
            renoise_count=len(sigmas) - 1,
        )

        bf16 = _oracle_latents(
            source,
            prompt,
            sigmas,
            initial_noise,
            renoise_noises,
        )
        q8_latents = _oracle_latents(
            q8,
            prompt,
            sigmas,
            initial_noise,
            renoise_noises,
        )
        _assert_q8_latent_gate(self, bf16, q8_latents, label="full")

    def test_transformer_isolated_latent_drift_stays_within_q8_gate(self) -> None:
        source, q8 = self._artifact_pair()
        prompt = "a glass library at sunrise"
        sigmas = sigma_schedule(timesteps=[1.0, 250.0])
        initial_noise, renoise_noises = _oracle_noises(
            source,
            denoise_height=512,
            denoise_width=512,
            renoise_count=len(sigmas) - 1,
        )

        pipeline = BooguTurboPipeline.from_pretrained(source)
        encoding = pipeline.load()._components.encoder.encode(prompt)
        mx.eval(encoding.hidden_states, encoding.attention_mask, encoding.input_ids)
        pipeline.unload()

        bf16 = _oracle_latents(
            source,
            prompt,
            sigmas,
            initial_noise,
            renoise_noises,
            encoding=encoding,
        )
        q8_latents = _oracle_latents(
            q8,
            prompt,
            sigmas,
            initial_noise,
            renoise_noises,
            encoding=encoding,
        )
        _assert_q8_latent_gate(self, bf16, q8_latents, label="transformer")

    def test_q8_512_smoke_png_is_non_flat_rgb(self) -> None:
        _, q8 = self._artifact_pair()
        output = ROOT / "outputs" / "q8-smoke.png"
        image = BooguTurboPipeline.from_pretrained(q8).generate(
            "a glass library at sunrise",
            height=512,
            width=512,
            seed=0,
            timesteps=[1.0, 250.0],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)

        self.assertEqual(image.mode, "RGB")
        extrema = image.getextrema()
        self.assertTrue(all(lo < hi for lo, hi in extrema))
        self.assertNotEqual(extrema, ((0, 0), (0, 0), (0, 0)))
        self.assertNotEqual(extrema, ((255, 255), (255, 255), (255, 255)))
        print(f"M7 q8 smoke output: {output}")

    def _artifact_pair(self) -> tuple[Path, Path]:
        source = Path(
            os.environ.get(
                "BOOGU_TURBO_MLX_QUANT_SOURCE",
                str(ROOT / "artifacts" / "boogu-mlx"),
            )
        )
        q8 = Path(
            os.environ.get(
                "BOOGU_TURBO_MLX_QUANT_OUTPUT",
                str(ROOT / "artifacts" / "boogu-mlx-q8"),
            )
        )
        if not (source / "artifact.json").exists():
            self.skipTest(f"M2 artifact not found at {source}")
        if not (q8 / "artifact.json").exists():
            quantize_artifact(source, q8)
        return source, q8


def _write_minimal_artifact_indexes(root: Path) -> None:
    (root / "weights" / "mllm").mkdir(parents=True)
    (root / "weights" / "transformer").mkdir(parents=True)
    (root / "weights" / "mllm" / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    OFFICIAL_TEXT_WEIGHT_PREFIX
                    + "layers.0.self_attn.q_proj.weight": "mllm.safetensors",
                    OFFICIAL_TEXT_WEIGHT_PREFIX
                    + "layers.0.self_attn.q_proj.scales": "mllm.safetensors",
                    OFFICIAL_TEXT_WEIGHT_PREFIX
                    + "layers.0.self_attn.q_proj.biases": "mllm.safetensors",
                    OFFICIAL_TEXT_WEIGHT_PREFIX
                    + "embed_tokens.weight": "mllm.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "weights" / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "block.to_q.weight": "transformer.safetensors",
                    "block.to_q.scales": "transformer.safetensors",
                    "block.to_q.biases": "transformer.safetensors",
                    "norm.weight": "transformer.safetensors",
                },
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
                        "weights_index": "weights/mllm/model.safetensors.index.json",
                    },
                    "transformer": {
                        "weights_index": "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_quantization_preflight_source(root: Path) -> None:
    for component, index_name, shard_name in (
        ("mllm", "model.safetensors.index.json", "mllm.safetensors"),
        (
            "transformer",
            "diffusion_pytorch_model.safetensors.index.json",
            "transformer.safetensors",
        ),
        ("vae", "diffusion_pytorch_model.safetensors.index.json", "vae.safetensors"),
    ):
        directory = root / "weights" / component
        directory.mkdir(parents=True)
        (directory / shard_name).write_bytes(b"weights")
        key = (
            OFFICIAL_TEXT_WEIGHT_PREFIX + "layers.0.weight"
            if component == "mllm"
            else "layers.0.weight"
        )
        (directory / index_name).write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 7},
                    "weight_map": {key: shard_name},
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
                        "weights_index": "weights/mllm/model.safetensors.index.json",
                    },
                    "transformer": {
                        "weights_index": "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
                    },
                    "vae": {
                        "weights_index": "weights/vae/diffusion_pytorch_model.safetensors.index.json",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_q8_source_artifact(root: Path) -> None:
    text_config = _q8_text_config()
    transformer_config = _q8_transformer_config()
    vae_config = _tiny_vae_config()

    (root / "processor").mkdir(parents=True)
    (root / "scheduler").mkdir(parents=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    _write_encoder_component(root, text_config)
    _write_transformer_component(root, transformer_config)
    _write_vae_component(root, vae_config)
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "boogu-turbo-mlx-artifact",
                "processor": "processor",
                "scheduler_config": "scheduler/scheduler_config.json",
                "components": {
                    "mllm": {
                        "config": "mllm/config.json",
                        "weights_index": "weights/mllm/model.safetensors.index.json",
                    },
                    "transformer": {
                        "config": "transformer/config.json",
                        "weights_index": "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
                    },
                    "vae": {
                        "config": "vae/config.json",
                        "weights_index": "weights/vae/diffusion_pytorch_model.safetensors.index.json",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_encoder_component(root: Path, config: Qwen3VLTextConfig) -> None:
    encoder = Qwen3VLInstructionEncoder(config)
    arrays = {
        OFFICIAL_TEXT_WEIGHT_PREFIX + key: value
        for key, value in flatten_parameters(encoder.parameters())
    }
    (root / "mllm").mkdir(parents=True)
    (root / "mllm" / "config.json").write_text(
        json.dumps(config.to_mllm_config()),
        encoding="utf-8",
    )
    _write_indexed_component(
        root / "weights" / "mllm",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        arrays,
    )


def _write_transformer_component(
    root: Path,
    config: BooguImageTransformerConfig,
) -> None:
    transformer = BooguImageTransformer(config)
    arrays = {
        key: value
        for key, value in flatten_parameters(transformer.parameters())
    }
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    _write_indexed_component(
        root / "weights" / "transformer",
        "diffusion_pytorch_model-00001-of-00001.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        arrays,
    )


def _write_vae_component(root: Path, config: AutoencoderKLDecoderConfig) -> None:
    vae = AutoencoderKLDecoder(config)
    arrays = {
        key: mx.zeros(_raw_pytorch_shape(shape), dtype=mx.float32)
        for key, shape in flatten_parameter_shapes(vae.parameters()).items()
    }
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    _write_indexed_component(
        root / "weights" / "vae",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        arrays,
    )


def _write_indexed_component(
    directory: Path,
    shard_name: str,
    index_name: str,
    arrays: dict[str, object],
) -> None:
    directory.mkdir(parents=True)
    mx.eval(*arrays.values())
    mx.save_safetensors(directory / shard_name, arrays)
    (directory / index_name).write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(int(array.nbytes) for array in arrays.values())
                },
                "weight_map": {key: shard_name for key in arrays},
            }
        ),
        encoding="utf-8",
    )


def _component_file_bytes(root: Path, component: str) -> dict[str, bytes]:
    index_name = {
        "mllm": "model.safetensors.index.json",
        "transformer": "diffusion_pytorch_model.safetensors.index.json",
        "vae": "diffusion_pytorch_model.safetensors.index.json",
    }[component]
    base = root / "weights" / component
    return {
        path.name: path.read_bytes()
        for path in sorted(base.glob("*"))
        if path.name == index_name or path.suffix == ".safetensors"
    }


def _q8_text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_mllm_config(
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "max_position_embeddings": 128,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_interleaved": True,
                    "mrope_section": [3, 3, 2],
                },
            },
        }
    )


def _q8_transformer_config() -> BooguImageTransformerConfig:
    return BooguImageTransformerConfig.from_dict(
        {
            "patch_size": 2,
            "in_channels": 8,
            "out_channels": None,
            "hidden_size": 32,
            "num_layers": 2,
            "num_double_stream_layers": 1,
            "num_refiner_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "multiple_of": 32,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "axes_dim_rope": [4, 4, 8],
            "axes_lens": [32, 32, 32],
            "instruction_feature_configs": {
                "instruction_feat_dim": 32,
                "num_instruction_feature_layers": 1,
                "reduce_type": "mean",
            },
            "timestep_scale": 1.0,
        }
    )


def _tiny_vae_config() -> AutoencoderKLDecoderConfig:
    return AutoencoderKLDecoderConfig.from_dict(
        {
            "_class_name": "AutoencoderKL",
            "act_fn": "silu",
            "block_out_channels": [4, 4],
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 1,
            "layers_per_block": 1,
            "mid_block_add_attention": True,
            "norm_num_groups": 1,
            "out_channels": 3,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "use_post_quant_conv": False,
            "use_quant_conv": False,
        }
    )


def _raw_pytorch_shape(mlx_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(mlx_shape) == 4:
        out_channels, kernel_height, kernel_width, in_channels = mlx_shape
        return (out_channels, in_channels, kernel_height, kernel_width)
    return mlx_shape


def _strip_official_text_weight_prefix(key: str) -> str | None:
    if not key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX):
        return None
    return key.removeprefix(OFFICIAL_TEXT_WEIGHT_PREFIX)


def _oracle_noises(
    artifact: Path,
    *,
    denoise_height: int,
    denoise_width: int,
    renoise_count: int,
) -> tuple[object, list[object]]:
    import numpy as np

    transformer_config = BooguImageTransformer.from_pretrained(artifact).config
    latent_shape = (
        1,
        transformer_config.in_channels,
        denoise_height // VAE_SCALE_FACTOR,
        denoise_width // VAE_SCALE_FACTOR,
    )
    count = int(np.prod(latent_shape))
    initial = np.linspace(-1.0, 1.0, num=count, dtype=np.float32).reshape(latent_shape)
    noises = [
        np.linspace(
            1.0 - index * 0.1,
            -1.0 + index * 0.1,
            num=count,
            dtype=np.float32,
        ).reshape(latent_shape)
        for index in range(renoise_count)
    ]
    return (
        mx.array(initial, dtype=mx.float32),
        [mx.array(noise, dtype=mx.float32) for noise in noises],
    )


def _oracle_latents(
    artifact: Path,
    prompt: str,
    sigmas: list[float],
    initial_noise: object,
    renoise_noises: list[object],
    *,
    encoding: object | None = None,
) -> object:
    pipeline = BooguTurboPipeline.from_pretrained(artifact)
    pipeline.load()
    if encoding is None:
        encoding = pipeline._components.encoder.encode(prompt)
    latents = pipeline._generate_latents(
        encoding=encoding,
        sigmas=sigmas,
        denoise_height=512,
        denoise_width=512,
        seed=None,
        initial_noise=initial_noise,
        renoise_noises=renoise_noises,
        runtime_dtype=mx.float32,
    )
    mx.eval(latents)
    pipeline.unload()
    return latents


def _assert_q8_latent_gate(
    test_case: unittest.TestCase,
    bf16: object,
    q8: object,
    *,
    label: str,
) -> None:
    import numpy as np

    bf16_arr = np.array(bf16, dtype=np.float32)
    q8_arr = np.array(q8, dtype=np.float32)
    test_case.assertEqual(bf16_arr.shape, q8_arr.shape)
    test_case.assertTrue(np.isfinite(q8_arr).all())
    diff = np.abs(bf16_arr - q8_arr)
    cosine = float(
        np.dot(bf16_arr.reshape(-1), q8_arr.reshape(-1))
        / (np.linalg.norm(bf16_arr.reshape(-1)) * np.linalg.norm(q8_arr.reshape(-1)))
    )
    mean_abs = float(np.mean(diff))
    max_abs = float(np.max(diff))
    print(
        f"M7 q8 latent drift [{label}] "
        f"cosine={cosine:.6f} mean_abs={mean_abs:.6f} max_abs={max_abs:.6f}"
    )
    test_case.assertGreaterEqual(cosine, 0.999)
    test_case.assertLessEqual(mean_abs, 0.12)
    test_case.assertLessEqual(max_abs, 2.0)


if __name__ == "__main__":
    unittest.main()
