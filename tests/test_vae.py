from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from boogu_turbo_mlx.errors import BooguTurboMlxError, ComponentNotImplementedError
from boogu_turbo_mlx.safetensors_header import read_safetensors_header
from boogu_turbo_mlx.selection import select_tensor
from boogu_turbo_mlx.vae import (
    AutoencoderKLDecoder,
    AutoencoderKLDecoderConfig,
    VaeAttentionBlock,
    VaeResnetBlock2D,
    VaeUpsample2D,
    _flatten_parameter_shapes,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - environment dependent.
    mx = None


class VaeConfigTests(unittest.TestCase):
    def test_official_config_fields_are_parsed(self) -> None:
        config = AutoencoderKLDecoderConfig.from_dict(
            {
                "_class_name": "AutoencoderKL",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": [
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ],
                "force_upcast": True,
                "in_channels": 3,
                "latent_channels": 16,
                "layers_per_block": 2,
                "mid_block_add_attention": True,
                "norm_num_groups": 32,
                "out_channels": 3,
                "scaling_factor": 0.3611,
                "shift_factor": 0.1159,
                "up_block_types": [
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
                "use_post_quant_conv": False,
                "use_quant_conv": False,
            }
        )

        self.assertEqual(config.decoder_channels, (512, 512, 256, 128))
        self.assertEqual(config.layers_per_block, 2)
        self.assertTrue(config.force_upcast)
        self.assertEqual(config.scaling_factor, 0.3611)
        self.assertEqual(config.shift_factor, 0.1159)

    def test_unsupported_config_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AutoencoderKLDecoderConfig.from_dict(
                {**_tiny_vae_config_dict(), "use_quant_conv": True}
            )

    def test_decode_without_mlx_raises_runtime_dependency_error(self) -> None:
        script = f"""
import sys

class BlockMlxImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ImportError("blocked mlx for test")
        return None

sys.meta_path.insert(0, BlockMlxImport())

from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.vae import AutoencoderKLDecoder, AutoencoderKLDecoderConfig

decoder = AutoencoderKLDecoder(
    AutoencoderKLDecoderConfig.from_dict({repr(_tiny_vae_config_dict())})
)
try:
    decoder.decode(None)
except BooguTurboMlxError:
    pass
else:
    raise SystemExit("expected BooguTurboMlxError when MLX is unavailable")
"""
        _run_python_subprocess(script)

    def test_encode_is_milestone_gated(self) -> None:
        decoder = AutoencoderKLDecoder(
            AutoencoderKLDecoderConfig.from_dict(_tiny_vae_config_dict())
        )
        with self.assertRaises(ComponentNotImplementedError):
            decoder.encode(None)


@unittest.skipIf(mx is None, "MLX is not installed")
class MlxVaeTests(unittest.TestCase):
    def test_prepare_latents_applies_official_scale_and_shift(self) -> None:
        config = AutoencoderKLDecoderConfig.from_dict(_tiny_vae_config_dict())
        decoder = AutoencoderKLDecoder(config)
        latents = mx.array([[[[0.3611]]]], dtype=mx.float32)

        prepared = decoder.prepare_latents_for_decode(latents)

        self.assertAlmostEqual(float(prepared.item()), 1.1159, places=5)

    def test_resnet_shortcut_uses_mlx_conv_weight_shape(self) -> None:
        resnet = VaeResnetBlock2D(8, 4, groups=2)
        shapes = _flatten_parameter_shapes(resnet.parameters())

        self.assertEqual(shapes["conv_shortcut.weight"], (4, 1, 1, 8))

    def test_mid_attention_preserves_nhwc_shape(self) -> None:
        attention = VaeAttentionBlock(8, groups=2)
        hidden_states = mx.zeros((1, 2, 3, 8), dtype=mx.float32)

        out = attention(hidden_states)

        mx.eval(out)
        self.assertEqual(out.shape, hidden_states.shape)

    def test_upsample_doubles_spatial_shape(self) -> None:
        upsample = VaeUpsample2D(4)
        hidden_states = mx.zeros((1, 2, 3, 4), dtype=mx.float32)

        out = upsample(hidden_states)

        mx.eval(out)
        self.assertEqual(out.shape, (1, 4, 6, 4))

    def test_tiny_full_decoder_forward_shape_uses_nchw_public_boundary(self) -> None:
        config = AutoencoderKLDecoderConfig.from_dict(_tiny_vae_config_dict())
        decoder = AutoencoderKLDecoder(config)
        latents = mx.zeros((1, config.latent_channels, 2, 2), dtype=mx.float32)

        out = decoder.decode(latents)

        mx.eval(out)
        self.assertEqual(out.shape, (1, config.out_channels, 4, 4))

    def test_from_pretrained_loads_strict_m2_artifact(self) -> None:
        config = AutoencoderKLDecoderConfig.from_dict(_tiny_vae_config_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_vae_artifact(root, config)

            loaded = AutoencoderKLDecoder.from_pretrained(root)

        self.assertEqual(loaded.config.latent_channels, config.latent_channels)

    def test_from_pretrained_rejects_missing_extra_and_mismatched_tensors(self) -> None:
        config = AutoencoderKLDecoderConfig.from_dict(_tiny_vae_config_dict())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = "decoder.conv_out.bias"
            _write_tiny_vae_artifact(root, config, missing={missing})
            with self.assertRaises(BooguTurboMlxError) as ctx:
                AutoencoderKLDecoder.from_pretrained(root)
            self.assertIn("missing selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_vae_artifact(root, config, extra={"encoder.conv_in.weight"})
            with self.assertRaises(BooguTurboMlxError) as ctx:
                AutoencoderKLDecoder.from_pretrained(root)
            self.assertIn("unexpected selected tensors", str(ctx.exception))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_vae_artifact(
                root,
                config,
                shape_overrides={"decoder.conv_out.bias": (config.out_channels + 1,)},
            )
            with self.assertRaises(BooguTurboMlxError) as ctx:
                AutoencoderKLDecoder.from_pretrained(root)
            self.assertIn("shape-mismatched tensors", str(ctx.exception))

    @unittest.skipUnless(
        os.environ.get("BOOGU_TURBO_MLX_VAE_ORACLE"),
        "set BOOGU_TURBO_MLX_VAE_ORACLE=1 to compare against Diffusers",
    )
    def test_oracle_decode_matches_diffusers_autoencoderkl(self) -> None:
        artifact = os.environ.get("BOOGU_TURBO_MLX_VAE_ORACLE_ARTIFACT")
        source = os.environ.get("BOOGU_TURBO_MLX_VAE_ORACLE_SOURCE")
        if not artifact or not source:
            self.skipTest("set VAE oracle artifact and source paths")

        try:
            import numpy as np
            import torch
            from diffusers import AutoencoderKL
        except ImportError as exc:
            self.skipTest(f"VAE oracle dependencies are unavailable: {exc}")

        mlx_decoder = AutoencoderKLDecoder.from_pretrained(artifact)
        torch_decoder = AutoencoderKL.from_pretrained(
            Path(source) / "vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        torch_decoder.eval()

        latent_shape = (1, mlx_decoder.config.latent_channels, 2, 2)
        torch_latents = torch.linspace(-1, 1, steps=int(np.prod(latent_shape))).reshape(
            latent_shape
        )
        mlx_latents = mx.array(torch_latents.numpy(), dtype=mx.float32)

        with torch.no_grad():
            expected = torch_decoder.decode(
                torch_latents / mlx_decoder.config.scaling_factor
                + mlx_decoder.config.shift_factor,
                return_dict=False,
            )[0]
        actual = mlx_decoder.decode(mlx_latents)
        mx.eval(actual)

        np.testing.assert_allclose(
            np.array(actual),
            expected.numpy(),
            rtol=2e-3,
            atol=2e-3,
        )

    @unittest.skipUnless(
        os.environ.get("BOOGU_TURBO_MLX_OFFICIAL_VAE_SMOKE"),
        "set BOOGU_TURBO_MLX_OFFICIAL_VAE_SMOKE=1 to load local official VAE weights",
    )
    def test_official_local_vae_weights_load_and_decode(self) -> None:
        source = Path(
            os.environ.get(
                "BOOGU_TURBO_MLX_OFFICIAL_SOURCE",
                "models/Boogu-Image-0.1-Turbo",
            )
        )
        if not (source / "vae" / "diffusion_pytorch_model.safetensors").exists():
            self.skipTest(f"local official VAE weights not found at {source}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifact"
            _write_official_vae_artifact(root, source)

            decoder = AutoencoderKLDecoder.from_pretrained(root)
            latents = mx.zeros(
                (1, decoder.config.latent_channels, 2, 2),
                dtype=mx.float32,
            )
            decoded = decoder.decode(latents)

        mx.eval(decoded)
        self.assertEqual(decoded.shape, (1, decoder.config.out_channels, 16, 16))
        self.assertEqual(decoded.dtype, mx.float32)
        self.assertFalse(bool(mx.any(mx.isnan(decoded)).item()))


def _tiny_vae_config_dict() -> dict[str, object]:
    return {
        "_class_name": "AutoencoderKL",
        "act_fn": "silu",
        "block_out_channels": [4, 8],
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 2,
        "layers_per_block": 1,
        "mid_block_add_attention": True,
        "norm_num_groups": 2,
        "out_channels": 3,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "use_post_quant_conv": False,
        "use_quant_conv": False,
    }


def _write_tiny_vae_artifact(
    root: Path,
    config: AutoencoderKLDecoderConfig,
    *,
    missing: set[str] | None = None,
    extra: set[str] | None = None,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
) -> None:
    missing = missing or set()
    extra = extra or set()
    shape_overrides = shape_overrides or {}

    decoder = AutoencoderKLDecoder(config)
    mlx_shapes = _flatten_parameter_shapes(decoder.parameters())
    arrays = {
        key: mx.zeros(
            _raw_pytorch_shape(shape_overrides.get(key, shape)),
            dtype=mx.float32,
        )
        for key, shape in mlx_shapes.items()
        if key not in missing
    }
    for key in extra:
        arrays[key] = mx.zeros((1,), dtype=mx.float32)

    (root / "vae").mkdir(parents=True)
    (root / "vae" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    (root / "weights" / "vae").mkdir(parents=True)
    shard_name = "diffusion_pytorch_model.safetensors"
    mx.save_safetensors(root / "weights" / "vae" / shard_name, arrays)
    (root / "weights" / "vae" / "diffusion_pytorch_model.safetensors.index.json").write_text(
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
                    "vae": {
                        "config": "vae/config.json",
                        "weights_index": "weights/vae/diffusion_pytorch_model.safetensors.index.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_official_vae_artifact(root: Path, source: Path) -> None:
    vae_file = source / "vae" / "diffusion_pytorch_model.safetensors"
    header = read_safetensors_header(vae_file)
    selected_keys = sorted(
        key for key in header.tensors if select_tensor("vae", key).selected
    )
    arrays = mx.load(vae_file)
    selected_arrays = {key: arrays[key] for key in selected_keys}
    shard_name = "diffusion_pytorch_model.safetensors"

    (root / "vae").mkdir(parents=True)
    shutil.copyfile(source / "vae" / "config.json", root / "vae" / "config.json")
    (root / "weights" / "vae").mkdir(parents=True)
    mx.save_safetensors(root / "weights" / "vae" / shard_name, selected_arrays)
    (root / "weights" / "vae" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        header.tensors[key].byte_count for key in selected_keys
                    )
                },
                "weight_map": {key: shard_name for key in selected_keys},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "format": "boogu-turbo-mlx-artifact",
                "schema_version": 2,
                "components": {
                    "vae": {
                        "config": "vae/config.json",
                        "weights_index": "weights/vae/diffusion_pytorch_model.safetensors.index.json",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _raw_pytorch_shape(mlx_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(mlx_shape) == 4:
        out_channels, kernel_height, kernel_width, in_channels = mlx_shape
        return (out_channels, in_channels, kernel_height, kernel_width)
    return mlx_shape


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
