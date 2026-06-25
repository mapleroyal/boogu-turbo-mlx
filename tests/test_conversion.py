from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from boogu_turbo_mlx.conversion import convert_weights, plan_conversion
from boogu_turbo_mlx.errors import BooguTurboMlxError
from safetensors_fixtures import write_safetensors_fixture


class ConversionPlanningTests(unittest.TestCase):
    def test_plan_preserves_source_shards_and_records_exclusions(self) -> None:
        manifest = {
            "schema_version": 2,
            "source": "/tmp/source",
            "safetensors_indexes": [
                {"component": "mllm", "path": "mllm/model.safetensors.index.json"},
                {
                    "component": "transformer",
                    "path": "transformer/diffusion_pytorch_model.safetensors.index.json",
                },
            ],
            "tensor_inventory": [
                {
                    "key": "model.language_model.embed.weight",
                    "component": "mllm",
                    "path": "mllm/model-00001-of-00002.safetensors",
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "byte_count": 8,
                    "selected": True,
                    "exclusion_reason": None,
                },
                {
                    "key": "lm_head.weight",
                    "component": "mllm",
                    "path": "mllm/model-00002-of-00002.safetensors",
                    "dtype": "BF16",
                    "shape": [2, 2],
                    "byte_count": 8,
                    "selected": False,
                    "exclusion_reason": "lm_head_excluded_m2",
                },
                {
                    "key": "transformer_blocks.0.weight",
                    "component": "transformer",
                    "path": "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
                    "dtype": "BF16",
                    "shape": [1, 2],
                    "byte_count": 4,
                    "selected": True,
                    "exclusion_reason": None,
                },
                {
                    "key": "ref_image_refiner.layers.0.weight",
                    "component": "transformer",
                    "path": "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
                    "dtype": "BF16",
                    "shape": [1, 2],
                    "byte_count": 4,
                    "selected": False,
                    "exclusion_reason": "reference_image_excluded_m2",
                },
                {
                    "key": "decoder.conv.weight",
                    "component": "vae",
                    "path": "vae/diffusion_pytorch_model.safetensors",
                    "dtype": "F32",
                    "shape": [1, 2],
                    "byte_count": 8,
                    "selected": True,
                    "exclusion_reason": None,
                },
                {
                    "key": "encoder.conv.weight",
                    "component": "vae",
                    "path": "vae/diffusion_pytorch_model.safetensors",
                    "dtype": "F32",
                    "shape": [1, 2],
                    "byte_count": 8,
                    "selected": False,
                    "exclusion_reason": "vae_encoder_excluded_m2",
                },
            ],
        }

        plan = plan_conversion(manifest)

        self.assertEqual(plan["dtype"], "auto")
        self.assertEqual(
            plan["components"]["mllm"]["index"],
            "weights/mllm/model.safetensors.index.json",
        )
        self.assertEqual(
            plan["components"]["mllm"]["shards"][0]["output"],
            "weights/mllm/model-00001-of-00002.safetensors",
        )
        self.assertEqual(
            plan["components"]["transformer"]["index"],
            "weights/transformer/diffusion_pytorch_model.safetensors.index.json",
        )
        self.assertEqual(
            plan["components"]["vae"]["index"],
            "weights/vae/diffusion_pytorch_model.safetensors.index.json",
        )
        self.assertEqual(
            plan["components"]["transformer"]["exclusion_reasons"],
            {"reference_image_excluded_m2": 1},
        )
        excluded = [
            item for item in plan["tensors"] if item["key"] == "encoder.conv.weight"
        ][0]
        self.assertIsNone(excluded["output_shard"])
        self.assertEqual(excluded["exclusion_reason"], "vae_encoder_excluded_m2")

    def test_convert_rejects_index_mismatch_before_loading_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "artifact"
            _write_required_metadata(source)
            write_safetensors_fixture(
                source / "mllm" / "model-00001-of-00001.safetensors",
                {"model.language_model.embed.weight": ("BF16", [2, 2])},
            )
            (source / "mllm" / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "missing.weight": "model-00001-of-00001.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_safetensors_fixture(
                source / "transformer" / "diffusion_pytorch_model.safetensors",
                {"transformer_blocks.0.weight": ("BF16", [2, 2])},
            )
            write_safetensors_fixture(
                source / "vae" / "diffusion_pytorch_model.safetensors",
                {"decoder.conv.weight": ("F32", [2, 2])},
            )

            with self.assertRaises(BooguTurboMlxError) as ctx:
                convert_weights(source, output)

        self.assertIn("index mismatches", str(ctx.exception))

    def test_convert_preflights_disk_space_before_loading_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "artifact"
            _write_required_metadata(source)

            with mock.patch(
                "boogu_turbo_mlx.conversion.generate_manifest",
                return_value=_minimal_conversion_manifest(source),
            ), mock.patch(
                "boogu_turbo_mlx.artifact_write.shutil.disk_usage",
                return_value=SimpleNamespace(free=1),
            ), mock.patch(
                "boogu_turbo_mlx.conversion._load_mlx",
            ) as load_mlx:
                with self.assertRaisesRegex(BooguTurboMlxError, "free disk space"):
                    convert_weights(source, output)

            load_mlx.assert_not_called()

    def test_convert_cleans_temporary_output_after_write_failure(self) -> None:
        class _FailingMx:
            def load(self, _path: Path) -> dict[str, object]:
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "artifact"
            _write_required_metadata(source)

            with mock.patch(
                "boogu_turbo_mlx.conversion.generate_manifest",
                return_value=_minimal_conversion_manifest(source),
            ), mock.patch(
                "boogu_turbo_mlx.artifact_write.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**12),
            ), mock.patch(
                "boogu_turbo_mlx.conversion._load_mlx",
                return_value=_FailingMx(),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    convert_weights(source, output)

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".artifact.tmp-*")), [])


class MlxConversionTests(unittest.TestCase):
    def test_convert_tiny_mlx_safetensors_round_trip(self) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "artifact"
            _write_required_metadata(source)

            mllm = source / "mllm"
            mx.save_safetensors(
                mllm / "model-00001-of-00001.safetensors",
                {
                    "model.language_model.embed.weight": mx.ones((2,), dtype=mx.bfloat16),
                    "lm_head.weight": mx.ones((2,), dtype=mx.bfloat16),
                },
            )
            (mllm / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "model.language_model.embed.weight": "model-00001-of-00001.safetensors",
                            "lm_head.weight": "model-00001-of-00001.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            transformer = source / "transformer"
            mx.save_safetensors(
                transformer / "diffusion_pytorch_model-00001-of-00001.safetensors",
                {
                    "transformer_blocks.0.weight": mx.ones((2,), dtype=mx.bfloat16),
                    "ref_image_patch_embedder.proj.weight": mx.ones((2,), dtype=mx.bfloat16),
                },
            )
            (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "transformer_blocks.0.weight": (
                                "diffusion_pytorch_model-00001-of-00001.safetensors"
                            ),
                            "ref_image_patch_embedder.proj.weight": (
                                "diffusion_pytorch_model-00001-of-00001.safetensors"
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            mx.save_safetensors(
                source / "vae" / "diffusion_pytorch_model.safetensors",
                {
                    "decoder.conv.weight": mx.ones((2,), dtype=mx.float32),
                    "encoder.conv.weight": mx.ones((2,), dtype=mx.float32),
                },
            )

            report = convert_weights(source, output)

            self.assertEqual(report["selected_tensor_count"], 3)
            self.assertEqual(report["excluded_tensor_count"], 3)
            converted_mllm = mx.load(output / "weights" / "mllm" / "model-00001-of-00001.safetensors")
            self.assertEqual(set(converted_mllm), {"model.language_model.embed.weight"})
            self.assertEqual(converted_mllm["model.language_model.embed.weight"].dtype, mx.bfloat16)

            converted_vae = mx.load(output / "weights" / "vae" / "diffusion_pytorch_model.safetensors")
            self.assertEqual(set(converted_vae), {"decoder.conv.weight"})
            self.assertEqual(converted_vae["decoder.conv.weight"].dtype, mx.float32)

            weight_map = json.loads((output / "weight_map.json").read_text(encoding="utf-8"))
            self.assertEqual(len(weight_map["tensors"]), 6)
            self.assertTrue((output / "conversion_report.json").exists())

    @unittest.skipUnless(
        os.environ.get("BOOGU_TURBO_MLX_OFFICIAL_CONVERSION"),
        "set BOOGU_TURBO_MLX_OFFICIAL_CONVERSION=1 to convert the official local source",
    )
    def test_official_local_source_counts(self) -> None:
        source = Path("models/Boogu-Image-0.1-Turbo")
        with tempfile.TemporaryDirectory() as temp_dir:
            report = convert_weights(source, Path(temp_dir) / "artifact")

        self.assertEqual(report["components"]["mllm"]["selected"], 398)
        self.assertEqual(report["components"]["mllm"]["excluded"], 352)
        self.assertEqual(report["components"]["transformer"]["selected"], 910)
        self.assertEqual(report["components"]["transformer"]["excluded"], 32)
        self.assertEqual(report["components"]["vae"]["selected"], 138)
        self.assertEqual(report["components"]["vae"]["excluded"], 106)


def _write_required_metadata(source: Path) -> None:
    (source / "processor").mkdir(parents=True)
    (source / "processor" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for component in ("mllm", "transformer", "vae"):
        (source / component).mkdir(parents=True)
        (source / component / "config.json").write_text("{}", encoding="utf-8")
    (source / "scheduler").mkdir()
    (source / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")


def _minimal_conversion_manifest(source: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source": str(source),
        "safetensors_headers": [],
        "safetensors_indexes": [],
        "tensor_inventory": [
            _selected_tensor("mllm", "mllm/model.safetensors", "model.language_model.embed.weight"),
            _selected_tensor("transformer", "transformer/model.safetensors", "block.weight"),
            _selected_tensor("vae", "vae/model.safetensors", "decoder.conv.weight"),
        ],
    }


def _selected_tensor(component: str, path: str, key: str) -> dict[str, object]:
    return {
        "key": key,
        "component": component,
        "path": path,
        "dtype": "BF16",
        "shape": [2, 2],
        "byte_count": 8,
        "selected": True,
        "exclusion_reason": None,
    }


if __name__ == "__main__":
    unittest.main()
