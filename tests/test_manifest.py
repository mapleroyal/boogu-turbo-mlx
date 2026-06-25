from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from boogu_turbo_mlx import manifest as manifest_module
from boogu_turbo_mlx.constants import OFFICIAL_TURBO_HF_ID, OFFICIAL_TURBO_REVISION
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.manifest import generate_manifest
from safetensors_fixtures import write_safetensors_fixture


@dataclass
class FakeSibling:
    rfilename: str
    size: int
    lfs: dict[str, str] | None = None


class FakeModelInfo:
    id = OFFICIAL_TURBO_HF_ID
    sha = OFFICIAL_TURBO_REVISION
    last_modified = datetime(2026, 6, 19, tzinfo=timezone.utc)
    tags = [
        "license:apache-2.0",
        "diffusers:BooguImageTurboPipeline",
        "base_model:Qwen/Qwen3-VL-8B-Instruct",
    ]
    cardData = {"license": "apache-2.0"}
    library_name = "diffusers"
    pipeline_tag = "text-to-image"
    siblings = [
        FakeSibling("model_index.json", 512),
        FakeSibling("mllm/config.json", 2048),
        FakeSibling("mllm/model.safetensors.index.json", 4096),
        FakeSibling("mllm/model-00001-of-00004.safetensors", 9_000_000_000, {"sha256": "mllm-sha"}),
        FakeSibling("processor/tokenizer_config.json", 1024),
        FakeSibling("processor/chat_template.jinja", 1024),
        FakeSibling("transformer/config.json", 2048),
        FakeSibling("transformer/model.safetensors.index.json", 4096),
        FakeSibling("transformer/model-00001-of-00003.safetensors", 8_000_000_000),
        FakeSibling("vae/config.json", 512),
        FakeSibling("scheduler/scheduler_config.json", 512),
    ]


class FakeHfApi:
    calls: list[dict[str, object]] = []

    def model_info(self, **kwargs: object) -> FakeModelInfo:
        self.calls.append(kwargs)
        return FakeModelInfo()


class RemoteManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeHfApi.calls = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.downloaded: list[str] = []
        self.payloads = {
            "model_index.json": {
                "_class_name": "BooguImageTurboPipeline",
                "_diffusers_version": "0.35.2",
                "mllm": ["transformers", "Qwen3VLForConditionalGeneration"],
                "processor": ["transformers", "Qwen3VLProcessor"],
                "scheduler": [
                    "scheduling_flow_match_euler_discrete_time_shifting",
                    "FlowMatchEulerDiscreteScheduler",
                ],
                "transformer": ["transformer_boogu", "BooguImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            },
            "mllm/config.json": {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "torch_dtype": "bfloat16",
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "hidden_size": 4096,
                    "num_hidden_layers": 36,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "intermediate_size": 12288,
                    "max_position_embeddings": 262144,
                    "rope_theta": 5000000,
                    "vocab_size": 151936,
                },
                "vision_config": {"hidden_size": 1152, "depth": 27, "out_hidden_size": 4096},
            },
            "mllm/model.safetensors.index.json": {
                "metadata": {"total_size": 17534247392},
                "weight_map": {
                    "model.language_model.layers.0.weight": "model-00001-of-00004.safetensors",
                    "lm_head.weight": "model-00004-of-00004.safetensors",
                },
            },
            "processor/tokenizer_config.json": {
                "tokenizer_class": "Qwen2Tokenizer",
                "processor_class": "Qwen3VLProcessor",
                "padding_side": "right",
                "pad_token": "<|endoftext|>",
                "pad_token_id": 151643,
            },
            "processor/chat_template.jinja": "{% for message in messages %}{{ message.role }}{% endfor %}",
            "transformer/config.json": {
                "_class_name": "BooguImageTransformer2DModel",
                "instruction_feat_dim": 4096,
                "num_instruction_feature_layers": 1,
                "reduce_type": "mean",
            },
            "transformer/model.safetensors.index.json": {
                "metadata": {"total_size": 20585112576},
                "weight_map": {
                    "transformer_blocks.0.weight": "model-00001-of-00003.safetensors"
                },
            },
            "vae/config.json": {"_class_name": "AutoencoderKL"},
            "scheduler/scheduler_config.json": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def fake_download(self, *, repo_id: str, revision: str, filename: str) -> str:
        self.downloaded.append(filename)
        path = self.root / filename.replace("/", "__")
        payload = self.payloads[filename]
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(payload, encoding="utf-8")
        return str(path)

    def test_official_remote_manifest_uses_pinned_revision_and_metadata_only(self) -> None:
        with mock.patch.object(
            manifest_module,
            "_load_hf_tools",
            return_value=(FakeHfApi, self.fake_download),
        ):
            manifest = generate_manifest(OFFICIAL_TURBO_HF_ID)

        self.assertEqual(manifest["status"], "remote_manifest_generated")
        self.assertEqual(FakeHfApi.calls[0]["revision"], OFFICIAL_TURBO_REVISION)
        self.assertEqual(manifest["resolved_revision"], OFFICIAL_TURBO_REVISION)
        self.assertEqual(manifest["license"], "apache-2.0")
        self.assertEqual(manifest["library_name"], "diffusers")
        self.assertNotIn("mllm/model-00001-of-00004.safetensors", self.downloaded)
        self.assertIn("mllm/model-00001-of-00004.safetensors", manifest["safetensors_files"])
        self.assertIn("mllm/model.safetensors.index.json", self.downloaded)

        root_config = manifest["component_configs"]["root"][0]
        self.assertEqual(root_config["_class_name"], "BooguImageTurboPipeline")
        self.assertEqual(
            root_config["component_classes"]["mllm"],
            ["transformers", "Qwen3VLForConditionalGeneration"],
        )

        mllm_config = manifest["component_configs"]["mllm"][0]
        self.assertEqual(mllm_config["text_config"]["hidden_size"], 4096)
        self.assertEqual(mllm_config["text_config"]["num_hidden_layers"], 36)

        processor_config = manifest["component_configs"]["processor"][0]
        self.assertEqual(processor_config["padding_side"], "right")
        self.assertEqual(processor_config["processor_class"], "Qwen3VLProcessor")

        indexes = {index["component"]: index for index in manifest["safetensors_indexes"]}
        self.assertEqual(indexes["mllm"]["tensor_count"], 2)
        self.assertEqual(indexes["mllm"]["shard_count"], 2)
        self.assertEqual(indexes["mllm"]["total_size"], 17534247392)

    def test_non_official_remote_manifest_requires_revision(self) -> None:
        with self.assertRaises(BooguTurboMlxError) as ctx:
            generate_manifest("someone/model")

        self.assertIn("requires --revision", str(ctx.exception))

    def test_non_official_remote_manifest_accepts_explicit_revision(self) -> None:
        with mock.patch.object(
            manifest_module,
            "_load_hf_tools",
            return_value=(FakeHfApi, self.fake_download),
        ):
            manifest = generate_manifest("someone/model", revision="abc123")

        self.assertEqual(FakeHfApi.calls[0]["repo_id"], "someone/model")
        self.assertEqual(FakeHfApi.calls[0]["revision"], "abc123")
        self.assertEqual(manifest["requested_revision"], "abc123")


class LocalManifestTests(unittest.TestCase):
    def test_local_manifest_ignores_audit_json_and_inventory_uses_selection_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model_index.json").write_text(
                json.dumps({"_class_name": "BooguImageTurboPipeline"}),
                encoding="utf-8",
            )
            (root / "download_report.json").write_text("{}", encoding="utf-8")
            (root / "manifest.local.json").write_text("{}", encoding="utf-8")

            mllm = root / "mllm"
            mllm.mkdir()
            (mllm / "config.json").write_text(
                json.dumps({"model_type": "qwen3_vl"}),
                encoding="utf-8",
            )
            (mllm / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {
                            "model.language_model.embed.weight": "model-00001-of-00002.safetensors",
                            "missing.weight": "model-00001-of-00002.safetensors",
                            "missing.shard.weight": "model-00002-of-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_safetensors_fixture(
                mllm / "model-00001-of-00002.safetensors",
                {
                    "model.language_model.embed.weight": ("BF16", [2, 2]),
                    "model.visual.patch.weight": ("BF16", [2, 2]),
                    "lm_head.weight": ("BF16", [2, 2]),
                },
            )

            transformer = root / "transformer"
            transformer.mkdir()
            (transformer / "config.json").write_text(
                json.dumps({"_class_name": "BooguImageTransformer2DModel"}),
                encoding="utf-8",
            )
            write_safetensors_fixture(
                transformer / "diffusion_pytorch_model.safetensors",
                {
                    "transformer_blocks.0.weight": ("BF16", [1, 2]),
                    "ref_image_patch_embedder.proj.weight": ("BF16", [1, 2]),
                    "ref_image_refiner.layers.0.weight": ("BF16", [1, 2]),
                },
            )

            vae = root / "vae"
            vae.mkdir()
            (vae / "config.json").write_text(
                json.dumps({"_class_name": "AutoencoderKL"}),
                encoding="utf-8",
            )
            write_safetensors_fixture(
                vae / "diffusion_pytorch_model.safetensors",
                {
                    "decoder.conv.weight": ("F32", [1, 2]),
                    "encoder.conv.weight": ("F32", [1, 2]),
                },
            )

            manifest = generate_manifest(root)

        self.assertEqual(manifest["schema_version"], 2)
        config_paths = {config["path"] for config in manifest["configs"]}
        self.assertIn("model_index.json", config_paths)
        self.assertIn("mllm/config.json", config_paths)
        self.assertNotIn("download_report.json", config_paths)
        self.assertNotIn("manifest.local.json", config_paths)

        inventory = {entry["key"]: entry for entry in manifest["tensor_inventory"]}
        self.assertTrue(inventory["model.language_model.embed.weight"]["selected"])
        self.assertEqual(
            inventory["model.visual.patch.weight"]["exclusion_reason"],
            "qwen_vision_excluded_m2",
        )
        self.assertEqual(inventory["lm_head.weight"]["exclusion_reason"], "lm_head_excluded_m2")
        self.assertTrue(inventory["transformer_blocks.0.weight"]["selected"])
        self.assertEqual(
            inventory["ref_image_patch_embedder.proj.weight"]["exclusion_reason"],
            "reference_image_excluded_m2",
        )
        self.assertTrue(inventory["decoder.conv.weight"]["selected"])
        self.assertEqual(
            inventory["encoder.conv.weight"]["exclusion_reason"],
            "vae_encoder_excluded_m2",
        )

        mllm_index = manifest["safetensors_indexes"][0]
        validation = mllm_index["validation"]
        self.assertEqual(validation["status"], "error")
        self.assertEqual(validation["missing_shards"], ["model-00002-of-00002.safetensors"])
        self.assertEqual(
            validation["missing_tensors"],
            [{"key": "missing.weight", "shard": "model-00001-of-00002.safetensors"}],
        )
        self.assertEqual(
            {item["key"] for item in validation["extra_tensors"]},
            {"lm_head.weight", "model.visual.patch.weight"},
        )


if __name__ == "__main__":
    unittest.main()
