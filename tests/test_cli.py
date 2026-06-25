from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from PIL import Image

from boogu_turbo_mlx import hf
from boogu_turbo_mlx import cli
from boogu_turbo_mlx.constants import OFFICIAL_TURBO_HF_ID, OFFICIAL_TURBO_REVISION
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.pipeline import BooguTurboGenerationResult, PipelineProgressEvent


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "boogu_turbo_mlx", *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


class CliTests(unittest.TestCase):
    def test_help_lists_public_commands(self) -> None:
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("setup", result.stdout)
        self.assertIn("manifest", result.stdout)
        self.assertIn("download", result.stdout)
        self.assertIn("convert", result.stdout)
        self.assertIn("generate", result.stdout)
        self.assertIn("doctor", result.stdout)
        self.assertIn("benchmark", result.stdout)

    def test_download_source_uses_pinned_official_revision_and_dest(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            hf,
            "_load_snapshot_download",
            return_value=fake_snapshot_download,
        ):
            local_path = hf.download_source(
                OFFICIAL_TURBO_HF_ID,
                dest_root=Path(temp_dir),
            )

        self.assertEqual(local_path.name, "Boogu-Image-0.1-Turbo")
        self.assertEqual(calls[0]["repo_id"], OFFICIAL_TURBO_HF_ID)
        self.assertEqual(calls[0]["revision"], OFFICIAL_TURBO_REVISION)

    def test_download_source_can_use_exact_local_dir(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(kwargs["local_dir"])

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            hf,
            "_load_snapshot_download",
            return_value=fake_snapshot_download,
        ):
            local_dir = Path(temp_dir) / "custom-source"
            local_path = hf.download_source(
                OFFICIAL_TURBO_HF_ID,
                local_dir=local_dir,
            )

        self.assertEqual(local_path, local_dir)
        self.assertEqual(calls[0]["local_dir"], local_dir)

    def test_download_source_returns_existing_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(hf.download_source(root), root)

    def test_download_source_requires_revision_for_non_official_repos(self) -> None:
        with self.assertRaises(BooguTurboMlxError) as ctx:
            hf.download_source("someone/model")
        self.assertIn("requires --revision", str(ctx.exception))

    def test_convert_downloads_remote_source_before_conversion(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "convert",
                "--source",
                "someone/model",
                "--revision",
                "abc123",
                "--output",
                "artifact",
                "--dtype",
                "float16",
            ]
        )
        with mock.patch(
            "boogu_turbo_mlx.cli.download_source",
            return_value=Path("models/model"),
        ) as download, mock.patch(
            "boogu_turbo_mlx.cli.convert_weights",
        ) as convert:
            code = args.handler(args)

        self.assertEqual(code, 0)
        download.assert_called_once_with("someone/model", revision="abc123")
        convert.assert_called_once_with(Path("models/model"), "artifact", dtype="float16")

    def test_manifest_inspects_local_indexes_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model_index.json").write_text(
                json.dumps({"_class_name": "BooguImageTurboPipeline"}),
                encoding="utf-8",
            )
            transformer_dir = root / "transformer"
            transformer_dir.mkdir()
            (transformer_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 1234},
                        "weight_map": {
                            "block.attn.q.weight": "model-00001-of-00002.safetensors",
                            "block.attn.k.weight": "model-00002-of-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_cli("manifest", "--source", str(root))

        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(result.stdout)
        self.assertEqual(manifest["status"], "local_manifest_generated")
        self.assertEqual(manifest["safetensors_indexes"][0]["tensor_count"], 2)
        self.assertEqual(manifest["safetensors_indexes"][0]["shard_count"], 2)

    def test_generate_validates_output_size_and_parses_timesteps(self) -> None:
        parser = cli.build_parser()

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass library",
                "--size",
                "16x32",
                "--output",
                "out.png",
                "--timesteps",
                "1,250,0.5",
            ]
        )
        self.assertEqual(cli._resolve_output_size(args), (32, 16))
        self.assertEqual(args.timesteps, [1.0, 250.0, 0.5])

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass library",
                "--height",
                "16",
                "--width",
                "16",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--output is required"):
            args.handler(args)

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a glass library",
                "--size",
                "16x16",
                "--height",
                "16",
                "--output",
                "out.png",
            ]
        )
        with self.assertRaisesRegex(ValueError, "either --size or --height/--width"):
            cli._resolve_output_size(args)

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                    "--prompt",
                    "a glass library",
                    "--size",
                    "16x16",
                    "--output",
                    "out.png",
                    "--memory-mode",
                    "low-memory",
                "--denoise-batch-size",
                "1",
                "--decode-batch-size",
                "1",
            ]
        )
        self.assertEqual(args.memory_mode, "low-memory")
        self.assertEqual(args.denoise_batch_size, 1)
        self.assertEqual(args.decode_batch_size, 1)

    def test_generate_rejects_invalid_inputs_without_traceback(self) -> None:
        result = run_cli(
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass library",
            "--height",
            "0",
            "--width",
            "16",
            "--output",
            "out.png",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("argument --height: must be a positive integer", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = run_cli(
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "",
            "--height",
            "16",
            "--width",
            "16",
            "--output",
            "out.png",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("prompt must be a non-empty string", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = run_cli(
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass library",
            "--height",
            "15",
            "--width",
            "16",
            "--output",
            "out.png",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("output size height must be a multiple of 16", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = run_cli(
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass library",
            "--size",
            "2048x2064",
            "--output",
            "out.png",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("output size height must be 2048 or smaller", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        result = run_cli(
            "generate",
            "--model",
            "artifact",
            "--prompt",
            "a glass library",
            "--size",
            "16x16",
            "--seed",
            "-1",
            "--output",
            "out.png",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("argument --seed: must be from 0 to 4294967295", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_generate_saves_image_and_prints_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "image.png"
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--model",
                    "artifact",
                    "--prompt",
                    "a glass library",
                    "--height",
                    "16",
                    "--width",
                    "16",
                    "--output",
                    str(output),
                    "--timesteps",
                    "0.1,0.5",
                    "--truncate-instruction-sequence",
                ]
            )
            result = _fake_result(
                Image.new("RGB", (1, 1), (9, 8, 7)),
                prompt="a glass library",
                seed=123456,
                height=16,
                width=16,
                output_height=1,
                output_width=1,
                timesteps=(0.1, 0.5),
                sigmas=(0.1, 0.5),
            )
            fake_pipeline = mock.Mock()
            fake_pipeline.generate_batch.return_value = [result]
            stream = StringIO()
            with mock.patch.object(
                cli.BooguTurboPipeline,
                "from_pretrained",
                return_value=fake_pipeline,
            ), mock.patch(
                "boogu_turbo_mlx.cli.secrets.randbits",
                return_value=123456,
            ) as randbits, redirect_stdout(stream), redirect_stderr(StringIO()):
                code = args.handler(args)

            self.assertEqual(code, 0)
            randbits.assert_called_once_with(32)
            self.assertTrue(output.exists())
            with Image.open(output) as saved:
                metadata = json.loads(saved.info["boogu-turbo-mlx"])
                parameters = saved.info["parameters"]
            self.assertEqual(metadata["generator"], "boogu-turbo-mlx")
            self.assertEqual(metadata["prompt"], "a glass library")
            self.assertEqual(metadata["model"], "artifact")
            self.assertEqual(metadata["height"], 16)
            self.assertEqual(metadata["width"], 16)
            self.assertEqual(metadata["output_height"], 1)
            self.assertEqual(metadata["output_width"], 1)
            self.assertEqual(metadata["seed"], 123456)
            self.assertEqual(metadata["timesteps"], [0.1, 0.5])
            self.assertEqual(metadata["sigmas"], [0.1, 0.5])
            self.assertIn("a glass library", parameters)
            self.assertIn("Steps: 4", parameters)
            self.assertIn("Seed: 123456", parameters)
            self.assertIn("Timesteps: 0.1,0.5", parameters)
            self.assertEqual(stream.getvalue().strip(), str(output))
            fake_pipeline.generate_batch.assert_called_once_with(
                "a glass library",
                height=16,
                width=16,
                steps=4,
                seed=123456,
                seeds=None,
                num_images_per_prompt=1,
                timesteps=[0.1, 0.5],
                max_sequence_length=1280,
                truncate_instruction_sequence=True,
                dmd_conditioning_sigma=0.001,
                progress_callback=None,
            )

    def test_generate_batch_writes_prompt_major_outputs_and_paths_only_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "batch"
            args = cli.build_parser().parse_args(
                [
                    "generate",
                    "--model",
                    "artifact",
                    "--prompt",
                    "a glass library",
                    "--prompt",
                    "a cedar observatory",
                    "--num-images-per-prompt",
                    "2",
                    "--size",
                    "16x16",
                    "--seeds",
                    "10,11,12,13",
                    "--output-dir",
                    str(output_dir),
                    "--output-template",
                    "item-{batch_index}-{prompt_index}-{image_index}-{seed}.png",
                    "--progress",
                    "never",
                ]
            )
            fake_pipeline = mock.Mock()
            fake_pipeline.generate_batch.return_value = [
                _fake_result(Image.new("RGB", (1, 1)), prompt="a glass library", seed=10, batch_index=0, prompt_index=0, image_index=0),
                _fake_result(Image.new("RGB", (1, 1)), prompt="a glass library", seed=11, batch_index=1, prompt_index=0, image_index=1),
                _fake_result(Image.new("RGB", (1, 1)), prompt="a cedar observatory", seed=12, batch_index=2, prompt_index=1, image_index=0),
                _fake_result(Image.new("RGB", (1, 1)), prompt="a cedar observatory", seed=13, batch_index=3, prompt_index=1, image_index=1),
            ]
            stream = StringIO()
            with mock.patch.object(
                cli.BooguTurboPipeline,
                "from_pretrained",
                return_value=fake_pipeline,
            ), redirect_stdout(stream):
                code = args.handler(args)

            self.assertEqual(code, 0)
            paths = stream.getvalue().splitlines()
            self.assertEqual(
                [Path(path).name for path in paths],
                [
                    "item-0-0-0-10.png",
                    "item-1-0-1-11.png",
                    "item-2-1-0-12.png",
                    "item-3-1-1-13.png",
                ],
            )
            self.assertTrue(all(Path(path).exists() for path in paths))
            fake_pipeline.generate_batch.assert_called_once()
            prompts, kwargs = fake_pipeline.generate_batch.call_args
            self.assertEqual(prompts[0], ["a glass library", "a cedar observatory"])
            self.assertIsNone(kwargs["seed"])
            self.assertEqual(kwargs["seeds"], (10, 11, 12, 13))
            self.assertEqual(kwargs["num_images_per_prompt"], 2)
            self.assertIsNone(kwargs["progress_callback"])

    def test_generate_batch_validates_output_contract_and_template(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a",
                "--prompt",
                "b",
                "--size",
                "16x16",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--output-dir is required"):
            args.handler(args)

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a",
                "--prompt",
                "b",
                "--size",
                "16x16",
                "--output",
                "out.png",
                "--output-dir",
                "out",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--output is only valid"):
            args.handler(args)

        args = parser.parse_args(
            [
                "generate",
                "--model",
                "artifact",
                "--prompt",
                "a",
                "--prompt",
                "b",
                "--size",
                "16x16",
                "--output-dir",
                "out",
                "--output-template",
                "../bad.png",
            ]
        )
        with self.assertRaisesRegex(ValueError, "filename template"):
            args.handler(args)

    def test_progress_reporter_writes_compact_stderr_lines(self) -> None:
        stream = StringIO()
        reporter = cli._CliProgressReporter(stream)
        reporter(
            PipelineProgressEvent(
                kind="load_component_start",
                stage="model_load",
                message="Loading image transformer",
                details={"component": "transformer"},
            )
        )
        reporter(
            PipelineProgressEvent(
                kind="denoise_step_start",
                stage="denoise",
                message="Denoising",
                step_index=1,
                step_count=4,
                sigma=0.25,
            )
        )
        reporter(
            PipelineProgressEvent(
                kind="denoise_compute_start",
                stage="denoise",
                message="Running denoising",
            )
        )
        reporter(
            PipelineProgressEvent(
                kind="complete",
                stage="complete",
                message="Done",
            )
        )

        self.assertEqual(
            stream.getvalue().splitlines(),
            [
                "[load] transformer",
                "[denoise 2/4] sigma=0.25",
                "[denoise] running",
                "[complete] generation finished",
            ],
        )

    def test_benchmark_outputs_json_shape(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "benchmark",
                "--model",
                "artifact",
                "--prompt",
                "a glass library",
                "--height",
                "16",
                "--width",
                "16",
                "--runs",
                "1",
                "--warmup-runs",
                "0",
                "--seed",
                "7",
                "--num-images-per-prompt",
                "2",
                "--memory-mode",
                "low-memory",
                "--decode-batch-size",
                "1",
            ]
        )
        fake_pipeline = mock.Mock()
        fake_pipeline.generate_batch.return_value = (object(), object())
        stream = StringIO()
        with mock.patch.object(
            cli.BooguTurboPipeline,
            "from_pretrained",
            return_value=fake_pipeline,
        ) as from_pretrained, redirect_stdout(stream):
            code = args.handler(args)

        self.assertEqual(code, 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["generator"], "boogu-turbo-mlx")
        self.assertEqual(payload["runs"][0]["image_count"], 2)
        self.assertEqual(payload["summary"]["run_count"], 1)
        self.assertEqual(payload["memory_mode"], "low-memory")
        from_pretrained.assert_called_once_with(
            "artifact",
            memory_mode="low-memory",
        )
        fake_pipeline.generate_batch.assert_called_once()
        _, kwargs = fake_pipeline.generate_batch.call_args
        self.assertEqual(kwargs["seed"], 7)
        self.assertEqual(kwargs["num_images_per_prompt"], 2)
        self.assertEqual(kwargs["decode_batch_size"], 1)
        self.assertIn("_benchmark_recorder", kwargs)


def _fake_result(
    image: Image.Image,
    *,
    prompt: str,
    seed: int,
    batch_index: int = 0,
    prompt_index: int = 0,
    image_index: int = 0,
    height: int = 16,
    width: int = 16,
    denoise_height: int = 16,
    denoise_width: int = 16,
    output_height: int = 16,
    output_width: int = 16,
    steps: int = 4,
    sigmas: tuple[float, ...] = (0.001, 0.25075, 0.5005, 0.75025),
    timesteps: tuple[float, ...] | None = None,
) -> BooguTurboGenerationResult:
    return BooguTurboGenerationResult(
        image=image,
        prompt=prompt,
        prompt_index=prompt_index,
        image_index=image_index,
        batch_index=batch_index,
        seed=seed,
        height=height,
        width=width,
        denoise_height=denoise_height,
        denoise_width=denoise_width,
        output_height=output_height,
        output_width=output_width,
        steps=steps,
        step_count=len(sigmas),
        sigmas=sigmas,
        timesteps=timesteps,
        max_sequence_length=1280,
        truncate_instruction_sequence=True,
        dmd_conditioning_sigma=0.001,
        token_count=12,
    )


if __name__ == "__main__":
    unittest.main()
