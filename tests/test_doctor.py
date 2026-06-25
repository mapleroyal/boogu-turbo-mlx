from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from boogu_turbo_mlx import cli
from boogu_turbo_mlx.doctor import run_doctor
from safetensors_fixtures import write_safetensors_fixture


class DoctorTests(unittest.TestCase):
    def test_healthy_artifact_reports_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifact"
            _write_artifact(root)

            report = run_doctor(model=root)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["error_count"], 0)
        check_names = {check["name"] for check in report["checks"]}
        self.assertIn("model.required_files", check_names)
        self.assertIn("model.mllm.index", check_names)
        self.assertIn("model.transformer.index", check_names)
        self.assertIn("model.vae.index", check_names)

    def test_missing_artifact_files_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifact"
            _write_artifact(root)
            (root / "transformer" / "config.json").unlink()

            report = run_doctor(model=root)

        self.assertEqual(report["status"], "error")
        self.assertGreater(report["error_count"], 0)
        errors = [check for check in report["checks"] if check["status"] == "error"]
        self.assertIn("model.required_files", {check["name"] for check in errors})

    def test_quantized_artifact_metadata_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifact"
            _write_artifact(
                root,
                quantization={"format": "mlx-q8", "mode": "affine", "bits": 8, "group_size": 32},
            )

            report = run_doctor(model=root)

        quantization = [
            check for check in report["checks"] if check["name"] == "model.quantization"
        ][0]
        self.assertEqual(quantization["status"], "ok")
        self.assertEqual(quantization["details"]["bits"], 8)

    def test_doctor_json_cli_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "artifact"
            _write_artifact(root)
            args = cli.build_parser().parse_args(["doctor", "--model", str(root), "--json"])
            stream = StringIO()

            with redirect_stdout(stream):
                code = args.handler(args)

        self.assertEqual(code, 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["error_count"], 0)

    def test_source_validation_uses_local_manifest_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            _write_source(root)

            report = run_doctor(source=root)

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["error_count"], 0)
        self.assertIn("source.manifest", {check["name"] for check in report["checks"]})

    def test_missing_jinja2_is_generation_error(self) -> None:
        real_import = __import__("importlib").import_module

        def fake_import(name: str) -> object:
            if name == "jinja2":
                raise ImportError("missing jinja2")
            return real_import(name)

        with mock.patch(
            "boogu_turbo_mlx.doctor.importlib.import_module",
            side_effect=fake_import,
        ):
            report = run_doctor()

        jinja2 = [check for check in report["checks"] if check["name"] == "jinja2"][0]
        self.assertEqual(report["status"], "error")
        self.assertEqual(jinja2["status"], "error")


def _write_artifact(
    root: Path,
    *,
    quantization: dict[str, object] | None = None,
) -> None:
    (root / "processor").mkdir(parents=True)
    (root / "scheduler").mkdir(parents=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    for component in ("mllm", "transformer", "vae"):
        (root / component).mkdir(parents=True)
        (root / component / "config.json").write_text("{}", encoding="utf-8")
        weights_dir = root / "weights" / component
        weights_dir.mkdir(parents=True)
        shard_name = f"{component}.safetensors"
        tensor_name = f"{component}.weight"
        (weights_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 8},
                    "weight_map": {tensor_name: shard_name},
                }
            ),
            encoding="utf-8",
        )
        write_safetensors_fixture(weights_dir / shard_name, {tensor_name: ("BF16", [2, 2])})

    artifact: dict[str, object] = {
        "schema_version": 2,
        "format": "boogu-turbo-mlx-artifact",
        "processor": "processor",
        "scheduler_config": "scheduler/scheduler_config.json",
        "components": {
            component: {
                "config": f"{component}/config.json",
                "weights_index": f"weights/{component}/model.safetensors.index.json",
            }
            for component in ("mllm", "transformer", "vae")
        },
    }
    if quantization is not None:
        artifact["quantization"] = quantization
    (root / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")


def _write_source(root: Path) -> None:
    (root / "processor").mkdir(parents=True)
    (root / "processor" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    for component in ("mllm", "transformer", "vae"):
        component_dir = root / component
        component_dir.mkdir(parents=True)
        component_dir.joinpath("config.json").write_text("{}", encoding="utf-8")
        shard_name = "model.safetensors"
        tensor_name = (
            "model.language_model.embed.weight"
            if component == "mllm"
            else f"{component}.weight"
        )
        component_dir.joinpath("model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 8},
                    "weight_map": {tensor_name: shard_name},
                }
            ),
            encoding="utf-8",
        )
        write_safetensors_fixture(component_dir / shard_name, {tensor_name: ("BF16", [2, 2])})


if __name__ == "__main__":
    unittest.main()
