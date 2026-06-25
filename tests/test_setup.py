from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from boogu_turbo_mlx import cli
from boogu_turbo_mlx.setup_flow import run_setup_cli


class SetupCliTests(unittest.TestCase):
    def test_dry_run_defaults_to_bf16_and_writes_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(config_path),
                    "--accept-defaults",
                    "--dry-run",
                ]
            )
            stream = StringIO()

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ):
                code = run_setup_cli(args, stream=stream)

            self.assertEqual(code, 0)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_type"], "bf16")
            self.assertEqual(payload["bf16_output"], "artifacts/boogu-mlx")
            self.assertFalse(payload["cleanup_source"])
            self.assertIn("Artifact: bf16", stream.getvalue())
            self.assertIn("Source cleanup: keep source", stream.getvalue())
            self.assertIn("[dry-run]", stream.getvalue())

    def test_completion_try_command_uses_repo_venv_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            venv_bin = root / ".venv" / "bin"
            venv_bin.mkdir(parents=True)
            python = venv_bin / "python"
            console_script = venv_bin / "boogu-turbo-mlx"
            python.write_text("", encoding="utf-8")
            console_script.write_text("", encoding="utf-8")
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--source-dir",
                    "source",
                    "--bf16-output",
                    "artifacts/boogu-mlx",
                ]
            )
            stream = StringIO()
            old_cwd = Path.cwd()

            try:
                os.chdir(root)
                with mock.patch(
                    "boogu_turbo_mlx.setup_flow._environment_errors",
                    return_value=[],
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.sys.executable",
                    str(python),
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.convert_weights",
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.run_doctor",
                    return_value={
                        "status": "ok",
                        "error_count": 0,
                        "warning_count": 0,
                        "checks": [],
                    },
                ):
                    code = run_setup_cli(args, stream=stream)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertIn(
                "Try: .venv/bin/boogu-turbo-mlx generate",
                stream.getvalue(),
            )
            self.assertNotIn(
                "Try: boogu-turbo-mlx generate",
                stream.getvalue(),
            )

    def test_bf16_setup_uses_existing_source_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "artifact"
            source.mkdir()
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--source-dir",
                    str(source),
                    "--bf16-output",
                    str(output),
                ]
            )

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.download_source",
            ) as download, mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
            ) as convert, mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ) as doctor:
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            download.assert_not_called()
            convert.assert_called_once()
            self.assertEqual(convert.call_args.args[:2], (source, output))
            self.assertEqual(convert.call_args.kwargs["dtype"], "auto")
            self.assertIn("progress_callback", convert.call_args.kwargs)
            doctor.assert_called_once_with(model=output)

    def test_bf16_setup_reuses_existing_artifact_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "missing-source"
            output = root / "artifact"
            output.mkdir()
            (output / "artifact.json").write_text("{}", encoding="utf-8")
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--cleanup-source",
                    "--source-dir",
                    str(source),
                    "--bf16-output",
                    str(output),
                ]
            )

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.download_source",
            ) as download, mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
            ) as convert, mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ) as doctor:
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            download.assert_not_called()
            convert.assert_not_called()
            doctor.assert_called_once_with(model=output)

    def test_bf16_setup_downloads_source_only_when_conversion_needs_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "models" / "source"
            output = root / "artifact"
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--source-dir",
                    str(source),
                    "--bf16-output",
                    str(output),
                ]
            )

            def fake_download(*_args: object, **_kwargs: object) -> Path:
                _write_source(source)
                return source

            def fake_convert(*_args: object, **_kwargs: object) -> None:
                output.mkdir()
                (output / "artifact.json").write_text("{}", encoding="utf-8")

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.download_source",
                side_effect=fake_download,
            ) as download, mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
                side_effect=fake_convert,
            ) as convert, mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ):
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            download.assert_called_once()
            self.assertEqual(download.call_args.kwargs["local_dir"], source)
            convert.assert_called_once()
            self.assertEqual(convert.call_args.args[:2], (source, output))

    def test_setup_passes_source_revision_to_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "models" / "custom-source"
            output = root / "artifact"
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--source-repo",
                    "someone/model",
                    "--source-revision",
                    "abc123",
                    "--source-dir",
                    str(source),
                    "--bf16-output",
                    str(output),
                ]
            )

            def fake_download(*_args: object, **_kwargs: object) -> Path:
                _write_source(source)
                return source

            def fake_convert(*_args: object, **_kwargs: object) -> None:
                output.mkdir()
                (output / "artifact.json").write_text("{}", encoding="utf-8")

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.download_source",
                side_effect=fake_download,
            ) as download, mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
                side_effect=fake_convert,
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ):
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            download.assert_called_once()
            self.assertEqual(download.call_args.args, ("someone/model",))
            self.assertEqual(download.call_args.kwargs["revision"], "abc123")
            payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["source_revision"], "abc123")

    def test_bf16_setup_removes_project_local_source_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "models" / "source"
            output = root / "artifact"
            _write_source(source)
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--cleanup-source",
                    "--source-dir",
                    "models/source",
                    "--bf16-output",
                    "artifact",
                ]
            )

            def fake_convert(*_args: object, **_kwargs: object) -> None:
                output.mkdir()
                (output / "artifact.json").write_text("{}", encoding="utf-8")

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with mock.patch(
                    "boogu_turbo_mlx.setup_flow._environment_errors",
                    return_value=[],
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.convert_weights",
                    side_effect=fake_convert,
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.run_doctor",
                    return_value={
                        "status": "ok",
                        "error_count": 0,
                        "warning_count": 0,
                        "checks": [],
                    },
                ):
                    code = run_setup_cli(args, stream=StringIO())
            finally:
                os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertFalse(source.exists())

    def test_source_cleanup_keeps_external_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = root / "project"
            external_source = root / "shared-source"
            output = project / "artifact"
            project.mkdir()
            _write_source(external_source)
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    "config.json",
                    "--accept-defaults",
                    "--cleanup-source",
                    "--source-dir",
                    str(external_source),
                    "--bf16-output",
                    "artifact",
                ]
            )

            def fake_convert(*_args: object, **_kwargs: object) -> None:
                output.mkdir()
                (output / "artifact.json").write_text("{}", encoding="utf-8")

            stream = StringIO()
            old_cwd = Path.cwd()
            try:
                os.chdir(project)
                with mock.patch(
                    "boogu_turbo_mlx.setup_flow._environment_errors",
                    return_value=[],
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.convert_weights",
                    side_effect=fake_convert,
                ), mock.patch(
                    "boogu_turbo_mlx.setup_flow.run_doctor",
                    return_value={
                        "status": "ok",
                        "error_count": 0,
                        "warning_count": 0,
                        "checks": [],
                    },
                ):
                    code = run_setup_cli(args, stream=stream)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(code, 0)
            self.assertTrue(external_source.exists())
            self.assertIn("outside this project", stream.getvalue())

    def test_q8_setup_uses_existing_bf16_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bf16 = root / "artifacts" / "boogu-mlx"
            q8 = root / "artifacts" / "boogu-mlx-q8"
            bf16.mkdir(parents=True)
            (bf16 / "artifact.json").write_text("{}", encoding="utf-8")
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / "config.json"),
                    "--accept-defaults",
                    "--artifact",
                    "q8",
                    "--source-dir",
                    str(root / "missing-source"),
                    "--bf16-output",
                    str(bf16),
                    "--q8-output",
                    str(q8),
                ]
            )

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.download_source",
            ) as download, mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
            ) as convert, mock.patch(
                "boogu_turbo_mlx.setup_flow.quantize_artifact",
            ) as quantize, mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ):
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            download.assert_not_called()
            convert.assert_not_called()
            quantize.assert_called_once()
            self.assertEqual(quantize.call_args.args[:2], (bf16, q8))

    def test_q8_setup_uses_and_cleans_temporary_bf16_after_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            bf16 = root / "artifacts" / "boogu-mlx"
            q8 = root / "artifacts" / "boogu-mlx-q8"
            temp_bf16 = root / ".boogu-turbo-mlx" / "intermediates" / "boogu-mlx-bf16"
            source.mkdir()
            args = cli.build_parser().parse_args(
                [
                    "setup",
                    "--config",
                    str(root / ".boogu-turbo-mlx" / "config.json"),
                    "--accept-defaults",
                    "--artifact",
                    "q8",
                    "--source-dir",
                    str(source),
                    "--bf16-output",
                    str(bf16),
                    "--q8-output",
                    str(q8),
                    "--q8-intermediate",
                    str(temp_bf16),
                ]
            )

            def fake_convert(*_args: object, **_kwargs: object) -> None:
                temp_bf16.mkdir(parents=True)
                (temp_bf16 / "artifact.json").write_text("{}", encoding="utf-8")

            with mock.patch(
                "boogu_turbo_mlx.setup_flow._environment_errors",
                return_value=[],
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.CONFIG_DIR",
                root / ".boogu-turbo-mlx",
            ), mock.patch(
                "boogu_turbo_mlx.setup_flow.convert_weights",
                side_effect=fake_convert,
            ) as convert, mock.patch(
                "boogu_turbo_mlx.setup_flow.quantize_artifact",
            ) as quantize, mock.patch(
                "boogu_turbo_mlx.setup_flow.run_doctor",
                return_value={"status": "ok", "error_count": 0, "warning_count": 0, "checks": []},
            ):
                code = run_setup_cli(args, stream=StringIO())

            self.assertEqual(code, 0)
            convert.assert_called_once()
            self.assertEqual(convert.call_args.args[:2], (source, temp_bf16))
            quantize.assert_called_once()
            self.assertEqual(quantize.call_args.args[:2], (temp_bf16, q8))
            self.assertFalse(temp_bf16.exists())


def _write_source(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model_index.json").write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
