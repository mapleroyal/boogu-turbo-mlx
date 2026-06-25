from __future__ import annotations

import html
import json
import platform
import shlex
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import parse_qs, urlencode

from .constants import OFFICIAL_TURBO_HF_ID
from .conversion import convert_weights
from .doctor import format_doctor_report, run_doctor
from .errors import BooguTurboMlxError
from .hf import download_source
from .local_server_security import (
    SESSION_TOKEN_FIELD,
    new_session_token,
    token_from_path,
    validate_local_request,
    validate_session_token,
)
from .quantization import quantize_artifact


CONFIG_DIR = Path(".boogu-turbo-mlx")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_SOURCE_DIR = Path("models") / "Boogu-Image-0.1-Turbo"
DEFAULT_BF16_ARTIFACT = Path("artifacts") / "boogu-mlx"
DEFAULT_Q8_ARTIFACT = Path("artifacts") / "boogu-mlx-q8"
DEFAULT_Q8_INTERMEDIATE = CONFIG_DIR / "intermediates" / "boogu-mlx-bf16"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_GUI_LAUNCHER = "Launch Boogu Turbo.command"

ARTIFACT_TYPES = ("bf16", "q8")
MEMORY_MODES = ("resident", "low-memory")


@dataclass(frozen=True)
class SetupConfig:
    source_repo: str = OFFICIAL_TURBO_HF_ID
    source_revision: str | None = None
    source_dir: Path = DEFAULT_SOURCE_DIR
    artifact_type: str = "bf16"
    bf16_output: Path = DEFAULT_BF16_ARTIFACT
    q8_output: Path = DEFAULT_Q8_ARTIFACT
    q8_intermediate: Path = DEFAULT_Q8_INTERMEDIATE
    output_dir: Path = DEFAULT_OUTPUT_DIR
    memory_mode: str = "resident"
    cleanup_intermediates: bool = True
    cleanup_source: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SetupConfig":
        config = cls(
            source_repo=_non_empty_str(payload.get("source_repo"), OFFICIAL_TURBO_HF_ID),
            source_revision=_optional_str(payload.get("source_revision")),
            source_dir=_path(payload.get("source_dir"), DEFAULT_SOURCE_DIR),
            artifact_type=_non_empty_str(payload.get("artifact_type"), "bf16"),
            bf16_output=_path(payload.get("bf16_output"), DEFAULT_BF16_ARTIFACT),
            q8_output=_path(payload.get("q8_output"), DEFAULT_Q8_ARTIFACT),
            q8_intermediate=_path(
                payload.get("q8_intermediate"),
                DEFAULT_Q8_INTERMEDIATE,
            ),
            output_dir=_path(payload.get("output_dir"), DEFAULT_OUTPUT_DIR),
            memory_mode=_non_empty_str(payload.get("memory_mode"), "resident"),
            cleanup_intermediates=_bool(
                payload.get("cleanup_intermediates"),
                default=True,
            ),
            cleanup_source=_bool(payload.get("cleanup_source"), default=False),
        )
        return config.validate()

    @classmethod
    def from_form(cls, fields: dict[str, list[str]]) -> "SetupConfig":
        payload: dict[str, Any] = {
            "source_repo": _field(fields, "source_repo"),
            "source_revision": _field(fields, "source_revision"),
            "source_dir": _field(fields, "source_dir"),
            "artifact_type": _field(fields, "artifact_type"),
            "bf16_output": _field(fields, "bf16_output"),
            "q8_output": _field(fields, "q8_output"),
            "q8_intermediate": _field(fields, "q8_intermediate"),
            "output_dir": _field(fields, "output_dir"),
            "memory_mode": _field(fields, "memory_mode"),
            "cleanup_intermediates": "cleanup_intermediates" in fields,
            "cleanup_source": "cleanup_source" in fields,
        }
        return cls.from_mapping(payload)

    def validate(self) -> "SetupConfig":
        if self.artifact_type not in ARTIFACT_TYPES:
            raise BooguTurboMlxError(
                f"Setup artifact must be one of {', '.join(ARTIFACT_TYPES)}."
            )
        if self.memory_mode not in MEMORY_MODES:
            raise BooguTurboMlxError(
                f"Setup memory mode must be one of {', '.join(MEMORY_MODES)}."
            )
        if not self.source_repo:
            raise BooguTurboMlxError("Setup source repo must not be empty.")
        for label, path in (
            ("source_dir", self.source_dir),
            ("bf16_output", self.bf16_output),
            ("q8_output", self.q8_output),
            ("q8_intermediate", self.q8_intermediate),
            ("output_dir", self.output_dir),
        ):
            if str(path).strip() in {"", "."}:
                raise BooguTurboMlxError(f"Setup {label} must be a real path.")
        return self

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "source_dir": str(self.source_dir),
            "artifact_type": self.artifact_type,
            "bf16_output": str(self.bf16_output),
            "q8_output": str(self.q8_output),
            "q8_intermediate": str(self.q8_intermediate),
            "output_dir": str(self.output_dir),
            "memory_mode": self.memory_mode,
            "cleanup_intermediates": self.cleanup_intermediates,
            "cleanup_source": self.cleanup_source,
        }

    @property
    def selected_model(self) -> Path:
        return self.bf16_output if self.artifact_type == "bf16" else self.q8_output


def run_setup_cli(args: Any, *, stream: TextIO = sys.stderr) -> int:
    config_path = Path(args.config).expanduser()
    config = _load_config(config_path) if config_path.exists() else SetupConfig()
    config = _apply_cli_overrides(config, args)

    if not args.accept_defaults and not args.dry_run:
        config = collect_setup_config(
            config,
            open_browser=not args.no_browser,
            stream=stream,
        )

    _save_config(config, config_path)
    return run_setup(config, config_path=config_path, dry_run=args.dry_run, stream=stream)


def run_setup(
    config: SetupConfig,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dry_run: bool = False,
    stream: TextIO = sys.stderr,
) -> int:
    errors = _environment_errors()
    if errors:
        raise BooguTurboMlxError("; ".join(errors))

    config = config.validate()
    _print_setup_summary(config, config_path, stream=stream)
    if dry_run:
        _print(stream, "[dry-run] setup would download, convert, validate, and clean up as needed")
        return 0

    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_to_cleanup: Path | None = None
    if config.artifact_type == "bf16":
        model, source_to_cleanup = _ensure_bf16_artifact(
            config,
            config.bf16_output,
            stream=stream,
        )
        _doctor_or_raise(model, stream=stream)
    else:
        model, temporary_bf16, source_to_cleanup = _ensure_q8_artifact(
            config,
            stream=stream,
        )
        _doctor_or_raise(model, stream=stream)
        if temporary_bf16 is not None and config.cleanup_intermediates:
            _cleanup_intermediate(temporary_bf16, stream=stream)
    if source_to_cleanup is not None and config.cleanup_source:
        _cleanup_source(source_to_cleanup, stream=stream)

    _print(stream, "")
    _print(stream, "Setup complete.")
    _print(stream, f"Try: {_format_try_command(model, config)}")
    launcher = _write_gui_launcher(config_path)
    _print_launcher_notice(stream, launcher)
    return 0


def _format_try_command(model: Path, config: SetupConfig) -> str:
    output = config.output_dir / "glass-library.png"
    parts = [
        _recommended_cli_command(),
        "generate",
        "--model",
        str(model),
        "--prompt",
        "a glass library at sunrise",
        "--size",
        "512x512",
        "--seed",
        "42",
        "--memory-mode",
        config.memory_mode,
        "--output",
        str(output),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _recommended_cli_command() -> str:
    script = Path(sys.executable).with_name("boogu-turbo-mlx")
    if not script.exists():
        return "boogu-turbo-mlx"
    try:
        return str(script.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        return str(script)


def _write_gui_launcher(config_path: Path) -> Path:
    launcher = Path.cwd() / DEFAULT_GUI_LAUNCHER
    config_arg = shlex.quote(str(config_path))
    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
clear

echo "Starting Boogu Turbo GUI..."

GENERATED_CONFIG_PATH={config_arg}
PROJECT_CONFIG_PATH=".boogu-turbo-mlx/config.json"

if [[ -f "$PROJECT_CONFIG_PATH" ]]; then
  CONFIG_PATH="$PROJECT_CONFIG_PATH"
else
  CONFIG_PATH="$GENERATED_CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo
  echo "Setup config not found: $CONFIG_PATH"
  echo "Run ./setup.sh first, then open this launcher again."
  exit 1
fi

if [[ -d "src/boogu_turbo_mlx" ]]; then
  export PYTHONPATH="$PWD/src${{PYTHONPATH:+:$PYTHONPATH}}"
fi

if [[ -x ".venv/bin/python" ]]; then
  BOOGU_CMD=(".venv/bin/python" "-m" "boogu_turbo_mlx")
elif [[ -x ".venv/bin/boogu-turbo-mlx" ]]; then
  BOOGU_CMD=(".venv/bin/boogu-turbo-mlx")
else
  BOOGU_CMD=("boogu-turbo-mlx")
fi

exec "${{BOOGU_CMD[@]}}" gui --config "$CONFIG_PATH"
"""
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _print_launcher_notice(stream: TextIO, launcher: Path) -> None:
    lines = [
        "GUI launcher ready",
        "",
        f"Double-click: {launcher.name}",
        f"Location:     {launcher}",
        "",
        "The launcher opens a Terminal log window and the browser GUI.",
        "Keep that Terminal window open while using the GUI.",
    ]
    width = max(len(line) for line in lines) + 4
    border = "+" + "-" * (width - 2) + "+"
    _print(stream, "")
    _print(stream, border)
    for line in lines:
        _print(stream, "| " + line.ljust(width - 4) + " |")
    _print(stream, border)


def collect_setup_config(
    default: SetupConfig,
    *,
    open_browser: bool,
    stream: TextIO,
) -> SetupConfig:
    done = threading.Event()
    result: dict[str, Any] = {}
    session_token = new_session_token()
    handler = _setup_handler(default, done, result, session_token=session_token)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/?{urlencode({'token': session_token})}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _print(stream, f"Setup choices: {url}")
    if open_browser:
        webbrowser.open(url)
    _print(stream, "Choose setup options in the browser, then return here.")
    try:
        done.wait()
    except KeyboardInterrupt:
        raise BooguTurboMlxError("Setup cancelled before choices were submitted.") from None
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    if "error" in result:
        raise result["error"]
    config = result.get("config")
    if not isinstance(config, SetupConfig):
        raise BooguTurboMlxError("Setup browser closed before options were submitted.")
    _print(stream, "Setup choices saved; continuing in Terminal.")
    return config


def _ensure_source(config: SetupConfig, *, stream: TextIO) -> Path:
    if config.source_dir.exists():
        if not config.source_dir.is_dir():
            raise BooguTurboMlxError(
                f"Source path must be a directory: {config.source_dir}"
            )
        _print(stream, f"[download] reusing source at {config.source_dir}")
        return config.source_dir
    _print(stream, f"[download] fetching {config.source_repo} into {config.source_dir.parent}")
    source = download_source(
        config.source_repo,
        revision=config.source_revision,
        dest_root=config.source_dir.parent,
        local_dir=config.source_dir,
    )
    _print(stream, f"[download] ready at {source}")
    return source


def _ensure_bf16_artifact(
    config: SetupConfig,
    output: Path,
    *,
    stream: TextIO,
) -> tuple[Path, Path | None]:
    status = _artifact_status(output)
    if status == "ready":
        _print(stream, f"[convert] reusing bf16 artifact at {output}")
        return output, None
    if status == "blocked":
        raise BooguTurboMlxError(
            f"Cannot write bf16 artifact because {output} is not empty and "
            "does not look like a boogu-turbo-mlx artifact."
        )
    source = _ensure_source(config, stream=stream)
    _print(stream, f"[convert] writing bf16 MLX artifact to {output}")
    convert_weights(
        source,
        output,
        dtype="auto",
        progress_callback=lambda message: _print(stream, f"[convert] {message}"),
    )
    _print(stream, f"[convert] ready at {output}")
    return output, source


def _ensure_q8_artifact(
    config: SetupConfig,
    *,
    stream: TextIO,
) -> tuple[Path, Path | None, Path | None]:
    q8_status = _artifact_status(config.q8_output)
    if q8_status == "ready":
        _print(stream, f"[quantize] reusing q8 artifact at {config.q8_output}")
        return config.q8_output, None, None
    if q8_status == "blocked":
        raise BooguTurboMlxError(
            f"Cannot write q8 artifact because {config.q8_output} is not empty "
            "and does not look like a boogu-turbo-mlx artifact."
        )

    bf16_status = _artifact_status(config.bf16_output)
    if bf16_status == "ready":
        bf16_source = config.bf16_output
        temporary_bf16 = None
        source_to_cleanup = None
        _print(stream, f"[convert] using existing bf16 artifact at {bf16_source}")
    else:
        bf16_source = config.q8_intermediate
        temporary_bf16 = config.q8_intermediate
        if _artifact_status(bf16_source) == "blocked" and _is_inside_config_dir(bf16_source):
            shutil.rmtree(bf16_source)
        bf16_source, source_to_cleanup = _ensure_bf16_artifact(
            config,
            bf16_source,
            stream=stream,
        )

    _print(stream, f"[quantize] writing q8 artifact to {config.q8_output}")
    quantize_artifact(
        bf16_source,
        config.q8_output,
        progress_callback=lambda message: _print(stream, f"[quantize] {message}"),
    )
    _print(stream, f"[quantize] ready at {config.q8_output}")
    return config.q8_output, temporary_bf16, source_to_cleanup


def _doctor_or_raise(model: Path, *, stream: TextIO) -> None:
    _print(stream, f"[doctor] validating {model}")
    report = run_doctor(model=model)
    for line in format_doctor_report(report).splitlines():
        _print(stream, f"[doctor] {line}")
    if int(report.get("error_count", 0)):
        raise BooguTurboMlxError(f"doctor failed for {model}")


def _cleanup_intermediate(path: Path, *, stream: TextIO) -> None:
    if not path.exists():
        return
    if not _is_inside_config_dir(path):
        _print(stream, f"[cleanup] keeping {path}; it is outside .boogu-turbo-mlx")
        return
    shutil.rmtree(path)
    _print(stream, f"[cleanup] removed temporary bf16 artifact at {path}")


def _cleanup_source(path: Path, *, stream: TextIO) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    project = Path.cwd().resolve()
    if resolved == project:
        _print(
            stream,
            f"[cleanup] keeping source at {path}; refusing to remove project root",
        )
        return
    try:
        resolved.relative_to(project)
    except ValueError:
        _print(stream, f"[cleanup] keeping source at {path}; it is outside this project")
        return
    if not path.is_dir() or not (path / "model_index.json").exists():
        _print(
            stream,
            f"[cleanup] keeping source at {path}; it does not look like a model source",
        )
        return
    shutil.rmtree(path)
    _print(stream, f"[cleanup] removed source at {path}")


def _artifact_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "blocked"
    if (path / "artifact.json").exists():
        return "ready"
    return "empty" if not any(path.iterdir()) else "blocked"


def _environment_errors() -> list[str]:
    errors = []
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10 or newer is required; found {platform.python_version()}"
        )
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin":
        errors.append(f"macOS is required for MLX setup; found {system}")
    if machine not in {"arm64", "aarch64"}:
        errors.append(f"Apple Silicon is required for MLX setup; found {machine}")
    return errors


def _print_setup_summary(config: SetupConfig, config_path: Path, *, stream: TextIO) -> None:
    _print(stream, "boogu-turbo-mlx setup")
    _print(stream, f"Config: {config_path}")
    source = config.source_repo
    if config.source_revision:
        source = f"{source}@{config.source_revision}"
    _print(stream, f"Source: {source} -> {config.source_dir}")
    _print(stream, f"Artifact: {config.artifact_type} -> {config.selected_model}")
    _print(stream, f"Memory mode: {config.memory_mode}")
    _print(
        stream,
        "Source cleanup: "
        + (
            "remove project-local source after validation"
            if config.cleanup_source
            else "keep source after validation"
        ),
    )
    _print(stream, f"Outputs: {config.output_dir}")


def _apply_cli_overrides(config: SetupConfig, args: Any) -> SetupConfig:
    updates: dict[str, Any] = {}
    for field_name, arg_name in (
        ("source_repo", "source_repo"),
        ("source_revision", "source_revision"),
        ("source_dir", "source_dir"),
        ("artifact_type", "artifact"),
        ("bf16_output", "bf16_output"),
        ("q8_output", "q8_output"),
        ("q8_intermediate", "q8_intermediate"),
        ("memory_mode", "memory_mode"),
        ("output_dir", "output_dir"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value
    cleanup = getattr(args, "cleanup_intermediates", None)
    if cleanup is not None:
        updates["cleanup_intermediates"] = bool(cleanup)
    source_cleanup = getattr(args, "cleanup_source", None)
    if source_cleanup is not None:
        updates["cleanup_source"] = bool(source_cleanup)
    if updates:
        config = replace(config, **updates)
    return config.validate()


def _load_config(path: Path) -> SetupConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BooguTurboMlxError(f"Invalid setup config JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BooguTurboMlxError(f"Invalid setup config JSON: {path}")
    return SetupConfig.from_mapping(payload)


def _save_config(config: SetupConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.to_mapping(), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


def _setup_handler(
    default: SetupConfig,
    done: threading.Event,
    result: dict[str, Any],
    *,
    session_token: str,
) -> type[BaseHTTPRequestHandler]:
    class SetupHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                validate_session_token(token_from_path(self.path), session_token)
            except BooguTurboMlxError as exc:
                self.send_error(403, str(exc))
                return
            self._write_html(_form_html(default, session_token))

        def do_POST(self) -> None:
            try:
                validate_local_request(
                    headers=self.headers,
                    path=self.path,
                    expected_token=session_token,
                    allow_unsafe_host=False,
                    require_same_origin=True,
                )
            except BooguTurboMlxError as exc:
                self.send_error(403, str(exc))
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 100_000:
                    raise BooguTurboMlxError("Setup form submission is too large.")
                body = self.rfile.read(length).decode("utf-8")
                fields = parse_qs(body, keep_blank_values=True)
                validate_session_token(_field(fields, SESSION_TOKEN_FIELD), session_token)
                result["config"] = SetupConfig.from_form(fields)
                self._write_html(_done_html())
            except Exception as exc:
                result["error"] = exc
                self._write_html(_done_html(error=str(exc)))
            finally:
                done.set()

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _write_html(self, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return SetupHandler


def _form_html(config: SetupConfig, session_token: str) -> str:
    bf16_checked = " checked" if config.artifact_type == "bf16" else ""
    q8_checked = " checked" if config.artifact_type == "q8" else ""
    resident_checked = " checked" if config.memory_mode == "resident" else ""
    low_memory_checked = " checked" if config.memory_mode == "low-memory" else ""
    cleanup_checked = " checked" if config.cleanup_intermediates else ""
    cleanup_source_checked = " checked" if config.cleanup_source else ""
    source_revision = "" if config.source_revision is None else config.source_revision
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>boogu-turbo-mlx setup</title>
  <style>
    body {{ font: 16px/1.45 -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; max-width: 760px; }}
    fieldset {{ border: 1px solid #ccc; margin: 0 0 20px; padding: 16px; }}
    label {{ display: block; margin: 10px 0; }}
    p.note {{ color: #555; margin: 8px 0 14px; }}
    input[type=text] {{ box-sizing: border-box; font: inherit; padding: 8px; width: 100%; }}
    button {{ font: inherit; padding: 10px 14px; }}
  </style>
</head>
<body>
  <h1>boogu-turbo-mlx setup</h1>
  <form method="post" action="?{urlencode({'token': session_token})}">
    <input type="hidden" name="{SESSION_TOKEN_FIELD}" value="{_escape_attr(session_token)}">
    <fieldset>
      <legend>Artifact</legend>
      <label><input type="radio" name="artifact_type" value="bf16"{bf16_checked}> bf16, default</label>
      <label><input type="radio" name="artifact_type" value="q8"{q8_checked}> q8, smaller optional artifact</label>
    </fieldset>
    <fieldset>
      <legend>Folders</legend>
      <p class="note">If the source folder exists, setup reuses it. If it is missing and conversion needs source weights, setup downloads the source to that path. To reuse an existing download from another clone or disk, enter that folder here.</p>
      {_text_input("source_repo", "Hugging Face source", config.source_repo)}
      {_text_input("source_revision", "Hugging Face revision (optional)", source_revision)}
      {_text_input("source_dir", "Downloaded source folder", config.source_dir)}
      {_text_input("bf16_output", "bf16 artifact folder", config.bf16_output)}
      {_text_input("q8_output", "q8 artifact folder", config.q8_output)}
      {_text_input("output_dir", "Generated image folder", config.output_dir)}
      <input type="hidden" name="q8_intermediate" value="{_escape_attr(config.q8_intermediate)}">
    </fieldset>
    <fieldset>
      <legend>Runtime</legend>
      <label><input type="radio" name="memory_mode" value="resident"{resident_checked}> Resident</label>
      <label><input type="radio" name="memory_mode" value="low-memory"{low_memory_checked}> Low memory</label>
      <label><input type="checkbox" name="cleanup_intermediates"{cleanup_checked}> Remove temporary conversion files after validation</label>
      <label><input type="checkbox" name="cleanup_source"{cleanup_source_checked}> Remove project-local source folder after artifact validation</label>
      <p class="note">Removing the source saves about 36 GB. Preparing a new converted artifact later, such as q8 after an initial bf16 setup, will require downloading the source again or choosing an existing source folder.</p>
    </fieldset>
    <button type="submit">Start setup</button>
  </form>
</body>
</html>
"""


def _done_html(error: str | None = None) -> str:
    if error:
        heading = "Setup could not start"
        body = html.escape(error)
    else:
        heading = "Setup is ready to continue"
        body = "Return to Terminal to watch download, conversion, validation, and cleanup."
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>boogu-turbo-mlx setup</title></head>
<body>
  <h1>{heading}</h1>
  <p>{body}</p>
</body>
</html>
"""


def _text_input(name: str, label: str, value: str | Path) -> str:
    return (
        f'<label>{html.escape(label)}'
        f'<input type="text" name="{html.escape(name)}" '
        f'value="{_escape_attr(value)}"></label>'
    )


def _escape_attr(value: str | Path) -> str:
    return html.escape(str(value), quote=True)


def _field(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name, [""])
    return values[0] if values else ""


def _path(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default
    return Path(str(value)).expanduser()


def _non_empty_str(value: Any, default: str) -> str:
    text = default if value is None else str(value).strip()
    return text or default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_inside_config_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(CONFIG_DIR.resolve())
        return True
    except ValueError:
        return False


def _print(stream: TextIO, message: str) -> None:
    print(message, file=stream, flush=True)
