from __future__ import annotations

import json
import secrets
import sys
import subprocess
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlencode, urlparse

from .constants import (
    DEFAULT_DMD_CONDITIONING_SIGMA,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_STEPS,
    MAX_GENERATION_SEED,
    MAX_GENERATION_SIZE,
    OUTPUT_ALIGNMENT,
)
from .errors import BooguTurboGenerationCancelled, BooguTurboMlxError
from .local_server_security import (
    new_session_token,
    token_from_path,
    validate_local_request,
    validate_loopback_bind_host,
    validate_session_token,
)
from .pipeline import BooguTurboPipeline, PipelineProgressEvent
from .png import PNG_METADATA_KEY, save_generation_png
from .setup_flow import DEFAULT_CONFIG_PATH, SetupConfig


DEFAULT_GUI_SIZE = 512
MAX_GUI_SIZE = MAX_GENERATION_SIZE
MAX_SEED = MAX_GENERATION_SEED
MAX_REQUEST_BYTES = 64_000
MAX_GUI_BATCH_JOBS = 100
MAX_GUI_BATCH_BYTES = 256_000
MAX_EVENT_HISTORY = 200
MAX_INITIAL_RECENT_GENERATIONS = 500
_BATCH_JOB_KEYS = frozenset({"prompt", "width", "height", "steps", "seed"})


@dataclass(frozen=True)
class _GenerationRequest:
    prompt: str
    height: int
    width: int
    steps: int
    seed: int


@dataclass(frozen=True)
class _GenerationRecord:
    id: int
    path: Path
    prompt: str
    seed: int
    height: int
    width: int
    steps: int
    model: str
    created_at: str

    @property
    def url(self) -> str:
        return f"/api/image/{self.id}"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "prompt": self.prompt,
            "seed": self.seed,
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "model": self.model,
            "created_at": self.created_at,
        }


def run_gui(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    host: str = "127.0.0.1",
    port: int = 0,
    allow_unsafe_host: bool = False,
    open_browser: bool = True,
    preload: bool = True,
    stream: TextIO = sys.stdout,
) -> int:
    """Run the local browser GUI backed by the configured pipeline mode."""

    config_path = Path(config_path).expanduser()
    validate_loopback_bind_host(
        host,
        allow_unsafe_host=allow_unsafe_host,
        server_name="GUI",
    )
    config = _load_config(config_path)
    pipeline = BooguTurboPipeline.from_pretrained(
        config.selected_model,
        memory_mode=config.memory_mode,
    )
    state = _GuiState(config=config, pipeline=pipeline, stream=stream)
    session_token = new_session_token()
    handler = _gui_handler(
        state,
        session_token=session_token,
        allow_unsafe_host=allow_unsafe_host,
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    resolved_port = int(server.server_address[1])
    url = f"http://{_url_host(host)}:{resolved_port}/?{urlencode({'token': session_token})}"

    _print_gui_banner(
        stream,
        url=url,
        model=config.selected_model,
        config_path=config_path,
        output_dir=config.output_dir,
        memory_mode=config.memory_mode,
    )
    state.record_system_event(
        "server",
        "GUI server started",
        progress=1.0,
        details={"url": url, "host": host, "port": resolved_port},
    )

    if preload and config.memory_mode == "resident":
        threading.Timer(0.1, state.start_load).start()

    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, stream)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log(stream, "Shutdown requested from Terminal.")
    finally:
        server.server_close()
        _log(stream, "GUI server stopped.")
    return 0


@dataclass
class _GuiState:
    config: SetupConfig
    pipeline: BooguTurboPipeline
    stream: TextIO
    lock: threading.RLock = field(default_factory=threading.RLock)
    pipeline_lock: threading.Lock = field(default_factory=threading.Lock)
    events: list[dict[str, Any]] = field(default_factory=list)
    phase: str = "starting"
    message: str = "Starting"
    progress: float = 0.0
    error: str | None = None
    load_running: bool = False
    generation_running: bool = False
    cancel_requested: bool = False
    batch_active: bool = False
    batch_index: int = 0
    batch_total: int = 0
    batch_current: _GenerationRequest | None = None
    recent_generations: list[_GenerationRecord] = field(default_factory=list)
    session_image: _GenerationRecord | None = None
    next_generation_id: int = 1

    def __post_init__(self) -> None:
        self._load_existing_generations()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            image = self.session_image.to_mapping() if self.session_image else None
            busy = self.load_running or self.generation_running
            batch = self._batch_snapshot_locked()
            return {
                "server": {"connected": True, "status": "running"},
                "model": {
                    "loaded": self.pipeline.is_loaded,
                    "status": (
                        "low-memory mode"
                        if self.config.memory_mode == "low-memory"
                        else ("in memory" if self.pipeline.is_loaded else "not loaded")
                    ),
                    "path": str(self.config.selected_model),
                    "type": self.config.artifact_type,
                    "memory_mode": self.config.memory_mode,
                },
                "busy": busy,
                "load_running": self.load_running,
                "generation_running": self.generation_running,
                "cancel_requested": self.cancel_requested,
                "batch": batch,
                "phase": self.phase,
                "message": self.message,
                "progress": self.progress,
                "error": self.error,
                "image": image,
                "recent": [
                    item.to_mapping()
                    for item in self.recent_generations
                ],
                "recent_hidden_count": 0,
                "output_dir": {
                    "path": str(self.config.output_dir),
                },
                "constraints": {
                    "alignment": OUTPUT_ALIGNMENT,
                    "max_size": MAX_GUI_SIZE,
                    "max_seed": MAX_SEED,
                    "default_size": DEFAULT_GUI_SIZE,
                    "default_steps": DEFAULT_STEPS,
                },
                "events": list(self.events[-40:]),
            }

    def start_load(self) -> tuple[bool, str]:
        if self.config.memory_mode == "low-memory":
            return False, "Low-memory mode loads components only during generation."
        with self.lock:
            if self.pipeline.is_loaded:
                self.record_system_event_locked(
                    "model_load",
                    "Model is already in memory",
                    progress=1.0,
                )
                return False, "Model is already in memory."
            if self.load_running or self.generation_running:
                return False, "The model is busy right now."
            self.load_running = True
            self.phase = "model_load"
            self.message = "Loading model into memory"
            self.progress = 0.0
            self.error = None
        thread = threading.Thread(target=self._load_worker, daemon=True)
        thread.start()
        return True, "Model load started."

    def start_eject(self) -> tuple[bool, str]:
        if self.config.memory_mode == "low-memory" and not self.pipeline.is_loaded:
            return False, "Low-memory mode has no resident model to eject."
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "Wait for the current task to finish before ejecting."
            self.phase = "model_load"
            self.message = "Ejecting model from memory"
            self.progress = 0.0
            self.error = None
        with self.pipeline_lock:
            self.pipeline.unload(progress_callback=self.record_pipeline_event)
        if not self.pipeline.is_loaded:
            self.record_system_event(
                "model_load",
                "Model is not in memory",
                progress=0.0,
            )
        return True, "Model ejected."

    def start_generation(self, request: _GenerationRequest) -> tuple[bool, str]:
        if not request.prompt:
            return False, "Prompt cannot be empty."
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "A task is already running."
            self.generation_running = True
            self.cancel_requested = False
            self.phase = "generate"
            self.message = "Generation queued"
            self.progress = 0.0
            self.error = None
        thread = threading.Thread(
            target=self._generation_worker,
            args=(request,),
            daemon=True,
        )
        thread.start()
        return True, "Generation started."

    def start_batch_generation(
        self,
        requests: list[_GenerationRequest],
    ) -> tuple[bool, str]:
        if not requests:
            return False, "Batch must include at least one job."
        with self.lock:
            if self.load_running or self.generation_running:
                return False, "A task is already running."
            self.generation_running = True
            self.cancel_requested = False
            self.batch_active = True
            self.batch_index = 0
            self.batch_total = len(requests)
            self.batch_current = None
            self.phase = "generate"
            self.message = "Batch queued"
            self.progress = 0.0
            self.error = None
        thread = threading.Thread(
            target=self._batch_worker,
            args=(requests,),
            daemon=True,
        )
        thread.start()
        return True, "Batch generation started."

    def start_cancel_generation(self) -> tuple[bool, str]:
        with self.lock:
            if not self.generation_running:
                return False, "No generation is running."
            if self.cancel_requested:
                return True, "Cancellation is already requested."
            self.cancel_requested = True
            self.message = "Cancelling generation"
            self.record_system_event_locked(
                "generate",
                "Cancel requested",
                progress=self.progress,
            )
        return True, "Cancellation requested."

    def record_pipeline_event(self, event: PipelineProgressEvent) -> None:
        payload = _event_payload(event)
        with self.lock:
            self.events.append(payload)
            del self.events[:-MAX_EVENT_HISTORY]
            self.phase = event.stage
            self.message = event.message
            if event.progress is not None:
                self.progress = _clamp_progress(event.progress)
            if event.kind == "error":
                self.error = event.message
        _log(self.stream, _format_event_for_terminal(payload))

    def record_generation_event(self, event: PipelineProgressEvent) -> None:
        self.record_pipeline_event(event)
        if event.kind not in {"cancelled", "complete", "error"}:
            self._raise_if_generation_cancelled()

    def record_system_event(
        self,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.lock:
            self.record_system_event_locked(
                stage,
                message,
                progress=progress,
                details=details,
            )

    def record_system_event_locked(
        self,
        stage: str,
        message: str,
        *,
        progress: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "time": _timestamp(),
            "kind": "system",
            "stage": stage,
            "message": message,
            "progress": progress,
            "details": _json_safe(details or {}),
        }
        self.events.append(payload)
        del self.events[:-MAX_EVENT_HISTORY]
        self.phase = stage
        self.message = message
        if progress is not None:
            self.progress = _clamp_progress(progress)
        _log(self.stream, _format_event_for_terminal(payload))

    def _load_worker(self) -> None:
        try:
            with self.pipeline_lock:
                self.pipeline.load(progress_callback=self.record_pipeline_event)
        except Exception as exc:
            self._finish_with_error(exc, phase="model_load")
        finally:
            with self.lock:
                self.load_running = False
                if self.error is None and self.pipeline.is_loaded:
                    self.phase = "model_load"
                    self.message = "Model ready in memory"
                    self.progress = 1.0

    def _generation_worker(self, request: _GenerationRequest) -> None:
        try:
            with self.pipeline_lock:
                self._run_single_job_locked(request)
        except BooguTurboGenerationCancelled:
            self._finish_cancelled()
        except Exception as exc:
            self._finish_with_error(exc, phase="generate")
        finally:
            with self.lock:
                self.generation_running = False
                self.cancel_requested = False
                self._clear_batch_locked()

    def _batch_worker(self, requests: list[_GenerationRequest]) -> None:
        try:
            with self.lock:
                self.batch_active = True
                self.batch_total = len(requests)
            for index, request in enumerate(requests, start=1):
                self._raise_if_generation_cancelled()
                with self.lock:
                    self.batch_index = index
                    self.batch_total = len(requests)
                    self.batch_current = request
                    self.phase = "generate"
                    self.message = f"Job {index} of {len(requests)}"
                    self.progress = 0.0
                    self.record_system_event_locked(
                        "generate",
                        self.message,
                        progress=0.0,
                        details={
                            "job": index,
                            "total": len(requests),
                            "seed": request.seed,
                            "height": request.height,
                            "width": request.width,
                            "steps": request.steps,
                            "model": self.config.artifact_type,
                        },
                    )
                with self.pipeline_lock:
                    self._run_single_job_locked(request)
        except BooguTurboGenerationCancelled:
            self._finish_cancelled()
        except Exception as exc:
            with self.lock:
                index = self.batch_index
                total = self.batch_total
            message = f"Job {index} of {total} failed: {exc}"
            self._finish_with_error(BooguTurboMlxError(message), phase="generate")
        finally:
            with self.lock:
                self.generation_running = False
                self.cancel_requested = False
                self._clear_batch_locked()

    def _run_single_job_locked(self, request: _GenerationRequest) -> None:
        self._raise_if_generation_cancelled()
        batch = self.pipeline.generate_batch(
            request.prompt,
            height=request.height,
            width=request.width,
            steps=request.steps,
            seed=request.seed,
            max_sequence_length=DEFAULT_MAX_SEQUENCE_LENGTH,
            truncate_instruction_sequence=False,
            dmd_conditioning_sigma=DEFAULT_DMD_CONDITIONING_SIGMA,
            progress_callback=self.record_generation_event,
        )
        result = batch.items[0]
        output_path = self._next_output_path(request)
        self._raise_if_generation_cancelled()
        self.record_system_event(
            "output",
            "Saving image",
            progress=0.98,
            details={"path": str(output_path)},
        )
        save_generation_png(
            result,
            output_path,
            model=self.config.selected_model,
            memory_mode=self.config.memory_mode,
        )
        self._raise_if_generation_cancelled()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            record = _GenerationRecord(
                id=self.next_generation_id,
                path=output_path,
                prompt=request.prompt,
                seed=request.seed,
                height=request.height,
                width=request.width,
                steps=request.steps,
                model=self.config.artifact_type,
                created_at=created_at,
            )
            self.next_generation_id += 1
            self.recent_generations.insert(0, record)
            self.session_image = record
            self.phase = "complete"
            self.message = f"Saved {output_path.name}"
            self.progress = 1.0
        self.record_system_event(
            "output",
            "Image saved",
            progress=1.0,
            details={
                "path": str(output_path),
                "seed": request.seed,
                "height": request.height,
                "width": request.width,
                "steps": request.steps,
                "model": self.config.artifact_type,
            },
        )

    def _raise_if_generation_cancelled(self) -> None:
        with self.lock:
            if self.cancel_requested:
                raise BooguTurboGenerationCancelled("Generation cancelled by user")

    def _finish_cancelled(self) -> None:
        with self.lock:
            self.phase = "cancelled"
            self.message = "Generation cancelled"
            self.error = None
            already_recorded = bool(
                self.events
                and self.events[-1].get("kind") == "cancelled"
            )
            if not already_recorded:
                self.record_system_event_locked(
                    "cancelled",
                    "Generation cancelled",
                    progress=self.progress,
                )

    def _finish_with_error(self, exc: Exception, *, phase: str) -> None:
        message = str(exc)
        with self.lock:
            self.phase = phase
            self.message = message
            self.error = message
        self.record_system_event(
            phase,
            message,
            details={"error_type": type(exc).__name__},
        )

    def _next_output_path(self, request: _GenerationRequest) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = (
            self.config.output_dir
            / (
                f"boogu-gui-{stamp}-{request.width}x{request.height}-"
                f"steps{request.steps}-seed{request.seed}.png"
            )
        )
        counter = 2
        while path.exists():
            path = (
                self.config.output_dir
                / (
                    f"boogu-gui-{stamp}-{counter}-{request.width}x{request.height}-"
                    f"steps{request.steps}-seed{request.seed}.png"
                )
            )
            counter += 1
        return path

    def _batch_snapshot_locked(self) -> dict[str, Any] | None:
        if not self.batch_active:
            return None
        current = self.batch_current
        payload: dict[str, Any] = {
            "index": self.batch_index,
            "total": self.batch_total,
            "model": self.config.artifact_type,
        }
        if current is None:
            payload.update(
                {
                    "prompt": "",
                    "width": None,
                    "height": None,
                    "steps": None,
                    "seed": None,
                }
            )
        else:
            payload.update(
                {
                    "prompt": current.prompt,
                    "width": current.width,
                    "height": current.height,
                    "steps": current.steps,
                    "seed": current.seed,
                }
            )
        return payload

    def _clear_batch_locked(self) -> None:
        self.batch_active = False
        self.batch_index = 0
        self.batch_total = 0
        self.batch_current = None

    def image_path_for_id(self, image_id: int) -> Path | None:
        with self.lock:
            for record in self.recent_generations:
                if record.id == image_id:
                    return record.path
        return None

    def open_output_dir(self) -> None:
        path = self.config.output_dir
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(path)], check=False)
        self.record_system_event(
            "output",
            "Opened output folder",
            details={"path": str(path)},
        )

    def _load_existing_generations(self) -> None:
        output_dir = self.config.output_dir
        if not output_dir.exists() or not output_dir.is_dir():
            return
        paths = sorted(
            output_dir.glob("*.png"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        records: list[_GenerationRecord] = []
        for path in paths[:MAX_INITIAL_RECENT_GENERATIONS]:
            record = _record_from_png(
                path,
                record_id=len(records) + 1,
                default_model=self.config.artifact_type,
            )
            if record is not None:
                records.append(record)
        self.recent_generations = records
        self.next_generation_id = len(records) + 1
        if records:
            self.message = f"Loaded {len(records)} image(s) from outputs"


def _gui_handler(
    state: _GuiState,
    *,
    session_token: str,
    allow_unsafe_host: bool,
) -> type[BaseHTTPRequestHandler]:
    html_text = GUI_HTML.replace(
        "__BOOGU_SESSION_TOKEN_JSON__",
        json.dumps(session_token),
    )

    class GuiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                try:
                    validate_session_token(token_from_path(self.path), session_token)
                except BooguTurboMlxError as exc:
                    self.send_error(403, str(exc))
                    return
                self._write_html(html_text)
            elif parsed.path.startswith("/api/"):
                try:
                    self._validate_api_request(require_same_origin=False)
                except BooguTurboMlxError as exc:
                    self._write_json({"ok": False, "message": str(exc)}, status=403)
                    return
                if parsed.path == "/api/status":
                    self._write_json(state.snapshot())
                elif parsed.path == "/api/image/latest":
                    self._write_latest_image()
                elif parsed.path.startswith("/api/image/"):
                    self._write_image_by_id(parsed.path)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                self._validate_api_request(require_same_origin=True)
            except BooguTurboMlxError as exc:
                state.record_system_event(
                    "request",
                    str(exc),
                    details={"error_type": type(exc).__name__},
                )
                self._write_json({"ok": False, "message": str(exc)}, status=403)
                return
            try:
                if parsed.path == "/api/load":
                    accepted, message = state.start_load()
                    self._write_json({"ok": accepted, "message": message}, status=202 if accepted else 409)
                elif parsed.path == "/api/eject":
                    accepted, message = state.start_eject()
                    self._write_json({"ok": accepted, "message": message}, status=200 if accepted else 409)
                elif parsed.path == "/api/open-output-dir":
                    state.open_output_dir()
                    self._write_json({"ok": True, "message": "Output folder opened."})
                elif parsed.path == "/api/generate":
                    payload = self._read_json()
                    request = _generation_request_from_payload(payload)
                    accepted, message = state.start_generation(request)
                    self._write_json({"ok": accepted, "message": message}, status=202 if accepted else 409)
                elif parsed.path == "/api/validate-batch":
                    payload = self._read_json(max_bytes=MAX_GUI_BATCH_BYTES)
                    requests = _batch_jobs_from_payload(payload)
                    self._write_json({"ok": True, "count": len(requests)})
                elif parsed.path == "/api/generate-batch":
                    payload = self._read_json(max_bytes=MAX_GUI_BATCH_BYTES)
                    requests = _batch_jobs_from_payload(payload)
                    accepted, message = state.start_batch_generation(requests)
                    self._write_json({"ok": accepted, "message": message}, status=202 if accepted else 409)
                elif parsed.path == "/api/cancel":
                    accepted, message = state.start_cancel_generation()
                    self._write_json({"ok": accepted, "message": message}, status=202 if accepted else 409)
                else:
                    self.send_error(404)
            except Exception as exc:
                state.record_system_event(
                    "request",
                    str(exc),
                    details={"error_type": type(exc).__name__},
                )
                self._write_json({"ok": False, "message": str(exc)}, status=400)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self, *, max_bytes: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > max_bytes:
                raise BooguTurboMlxError("Request is too large.")
            if length <= 0:
                return {}
            data = self.rfile.read(length)
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise BooguTurboMlxError("Request body must be a JSON object.")
            return payload

        def _validate_api_request(self, *, require_same_origin: bool) -> None:
            validate_local_request(
                headers=self.headers,
                path=self.path,
                expected_token=session_token,
                allow_unsafe_host=allow_unsafe_host,
                require_same_origin=require_same_origin,
            )

        def _write_html(self, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_latest_image(self) -> None:
            with state.lock:
                path = state.session_image.path if state.session_image else None
            self._write_image_path(path)

        def _write_image_by_id(self, path_text: str) -> None:
            try:
                image_id = int(path_text.rsplit("/", 1)[1])
            except ValueError:
                self.send_error(404)
                return
            self._write_image_path(state.image_path_for_id(image_id))

        def _write_image_path(self, path: Path | None) -> None:
            if path is None or not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return GuiHandler


def _load_config(path: Path) -> SetupConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BooguTurboMlxError(f"Setup config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BooguTurboMlxError(f"Invalid setup config JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BooguTurboMlxError(f"Invalid setup config JSON: {path}")
    return SetupConfig.from_mapping(payload)


def _generation_request_from_payload(payload: dict[str, Any]) -> _GenerationRequest:
    return _generation_request_from_job(payload, allow_random_seed=False)


def _generation_request_from_job(
    job: dict[str, Any],
    *,
    allow_random_seed: bool,
) -> _GenerationRequest:
    prompt = str(job.get("prompt", "")).strip()
    width = _parse_gui_int(job.get("width"), "width")
    height = _parse_gui_int(job.get("height"), "height")
    steps = _parse_gui_int(job.get("steps"), "steps")
    if allow_random_seed and "seed" not in job:
        seed = secrets.randbits(32)
    else:
        seed = _parse_gui_int(job.get("seed"), "seed")
    if not prompt:
        raise BooguTurboMlxError("Prompt cannot be empty.")
    _validate_dimension(width, "width")
    _validate_dimension(height, "height")
    if steps <= 0:
        raise BooguTurboMlxError("Steps must be a positive integer.")
    if seed < 0 or seed > MAX_SEED:
        raise BooguTurboMlxError(f"Seed must be an integer from 0 to {MAX_SEED}.")
    return _GenerationRequest(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
    )


def _batch_jobs_from_payload(payload: dict[str, Any]) -> list[_GenerationRequest]:
    if not isinstance(payload, dict):
        raise BooguTurboMlxError("Batch payload must be a JSON object.")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise BooguTurboMlxError("Batch payload must include a jobs array.")
    if not jobs:
        raise BooguTurboMlxError("Batch must include at least one job.")
    if len(jobs) > MAX_GUI_BATCH_JOBS:
        raise BooguTurboMlxError(
            f"Batch cannot include more than {MAX_GUI_BATCH_JOBS} jobs."
        )

    requests: list[_GenerationRequest] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise BooguTurboMlxError(f"Job {index} must be a JSON object.")
        unsupported = sorted(set(job) - _BATCH_JOB_KEYS)
        if unsupported:
            label = "field" if len(unsupported) == 1 else "fields"
            names = ", ".join(unsupported)
            raise BooguTurboMlxError(
                f"Job {index} has unsupported {label}: {names}."
            )
        try:
            requests.append(
                _generation_request_from_job(job, allow_random_seed=True)
            )
        except BooguTurboMlxError as exc:
            raise BooguTurboMlxError(f"Job {index}: {exc}") from exc
    return requests


def _parse_gui_int(value: Any, label: str) -> int:
    try:
        if isinstance(value, str):
            value = value.strip()
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BooguTurboMlxError(f"{label.capitalize()} must be an integer.") from exc


def _validate_dimension(value: int, label: str) -> None:
    if value <= 0:
        raise BooguTurboMlxError(f"{label.capitalize()} must be positive.")
    if value > MAX_GUI_SIZE:
        raise BooguTurboMlxError(
            f"{label.capitalize()} must be {MAX_GUI_SIZE} or smaller."
        )
    if value % OUTPUT_ALIGNMENT:
        raise BooguTurboMlxError(
            f"{label.capitalize()} must be a multiple of {OUTPUT_ALIGNMENT}."
        )


def _record_from_png(
    path: Path,
    *,
    record_id: int,
    default_model: str,
) -> _GenerationRecord | None:
    try:
        metadata, image_size = _read_png_details(path)
        stat = path.stat()
    except OSError:
        return None
    created_at = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    width = _optional_int(metadata.get("width")) or _optional_int(metadata.get("output_width"))
    height = _optional_int(metadata.get("height")) or _optional_int(metadata.get("output_height"))
    if image_size is not None:
        width = width or image_size[0]
        height = height or image_size[1]
    if width is None or height is None:
        return None
    seed = _optional_int(metadata.get("seed"))
    steps = _optional_int(metadata.get("steps")) or DEFAULT_STEPS
    model = _model_label(metadata.get("model"), default_model=default_model)
    prompt = str(metadata.get("prompt") or "")
    filename_parts = _details_from_filename(path.name)
    if seed is None:
        seed = filename_parts.get("seed")
    if seed is None:
        seed = 0
    if "steps" in filename_parts and not metadata.get("steps"):
        steps = int(filename_parts["steps"])
    return _GenerationRecord(
        id=record_id,
        path=path,
        prompt=prompt,
        seed=int(seed),
        height=int(height),
        width=int(width),
        steps=int(steps),
        model=model,
        created_at=created_at,
    )


def _read_png_details(path: Path) -> tuple[dict[str, Any], tuple[int, int] | None]:
    try:
        from PIL import Image
    except ImportError:
        return {}, None
    with Image.open(path) as image:
        metadata_text = image.info.get(PNG_METADATA_KEY)
        metadata = {}
        if isinstance(metadata_text, str):
            try:
                payload = json.loads(metadata_text)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                metadata = payload
        return metadata, tuple(int(item) for item in image.size)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _details_from_filename(filename: str) -> dict[str, int]:
    details: dict[str, int] = {}
    for part in filename.replace(".", "-").split("-"):
        if part.startswith("seed"):
            seed = _optional_int(part[4:])
            if seed is not None:
                details["seed"] = seed
        elif part.startswith("steps"):
            steps = _optional_int(part[5:])
            if steps is not None:
                details["steps"] = steps
    return details


def _model_label(value: Any, *, default_model: str) -> str:
    text = "" if value is None else str(value)
    if "q8" in text.lower():
        return "q8"
    if "bf16" in text.lower() or "boogu-mlx" in text.lower():
        return "bf16"
    return default_model


def _url_host(host: str) -> str:
    text = str(host)
    if ":" in text and not text.startswith("["):
        return f"[{text}]"
    return text


def _event_payload(event: PipelineProgressEvent) -> dict[str, Any]:
    return {
        "time": _timestamp(),
        "kind": event.kind,
        "stage": event.stage,
        "message": event.message,
        "progress": event.progress,
        "step_index": event.step_index,
        "step_count": event.step_count,
        "sigma": event.sigma,
        "details": _json_safe(dict(event.details)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_event_for_terminal(payload: dict[str, Any]) -> str:
    progress = payload.get("progress")
    progress_text = "" if progress is None else f" progress={float(progress):.0%}"
    step_text = ""
    if payload.get("step_index") is not None and payload.get("step_count") is not None:
        step_text = f" step={int(payload['step_index']) + 1}/{payload['step_count']}"
    sigma_text = "" if payload.get("sigma") is None else f" sigma={payload['sigma']:.6g}"
    details = payload.get("details") or {}
    detail_text = "" if not details else " " + json.dumps(details, sort_keys=True)
    return (
        f"[{payload['time']}] "
        f"{payload['stage']}/{payload['kind']}: {payload['message']}"
        f"{progress_text}{step_text}{sigma_text}{detail_text}"
    )


def _print_gui_banner(
    stream: TextIO,
    *,
    url: str,
    model: Path,
    config_path: Path,
    output_dir: Path,
    memory_mode: str,
) -> None:
    lines = [
        "Boogu Turbo GUI",
        "",
        f"Browser: {url}",
        f"Model:   {model}",
        f"Memory:  {memory_mode}",
        f"Config:  {config_path}",
        f"Outputs: {output_dir}",
        "",
        "Keep this Terminal window open while using the browser GUI.",
        "Close this window or press Ctrl-C to stop the local process.",
    ]
    width = max(len(line) for line in lines) + 4
    border = "+" + "-" * (width - 2) + "+"
    stream.write("\n")
    stream.write(border + "\n")
    for line in lines:
        stream.write("| " + line.ljust(width - 4) + " |\n")
    stream.write(border + "\n\n")
    stream.flush()


def _log(stream: TextIO, message: str) -> None:
    stream.write(message + "\n")
    stream.flush()


def _open_browser(url: str, stream: TextIO) -> None:
    webbrowser.open(url)
    _log(stream, "Browser requested. If it does not open, paste the URL above.")


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _clamp_progress(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


GUI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boogu Turbo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0d1015;
      --surface: #171b22;
      --surface-2: #202630;
      --field: #12161d;
      --line: #333c49;
      --line-soft: #252d37;
      --text: #f4f7fb;
      --muted: #9aa8ba;
      --green: #45d483;
      --amber: #f2c14e;
      --red: #ff6b6b;
      --cyan: #70d7ff;
      --ink: #080a0d;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1360px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 22px 0 44px;
    }
    .seg-button {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--muted);
      padding: 0 11px;
      font: inherit;
      white-space: nowrap;
    }
    .status-stack {
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }
    .server-status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--muted);
      padding: 0 11px;
      font: inherit;
      white-space: nowrap;
    }
    .status-stack .button-group {
      width: 100%;
    }
    .button-group {
      display: inline-flex;
      align-items: stretch;
      border-radius: 8px;
      overflow: hidden;
    }
    .model-controls {
      width: 100%;
    }
    .model-controls .seg-button:first-child {
      flex: 1 1 auto;
      justify-content: flex-start;
    }
    .seg-button {
      border-radius: 0;
      cursor: pointer;
    }
    .seg-button + .seg-button { border-left: 0; }
    .seg-button:first-child { border-radius: 8px 0 0 8px; }
    .seg-button:last-child { border-radius: 0 8px 8px 0; }
    .seg-button:hover:not(:disabled) {
      border-color: var(--cyan);
      color: var(--text);
    }
    .seg-button:disabled {
      color: #667386;
      cursor: not-allowed;
      border-color: #29313b;
    }
    .model-controls #loadButton:disabled {
      color: var(--muted);
      cursor: default;
      border-color: var(--line);
    }
    .icon-button {
      justify-content: center;
      min-width: 42px;
      font-size: 18px;
      line-height: 1;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--amber);
      box-shadow: 0 0 12px color-mix(in srgb, var(--amber) 70%, transparent);
      flex: 0 0 auto;
    }
    .dot.good {
      background: var(--green);
      box-shadow: 0 0 12px color-mix(in srgb, var(--green) 70%, transparent);
    }
    .dot.bad {
      background: var(--red);
      box-shadow: 0 0 12px color-mix(in srgb, var(--red) 70%, transparent);
    }
    .dot.idle {
      background: #667386;
      box-shadow: none;
    }
    .workspace {
      display: grid;
      grid-template-columns: 270px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }
    .control-rail {
      border-right: 1px solid var(--line);
      padding-right: 20px;
      min-width: 0;
    }
    .control-section {
      padding: 0 0 18px;
      margin: 0 0 18px;
      border-bottom: 1px solid var(--line-soft);
    }
    .control-section:last-child {
      border-bottom: 0;
      margin-bottom: 0;
      padding-bottom: 0;
    }
    .section-label {
      color: var(--text);
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 10px;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    select,
    input,
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--field);
      color: var(--text);
      font: inherit;
      outline: none;
    }
    select,
    input {
      height: 38px;
      padding: 0 10px;
    }
    select:focus,
    input:focus,
    textarea:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 3px rgba(112, 215, 255, 0.14);
    }
    .dimension-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .field-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 42px;
      gap: 8px;
      align-items: end;
    }
    .small-button {
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: 700 18px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
    }
    .small-button:hover { border-color: var(--cyan); }
    .rail-button {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    .rail-button:hover:not(:disabled) {
      border-color: var(--cyan);
      color: var(--text);
    }
    .rail-button:disabled {
      color: #667386;
      cursor: not-allowed;
      border-color: #29313b;
    }
    .hint {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .content {
      min-width: 0;
    }
    .prompt-shell {
      position: relative;
      width: 100%;
    }
    #prompt {
      display: block;
      min-height: 56px;
      resize: none;
      overflow: hidden;
      padding: 16px 64px 16px 16px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .send-button {
      position: absolute;
      right: 8px;
      top: 8px;
      width: 40px;
      height: 40px;
      border: 0;
      border-radius: 8px;
      background: var(--green);
      color: var(--ink);
      font: 700 22px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
    }
    .send-button:disabled {
      background: #3b4652;
      color: #778395;
      cursor: not-allowed;
    }
    .send-button.cancel {
      background: var(--red);
      color: #160608;
    }
    .batch-status {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .batch-status-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .batch-job {
      color: var(--text);
      font-size: 16px;
      font-weight: 650;
    }
    .batch-cancel {
      width: 34px;
      height: 34px;
      border: 0;
      border-radius: 8px;
      background: var(--red);
      color: #160608;
      font: 700 20px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      cursor: pointer;
      flex: 0 0 auto;
    }
    .batch-cancel:disabled {
      background: #3b4652;
      color: #778395;
      cursor: not-allowed;
    }
    .batch-prompt {
      margin-top: 8px;
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .batch-meta {
      margin-top: 8px;
      color: var(--muted);
      white-space: pre-line;
      overflow-wrap: anywhere;
    }
    .progress-wrap {
      width: 100%;
      margin-top: 12px;
    }
    .progress-track {
      width: 100%;
      height: 10px;
      overflow: hidden;
      border-radius: 8px;
      background: #252d38;
      border: 1px solid #303a47;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      transition: width 180ms ease;
    }
    .log-panel {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }
    .log-toggle {
      width: 30px;
      height: 28px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--field);
      color: var(--muted);
      cursor: pointer;
      font: 700 16px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      flex: 0 0 auto;
    }
    .log-toggle:hover {
      border-color: var(--cyan);
      color: var(--text);
    }
    .event-log {
      height: 126px;
      overflow-y: auto;
      scrollbar-color: #4a5564 var(--surface);
    }
    .event-log.expanded { height: 504px; }
    .event-row {
      display: grid;
      grid-template-columns: 74px minmax(0, 1fr) 34px;
      gap: 10px;
      height: 42px;
      padding: 9px 12px;
      border-top: 1px solid var(--line-soft);
      color: var(--muted);
      align-items: center;
    }
    .event-row:first-child { border-top: 0; }
    .event-row.current {
      background: rgba(112, 215, 255, 0.06);
    }
    .event-message {
      color: var(--text);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .event-message.error { color: var(--red); }
    .event-action {
      display: flex;
      justify-content: flex-end;
    }
    .image-stage {
      margin-top: 22px;
      border-radius: 8px;
      background: transparent;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      overflow: visible;
    }
    .image-stage.empty {
      min-height: 160px;
      color: var(--muted);
      background: var(--surface);
      align-items: center;
    }
    .image-link {
      display: inline-block;
      max-width: min(100%, var(--preview-max-width, 1068px));
      text-align: center;
      border-radius: 8px;
      overflow: hidden;
    }
    #resultImage {
      display: block;
      max-width: min(100%, var(--preview-max-width, 1068px));
      width: auto;
      height: auto;
      background: #090b0f;
      border-radius: 8px;
    }
    .image-meta,
    .recent-meta {
      color: var(--muted);
      overflow-wrap: anywhere;
      white-space: pre-line;
    }
    .image-meta {
      max-width: 100%;
      margin: 8px auto 0;
    }
    .recent-section {
      margin-top: 24px;
      padding-top: 18px;
      border-top: 1px solid var(--line-soft);
    }
    .recent-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .recent-title-row {
      display: inline-flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }
    .recent-title-row .section-label {
      margin-bottom: 0;
    }
    .recent-controls {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .directory-link {
      border: 0;
      background: transparent;
      color: var(--cyan);
      text-decoration: none;
      font-size: 13px;
      font: inherit;
      cursor: pointer;
      padding: 0;
    }
    .directory-link:hover { text-decoration: underline; }
    .count-stepper {
      display: inline-flex;
      border-radius: 8px;
    }
    .count-button {
      width: 30px;
      height: 28px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
      font: 700 15px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .count-button + .count-button { border-left: 0; }
    .count-button:first-child { border-radius: 8px 0 0 8px; }
    .count-button:last-child { border-radius: 0 8px 8px 0; }
    .count-value {
      display: inline-grid;
      place-items: center;
      min-width: 38px;
      height: 28px;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      background: var(--surface);
      font-size: 13px;
    }
    .count-button:disabled {
      color: #667386;
      cursor: not-allowed;
    }
    .recent-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(164px, 1fr));
      gap: 10px;
    }
    .recent-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }
    .recent-card.selected {
      border-color: var(--cyan);
      box-shadow: 0 0 0 1px rgba(112, 215, 255, 0.22);
    }
    .recent-card a {
      display: block;
      background: #090b0f;
    }
    .recent-card img {
      display: block;
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
    }
    .recent-meta {
      padding: 8px 9px 9px;
      font-size: 12px;
      line-height: 1.35;
    }
    .recent-empty {
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--surface);
    }
    .recent-more {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: grid;
      place-items: center;
      padding: 16px;
      background: rgba(0, 0, 0, 0.68);
    }
    .batch-modal {
      width: min(520px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.48);
      padding: 16px;
    }
    .batch-modal-head,
    .batch-modal-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .batch-modal-actions {
      justify-content: flex-end;
      margin-top: 14px;
    }
    .batch-close {
      width: 32px;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--field);
      color: var(--muted);
      cursor: pointer;
      font: 700 18px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .batch-close:hover {
      border-color: var(--cyan);
      color: var(--text);
    }
    .batch-drop {
      margin-top: 14px;
      min-height: 132px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--field);
      padding: 22px;
      text-align: center;
      color: var(--muted);
    }
    .batch-drop.dragging {
      border-color: var(--cyan);
      color: var(--text);
      background: #151d26;
    }
    .batch-input-actions {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .batch-file-button,
    .batch-action-button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--text);
      font: inherit;
      padding: 0 12px;
      cursor: pointer;
    }
    .batch-file-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .batch-button-icon {
      width: 16px;
      height: 16px;
      flex: 0 0 auto;
      fill: none;
      stroke: currentColor;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .batch-file-button:hover,
    .batch-action-button:hover:not(:disabled) {
      border-color: var(--cyan);
    }
    .batch-action-button.primary {
      border-color: transparent;
      background: var(--green);
      color: var(--ink);
      font-weight: 650;
    }
    .batch-action-button:disabled {
      background: #3b4652;
      color: #778395;
      cursor: not-allowed;
      border-color: #29313b;
    }
    .batch-file-name {
      margin-top: 10px;
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .batch-file-name:empty {
      display: none;
    }
    .batch-drop-copy {
      margin-top: 10px;
      color: var(--muted);
    }
    .batch-validation {
      min-height: 20px;
      margin-top: 10px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .batch-validation:empty {
      display: none;
    }
    .batch-validation.valid { color: var(--green); }
    .batch-validation.invalid { color: var(--red); }
    .batch-agent-hint {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }
    .batch-example-link {
      border: 0;
      background: transparent;
      color: var(--cyan);
      cursor: pointer;
      font: inherit;
      padding: 0;
    }
    .batch-example-link:hover {
      text-decoration: underline;
    }
    .batch-copy-status {
      color: var(--green);
    }
    .prompt-tooltip {
      position: fixed;
      z-index: 30;
      max-width: min(420px, calc(100vw - 24px));
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--text);
      box-shadow: 0 12px 42px rgba(0, 0, 0, 0.42);
      font-size: 13px;
      line-height: 1.35;
      pointer-events: none;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      main { width: min(100vw - 20px, 1360px); padding-top: 16px; }
      .workspace { grid-template-columns: 1fr; }
      .control-rail {
        border-right: 0;
        border-bottom: 1px solid var(--line);
        padding-right: 0;
        padding-bottom: 16px;
      }
      .event-row { grid-template-columns: 58px 92px minmax(0, 1fr); gap: 8px; }
      .image-stage.empty { min-height: 140px; }
      .recent-head { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main>
    <div class="workspace">
      <aside class="control-rail">
        <section class="control-section">
          <div class="status-stack">
            <div class="server-status"><span id="serverDot" class="dot"></span><span id="serverStatus">Server connecting</span></div>
            <div class="button-group model-controls" role="group" aria-label="Model memory controls">
              <button id="loadButton" class="seg-button" type="button"><span id="loadDot" class="dot idle"></span><span id="loadLabel">Load model</span></button>
              <button id="ejectButton" class="seg-button icon-button" type="button" title="Eject model" aria-label="Eject model">&#9167;</button>
            </div>
          </div>
        </section>

        <section class="control-section">
          <div class="section-label">Dimensions</div>
          <label for="dimensionPreset">Preset</label>
          <select id="dimensionPreset">
            <option value="512x512">512 x 512</option>
            <option value="768x512">768 x 512</option>
            <option value="512x768">512 x 768</option>
            <option value="1024x1024">1024 x 1024</option>
            <option value="1216x832">1216 x 832</option>
            <option value="832x1216">832 x 1216</option>
            <option value="1536x1024">1536 x 1024</option>
            <option value="1024x1536">1024 x 1536</option>
            <option value="2048x2048">2048 x 2048</option>
            <option value="custom">Custom</option>
          </select>
          <div class="dimension-grid">
            <div>
              <label for="widthInput">Width</label>
              <input id="widthInput" type="number" inputmode="numeric" min="16" max="2048" step="16" value="512">
            </div>
            <div>
              <label for="heightInput">Height</label>
              <input id="heightInput" type="number" inputmode="numeric" min="16" max="2048" step="16" value="512">
            </div>
          </div>
          <p class="hint">Multiples of 16. Max 2048.</p>
        </section>

        <section class="control-section">
          <div class="section-label">Seed</div>
          <div class="field-row">
            <div>
              <label for="seedInput">Value</label>
              <input id="seedInput" type="text" inputmode="numeric">
            </div>
            <button id="randomSeedButton" class="small-button" type="button" title="Randomize seed" aria-label="Randomize seed">&#8635;</button>
          </div>
          <p class="hint">Integer from 0 to 4294967295. Empty uses the placeholder seed.</p>
        </section>

        <section class="control-section">
          <div class="section-label">Steps</div>
          <label for="stepsInput">Count</label>
          <input id="stepsInput" type="number" inputmode="numeric" min="1" step="1" value="4">
          <p class="hint">Positive integer. Turbo default is 4.</p>
        </section>

        <section class="control-section">
          <div class="section-label">Batch</div>
          <button id="batchOpenButton" class="rail-button" type="button">Run batch...</button>
        </section>
      </aside>

      <section class="content">
        <form id="promptForm">
          <div class="prompt-shell">
            <textarea id="prompt" rows="1" autocomplete="off" placeholder="Describe the image to generate"></textarea>
            <button id="sendButton" class="send-button" type="submit" title="Generate" aria-label="Generate">&#8593;</button>
          </div>
        </form>
        <section id="batchStatusArea" class="batch-status" hidden>
          <div class="batch-status-head">
            <div id="batchJob" class="batch-job">Job 0 of 0</div>
            <button id="batchCancelButton" class="batch-cancel" type="button" title="Cancel batch" aria-label="Cancel batch">&#215;</button>
          </div>
          <div id="batchPrompt" class="batch-prompt"></div>
          <div id="batchMeta" class="batch-meta"></div>
        </section>
        <div class="progress-wrap">
          <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
        </div>
        <div class="log-panel">
          <div id="eventLog" class="event-log"></div>
        </div>

        <section id="imageStage" class="image-stage empty">
          <div id="emptyState">Generate an image or select one from the gallery.</div>
          <a id="imageLink" class="image-link" target="_blank" rel="noopener" hidden>
            <img id="resultImage" alt="Preview image">
          </a>
        </section>
        <div id="imageMeta" class="image-meta"></div>

        <div class="recent-section">
          <div class="recent-head">
            <div class="recent-title-row">
              <div class="section-label">Gallery</div>
              <div class="count-stepper" role="group" aria-label="Recent generation count">
                <button id="recentLess" class="count-button" type="button" title="Show fewer recent generations" aria-label="Show fewer recent generations">-</button>
                <span id="recentCount" class="count-value">25</span>
                <button id="recentMoreButton" class="count-button" type="button" title="Show more recent generations" aria-label="Show more recent generations">+</button>
              </div>
            </div>
            <div class="recent-controls">
              <button id="directoryButton" class="directory-link" type="button">Open output folder</button>
            </div>
          </div>
          <div id="recentGrid" class="recent-grid"></div>
          <div id="recentMore" class="recent-more"></div>
        </div>
      </section>
    </div>
  </main>

  <div id="batchModal" class="modal-backdrop" hidden>
    <div class="batch-modal" role="dialog" aria-modal="true" aria-labelledby="batchModalTitle">
      <div class="batch-modal-head">
        <div id="batchModalTitle" class="section-label">Run batch from JSON</div>
        <button id="batchCloseButton" class="batch-close" type="button" title="Close" aria-label="Close">&#215;</button>
      </div>
      <div id="batchDrop" class="batch-drop">
        <div class="batch-input-actions">
          <button id="batchFileButton" class="batch-file-button" type="button">
            <svg class="batch-button-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <path d="M14 2v6h6"></path>
            </svg>
            <span>Choose file</span>
          </button>
          <button id="batchPasteButton" class="batch-file-button" type="button">
            <svg class="batch-button-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M9 4h6"></path>
              <path d="M9 2h6a2 2 0 0 1 2 2v2H7V4a2 2 0 0 1 2-2z"></path>
              <path d="M7 4H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2"></path>
            </svg>
            <span>Paste</span>
          </button>
        </div>
        <div class="batch-drop-copy">Or drop the file here.</div>
        <input id="batchFileInput" type="file" accept=".json,.boogu-batch.json,application/json" hidden>
        <div id="batchFileName" class="batch-file-name"></div>
        <div id="batchValidation" class="batch-validation"></div>
      </div>
      <div class="batch-agent-hint">
        Copy
        <button id="batchExampleButton" class="batch-example-link" type="button">this example prompt</button>
        to have your agent generate your batch file.
        <span id="batchCopyStatus" class="batch-copy-status"></span>
      </div>
      <div class="batch-modal-actions">
        <button id="batchCancelModalButton" class="batch-action-button" type="button">Cancel</button>
        <button id="batchGenerateButton" class="batch-action-button primary" type="button" disabled>Generate</button>
      </div>
    </div>
  </div>
  <div id="promptTooltip" class="prompt-tooltip" hidden></div>

  <script>
    const serverDot = document.getElementById("serverDot");
    const serverStatus = document.getElementById("serverStatus");
    const loadDot = document.getElementById("loadDot");
    const loadLabel = document.getElementById("loadLabel");
    const loadButton = document.getElementById("loadButton");
    const ejectButton = document.getElementById("ejectButton");
    const dimensionPreset = document.getElementById("dimensionPreset");
    const widthInput = document.getElementById("widthInput");
    const heightInput = document.getElementById("heightInput");
    const seedInput = document.getElementById("seedInput");
    const randomSeedButton = document.getElementById("randomSeedButton");
    const stepsInput = document.getElementById("stepsInput");
    const batchOpenButton = document.getElementById("batchOpenButton");
    const promptForm = document.getElementById("promptForm");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("sendButton");
    const batchStatusArea = document.getElementById("batchStatusArea");
    const batchJob = document.getElementById("batchJob");
    const batchPrompt = document.getElementById("batchPrompt");
    const batchMeta = document.getElementById("batchMeta");
    const batchCancelButton = document.getElementById("batchCancelButton");
    const progressFill = document.getElementById("progressFill");
    const eventLog = document.getElementById("eventLog");
    const imageStage = document.getElementById("imageStage");
    const emptyState = document.getElementById("emptyState");
    const imageLink = document.getElementById("imageLink");
    const resultImage = document.getElementById("resultImage");
    const imageMeta = document.getElementById("imageMeta");
    const recentCount = document.getElementById("recentCount");
    const recentLess = document.getElementById("recentLess");
    const recentMoreButton = document.getElementById("recentMoreButton");
    const directoryButton = document.getElementById("directoryButton");
    const recentGrid = document.getElementById("recentGrid");
    const recentMore = document.getElementById("recentMore");
    const batchModal = document.getElementById("batchModal");
    const batchCloseButton = document.getElementById("batchCloseButton");
    const batchCancelModalButton = document.getElementById("batchCancelModalButton");
    const batchDrop = document.getElementById("batchDrop");
    const batchFileButton = document.getElementById("batchFileButton");
    const batchPasteButton = document.getElementById("batchPasteButton");
    const batchFileInput = document.getElementById("batchFileInput");
    const batchFileName = document.getElementById("batchFileName");
    const batchValidation = document.getElementById("batchValidation");
    const batchExampleButton = document.getElementById("batchExampleButton");
    const batchCopyStatus = document.getElementById("batchCopyStatus");
    const batchGenerateButton = document.getElementById("batchGenerateButton");
    const promptTooltip = document.getElementById("promptTooltip");
    const SESSION_TOKEN = __BOOGU_SESSION_TOKEN_JSON__;
    const PROMPT_TOOLTIP_DELAY_MS = 400;
    const EXAMPLE_BATCH_AGENT_PROMPT = [
      "Create a JSON file for batch image generation.",
      "",
      "The format is:",
      "",
      "```json",
      "[",
      '  { "prompt": "a glass library at sunrise", "width": 768, "height": 768, "steps": 8, "seed": 1933305333 },',
      '  { "prompt": "a cedar observatory at night", "width": 1024, "height": 1024, "steps": 6 }',
      "]",
      "```",
      "",
      "`prompt`, `width`, `height`, and `steps` are required. `seed` is optional and must be from `0` to `4294967295`; when omitted, a random seed is applied. Dimensions must be positive multiples of `16` up to `2048`. `steps` must be positive, typically from `4` to `12`. Up to `100` jobs can be batched.",
      "",
      "My requirements are: [user's requirements here].",
    ].join("\\n");

    let busy = false;
    let generationRunning = false;
    let batchRunning = false;
    let cancelRequested = false;
    let previewImageId = null;
    let lastSessionImageId = null;
    let logExpanded = false;
    let lastEvents = [];
    let renderedEventCount = 0;
    let lastEventSignature = "";
    let lastRecentSignature = "";
    let currentLogMessage = "Waiting for the local process.";
    let currentLogIsError = false;
    let recentLimit = 25;
    let validatedBatchJobs = null;
    let promptTooltipTimer = null;
    let promptTooltipTarget = null;
    let promptTooltipPoint = { x: 0, y: 0 };
    let seedPlaceholder = randomSeed();
    seedInput.placeholder = String(seedPlaceholder);

    function randomSeed() {
      const values = new Uint32Array(1);
      crypto.getRandomValues(values);
      return values[0];
    }

    function apiPath(path, params = {}) {
      const url = new URL(path, window.location.href);
      url.searchParams.set("token", SESSION_TOKEN);
      for (const [key, value] of Object.entries(params)) {
        url.searchParams.set(key, String(value));
      }
      return `${url.pathname}${url.search}`;
    }

    function apiFetch(path, options = {}) {
      const headers = new Headers(options.headers || {});
      headers.set("X-Boogu-Session-Token", SESSION_TOKEN);
      return fetch(apiPath(path), { ...options, headers });
    }

    function setDot(dot, state) {
      dot.classList.remove("good", "bad", "idle");
      if (state === "good") dot.classList.add("good");
      else if (state === "bad") dot.classList.add("bad");
      else if (state === "idle") dot.classList.add("idle");
    }

    function renderStatus(data) {
      busy = Boolean(data.busy);
      generationRunning = Boolean(data.generation_running);
      batchRunning = Boolean(data.batch);
      cancelRequested = Boolean(data.cancel_requested);
      setDot(serverDot, "good");
      serverStatus.textContent = "Server connected";

      const modelLoaded = Boolean(data.model.loaded);
      const lowMemoryMode = data.model.memory_mode === "low-memory";
      const loadingModel = busy && data.phase === "model_load" && !modelLoaded;
      setDot(loadDot, modelLoaded ? "good" : (loadingModel ? "warn" : "idle"));
      loadLabel.textContent = lowMemoryMode ? "Low-memory mode" : (modelLoaded ? "Model loaded" : (loadingModel ? "Loading model" : "Load model"));
      loadButton.disabled = lowMemoryMode || busy || modelLoaded;
      ejectButton.disabled = lowMemoryMode || busy || !modelLoaded;
      batchOpenButton.disabled = busy;
      promptForm.hidden = batchRunning;
      renderBatchStatus(data.batch || null);
      updateBatchGenerateEnabled();
      updateGenerateEnabled();

      const pct = Math.max(0, Math.min(1, Number(data.progress || 0)));
      progressFill.style.width = `${Math.round(pct * 100)}%`;
      currentLogMessage = data.error || data.message || "Ready";
      currentLogIsError = Boolean(data.error);

      lastEvents = data.events || [];
      renderEvents(lastEvents);
      renderSessionImage(data.image);
      renderRecent(data.recent || [], data.recent_hidden_count || 0);
    }

    function renderBatchStatus(batch) {
      if (!batch) {
        batchStatusArea.hidden = true;
        batchJob.textContent = "Job 0 of 0";
        batchPrompt.textContent = "";
        batchMeta.textContent = "";
        return;
      }
      batchStatusArea.hidden = false;
      const index = Number(batch.index || 0);
      const total = Number(batch.total || 0);
      batchJob.textContent = `Job ${index} of ${total}`;
      batchPrompt.textContent = batch.prompt || "";
      batchMeta.textContent = batch.width && batch.height ? generationDetails(batch) : "";
      batchCancelButton.disabled = cancelRequested;
      batchCancelButton.textContent = cancelRequested ? "\u2026" : "\u00d7";
      batchCancelButton.title = cancelRequested ? "Cancelling batch" : "Cancel batch";
      batchCancelButton.setAttribute("aria-label", batchCancelButton.title);
    }

    function selectionTouches(...roots) {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return false;
      return roots.some((root) => {
        if (!root) return false;
        return (
          (selection.anchorNode && root.contains(selection.anchorNode)) ||
          (selection.focusNode && root.contains(selection.focusNode))
        );
      });
    }

    function eventSignature(events, visibleEvents) {
      return visibleEvents.map((event) => [
        event.time || "",
        event.kind || "",
        event.message || "",
        event.progress ?? "",
      ].join("\u001f")).join("\u001e");
    }

    function renderEvents(events, options = {}) {
      const wasAtNewest = eventLog.scrollTop <= 2;
      const previousScroll = eventLog.scrollTop;
      const visibleEvents = events.length
        ? events.slice().reverse()
        : [{ time: "", message: currentLogMessage, kind: currentLogIsError ? "error" : "system" }];
      const signature = eventSignature(events, visibleEvents);
      if (!options.force && signature === lastEventSignature) return;
      if (!options.force && selectionTouches(eventLog)) return;
      eventLog.replaceChildren(...visibleEvents.map((event, index) => {
        const row = document.createElement("div");
        row.className = "event-row";
        if (index === 0) row.classList.add("current");
        const time = document.createElement("div");
        time.textContent = event.time || "";
        const message = document.createElement("div");
        message.className = "event-message";
        if (event.kind === "error" || (index === 0 && currentLogIsError)) {
          message.classList.add("error");
        }
        message.textContent = event.message || "";
        const action = document.createElement("div");
        action.className = "event-action";
        if (index === 0) {
          const toggle = document.createElement("button");
          toggle.className = "log-toggle";
          toggle.type = "button";
          toggle.title = logExpanded ? "Collapse log" : "Expand log";
          toggle.setAttribute("aria-label", logExpanded ? "Collapse log" : "Expand log");
          toggle.textContent = "\u2195";
          toggle.addEventListener("click", toggleLog);
          action.append(toggle);
        }
        row.append(time, message, action);
        return row;
      }));
      if (wasAtNewest) {
        eventLog.scrollTop = 0;
      } else {
        const newRows = Math.max(0, visibleEvents.length - renderedEventCount);
        eventLog.scrollTop = previousScroll + newRows * 42;
      }
      renderedEventCount = visibleEvents.length;
      lastEventSignature = signature;
    }

    function renderSessionImage(image) {
      if (!image || !image.url || image.id === lastSessionImageId) return;
      lastSessionImageId = image.id;
      showPreview(image, { cacheBust: true });
    }

    function showPreview(image, options = {}) {
      if (!image || !image.url) return;
      const cacheBust = Boolean(options.cacheBust);
      if (!cacheBust && image.id === previewImageId) return;
      previewImageId = image.id;
      const previewMaxWidth = Math.min(Number(image.width) || 1068, 1068);
      imageStage.style.setProperty("--preview-max-width", `${previewMaxWidth}px`);
      imageStage.classList.remove("empty");
      emptyState.hidden = true;
      imageLink.hidden = false;
      imageLink.href = apiPath(image.url);
      imageLink.setAttribute("aria-label", image.prompt || "Preview image");
      setPromptTooltip(imageLink, image.prompt || "");
      resultImage.src = apiPath(image.url, cacheBust ? { ts: Date.now() } : {});
      resultImage.removeAttribute("title");
      imageMeta.style.maxWidth = `${previewMaxWidth}px`;
      imageMeta.textContent = generationDetails(image);
      syncPreviewMetaWidth();
      requestAnimationFrame(syncPreviewMetaWidth);
      markSelectedGalleryImage();
    }

    function syncPreviewMetaWidth() {
      if (imageLink.hidden) return;
      const width = Math.round(resultImage.getBoundingClientRect().width);
      if (width > 0) imageMeta.style.width = `${width}px`;
    }

    function setPromptTooltip(element, prompt) {
      element.removeAttribute("title");
      if (prompt) {
        element.dataset.promptTooltip = prompt;
      } else {
        delete element.dataset.promptTooltip;
      }
    }

    function promptTooltipAnchor(event) {
      if (!(event.target instanceof Element)) return null;
      return event.target.closest("[data-prompt-tooltip]");
    }

    function schedulePromptTooltip(element, x, y) {
      hidePromptTooltip();
      const prompt = element.dataset.promptTooltip || "";
      if (!prompt) return;
      promptTooltipTarget = element;
      promptTooltipPoint = { x, y };
      promptTooltipTimer = setTimeout(() => {
        if (promptTooltipTarget !== element) return;
        promptTooltip.textContent = prompt;
        promptTooltip.hidden = false;
        positionPromptTooltip(promptTooltipPoint.x, promptTooltipPoint.y);
      }, PROMPT_TOOLTIP_DELAY_MS);
    }

    function positionPromptTooltip(x, y) {
      if (promptTooltip.hidden) return;
      const margin = 12;
      const offset = 14;
      const rect = promptTooltip.getBoundingClientRect();
      let left = x + offset;
      let top = y + offset;
      if (left + rect.width + margin > window.innerWidth) {
        left = Math.max(margin, x - rect.width - offset);
      }
      if (top + rect.height + margin > window.innerHeight) {
        top = Math.max(margin, y - rect.height - offset);
      }
      promptTooltip.style.left = `${left}px`;
      promptTooltip.style.top = `${top}px`;
    }

    function hidePromptTooltip() {
      if (promptTooltipTimer !== null) clearTimeout(promptTooltipTimer);
      promptTooltipTimer = null;
      promptTooltipTarget = null;
      promptTooltip.hidden = true;
    }

    function markSelectedGalleryImage() {
      for (const card of recentGrid.querySelectorAll(".recent-card")) {
        card.classList.toggle("selected", card.dataset.imageId === String(previewImageId));
      }
    }

    function recentSignature(items, visibleItems, hiddenCount) {
      return [
        recentLimit,
        hiddenCount,
        previewImageId ?? "",
        visibleItems.map((item) => [
          item.id,
          item.seed,
          item.width,
          item.height,
          item.steps,
          item.model,
          item.prompt || "",
        ].join("\u001f")).join("\u001e"),
      ].join("\u001d");
    }

    function renderRecent(items, hiddenCount) {
      recentLimit = Math.max(1, recentLimit);
      recentCount.textContent = String(recentLimit);
      recentLess.disabled = recentLimit <= 1;
      recentMoreButton.disabled = false;
      const visibleItems = items.slice(0, recentLimit);
      const signature = recentSignature(items, visibleItems, hiddenCount);
      if (signature === lastRecentSignature) return;
      if (selectionTouches(recentGrid)) return;
      if (!visibleItems.length) {
        recentGrid.innerHTML = "";
        const empty = document.createElement("div");
        empty.className = "recent-empty";
        empty.textContent = "Gallery is empty.";
        recentGrid.append(empty);
      } else {
        recentGrid.replaceChildren(...visibleItems.map((item) => {
          const card = document.createElement("article");
          card.className = "recent-card";
          card.dataset.imageId = String(item.id);
          if (item.id === previewImageId) card.classList.add("selected");
          setPromptTooltip(card, item.prompt || "");
          const link = document.createElement("a");
          link.href = apiPath(item.url);
          link.target = "_blank";
          link.rel = "noopener";
          link.setAttribute("aria-label", item.prompt || "Generated image");
          link.addEventListener("click", (event) => {
            if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
              return;
            }
            event.preventDefault();
            showPreview(item);
          });
          const image = document.createElement("img");
          image.alt = item.prompt || "Generated image";
          image.src = apiPath(item.url);
          link.append(image);
          const meta = document.createElement("div");
          meta.className = "recent-meta";
          meta.textContent = generationDetails(item);
          card.append(link, meta);
          return card;
        }));
      }
      const hiddenByLimit = Math.max(0, items.length - visibleItems.length);
      const totalHidden = hiddenByLimit + hiddenCount;
      recentMore.textContent = totalHidden > 0
        ? `${totalHidden} more in the gallery are not shown.`
        : "";
      lastRecentSignature = signature;
    }

    function generationDetails(item) {
      return `${item.model} @ ${item.width}x${item.height}\nSeed: ${item.seed}\nSteps: ${item.steps}`;
    }

    async function refreshStatus() {
      try {
        const response = await apiFetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderStatus(await response.json());
      } catch (error) {
        setDot(serverDot, "bad");
        serverStatus.textContent = "Server disconnected";
        setDot(loadDot, "bad");
        loadLabel.textContent = "Model unavailable";
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
        busy = false;
        sendButton.disabled = true;
        batchOpenButton.disabled = true;
        batchGenerateButton.disabled = true;
        loadButton.disabled = true;
        ejectButton.disabled = true;
      }
    }

    async function postJSON(path, payload = {}) {
      const response = await apiFetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) {
        throw new Error(data.message || `HTTP ${response.status}`);
      }
      await refreshStatus();
    }

    function openBatchModal() {
      batchModal.hidden = false;
      batchFileInput.value = "";
      resetBatchValidation();
      batchFileButton.focus();
    }

    function closeBatchModal() {
      batchModal.hidden = true;
      batchDrop.classList.remove("dragging");
    }

    function resetBatchValidation() {
      validatedBatchJobs = null;
      batchFileName.textContent = "";
      batchValidation.textContent = "";
      batchValidation.classList.remove("valid", "invalid");
      updateBatchGenerateEnabled();
    }

    function setBatchValidation(message, state) {
      batchValidation.textContent = message;
      batchValidation.classList.remove("valid", "invalid");
      if (state) batchValidation.classList.add(state);
      updateBatchGenerateEnabled();
    }

    function isBatchStatusText(text) {
      const trimmed = text.trim();
      return trimmed.startsWith("Paste failed \u00d7 -")
        || trimmed.startsWith("Invalid \u00d7 -")
        || trimmed.startsWith("Validated \u2713 -");
    }

    async function copyExampleBatchPrompt() {
      batchCopyStatus.textContent = "";
      try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) {
          throw new Error("Clipboard is unavailable");
        }
        await navigator.clipboard.writeText(EXAMPLE_BATCH_AGENT_PROMPT);
        batchCopyStatus.textContent = " Copied.";
      } catch (error) {
        const blob = new Blob([EXAMPLE_BATCH_AGENT_PROMPT], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank", "noopener");
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
        batchCopyStatus.textContent = " Opened.";
      }
    }

    async function validateBatchText(text, sourceName) {
      resetBatchValidation();
      batchFileInput.value = "";
      batchFileName.textContent = sourceName;
      setBatchValidation("Validating...", "");
      try {
        const parsed = JSON.parse(text);
        const response = await apiFetch("/api/validate-batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jobs: parsed }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.message || `HTTP ${response.status}`);
        }
        validatedBatchJobs = parsed;
        setBatchValidation(`Validated \u2713 - ${data.count} jobs`, "valid");
      } catch (error) {
        validatedBatchJobs = null;
        setBatchValidation(`Invalid \u00d7 - ${error.message || String(error)}`, "invalid");
      }
    }

    async function validateBatchFile(file) {
      resetBatchValidation();
      if (!file) return;
      try {
        await validateBatchText(await file.text(), file.name);
      } catch (error) {
        batchFileName.textContent = file.name;
        setBatchValidation(`Invalid \u00d7 - ${error.message || String(error)}`, "invalid");
      }
    }

    async function pasteBatchText() {
      resetBatchValidation();
      batchFileName.textContent = "Pasted JSON";
      try {
        if (!navigator.clipboard || !navigator.clipboard.readText) {
          throw new Error("Clipboard is unavailable.");
        }
        const text = await navigator.clipboard.readText();
        if (!text.trim()) {
          throw new Error("Clipboard is empty.");
        }
        if (isBatchStatusText(text)) {
          setBatchValidation("Paste failed \u00d7 - Your clipboard contains a previous status message. Copy your batch JSON, then try again.", "invalid");
          return;
        }
        await validateBatchText(text, "Pasted JSON");
      } catch (error) {
        validatedBatchJobs = null;
        const message = error.message || String(error);
        const clipboardBlocked = error.name === "NotAllowedError"
          || error.name === "SecurityError"
          || message === "Clipboard is unavailable.";
        if (clipboardBlocked) {
          setBatchValidation("Paste failed \u00d7 - Clipboard access was blocked. Enable the clipboard permission for this site in your browser, then try again.", "invalid");
        } else {
          setBatchValidation(`Paste failed \u00d7 - ${message}`, "invalid");
        }
      }
    }

    async function requestGenerationCancel(message) {
      if (cancelRequested) return;
      currentLogMessage = message;
      currentLogIsError = false;
      renderEvents([], { force: true });
      try {
        await postJSON("/api/cancel");
      } catch (error) {
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
      }
    }

    function updateGenerateEnabled() {
      if (batchRunning) {
        sendButton.disabled = true;
        return;
      }
      sendButton.classList.toggle("cancel", generationRunning);
      if (generationRunning) {
        sendButton.disabled = cancelRequested;
        sendButton.textContent = cancelRequested ? "\u2026" : "\u00d7";
        sendButton.title = cancelRequested ? "Cancelling" : "Cancel generation";
        sendButton.setAttribute("aria-label", sendButton.title);
        return;
      }
      sendButton.disabled = busy || !promptInput.value.trim();
      sendButton.textContent = "\u2191";
      sendButton.title = "Generate";
      sendButton.setAttribute("aria-label", "Generate");
    }

    function updateBatchGenerateEnabled() {
      batchGenerateButton.disabled = !validatedBatchJobs || busy;
    }

    function autoResizePrompt() {
      promptInput.style.height = "auto";
      promptInput.style.height = `${promptInput.scrollHeight}px`;
    }

    function selectedSeed() {
      const value = seedInput.value.trim();
      return value ? value : seedInput.placeholder;
    }

    function generationPayload() {
      return {
        prompt: promptInput.value.trim(),
        width: widthInput.value,
        height: heightInput.value,
        steps: stepsInput.value,
        seed: selectedSeed(),
      };
    }

    function applyPreset() {
      const value = dimensionPreset.value;
      if (value === "custom") return;
      const [width, height] = value.split("x");
      widthInput.value = width;
      heightInput.value = height;
    }

    function syncPresetFromFields() {
      const value = `${widthInput.value}x${heightInput.value}`;
      const option = Array.from(dimensionPreset.options).find((item) => item.value === value);
      dimensionPreset.value = option ? value : "custom";
    }

    promptInput.addEventListener("input", () => {
      autoResizePrompt();
      updateGenerateEnabled();
    });

    resultImage.addEventListener("load", syncPreviewMetaWidth);
    window.addEventListener("resize", () => {
      syncPreviewMetaWidth();
      hidePromptTooltip();
    });

    document.addEventListener("pointerover", (event) => {
      const target = promptTooltipAnchor(event);
      if (!target || target === promptTooltipTarget) return;
      schedulePromptTooltip(target, event.clientX, event.clientY);
    });

    document.addEventListener("pointermove", (event) => {
      const target = promptTooltipAnchor(event);
      if (!target || target !== promptTooltipTarget) return;
      promptTooltipPoint = { x: event.clientX, y: event.clientY };
      positionPromptTooltip(event.clientX, event.clientY);
    });

    document.addEventListener("pointerout", (event) => {
      const target = promptTooltipAnchor(event);
      if (!target) return;
      if (event.relatedTarget instanceof Node && target.contains(event.relatedTarget)) return;
      hidePromptTooltip();
    });

    document.addEventListener("focusin", (event) => {
      const target = promptTooltipAnchor(event);
      if (!target) return;
      const rect = target.getBoundingClientRect();
      schedulePromptTooltip(target, rect.left + Math.min(rect.width / 2, 240), rect.top + 8);
    });

    document.addEventListener("focusout", (event) => {
      if (promptTooltipAnchor(event)) hidePromptTooltip();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !batchModal.hidden) {
        event.preventDefault();
        closeBatchModal();
        return;
      }
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        if (batchRunning || generationRunning || cancelRequested || !promptInput.value.trim()) return;
        event.preventDefault();
        promptForm.requestSubmit();
      }
    });

    dimensionPreset.addEventListener("change", applyPreset);
    widthInput.addEventListener("input", syncPresetFromFields);
    heightInput.addEventListener("input", syncPresetFromFields);
    randomSeedButton.addEventListener("click", () => {
      seedInput.value = String(randomSeed());
    });

    promptForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (batchRunning) return;
      if (generationRunning) {
        if (cancelRequested) return;
        sendButton.disabled = true;
        await requestGenerationCancel("Requesting cancellation.");
        return;
      }
      if (!promptInput.value.trim() || busy) return;
      sendButton.disabled = true;
      currentLogMessage = "Sending prompt.";
      currentLogIsError = false;
      renderEvents([], { force: true });
      try {
        await postJSON("/api/generate", generationPayload());
      } catch (error) {
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
      }
    });

    batchOpenButton.addEventListener("click", openBatchModal);
    batchCloseButton.addEventListener("click", closeBatchModal);
    batchCancelModalButton.addEventListener("click", closeBatchModal);
    batchFileButton.addEventListener("click", () => batchFileInput.click());
    batchPasteButton.addEventListener("click", pasteBatchText);
    batchFileInput.addEventListener("change", () => {
      validateBatchFile(batchFileInput.files && batchFileInput.files[0]);
    });
    batchExampleButton.addEventListener("click", copyExampleBatchPrompt);
    batchCancelButton.addEventListener("click", async () => {
      batchCancelButton.disabled = true;
      await requestGenerationCancel("Requesting batch cancellation.");
    });
    batchGenerateButton.addEventListener("click", async () => {
      if (!validatedBatchJobs || busy) return;
      batchGenerateButton.disabled = true;
      currentLogMessage = "Starting batch.";
      currentLogIsError = false;
      renderEvents([], { force: true });
      try {
        await postJSON("/api/generate-batch", { jobs: validatedBatchJobs });
        closeBatchModal();
      } catch (error) {
        setBatchValidation(`Invalid \u00d7 - ${error.message || String(error)}`, "invalid");
      } finally {
        updateBatchGenerateEnabled();
      }
    });

    for (const eventName of ["dragenter", "dragover"]) {
      batchDrop.addEventListener(eventName, (event) => {
        event.preventDefault();
        batchDrop.classList.add("dragging");
      });
    }
    for (const eventName of ["dragleave", "drop"]) {
      batchDrop.addEventListener(eventName, (event) => {
        event.preventDefault();
        batchDrop.classList.remove("dragging");
      });
    }
    batchDrop.addEventListener("drop", (event) => {
      const file = event.dataTransfer && event.dataTransfer.files[0];
      validateBatchFile(file);
    });

    loadButton.addEventListener("click", async () => {
      try { await postJSON("/api/load"); }
      catch (error) {
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
      }
    });

    ejectButton.addEventListener("click", async () => {
      try { await postJSON("/api/eject"); }
      catch (error) {
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
      }
    });

    function toggleLog() {
      logExpanded = !logExpanded;
      eventLog.classList.toggle("expanded", logExpanded);
      eventLog.scrollTop = 0;
      renderEvents(lastEvents, { force: true });
    }

    function adjustRecentLimit(delta) {
      recentLimit = Math.max(1, recentLimit + delta);
      refreshStatus();
    }

    function attachHoldStepper(button, action) {
      let startTimer = null;
      let repeatTimer = null;
      let delay = 220;
      let suppressClick = false;

      function clearTimers() {
        if (startTimer !== null) clearTimeout(startTimer);
        if (repeatTimer !== null) clearTimeout(repeatTimer);
        startTimer = null;
        repeatTimer = null;
      }

      function repeat() {
        action();
        delay = Math.max(55, delay * 0.82);
        repeatTimer = setTimeout(repeat, delay);
      }

      button.addEventListener("pointerdown", (event) => {
        if (button.disabled) return;
        event.preventDefault();
        suppressClick = true;
        delay = 220;
        action();
        clearTimers();
        startTimer = setTimeout(repeat, 300);
        if (button.setPointerCapture) {
          button.setPointerCapture(event.pointerId);
        }
      });

      for (const eventName of ["pointerup", "pointercancel", "pointerleave", "lostpointercapture", "blur"]) {
        button.addEventListener(eventName, clearTimers);
      }

      button.addEventListener("click", (event) => {
        if (suppressClick) {
          event.preventDefault();
          suppressClick = false;
          return;
        }
        if (!button.disabled) action();
      });
    }

    attachHoldStepper(recentLess, () => {
      recentLimit = Math.max(1, recentLimit - 1);
      refreshStatus();
    });

    attachHoldStepper(recentMoreButton, () => adjustRecentLimit(1));

    directoryButton.addEventListener("click", async () => {
      try { await postJSON("/api/open-output-dir"); }
      catch (error) {
        currentLogMessage = String(error);
        currentLogIsError = true;
        renderEvents([], { force: true });
      }
    });

    autoResizePrompt();
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
