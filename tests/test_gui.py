from __future__ import annotations

import tempfile
import threading
import time
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

import boogu_turbo_mlx.gui as gui
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.pipeline import BooguTurboGenerationResult
from boogu_turbo_mlx.setup_flow import SetupConfig


class GuiBatchValidationTests(unittest.TestCase):
    def test_batch_validation_accepts_valid_jobs_and_randomizes_missing_seed(self) -> None:
        payload = {
            "jobs": [
                {
                    "prompt": "a glass library",
                    "width": 16,
                    "height": 16,
                    "steps": 4,
                    "seed": 7,
                },
                {
                    "prompt": "a cedar observatory",
                    "width": 32,
                    "height": 16,
                    "steps": 2,
                },
            ]
        }

        with mock.patch.object(gui.secrets, "randbits", return_value=123456):
            requests = gui._batch_jobs_from_payload(payload)

        self.assertEqual([request.prompt for request in requests], [
            "a glass library",
            "a cedar observatory",
        ])
        self.assertEqual([request.seed for request in requests], [7, 123456])
        self.assertTrue(0 <= requests[1].seed <= gui.MAX_SEED)

    def test_batch_validation_rejects_invalid_payloads(self) -> None:
        valid_job = {
            "prompt": "a glass library",
            "width": 16,
            "height": 16,
            "steps": 1,
            "seed": 1,
        }
        cases = [
            ([], "JSON object"),
            ({"jobs": {}}, "jobs array"),
            ({"jobs": []}, "at least one job"),
            ({"jobs": ["nope"]}, "Job 1 must be"),
            ({"jobs": [{**valid_job, "surprise": True}]}, "unsupported field"),
            ({"jobs": [{key: value for key, value in valid_job.items() if key != "prompt"}]}, "Prompt cannot"),
            ({"jobs": [{**valid_job, "width": 15}]}, "multiple of 16"),
            ({"jobs": [{**valid_job, "height": gui.MAX_GUI_SIZE + 16}]}, "2048 or smaller"),
            ({"jobs": [{**valid_job, "width": 0}]}, "positive"),
            ({"jobs": [{**valid_job, "steps": 0}]}, "positive integer"),
            ({"jobs": [{**valid_job, "seed": gui.MAX_SEED + 1}]}, "Seed must"),
            ({"jobs": [valid_job] * (gui.MAX_GUI_BATCH_JOBS + 1)}, "more than"),
        ]

        for payload, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(BooguTurboMlxError) as ctx:
                    gui._batch_jobs_from_payload(payload)  # type: ignore[arg-type]
                self.assertIn(expected, str(ctx.exception))


class GuiBatchWorkerTests(unittest.TestCase):
    def test_batch_worker_runs_jobs_in_order_and_records_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = _FakePipeline()
            state = _state(Path(temp_dir), pipeline)
            requests = [
                gui._GenerationRequest("one", height=16, width=16, steps=2, seed=11),
                gui._GenerationRequest("two", height=32, width=16, steps=3, seed=22),
            ]

            accepted, _ = state.start_batch_generation(requests)
            self.assertTrue(accepted)
            _wait_for_idle(state)

            self.assertEqual([call["prompt"] for call in pipeline.calls], ["one", "two"])
            self.assertEqual([call["kwargs"]["seed"] for call in pipeline.calls], [11, 22])
            self.assertEqual(len(list(Path(temp_dir).glob("*.png"))), 2)
            self.assertEqual([record.prompt for record in state.recent_generations], ["two", "one"])
            self.assertEqual(state.session_image.prompt, "two")
            self.assertFalse(state.generation_running)
            self.assertIsNone(state.snapshot()["batch"])

    def test_batch_worker_aborts_on_first_failure_and_keeps_prior_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = _FakePipeline(fail_on=2)
            state = _state(Path(temp_dir), pipeline)
            requests = [
                gui._GenerationRequest("one", height=16, width=16, steps=1, seed=1),
                gui._GenerationRequest("two", height=16, width=16, steps=1, seed=2),
                gui._GenerationRequest("three", height=16, width=16, steps=1, seed=3),
            ]

            accepted, _ = state.start_batch_generation(requests)
            self.assertTrue(accepted)
            _wait_for_idle(state)

            self.assertEqual([call["prompt"] for call in pipeline.calls], ["one", "two"])
            self.assertEqual(len(list(Path(temp_dir).glob("*.png"))), 1)
            self.assertEqual([record.prompt for record in state.recent_generations], ["one"])
            self.assertFalse(state.generation_running)
            self.assertIn("Job 2 of 3 failed", state.error or "")
            self.assertIsNone(state.snapshot()["batch"])

    def test_batch_worker_cancels_between_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = _FakePipeline()
            state = _state(Path(temp_dir), pipeline)
            original_run_single = state._run_single_job_locked

            def run_single_and_cancel(request: gui._GenerationRequest) -> None:
                original_run_single(request)
                if request.prompt == "one":
                    state.start_cancel_generation()

            state._run_single_job_locked = run_single_and_cancel  # type: ignore[method-assign]
            requests = [
                gui._GenerationRequest("one", height=16, width=16, steps=1, seed=1),
                gui._GenerationRequest("two", height=16, width=16, steps=1, seed=2),
            ]

            accepted, _ = state.start_batch_generation(requests)
            self.assertTrue(accepted)
            _wait_for_idle(state)

            self.assertEqual([call["prompt"] for call in pipeline.calls], ["one"])
            self.assertEqual(len(list(Path(temp_dir).glob("*.png"))), 1)
            self.assertEqual(state.phase, "cancelled")
            self.assertIsNone(state.error)
            self.assertFalse(state.generation_running)
            self.assertIsNone(state.snapshot()["batch"])

    def test_snapshot_exposes_batch_block_only_while_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = _BlockingPipeline()
            state = _state(Path(temp_dir), pipeline)
            request = gui._GenerationRequest(
                "a glass library",
                height=32,
                width=16,
                steps=4,
                seed=99,
            )

            accepted, _ = state.start_batch_generation([request])
            self.assertTrue(accepted)
            self.assertTrue(pipeline.started.wait(timeout=2.0))
            snapshot = state.snapshot()

            self.assertEqual(
                snapshot["batch"],
                {
                    "index": 1,
                    "total": 1,
                    "prompt": "a glass library",
                    "width": 16,
                    "height": 32,
                    "steps": 4,
                    "seed": 99,
                    "model": "bf16",
                },
            )

            pipeline.release.set()
            _wait_for_idle(state)
            self.assertIsNone(state.snapshot()["batch"])


class GuiMemoryModeTests(unittest.TestCase):
    def test_low_memory_mode_does_not_resident_load_and_marks_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = _FakePipeline()
            pipeline.is_loaded = False
            state = _state(Path(temp_dir), pipeline, memory_mode="low-memory")

            accepted, message = state.start_load()
            self.assertFalse(accepted)
            self.assertIn("Low-memory mode", message)
            self.assertFalse(pipeline.is_loaded)
            self.assertEqual(state.snapshot()["model"]["memory_mode"], "low-memory")

            request = gui._GenerationRequest(
                "a glass library",
                height=16,
                width=16,
                steps=1,
                seed=1,
            )
            state._run_single_job_locked(request)

            output = next(Path(temp_dir).glob("*.png"))
            with Image.open(output) as image:
                metadata = image.info["boogu-turbo-mlx"]
            self.assertIn('"memory_mode": "low-memory"', metadata)


class _FakePipeline:
    is_loaded = True

    def __init__(self, *, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[dict[str, object]] = []

    def load(self, *args: object, **kwargs: object) -> "_FakePipeline":
        self.is_loaded = True
        return self

    def unload(self, *args: object, **kwargs: object) -> None:
        self.is_loaded = False

    def generate_batch(self, prompt: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.fail_on == len(self.calls):
            raise RuntimeError("boom")
        seed = int(kwargs["seed"])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        steps = int(kwargs["steps"])
        return SimpleNamespace(
            items=(
                _fake_result(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    steps=steps,
                ),
            )
        )


class _BlockingPipeline(_FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate_batch(self, prompt: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("timed out waiting for test release")
        seed = int(kwargs["seed"])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        steps = int(kwargs["steps"])
        return SimpleNamespace(
            items=(
                _fake_result(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    steps=steps,
                ),
            )
        )


def _state(
    output_dir: Path,
    pipeline: _FakePipeline,
    *,
    memory_mode: str = "resident",
) -> gui._GuiState:
    output_dir.mkdir(parents=True, exist_ok=True)
    return gui._GuiState(
        config=SetupConfig(
            artifact_type="bf16",
            bf16_output=Path("artifact"),
            output_dir=output_dir,
            memory_mode=memory_mode,
        ),
        pipeline=pipeline,  # type: ignore[arg-type]
        stream=StringIO(),
    )


def _wait_for_idle(state: gui._GuiState, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with state.lock:
            if not state.generation_running:
                return
        time.sleep(0.01)
    raise AssertionError("GUI state did not become idle")


def _fake_result(
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    steps: int,
) -> BooguTurboGenerationResult:
    return BooguTurboGenerationResult(
        image=Image.new("RGB", (1, 1), (seed % 255, 8, 7)),
        prompt=prompt,
        prompt_index=0,
        image_index=0,
        batch_index=0,
        seed=seed,
        height=height,
        width=width,
        denoise_height=height,
        denoise_width=width,
        output_height=1,
        output_width=1,
        steps=steps,
        step_count=steps,
        sigmas=tuple(0.1 for _ in range(steps)),
        timesteps=None,
        max_sequence_length=1280,
        truncate_instruction_sequence=False,
        dmd_conditioning_sigma=0.001,
        token_count=12,
    )


if __name__ == "__main__":
    unittest.main()
