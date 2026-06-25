from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from boogu_turbo_mlx import (
    BooguTurboGenerationCancelled,
    BooguTurboPipeline,
    InstructionEncoding,
    PipelineProgressEvent,
)
from boogu_turbo_mlx.constants import (
    MAX_GENERATION_SEED,
    OFFICIAL_BOOGU_GITHUB_REVISION,
    VAE_SCALE_FACTOR,
)
from boogu_turbo_mlx.encoding import (
    OFFICIAL_TEXT_WEIGHT_PREFIX,
    Qwen3VLInstructionEncoder,
    Qwen3VLTextConfig,
    _flatten_parameter_shapes as _encoder_parameter_shapes,
)
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.pipeline import (
    MAX_DENOISE_PIXELS,
    _BenchmarkRecorder,
    _denoise_size_for_request,
    _normalize_generation_seeds,
    _postprocess_batch_to_pil,
    _postprocess_to_pil,
    _validate_generation_dimensions,
    _validate_memory_mode,
)
from boogu_turbo_mlx.transformer import (
    BooguImageTransformer,
    BooguImageTransformerConfig,
    _flatten_parameter_shapes as _transformer_parameter_shapes,
    build_transformer_freqs_cis,
)
from boogu_turbo_mlx.vae import (
    AutoencoderKLDecoder,
    AutoencoderKLDecoderConfig,
    _flatten_parameter_shapes as _vae_parameter_shapes,
)

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - environment dependent.
    mx = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_BOOGU_SOURCE = (
    ROOT / "models" / f"Boogu-Image-source-{OFFICIAL_BOOGU_GITHUB_REVISION}"
)


class PipelineSizingTests(unittest.TestCase):
    def test_denoise_size_preserves_request_separately_from_cap_and_alignment(self) -> None:
        self.assertEqual(_denoise_size_for_request(513, 520), (512, 512))
        self.assertEqual(_denoise_size_for_request(4096, 4096), (2048, 2048))

        height, width = _denoise_size_for_request(4096, 2048)

        self.assertLessEqual(height * width, MAX_DENOISE_PIXELS)
        self.assertEqual(height % 16, 0)
        self.assertEqual(width % 16, 0)

    def test_generation_seed_normalization_expands_or_validates_batch(self) -> None:
        self.assertEqual(
            _normalize_generation_seeds(seed=7, seeds=None, batch_size=3),
            (7, 8, 9),
        )
        self.assertEqual(
            _normalize_generation_seeds(seed=None, seeds=[4, 4], batch_size=2),
            (4, 4),
        )
        self.assertIsNone(
            _normalize_generation_seeds(seed=None, seeds=None, batch_size=2)
        )

        with self.assertRaises(ValueError):
            _normalize_generation_seeds(seed=1, seeds=[2], batch_size=1)
        with self.assertRaises(ValueError):
            _normalize_generation_seeds(seed=None, seeds=[1], batch_size=2)
        with self.assertRaises(ValueError):
            _normalize_generation_seeds(seed=-1, seeds=None, batch_size=1)
        with self.assertRaises(ValueError):
            _normalize_generation_seeds(
                seed=MAX_GENERATION_SEED,
                seeds=None,
                batch_size=2,
            )

    def test_public_generation_dimensions_match_model_limits(self) -> None:
        self.assertEqual(_validate_generation_dimensions(2048, 16), (2048, 16))

        for height, width, expected in (
            (15, 16, "multiple of 16"),
            (16, 15, "multiple of 16"),
            (2064, 16, "2048 or smaller"),
            (16, 0, "positive"),
        ):
            with self.subTest(height=height, width=width):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_generation_dimensions(height, width)

    def test_runtime_control_validators_normalize_and_reject_conflicts(self) -> None:
        self.assertEqual(_validate_memory_mode("low-memory"), "low_memory")

        with self.assertRaises(BooguTurboMlxError):
            _validate_memory_mode("streaming")


@unittest.skipIf(mx is None, "MLX is not installed")
class MlxPipelineTests(unittest.TestCase):
    def test_postprocess_maps_minus_one_to_one_into_uint8_pil(self) -> None:
        tensor = mx.array(
            [[[[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]]]],
            dtype=mx.float32,
        )

        image = _postprocess_to_pil(tensor)

        self.assertEqual(image.size, (3, 1))
        self.assertEqual(
            list(image.tobytes()),
            [0, 0, 0, 128, 128, 128, 255, 255, 255],
        )

    def test_postprocess_batch_returns_all_pil_images(self) -> None:
        tensor = mx.array(
            [
                [[[-1.0]], [[-1.0]], [[-1.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
            ],
            dtype=mx.float32,
        )

        images = _postprocess_batch_to_pil(tensor)

        self.assertEqual(len(images), 2)
        self.assertEqual([image.size for image in images], [(1, 1), (1, 1)])
        self.assertEqual(images[0].tobytes(), bytes([0, 0, 0]))
        self.assertEqual(images[1].tobytes(), bytes([255, 255, 255]))

    def test_generate_runs_dmd_steps_with_batched_timesteps(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)

        image = pipeline.generate(
            "a glass library",
            height=16,
            width=32,
            seed=3,
            timesteps=[1.0, 500.0],
            max_sequence_length=7,
            truncate_instruction_sequence=True,
        )

        self.assertEqual(image.size, (32, 16))
        self.assertEqual([name for name, _ in calls], ["encode", "transform", "transform", "vae"])
        self.assertEqual(calls[0][1]["max_sequence_length"], 7)
        self.assertTrue(calls[0][1]["truncate"])
        transformer_calls = [payload for name, payload in calls if name == "transform"]
        self.assertEqual([payload["timestep_shape"] for payload in transformer_calls], [(1,), (1,)])
        self.assertEqual([payload["sigma"] for payload in transformer_calls], [0.001, 0.5])

    def test_generate_batch_expands_prompts_images_and_seeds_in_order(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)

        batch = pipeline.generate_batch(
            ["a glass library", "a cedar observatory"],
            height=16,
            width=16,
            seed=10,
            num_images_per_prompt=2,
            steps=1,
        )

        self.assertEqual(len(batch), 4)
        self.assertEqual(len(batch.images), 4)
        self.assertEqual(
            [item.prompt for item in batch],
            [
                "a glass library",
                "a glass library",
                "a cedar observatory",
                "a cedar observatory",
            ],
        )
        self.assertEqual([item.seed for item in batch], [10, 11, 12, 13])
        self.assertEqual([item.prompt_index for item in batch], [0, 0, 1, 1])
        self.assertEqual([item.image_index for item in batch], [0, 1, 0, 1])
        self.assertEqual([item.batch_index for item in batch], [0, 1, 2, 3])
        self.assertEqual([item.step_count for item in batch], [1, 1, 1, 1])
        self.assertEqual([item.sigmas for item in batch], [(0.001,)] * 4)
        self.assertEqual([item.output_width for item in batch], [16] * 4)
        self.assertEqual([item.output_height for item in batch], [16] * 4)
        self.assertEqual(batch[2].to_metadata()["prompt"], "a cedar observatory")
        self.assertEqual(batch.to_metadata()[2]["seed"], 12)

        self.assertEqual([name for name, _ in calls], ["encode", "transform", "vae"])
        self.assertEqual(calls[0][1]["prompt"], ["a glass library", "a cedar observatory"])
        transformer_call = next(payload for name, payload in calls if name == "transform")
        self.assertEqual(transformer_call["latent_shape"], (4, 1, 2, 2))
        self.assertEqual(transformer_call["timestep_shape"], (4,))
        self.assertEqual(transformer_call["mask_shape"], (4, 2))
        vae_call = next(payload for name, payload in calls if name == "vae")
        self.assertEqual(vae_call["latent_shape"], (4, 1, 2, 2))

    def test_generate_emits_ordered_progress_for_fake_two_step_generation(self) -> None:
        pipeline = _fake_pipeline([])
        events: list[PipelineProgressEvent] = []

        image = pipeline.generate(
            "a glass library",
            height=16,
            width=32,
            seed=3,
            timesteps=[1.0, 500.0],
            progress_callback=events.append,
        )

        self.assertEqual(image.size, (32, 16))
        self.assertEqual(
            [event.kind for event in events],
            [
                "encode_start",
                "encode_end",
                "denoise_step_start",
                "denoise_step_end",
                "denoise_step_start",
                "denoise_step_end",
                "denoise_compute_start",
                "denoise_compute_end",
                "decode_start",
                "decode_end",
                "complete",
            ],
        )
        self.assertEqual(events[2].step_index, 0)
        self.assertEqual(events[2].step_count, 2)
        self.assertEqual(events[2].sigma, 0.001)
        self.assertEqual(events[4].sigma, 0.5)

    def test_generate_omits_resize_progress_for_noop_resize(self) -> None:
        pipeline = _fake_pipeline([])
        events: list[PipelineProgressEvent] = []

        image = pipeline.generate(
            "a glass library",
            height=16,
            width=16,
            seed=3,
            steps=1,
            progress_callback=events.append,
        )

        self.assertEqual(image.size, (16, 16))
        self.assertNotIn("resize_start", [event.kind for event in events])
        self.assertNotIn("resize_end", [event.kind for event in events])
        self.assertEqual(events[-1].kind, "complete")

    def test_progress_callback_can_cancel_between_denoise_steps(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)
        events: list[str] = []

        def cancel_on_second_step(event: PipelineProgressEvent) -> None:
            events.append(event.kind)
            if event.kind == "denoise_step_start" and event.step_index == 1:
                raise BooguTurboGenerationCancelled("stop requested")

        with mock.patch("boogu_turbo_mlx.pipeline._clear_runtime_cache") as clear_cache:
            with self.assertRaises(BooguTurboGenerationCancelled):
                pipeline.generate(
                    "a glass library",
                    height=16,
                    width=16,
                    seed=3,
                    timesteps=[1.0, 500.0],
                    progress_callback=cancel_on_second_step,
                )

        self.assertEqual([name for name, _ in calls], ["encode", "transform"])
        self.assertEqual(events[-1], "cancelled")
        clear_cache.assert_called_once_with()

    def test_progress_callback_can_cancel_before_denoise_compute(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)
        events: list[str] = []

        def cancel_on_compute(event: PipelineProgressEvent) -> None:
            events.append(event.kind)
            if event.kind == "denoise_compute_start":
                raise BooguTurboGenerationCancelled("stop requested")

        with mock.patch("boogu_turbo_mlx.pipeline._clear_runtime_cache") as clear_cache:
            with self.assertRaises(BooguTurboGenerationCancelled):
                pipeline.generate(
                    "a glass library",
                    height=16,
                    width=16,
                    seed=3,
                    steps=1,
                    progress_callback=cancel_on_compute,
                )

        self.assertEqual([name for name, _ in calls], ["encode", "transform"])
        self.assertEqual(events[-1], "cancelled")
        clear_cache.assert_called_once_with()

    def test_seeded_generation_is_deterministic(self) -> None:
        pipeline = _fake_pipeline([])

        first = pipeline.generate("a glass library", height=16, width=16, seed=11, steps=1)
        second = pipeline.generate("a glass library", height=16, width=16, seed=11, steps=1)

        self.assertEqual(first.tobytes(), second.tobytes())

    def test_chunked_generation_matches_full_batch_with_explicit_seeds(self) -> None:
        full = _fake_pipeline([]).generate_batch(
            ["a glass library", "a cedar observatory"],
            height=16,
            width=16,
            seed=21,
            num_images_per_prompt=2,
            timesteps=[1.0, 500.0],
        )
        chunked = _fake_pipeline([]).generate_batch(
            ["a glass library", "a cedar observatory"],
            height=16,
            width=16,
            seed=21,
            num_images_per_prompt=2,
            timesteps=[1.0, 500.0],
            denoise_batch_size=1,
            decode_batch_size=1,
        )

        self.assertEqual(
            [image.tobytes() for image in chunked.images],
            [image.tobytes() for image in full.images],
        )
        self.assertEqual(chunked.to_metadata(), full.to_metadata())

    def test_transformer_prepared_inputs_are_built_once_per_latent_generation(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)
        pipeline._components.transformer = _PreparedFakeTransformer(calls)

        pipeline.generate(
            "a glass library",
            height=16,
            width=16,
            seed=3,
            timesteps=[1.0, 500.0],
        )

        self.assertEqual(
            [name for name, _ in calls],
            ["encode", "prepare", "transform", "transform", "vae"],
        )
        prepare_payload = next(payload for name, payload in calls if name == "prepare")
        self.assertEqual(prepare_payload["height"], 2)
        self.assertEqual(prepare_payload["width"], 2)
        self.assertTrue(
            all(payload["prepared"] for name, payload in calls if name == "transform")
        )

    def test_benchmark_recorder_run_peak_is_max_over_stage_peaks(self) -> None:
        # Models MLX's peak counter: reset_peak_memory() snaps peak down to the
        # current active memory, and get_peak_memory() tracks the max active
        # since the last reset. The recorder resets per stage, so the global
        # counter is clobbered by the last (light) stage once a run finishes.
        class _FakeMx:
            def __init__(self) -> None:
                self.active = 0
                self.peak = 0
                self.cache = 0

            def allocate(self, n: int) -> None:
                self.active = n
                self.peak = max(self.peak, n)

            def reset_peak_memory(self) -> None:
                self.peak = self.active

            def get_peak_memory(self) -> int:
                return self.peak

            def get_active_memory(self) -> int:
                return self.active

            def get_cache_memory(self) -> int:
                return self.cache

        fake = _FakeMx()
        recorder = _BenchmarkRecorder()
        with mock.patch("boogu_turbo_mlx.pipeline._load_mlx", return_value=fake):
            with recorder.stage("denoise"):
                fake.allocate(20)  # heavy transformer peak
                fake.active = 1  # freed before the stage ends (low_memory)
            with recorder.stage("postprocess"):
                fake.allocate(2)  # light final stage
                fake.active = 1

        # The global counter only holds the final stage's peak — this is the bug.
        self.assertEqual(fake.get_peak_memory(), 2)
        # The recorder reconstructs the true run-wide peak from per-stage peaks.
        self.assertEqual(recorder.run_peak_memory_bytes(), 20)
        self.assertEqual(recorder.to_dict()["denoise"]["peak_memory_bytes"], 20)

    def test_low_memory_generation_loads_and_drops_components_sequentially(self) -> None:
        calls: list[tuple[str, object]] = []
        order: list[str] = []
        pipeline = BooguTurboPipeline.from_pretrained("/unused", memory_mode="low-memory")
        pipeline._load_instruction_encoder = (
            lambda **_: order.append("load_encoder") or _FakeEncoder(calls)
        )
        pipeline._load_transformer_runtime = (
            lambda **_: order.append("load_transformer")
            or SimpleNamespace(transformer=_FakeTransformer(calls), freqs_cis="cached-freqs")
        )
        pipeline._load_vae_decoder = (
            lambda **_: order.append("load_vae") or _FakeVae(calls)
        )

        with mock.patch("boogu_turbo_mlx.pipeline._validate_artifact_root"), mock.patch(
            "boogu_turbo_mlx.pipeline._clear_runtime_cache",
            side_effect=lambda: order.append("clear"),
        ):
            image = pipeline.generate(
                "a glass library",
                height=16,
                width=16,
                seed=3,
                steps=1,
            )

        self.assertEqual(image.size, (16, 16))
        self.assertFalse(pipeline.is_loaded)
        self.assertEqual([name for name, _ in calls], ["encode", "transform", "vae"])
        self.assertLess(order.index("clear"), order.index("load_transformer"))
        self.assertLess(
            max(index for index, item in enumerate(order) if item == "clear" and index < order.index("load_vae")),
            order.index("load_vae"),
        )

    def test_final_dmd_step_does_not_renoise(self) -> None:
        calls: list[tuple[str, object]] = []
        pipeline = _fake_pipeline(calls)
        encoding = pipeline._components.encoder.encode("a glass library")
        initial = mx.ones((1, 1, 2, 2), dtype=mx.float32)
        renoise = [mx.zeros((1, 1, 2, 2), dtype=mx.float32)]

        latents = pipeline._generate_latents(
            encoding=encoding,
            sigmas=[0.25, 0.75],
            denoise_height=16,
            denoise_width=16,
            seed=None,
            initial_noise=initial,
            renoise_noises=renoise,
            runtime_dtype=mx.float32,
        )

        expected = mx.full((1, 1, 2, 2), 0.75, dtype=mx.float32)
        self.assertTrue(bool(mx.all(latents == expected).item()))

    def test_tiny_artifact_end_to_end_uses_real_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_pipeline_artifact(root)
            with mock.patch(
                "boogu_turbo_mlx.encoding._load_tokenizer",
                return_value=_FakeTokenizer(),
            ):
                image = BooguTurboPipeline.from_pretrained(root).generate(
                    "a glass library",
                    height=16,
                    width=16,
                    seed=0,
                    steps=1,
                )

        self.assertEqual(image.size, (16, 16))
        self.assertEqual(image.mode, "RGB")

    def test_tiny_artifact_uses_metadata_weight_index_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_tiny_pipeline_artifact(root)
            _move_weight_indexes_to_metadata_paths(root)
            with mock.patch(
                "boogu_turbo_mlx.encoding._load_tokenizer",
                return_value=_FakeTokenizer(),
            ):
                image = BooguTurboPipeline.from_pretrained(root).generate(
                    "a glass library",
                    height=16,
                    width=16,
                    seed=0,
                    steps=1,
                )

        self.assertEqual(image.size, (16, 16))


class ManualM6OracleTests(unittest.TestCase):
    ORACLE_ENV = "BOOGU_TURBO_MLX_PIPELINE_ORACLE"
    ORACLE_ARTIFACT_ENV = "BOOGU_TURBO_MLX_PIPELINE_ORACLE_ARTIFACT"
    ORACLE_SOURCE_ENV = "BOOGU_TURBO_MLX_PIPELINE_ORACLE_SOURCE"
    ORACLE_SOURCE_WEIGHTS_ENV = "BOOGU_TURBO_MLX_PIPELINE_ORACLE_SOURCE_WEIGHTS"

    @unittest.skipUnless(
        os.environ.get(ORACLE_ENV),
        f"set {ORACLE_ENV}=1 to run the official M6 pipeline oracle",
    )
    def test_pre_vae_dmd_latents_match_official_torch_loop(self) -> None:
        if mx is None:
            self.skipTest("MLX is not installed")
        try:
            import numpy as np
            import torch
            from mlx.utils import tree_map
        except ImportError as exc:
            self.skipTest(f"M6 pipeline oracle dependencies are unavailable: {exc}")

        official = _official_boogu_modules(self._oracle_source())
        artifact = self._oracle_artifact()
        source_weights = self._oracle_source_weights(artifact)
        config = BooguImageTransformerConfig.from_dict(
            json.loads((artifact / "transformer" / "config.json").read_text(encoding="utf-8"))
        )

        mlx_transformer = BooguImageTransformer.from_pretrained(artifact)
        mlx_transformer.update(
            tree_map(lambda value: value.astype(mx.float32), mlx_transformer.parameters())
        )
        mx.eval(mlx_transformer.parameters())

        torch_transformer = official.Model(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_double_stream_layers=config.num_double_stream_layers,
            num_refiner_layers=config.num_refiner_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            norm_eps=config.norm_eps,
            axes_dim_rope=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            instruction_feature_configs={
                "instruction_feat_dim": config.instruction_feat_dim,
                "num_instruction_feature_layers": config.num_instruction_feature_layers,
                "reduce_type": config.reduce_type,
            },
            timestep_scale=config.timestep_scale,
        )
        torch_transformer.load_state_dict(
            _load_torch_safetensor_prefix(
                source_weights,
                "diffusion_pytorch_model.safetensors.index.json",
                "",
            ),
            strict=True,
        )
        torch_transformer.eval()

        denoise_height = denoise_width = 16
        latent_shape = (
            1,
            config.in_channels,
            denoise_height // VAE_SCALE_FACTOR,
            denoise_width // VAE_SCALE_FACTOR,
        )
        instruction_shape = (1, 6, config.instruction_feat_dim)
        initial_noise_np = np.linspace(
            -0.75,
            0.75,
            num=int(np.prod(latent_shape)),
            dtype=np.float32,
        ).reshape(latent_shape)
        renoise_noise_np = np.linspace(
            0.5,
            -0.5,
            num=int(np.prod(latent_shape)),
            dtype=np.float32,
        ).reshape(latent_shape)
        instruction_np = np.linspace(
            -0.25,
            0.25,
            num=int(np.prod(instruction_shape)),
            dtype=np.float32,
        ).reshape(instruction_shape)
        mask_np = np.ones(instruction_shape[:2], dtype=np.int32)
        sigmas = [0.001, 0.5]

        pipeline = BooguTurboPipeline.from_pretrained(artifact)
        pipeline._components = SimpleNamespace(
            encoder=None,
            transformer=mlx_transformer,
            vae=None,
            freqs_cis=build_transformer_freqs_cis(config),
        )
        encoding = InstructionEncoding(
            hidden_states=mx.array(instruction_np, dtype=mx.float32),
            attention_mask=mx.array(mask_np, dtype=mx.int32),
            input_ids=mx.ones(mask_np.shape, dtype=mx.int32),
        )
        actual = pipeline._generate_latents(
            encoding=encoding,
            sigmas=sigmas,
            denoise_height=denoise_height,
            denoise_width=denoise_width,
            seed=None,
            initial_noise=mx.array(initial_noise_np, dtype=mx.float32),
            renoise_noises=[mx.array(renoise_noise_np, dtype=mx.float32)],
            runtime_dtype=mx.float32,
        )
        mx.eval(actual)

        rope_embedder = official.Rope(
            theta=config.theta,
            axes_dim=config.axes_dim_rope,
            axes_lens=config.axes_lens,
            patch_size=config.patch_size,
        )
        freqs_cis = rope_embedder.get_freqs_cis(
            config.axes_dim_rope,
            config.axes_lens,
            config.theta,
        )
        expected = torch.from_numpy(initial_noise_np).to(torch.float32)
        instruction = torch.from_numpy(instruction_np).to(torch.float32)
        mask = torch.from_numpy(mask_np).to(torch.int64)
        renoise_noise = torch.from_numpy(renoise_noise_np).to(torch.float32)
        with torch.no_grad():
            for index, sigma in enumerate(sigmas):
                model_prediction = torch_transformer(
                    expected,
                    torch.tensor([sigma], dtype=torch.float32),
                    instruction,
                    freqs_cis,
                    mask,
                    ref_image_hidden_states=None,
                    return_dict=False,
                )
                expected = expected + (1.0 - sigma) * model_prediction
                if index < len(sigmas) - 1:
                    next_sigma = sigmas[index + 1]
                    expected = (1.0 - next_sigma) * renoise_noise + next_sigma * expected

        diff = np.abs(np.array(actual) - expected.detach().cpu().numpy())
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        print(
            "M6 pipeline oracle pre-VAE final latents "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )
        self.assertLessEqual(max_abs, 2e-2)
        self.assertLessEqual(mean_abs, 5e-3)

    def _oracle_artifact(self) -> Path:
        artifact = os.environ.get(
            self.ORACLE_ARTIFACT_ENV,
            str(ROOT / "artifacts" / "boogu-mlx"),
        )
        path = Path(artifact)
        if not (path / "artifact.json").exists():
            self.skipTest(f"M2 artifact not found at {path}; set {self.ORACLE_ARTIFACT_ENV}")
        return path

    def _oracle_source(self) -> Path:
        source = os.environ.get(
            self.ORACLE_SOURCE_ENV,
            str(DEFAULT_OFFICIAL_BOOGU_SOURCE),
        )
        path = Path(source)
        if not (path / "boogu" / "models" / "transformers" / "transformer_boogu.py").exists():
            self.skipTest(f"official Boogu source not found at {path}; set {self.ORACLE_SOURCE_ENV}")
        return path

    def _oracle_source_weights(self, artifact: Path) -> Path:
        weights = os.environ.get(self.ORACLE_SOURCE_WEIGHTS_ENV)
        if weights is None:
            report_path = artifact / "conversion_report.json"
            if report_path.exists():
                source_path = json.loads(report_path.read_text(encoding="utf-8")).get("source_path")
                if source_path:
                    weights = str(Path(source_path) / "transformer")
        weights = weights or str(ROOT / "models" / "Boogu-Image-0.1-Turbo" / "transformer")
        path = Path(weights)
        if not (path / "diffusion_pytorch_model.safetensors.index.json").exists():
            self.skipTest(
                f"official source transformer weights not found at {path}; "
                f"set {self.ORACLE_SOURCE_WEIGHTS_ENV}"
            )
        return path


class _FakeEncoder:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def encode(
        self,
        prompt: object,
        *,
        max_sequence_length: int = 1280,
        truncate: bool = False,
    ) -> InstructionEncoding:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        self.calls.append(
            (
                "encode",
                {
                    "prompt": prompt,
                    "max_sequence_length": max_sequence_length,
                    "truncate": truncate,
                },
            )
        )
        return InstructionEncoding(
            hidden_states=mx.ones((len(prompts), 2, 3), dtype=mx.float32),
            attention_mask=mx.ones((len(prompts), 2), dtype=mx.int32),
            input_ids=mx.tile(mx.array([[1, 2]], dtype=mx.int32), (len(prompts), 1)),
        )


class _FakeTransformer:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls
        self.config = SimpleNamespace(in_channels=1, patch_size=2)

    def __call__(
        self,
        latents: object,
        timestep: object,
        instruction_hidden_states: object,
        freqs_cis: object,
        instruction_attention_mask: object,
    ) -> object:
        self.calls.append(
            (
                "transform",
                {
                    "latent_shape": tuple(latents.shape),
                    "timestep_shape": tuple(timestep.shape),
                    "sigma": round(float(timestep[0].item()), 6),
                    "mask_shape": tuple(instruction_attention_mask.shape),
                    "freqs_cis": freqs_cis,
                },
            )
        )
        return mx.zeros(latents.shape, dtype=latents.dtype)


class _PreparedFakeTransformer(_FakeTransformer):
    def prepare_forward_inputs(
        self,
        freqs_cis: object,
        instruction_attention_mask: object,
        height: int,
        width: int,
    ) -> object:
        prepared = SimpleNamespace(mlx_values=lambda: (mx.array(0),))
        self.calls.append(
            (
                "prepare",
                {
                    "freqs_cis": freqs_cis,
                    "mask_shape": tuple(instruction_attention_mask.shape),
                    "height": height,
                    "width": width,
                },
            )
        )
        return prepared

    def __call__(
        self,
        latents: object,
        timestep: object,
        instruction_hidden_states: object,
        freqs_cis: object,
        instruction_attention_mask: object,
        *,
        prepared_inputs: object | None = None,
    ) -> object:
        self.calls.append(
            (
                "transform",
                {
                    "latent_shape": tuple(latents.shape),
                    "timestep_shape": tuple(timestep.shape),
                    "sigma": round(float(timestep[0].item()), 6),
                    "mask_shape": tuple(instruction_attention_mask.shape),
                    "freqs_cis": freqs_cis,
                    "prepared": prepared_inputs is not None,
                },
            )
        )
        return mx.zeros(latents.shape, dtype=latents.dtype)


class _FakeVae:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    def decode(self, latents: object) -> object:
        self.calls.append(("vae", {"latent_shape": tuple(latents.shape)}))
        sample = mx.repeat(latents[:, :1], 3, axis=1)
        sample = mx.repeat(mx.repeat(sample, 8, axis=2), 8, axis=3)
        return mx.tanh(sample)


class _FakeTokenizer:
    def apply_chat_template(self, *args: object, **kwargs: object) -> str:
        return "rendered prompt"

    def __call__(self, *args: object, **kwargs: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}


def _fake_pipeline(calls: list[tuple[str, object]]) -> BooguTurboPipeline:
    pipeline = BooguTurboPipeline.from_pretrained("/unused")
    pipeline._components = SimpleNamespace(
        encoder=_FakeEncoder(calls),
        transformer=_FakeTransformer(calls),
        vae=_FakeVae(calls),
        freqs_cis="cached-freqs",
    )
    return pipeline


def _write_tiny_pipeline_artifact(root: Path) -> None:
    text_config = _tiny_text_config()
    transformer_config = _tiny_transformer_config()
    vae_config = _tiny_vae_config()

    (root / "processor").mkdir(parents=True)
    _write_tiny_encoder_weights(root, text_config)
    _write_tiny_transformer_weights(root, transformer_config)
    _write_tiny_vae_weights(root, vae_config)
    (root / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "format": "boogu-turbo-mlx-artifact",
                "processor": "processor",
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


def _tiny_text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_mllm_config(
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": 32,
                "hidden_size": 12,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 6,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000,
                "max_position_embeddings": 128,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_interleaved": True,
                    "mrope_section": [1, 1, 1],
                },
            },
        }
    )


def _tiny_transformer_config() -> BooguImageTransformerConfig:
    return BooguImageTransformerConfig.from_dict(
        {
            "patch_size": 2,
            "in_channels": 1,
            "out_channels": None,
            "hidden_size": 12,
            "num_layers": 2,
            "num_double_stream_layers": 1,
            "num_refiner_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "multiple_of": 4,
            "ffn_dim_multiplier": None,
            "norm_eps": 1e-5,
            "axes_dim_rope": [2, 2, 2],
            "axes_lens": [32, 32, 32],
            "instruction_feature_configs": {
                "instruction_feat_dim": 12,
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


def _write_tiny_encoder_weights(root: Path, config: Qwen3VLTextConfig) -> None:
    encoder = Qwen3VLInstructionEncoder(config)
    arrays = {
        OFFICIAL_TEXT_WEIGHT_PREFIX + key: mx.zeros(shape, dtype=mx.float32)
        for key, shape in _encoder_parameter_shapes(encoder.parameters()).items()
    }
    (root / "mllm").mkdir(parents=True)
    (root / "mllm" / "config.json").write_text(
        json.dumps(config.to_mllm_config()),
        encoding="utf-8",
    )
    _write_safetensor_indexed_component(
        root / "weights" / "mllm",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        arrays,
    )


def _write_tiny_transformer_weights(
    root: Path,
    config: BooguImageTransformerConfig,
) -> None:
    transformer = BooguImageTransformer(config)
    arrays = {
        key: mx.zeros(shape, dtype=mx.float32)
        for key, shape in _transformer_parameter_shapes(transformer.parameters()).items()
    }
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    _write_safetensor_indexed_component(
        root / "weights" / "transformer",
        "diffusion_pytorch_model-00001-of-00001.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        arrays,
    )


def _write_tiny_vae_weights(root: Path, config: AutoencoderKLDecoderConfig) -> None:
    vae = AutoencoderKLDecoder(config)
    arrays = {
        key: mx.zeros(_raw_pytorch_shape(shape), dtype=mx.float32)
        for key, shape in _vae_parameter_shapes(vae.parameters()).items()
    }
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "config.json").write_text(
        json.dumps(config.to_dict()),
        encoding="utf-8",
    )
    _write_safetensor_indexed_component(
        root / "weights" / "vae",
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
        arrays,
    )


def _write_safetensor_indexed_component(
    directory: Path,
    shard_name: str,
    index_name: str,
    arrays: dict[str, object],
) -> None:
    directory.mkdir(parents=True)
    mx.save_safetensors(directory / shard_name, arrays)
    (directory / index_name).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {key: shard_name for key in arrays},
            }
        ),
        encoding="utf-8",
    )


def _move_weight_indexes_to_metadata_paths(root: Path) -> None:
    artifact_path = root / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    for component in ("mllm", "transformer", "vae"):
        original = root / artifact["components"][component]["weights_index"]
        moved = original.with_name("artifact-" + original.name)
        original.rename(moved)
        artifact["components"][component]["weights_index"] = str(moved.relative_to(root))
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")


def _raw_pytorch_shape(mlx_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(mlx_shape) == 4:
        out_channels, kernel_height, kernel_width, in_channels = mlx_shape
        return (out_channels, in_channels, kernel_height, kernel_width)
    return mlx_shape


class _OfficialBooguModules:
    def __init__(self, source: Path) -> None:
        os.environ.setdefault("device", "cpu")
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
        from boogu.models.transformers.transformer_boogu import (
            BooguImageTransformer2DModel,
        )

        self.Model = BooguImageTransformer2DModel
        self.Rope = BooguImageDoubleStreamRotaryPosEmbed


def _official_boogu_modules(source: Path) -> _OfficialBooguModules:
    try:
        return _OfficialBooguModules(source)
    except ImportError as exc:
        raise unittest.SkipTest(
            f"official Boogu source dependencies are unavailable: {exc}"
        ) from exc


def _load_torch_safetensor_prefix(
    weights_dir: Path,
    index_name: str,
    prefix: str,
) -> dict[str, object]:
    import torch
    from safetensors import safe_open

    index = json.loads((weights_dir / index_name).read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not keys:
        raise AssertionError(f"No tensors found with prefix {prefix!r}")

    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(weights_dir / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key).to(torch.float32)
    return tensors


if __name__ == "__main__":
    unittest.main()
