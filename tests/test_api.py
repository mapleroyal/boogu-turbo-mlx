from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

from PIL import Image

from boogu_turbo_mlx import (
    BooguTurboGenerationResult,
    BooguTurboPipeline,
    save_generation_png,
    sigma_schedule,
)
from boogu_turbo_mlx.errors import BooguTurboMlxError
from boogu_turbo_mlx.geometry import aligned_size, image_token_grid, latent_shape
from boogu_turbo_mlx.png import (
    PNG_METADATA_KEY,
    PNG_PARAMETERS_KEY,
    generation_metadata_payload,
)


class ApiTests(unittest.TestCase):
    def test_default_dmd_sigma_schedule_matches_official_turbo_values(self) -> None:
        self.assertEqual(
            sigma_schedule(),
            [0.001, 0.25075, 0.5005, 0.75025],
        )

    def test_timestep_sigmas_are_normalized_when_given_diffusion_scale(self) -> None:
        self.assertEqual(sigma_schedule(timesteps=[1, 250, 500]), [0.001, 0.25, 0.5])

    def test_geometry_uses_vae_and_patch_alignment(self) -> None:
        self.assertEqual(aligned_size(513, 520), (512, 512))
        self.assertEqual(latent_shape(513, 520, batch_size=2), (2, 16, 64, 64))
        self.assertEqual(image_token_grid(1024, 768), (64, 48))

    def test_pipeline_reports_missing_artifact_before_runtime_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "future-artifact"
            pipeline = BooguTurboPipeline.from_pretrained(missing)
            with self.assertRaises(BooguTurboMlxError) as ctx:
                pipeline.generate("a quiet glass library", height=512, width=512)
        self.assertIn("Model artifact directory does not exist", str(ctx.exception))

    def test_save_generation_png_writes_cli_metadata_keys(self) -> None:
        result = _fake_result()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "image.png"
            save_generation_png(
                result,
                output,
                model="artifact",
                memory_mode="resident",
                denoise_batch_size=2,
                decode_batch_size=1,
            )

            with Image.open(output) as saved:
                metadata = json.loads(saved.info[PNG_METADATA_KEY])
                parameters = saved.info[PNG_PARAMETERS_KEY]

        self.assertEqual(
            metadata,
            generation_metadata_payload(
                result,
                model="artifact",
                memory_mode="resident",
                denoise_batch_size=2,
                decode_batch_size=1,
            ),
        )
        self.assertIn("a glass library", parameters)
        self.assertIn("Seed: 42", parameters)
        self.assertIn("Generator: boogu-turbo-mlx", parameters)


def _fake_result() -> BooguTurboGenerationResult:
    return BooguTurboGenerationResult(
        image=Image.new("RGB", (1, 1), (9, 8, 7)),
        prompt="a glass library",
        prompt_index=0,
        image_index=0,
        batch_index=0,
        seed=42,
        height=16,
        width=16,
        denoise_height=16,
        denoise_width=16,
        output_height=1,
        output_width=1,
        steps=4,
        step_count=2,
        sigmas=(0.1, 0.5),
        timesteps=(0.1, 0.5),
        max_sequence_length=1280,
        truncate_instruction_sequence=False,
        dmd_conditioning_sigma=0.001,
        token_count=12,
    )


if __name__ == "__main__":
    unittest.main()
