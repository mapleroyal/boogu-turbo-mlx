from __future__ import annotations

import unittest

from boogu_turbo_mlx.dmd import predict_dmd_update, renoise_dmd_latents, sigma_schedule


class DmdTests(unittest.TestCase):
    def test_predict_update_matches_official_student_math(self) -> None:
        self.assertEqual(predict_dmd_update(2.0, 0.25, 4.0), 5.0)

    def test_renoise_matches_official_student_math(self) -> None:
        self.assertEqual(renoise_dmd_latents(5.0, 0.5, 3.0), 4.0)

    def test_custom_timesteps_define_effective_step_count(self) -> None:
        self.assertEqual(
            sigma_schedule(num_inference_steps=99, timesteps=[1.0, 250.0]),
            [0.001, 0.25],
        )


if __name__ == "__main__":
    unittest.main()
