import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler


class TestSGLDiffusionProfiler(unittest.TestCase):
    def tearDown(self):
        SGLDiffusionProfiler._instance = None

    def test_scheduled_profile_uses_requested_active_timestep_count(self):
        fake_profiler = MagicMock()

        with (
            patch(
                "sglang.multimodal_gen.runtime.utils.profiler.torch.profiler.schedule"
            ) as schedule,
            patch(
                "sglang.multimodal_gen.runtime.utils.profiler.torch.profiler.profile",
                return_value=fake_profiler,
            ),
        ):
            profiler = SGLDiffusionProfiler(
                request_id="three-steps",
                rank=0,
                full_profile=False,
                num_steps=3,
                num_inference_steps=5,
            )
            for _ in range(5):
                profiler.step_denoising_step()

        schedule.assert_called_once_with(
            skip_first=0,
            wait=0,
            warmup=1,
            active=3,
            repeat=1,
        )
        self.assertEqual(fake_profiler.step.call_count, 4)


if __name__ == "__main__":
    unittest.main()
