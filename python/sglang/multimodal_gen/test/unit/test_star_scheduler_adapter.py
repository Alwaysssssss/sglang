import unittest

import torch

from sglang.multimodal_gen.runtime.models.schedulers.star_vpsde_dpmpp2m import (
    StarVPSDEDPMPP2MScheduler,
)


class TestStarSchedulerAdapter(unittest.TestCase):
    def test_set_timesteps_and_step_keep_shapes_stable(self):
        scheduler = StarVPSDEDPMPP2MScheduler(
            num_train_timesteps=32,
            num_steps=4,
        )
        scheduler.set_timesteps(4, device="cpu")

        self.assertEqual(tuple(scheduler.timesteps.shape), (4,))
        self.assertEqual(tuple(scheduler.alphas_cumprod_sqrt.shape), (5,))
        self.assertGreater(int(scheduler.timesteps[0].item()), int(scheduler.timesteps[-1].item()))

        sample = torch.randn(1, 4, 2, 8, 8)
        model_output = torch.randn_like(sample)

        scaled = scheduler.scale_model_input(sample, scheduler.timesteps[0])
        self.assertEqual(tuple(scaled.shape), tuple(sample.shape))
        self.assertTrue(torch.equal(scaled, sample))

        latents = sample
        generator = torch.Generator(device="cpu").manual_seed(0)
        for timestep in scheduler.timesteps:
            latents = scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
                generator=generator,
                return_dict=False,
            )[0]

        self.assertEqual(tuple(latents.shape), tuple(sample.shape))
        self.assertEqual(latents.dtype, sample.dtype)


if __name__ == "__main__":
    unittest.main()
