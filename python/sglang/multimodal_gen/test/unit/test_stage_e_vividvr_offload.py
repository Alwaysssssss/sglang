import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.cogvideox_dpm_vividvr import (
    CogVideoXDDIMScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRConditionEncodingStage,
    VividVRDecodingStage,
    VividVRDenoisingStage,
    VividVRLatentPreparationStage,
    VividVRTimestepPreparationStage,
    _VividVRLatentMixin,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


class _DummyVividVRLatentHarness(_VividVRLatentMixin):
    def __init__(self):
        self.vae = SimpleNamespace(
            config=SimpleNamespace(
                block_out_channels=(1, 1, 1, 1),
                temporal_compression_ratio=4,
            )
        )
        self.transformer = SimpleNamespace(config=SimpleNamespace(patch_size_t=None))


class _DummyScheduler:
    def step(self, sample, generator=None, eta=None, **kwargs):
        return sample


def _dummy_server_args():
    return SimpleNamespace(
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        pipeline_config=SimpleNamespace(
            dit_precision="fp32",
            vae_precision="fp32",
        ),
    )


def _dummy_transformer_module():
    module = torch.nn.Linear(1, 1, bias=False)
    module.config = SimpleNamespace(
        in_channels=16,
        patch_size_t=None,
        ofs_embed_dim=None,
        use_rotary_positional_embeddings=False,
        attention_head_dim=64,
    )
    return module


@unittest.skipUnless(torch.cuda.is_available(), "Offload parity test requires CUDA")
class TestStageEVividVROffload(unittest.TestCase):
    def test_prepare_latent_noise_preserves_cuda_rng_when_target_is_cpu(self):
        harness = _DummyVividVRLatentHarness()
        scheduler = SimpleNamespace(init_noise_sigma=1.0)
        control_video = torch.zeros(1, 3, 9, 32, 32, dtype=torch.float32)
        control_latents = torch.zeros(1, 3, 16, 4, 4, dtype=torch.float32)
        expected_shape = (1, 3, 16, 4, 4)

        expected_generator = torch.Generator(device="cuda").manual_seed(1234)
        expected_latents = randn_tensor(
            expected_shape,
            generator=expected_generator,
            device=torch.device("cuda"),
            dtype=torch.float32,
        ).cpu()

        actual_generator = torch.Generator(device="cuda").manual_seed(1234)
        actual_latents, actual_control_latents, num_padding_frames = (
            harness._prepare_latent_noise(
                control_video=control_video,
                control_latents=control_latents,
                batch_size=1,
                num_channels_latents=16,
                height=32,
                width=32,
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=actual_generator,
                scheduler=scheduler,
            )
        )

        self.assertEqual(actual_latents.device.type, "cpu")
        self.assertEqual(num_padding_frames, 0)
        self.assertEqual(
            tuple(actual_control_latents.shape), tuple(control_latents.shape)
        )
        torch.testing.assert_close(actual_latents, expected_latents)

    def test_scheduler_step_preserves_cuda_rng_when_sample_is_cpu(self):
        scheduler = CogVideoXDDIMScheduler()
        scheduler.set_timesteps(5)

        timestep = int(scheduler.timesteps[0])
        sample_cpu = torch.randn(1, 4, 2, 2, dtype=torch.float32)
        model_output_cpu = torch.randn(1, 4, 2, 2, dtype=torch.float32)

        cpu_generator = torch.Generator(device="cuda").manual_seed(2024)
        cpu_prev_sample, cpu_pred_original = scheduler.step(
            model_output_cpu,
            None,
            timestep,
            None,
            sample_cpu,
            generator=cpu_generator,
            return_dict=False,
        )

        cuda_generator = torch.Generator(device="cuda").manual_seed(2024)
        cuda_prev_sample, cuda_pred_original = scheduler.step(
            model_output_cpu.cuda(),
            None,
            timestep,
            None,
            sample_cpu.cuda(),
            generator=cuda_generator,
            return_dict=False,
        )

        self.assertEqual(cpu_prev_sample.device.type, "cpu")
        self.assertEqual(cpu_pred_original.device.type, "cpu")
        torch.testing.assert_close(cpu_prev_sample, cuda_prev_sample.cpu())
        torch.testing.assert_close(cpu_pred_original, cuda_pred_original.cpu())

    def test_prepare_latents_uses_local_cuda_device_even_when_transformer_is_cpu(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRLatentPreparationStage(
                vae=SimpleNamespace(
                    config=SimpleNamespace(
                        block_out_channels=(1, 1, 1, 1),
                        temporal_compression_ratio=4,
                    )
                ),
                transformer=_dummy_transformer_module(),
                scheduler=SimpleNamespace(init_noise_sigma=1.0),
            )
        control_video = torch.zeros(1, 3, 9, 32, 32, dtype=torch.float32)
        control_latents = torch.zeros(1, 3, 16, 4, 4, dtype=torch.float32)
        generator = torch.Generator(device="cuda").manual_seed(7)

        latents, prepared_control_latents, _ = stage.prepare_latents(
            control_video=control_video,
            control_latents=control_latents,
            generator=generator,
            height=32,
            width=32,
        )

        self.assertEqual(next(stage.transformer.parameters()).device.type, "cpu")
        self.assertEqual(latents.device.type, "cuda")
        self.assertEqual(prepared_control_latents.device.type, "cuda")

    def test_prepare_denoising_state_uses_local_cuda_device_even_when_modules_are_cpu(
        self,
    ):
        transformer = _dummy_transformer_module()
        controlnet = torch.nn.Linear(1, 1, bias=False)
        scheduler = _DummyScheduler()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRDenoisingStage(
                transformer=transformer,
                controlnet=controlnet,
                scheduler=scheduler,
            )
        batch = Req(sampling_params=VividVRSamplingParams())
        batch.generator = torch.Generator(device="cuda").manual_seed(11)
        batch.eta = 0.0

        state = stage.prepare_denoising_state(
            batch,
            _dummy_server_args(),
            latents=torch.zeros(1, 3, 16, 4, 4),
            control_latents=torch.zeros(1, 3, 16, 4, 4),
            prompt_embeds=torch.zeros(1, 2, 8),
            negative_prompt_embeds=None,
            do_classifier_free_guidance=False,
            timesteps=torch.tensor([1.0], device="cuda"),
            tiling_infos=[],
        )

        self.assertEqual(next(stage.transformer.parameters()).device.type, "cuda")
        self.assertEqual(next(stage.controlnet.parameters()).device.type, "cuda")
        self.assertEqual(state["latents"].device.type, "cuda")
        self.assertEqual(state["control_latents"].device.type, "cuda")
        self.assertEqual(state["prompt_embeds"].device.type, "cuda")
        self.assertFalse(state["autocast_enabled"])

    def test_build_runtime_attn_metadata_records_fa_debug_fields(self):
        transformer = _dummy_transformer_module()
        controlnet = torch.nn.Linear(1, 1, bias=False)
        scheduler = _DummyScheduler()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRDenoisingStage(
                transformer=transformer,
                controlnet=controlnet,
                scheduler=scheduler,
            )
        batch = Req(sampling_params=VividVRSamplingParams())
        batch.generator = torch.Generator(device="cuda").manual_seed(11)
        batch.eta = 0.0
        batch.raw_latent_shape = (1, 3, 16, 4, 4)

        stage.prepare_denoising_state(
            batch,
            _dummy_server_args(),
            latents=torch.zeros(1, 3, 16, 4, 4),
            control_latents=torch.zeros(1, 3, 16, 4, 4),
            prompt_embeds=torch.zeros(1, 2, 8),
            negative_prompt_embeds=None,
            do_classifier_free_guidance=False,
            timesteps=torch.tensor([1.0], device="cuda"),
            tiling_infos=[],
        )
        attn_metadata = stage._build_runtime_attn_metadata(
            batch,
            _dummy_server_args(),
            timestep_index=0,
        )

        self.assertIsNotNone(attn_metadata)
        debug = batch.extra["vividvr_debug"]
        self.assertTrue(debug["attn_metadata_enabled"])
        self.assertEqual(debug["attn_metadata_backend"], "fa")
        self.assertEqual(
            debug["attn_metadata_builder"],
            "FlashAttentionMetadataBuilder",
        )

    def test_prepare_timesteps_uses_local_cuda_device_even_when_transformer_is_cpu(
        self,
    ):
        scheduler = CogVideoXDDIMScheduler()
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRTimestepPreparationStage(
                scheduler=scheduler,
                transformer=_dummy_transformer_module(),
            )

        timesteps = stage.prepare_timesteps(5)

        self.assertEqual(next(stage.transformer.parameters()).device.type, "cpu")
        self.assertEqual(timesteps.device.type, "cuda")


class _DummyDecodeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=1.0)
        self.enable_tiling_calls = 0

    def enable_tiling(self):
        self.enable_tiling_calls += 1

    def decode(self, latents):
        return latents

    def get_last_spatial_decode_stats(self):
        return SimpleNamespace(
            to_debug_dict=lambda: {
                "vae_sp_requested": True,
                "vae_sp_effective": True,
                "vae_sp_fallback_reason": "effective",
                "vae_sp_world_size": 2,
                "vae_sp_group_type": "sp",
                "vae_total_tiles": 9,
                "vae_local_tiles_per_rank": [5, 4],
                "vae_tile_decode_seconds": 1.2,
                "vae_tile_gather_seconds": 0.2,
                "vae_tile_merge_seconds": 0.1,
                "vae_decode_seconds": 1.5,
            }
        )


class _DummyEncodeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            scaling_factor=1.0,
            block_out_channels=(1, 1, 1, 1),
            temporal_compression_ratio=4,
        )
        self.events = []

    def to(self, *args, **kwargs):
        destination = args[0] if args else kwargs.get("device")
        self.events.append(f"to:{destination}")
        return self

    def encode(self, video):
        self.events.append("encode")
        return SimpleNamespace(
            latent_dist=SimpleNamespace(sample=lambda generator: video)
        )

    def get_last_spatial_encode_stats(self):
        self.events.append("stats")
        return SimpleNamespace(
            to_debug_dict=lambda: {
                "vae_encode_sp_requested": True,
                "vae_encode_sp_effective": True,
                "vae_encode_sp_fallback_reason": "effective",
                "vae_encode_sp_world_size": 2,
                "vae_encode_sp_group_type": "sp",
                "vae_encode_total_tiles": 16,
                "vae_encode_local_tiles_per_rank": [8, 8],
                "vae_encode_tile_compute_seconds": 4.0,
                "vae_encode_tile_gather_seconds": 1.0,
                "vae_encode_tile_merge_seconds": 0.5,
                "vae_encode_seconds": 6.0,
            }
        )


class _DummyVideoProcessor:
    @staticmethod
    def preprocess_video(_video, *, height, width):
        return torch.zeros(1, 3, 1, height, width)


class TestStageEVividVRConditionEncoding(unittest.TestCase):
    def test_condition_stage_exposes_encode_stats_before_cpu_offload(self):
        vae = _DummyEncodeVAE()
        stage = object.__new__(VividVRConditionEncodingStage)
        stage.vae = vae
        stage.transformer = _dummy_transformer_module()
        stage.video_processor = _DummyVideoProcessor()
        stage._resolve_generator = lambda *_args, **_kwargs: torch.Generator()
        stage._resolve_control_video_info = lambda *_args, **_kwargs: {
            "video": object(),
            "reference_video": torch.zeros(1),
            "original_height": 1,
            "original_width": 1,
            "gen_height": 1,
            "gen_width": 1,
            "original_num_frames": 1,
            "num_padding_frames": 0,
            "fps": 8,
        }
        stage._sync_runtime_resolution = lambda params, _info: (
            setattr(params, "height", 1),
            setattr(params, "width", 1),
        )
        batch = SimpleNamespace(
            sampling_params=VividVRSamplingParams(
                prompt=" ",
                video_input_path="unused.mp4",
                seed=42,
                height=1,
                width=1,
            ),
            generator=None,
            extra={},
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(dit_precision="fp32", vae_precision="fp32"),
            vae_cpu_offload=True,
        )
        module = (
            "sglang.multimodal_gen.runtime.pipelines_core.stages."
            "model_specific_stages.vividvr"
        )
        with patch(
            f"{module}.get_local_torch_device", return_value=torch.device("cpu")
        ):
            prepared = stage.prepare_condition_inputs(batch, server_args)

        self.assertTrue(prepared["vae_encode_stats"]["vae_encode_sp_effective"])
        self.assertLess(vae.events.index("stats"), len(vae.events) - 1)
        self.assertEqual(vae.events[-1], "to:cpu")


class TestStageEVividVRDecoding(unittest.TestCase):
    def test_decode_latents_keeps_e2_decode_path_without_explicit_tiling(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRDecodingStage(vae=_DummyDecodeVAE())
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(vae_precision="fp32", vae_tiling=True),
            vae_cpu_offload=False,
        )

        decoded = stage.decode_latents(
            torch.zeros(1, 3, 16, 4, 4),
            0,
            server_args,
        )

        self.assertEqual(tuple(decoded.shape), (1, 16, 3, 4, 4))
        self.assertEqual(stage.vae.enable_tiling_calls, 0)

    def test_decode_stage_exposes_last_vae_spatial_stats(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()):
            stage = VividVRDecodingStage(vae=_DummyDecodeVAE())

        stage.decode_latents(
            torch.zeros(1, 3, 16, 4, 4),
            0,
            _dummy_server_args(),
        )

        self.assertTrue(stage.last_vae_decode_stats["vae_sp_effective"])
        self.assertEqual(
            stage.last_vae_decode_stats["vae_local_tiles_per_rank"], [5, 4]
        )


if __name__ == "__main__":
    unittest.main()
