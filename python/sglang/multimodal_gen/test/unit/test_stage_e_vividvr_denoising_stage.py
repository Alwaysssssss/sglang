import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRDenoisingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)
_DEVICE_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".vividvr._runtime_compute_device"
)
_ROPE_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".vividvr.prepare_rotary_positional_embeddings"
)


class _PassThroughScheduler:
    def scale_model_input(self, sample, timestep):
        del timestep
        return sample

    def step(
        self,
        noise_pred,
        old_pred_original_sample,
        timestep,
        previous_timestep,
        latents,
        **kwargs,
    ):
        del noise_pred, old_pred_original_sample, timestep, previous_timestep, kwargs
        return latents, torch.zeros_like(latents)


class _RecordingControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        control_states,
        image_rotary_emb,
        timestep,
        ofs,
        return_dict=False,
    ):
        del encoder_hidden_states, control_states, timestep, ofs, return_dict
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "rope_tokens": None
                if image_rotary_emb is None
                else int(image_rotary_emb[0].shape[0]),
            }
        )
        return ([(hidden_states,)],)


class _RecordingTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            attention_head_dim=64,
            patch_size=2,
            patch_size_t=None,
            ofs_embed_dim=None,
            use_rotary_positional_embeddings=True,
            sample_height=60,
            sample_width=90,
        )
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        control_hidden_states,
        image_rotary_emb,
        timestep,
        ofs,
        return_dict=False,
    ):
        del encoder_hidden_states, control_hidden_states, timestep, ofs, return_dict
        self.calls.append(
            {
                "hidden_shape": tuple(hidden_states.shape),
                "rope_tokens": None
                if image_rotary_emb is None
                else int(image_rotary_emb[0].shape[0]),
            }
        )
        latent_channels = hidden_states.shape[2] // 2
        return (hidden_states[:, :, :latent_channels].clone(),)


def _dummy_server_args():
    return SimpleNamespace(
        dit_cpu_offload=False,
        pipeline_config=SimpleNamespace(
            dit_precision="fp32",
            vae_precision="fp32",
        ),
    )


class TestVividVRDenoisingStageRotaryAlignment(unittest.TestCase):
    def test_run_denoising_step_recomputes_rotary_embeddings_for_each_tile(self):
        rope_calls: list[tuple[int, int]] = []

        def _fake_prepare_rotary_positional_embeddings(
            *,
            latent_height,
            latent_width,
            num_frames,
            patch_size,
            patch_size_t,
            attention_head_dim,
            device,
            sample_height,
            sample_width,
        ):
            del (
                num_frames,
                patch_size,
                patch_size_t,
                attention_head_dim,
                sample_height,
                sample_width,
            )
            rope_calls.append((int(latent_height), int(latent_width)))
            token_count = int(latent_height) * int(latent_width)
            return (
                torch.zeros(token_count, 1, device=device),
                torch.zeros(token_count, 1, device=device),
            )

        transformer = _RecordingTransformer()
        controlnet = _RecordingControlNet()
        scheduler = _PassThroughScheduler()

        with (
            patch(_GLOBAL_ARGS_PATCH, return_value=SimpleNamespace()),
            patch(_DEVICE_PATCH, return_value=torch.device("cpu")),
            patch(_ROPE_PATCH, side_effect=_fake_prepare_rotary_positional_embeddings),
        ):
            stage = VividVRDenoisingStage(
                transformer=transformer,
                controlnet=controlnet,
                scheduler=scheduler,
            )
            stage._build_runtime_attn_metadata = lambda *args, **kwargs: None

            batch = Req(sampling_params=VividVRSamplingParams())
            batch.generator = None
            batch.eta = 0.0

            state = stage.prepare_denoising_state(
                batch,
                _dummy_server_args(),
                latents=torch.zeros(1, 3, 16, 4, 8),
                control_latents=torch.zeros(1, 3, 16, 4, 8),
                prompt_embeds=torch.zeros(2, 2, 8),
                negative_prompt_embeds=None,
                do_classifier_free_guidance=False,
                timesteps=torch.tensor([1.0]),
                tiling_infos=[
                    (
                        (
                            slice(None),
                            slice(None),
                            slice(None),
                            slice(None),
                            slice(0, 4),
                        ),
                        torch.ones(1, 3, 16, 4, 4),
                    ),
                    (
                        (
                            slice(None),
                            slice(None),
                            slice(None),
                            slice(None),
                            slice(4, 8),
                        ),
                        torch.ones(1, 3, 16, 4, 4),
                    ),
                ],
            )

            stage.run_denoising_step(
                batch,
                _dummy_server_args(),
                state,
                0,
                guidance_scale=1.0,
                restoration_guidance_scale=-1.0,
            )

        self.assertEqual(
            rope_calls,
            [(4, 8), (4, 4), (4, 4)],
        )
        self.assertEqual(
            [call["rope_tokens"] for call in controlnet.calls],
            [16, 16],
        )
        self.assertEqual(
            [call["rope_tokens"] for call in transformer.calls],
            [16, 16],
        )
