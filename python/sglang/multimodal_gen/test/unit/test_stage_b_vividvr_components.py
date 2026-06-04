import json
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig
from sglang.multimodal_gen.configs.models.vaes.cogvideox import CogVideoXVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.runtime.models.dits.cogvideox import (
    CogVideoXTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr import (
    CogVideoXVividVRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_controlnet import (
    CogVideoXVividVRControlNetModel,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.cogvideox_dpm_vividvr import (
    CogVideoXDDIMScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.cogvideox import AutoencoderKLCogVideoX

BASE_CKPT_DIR = Path("/home/zhiheng/Vivid-VR/ckpts")
COGVIDEOX_DIR = BASE_CKPT_DIR / "CogVideoX1.5-5B"
VIVIDVR_DIR = BASE_CKPT_DIR / "Vivid-VR"

TRANSFORMER_CONFIG_PATH = COGVIDEOX_DIR / "transformer" / "config.json"
VAE_CONFIG_PATH = COGVIDEOX_DIR / "vae" / "config.json"
SCHEDULER_CONFIG_PATH = COGVIDEOX_DIR / "scheduler" / "scheduler_config.json"
CONTROLNET_CONFIG_PATH = VIVIDVR_DIR / "controlnet" / "config.json"
CONNECTORS_PATH = VIVIDVR_DIR / "connectors.pt"
CONTROL_FEAT_PROJ_PATH = VIVIDVR_DIR / "control_feat_proj.pt"
CONTROL_PATCH_EMBED_PATH = VIVIDVR_DIR / "control_patch_embed.pt"
CONTROLNET_WEIGHTS_PATH = VIVIDVR_DIR / "controlnet" / "diffusion_pytorch_model.safetensors"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _meta_device():
    return torch.device("meta")


class TestStageBVividVRComponents(unittest.TestCase):
    def test_model_registry_resolves_phase_b_classes(self):
        expected = {
            "CogVideoXTransformer3DModel": CogVideoXTransformer3DModel,
            "AutoencoderKLCogVideoX": AutoencoderKLCogVideoX,
            "CogVideoXDPMScheduler": None,
            "CogVideoXDDIMScheduler": None,
            "CogVideoXVividVRTransformer3DModel": CogVideoXVividVRTransformer3DModel,
            "CogVideoXVividVRControlNetModel": CogVideoXVividVRControlNetModel,
        }

        supported = ModelRegistry.get_supported_archs()
        for arch_name, expected_cls in expected.items():
            with self.subTest(arch_name=arch_name):
                self.assertIn(arch_name, supported)
                if expected_cls is not None:
                    resolved_cls, resolved_arch = ModelRegistry.resolve_model_cls(arch_name)
                    self.assertEqual(resolved_arch, arch_name)
                    self.assertIs(resolved_cls, expected_cls)

    def test_vividvr_pipeline_uses_cogvideox_configs(self):
        pipeline_config = VividVRPipelineConfig()

        self.assertIsInstance(pipeline_config.dit_config, CogVideoXConfig)
        self.assertIsInstance(pipeline_config.vae_config, CogVideoXVAEConfig)

    def test_base_transformer_initializes_from_real_config_on_meta(self):
        hf_config = _load_json(TRANSFORMER_CONFIG_PATH)
        config = CogVideoXConfig()
        config.update_model_arch(hf_config)

        with _meta_device():
            model = CogVideoXTransformer3DModel(config, hf_config=hf_config)

        self.assertTrue(next(model.parameters()).is_meta)
        self.assertEqual(model.config.num_layers, 42)
        self.assertEqual(model.config.num_attention_heads, 48)
        self.assertEqual(model.config.attention_head_dim, 64)
        self.assertEqual(model.config.in_channels, 16)
        self.assertTrue(model.config.use_rotary_positional_embeddings)
        self.assertFalse(model.config.use_learned_positional_embeddings)

    def test_vivid_transformer_initializes_from_real_config_on_meta(self):
        hf_config = _load_json(TRANSFORMER_CONFIG_PATH)
        config = CogVideoXConfig()
        config.update_model_arch(hf_config)

        with _meta_device():
            model = CogVideoXVividVRTransformer3DModel(config, hf_config=hf_config)

        self.assertTrue(next(model.parameters()).is_meta)
        self.assertEqual(len(model.connectors), 42)
        self.assertEqual(len(model.control_feat_proj), 3)

    def test_vivid_controlnet_initializes_from_real_config_on_meta(self):
        config = _load_json(CONTROLNET_CONFIG_PATH)
        config.pop("_class_name", None)

        with _meta_device():
            model = CogVideoXVividVRControlNetModel(**config)

        self.assertTrue(next(model.parameters()).is_meta)
        self.assertEqual(len(model.transformer_blocks), 6)
        self.assertEqual(len(model.control_feat_proj), 3)
        self.assertTrue(model.config.use_rotary_positional_embeddings)
        self.assertFalse(model.config.use_learned_positional_embeddings)

    def test_vae_initializes_from_real_config_on_meta(self):
        hf_config = _load_json(VAE_CONFIG_PATH)
        config = CogVideoXVAEConfig()
        config.update_model_arch(hf_config)

        with _meta_device():
            model = AutoencoderKLCogVideoX(config)

        self.assertTrue(next(model.parameters()).is_meta)
        self.assertEqual(model.config.latent_channels, 16)
        self.assertEqual(model.config.temporal_compression_ratio, 4)
        self.assertEqual(model.config.scaling_factor, 0.7)

    def test_vae_encode_decode_toy_roundtrip(self):
        config = CogVideoXVAEConfig()
        config.arch_config.block_out_channels = (8, 16, 16, 32)
        config.arch_config.layers_per_block = 1
        config.arch_config.latent_channels = 4
        config.arch_config.norm_num_groups = 8
        config.arch_config.sample_height = 32
        config.arch_config.sample_width = 32
        config.arch_config.temporal_compression_ratio = 4
        config.arch_config.scaling_factor = 0.7

        vae = AutoencoderKLCogVideoX(config).eval()
        video = torch.randn(1, 3, 9, 32, 32)

        with torch.no_grad():
            latent_dist = vae.encode(video).latent_dist
            latents = latent_dist.mode()
            decoded = vae.decode(latents).sample

        self.assertEqual(latents.shape[0], video.shape[0])
        self.assertEqual(decoded.shape[:3], video.shape[:3])
        self.assertGreater(decoded.shape[-2], 0)
        self.assertGreater(decoded.shape[-1], 0)
        self.assertTrue(torch.isfinite(latents).all())
        self.assertTrue(torch.isfinite(decoded).all())

    def test_scheduler_restoration_guidance_semantics(self):
        config = _load_json(SCHEDULER_CONFIG_PATH)
        config.pop("_class_name", None)
        scheduler = CogVideoXDDIMScheduler(**config)
        scheduler.set_timesteps(10)

        # The first trailing timestep can produce NaNs in prev_sample for this tiny
        # toy input, so probe a stable middle step and validate the restoration
        # guidance branch directly.
        timestep = scheduler.timesteps[1]
        timestep_back = scheduler.timesteps[2]
        sample = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(0))
        model_output = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(1))
        old_pred_original_sample = torch.randn(
            1, 4, 2, 2, generator=torch.Generator().manual_seed(2)
        )
        restoration_ori_latent = torch.randn(
            1, 4, 2, 2, generator=torch.Generator().manual_seed(3)
        )

        base_seed = 1234
        prev_sample_base, pred_original_base = scheduler.step(
            model_output=model_output,
            old_pred_original_sample=old_pred_original_sample,
            timestep=timestep,
            timestep_back=timestep_back,
            sample=sample,
            generator=torch.Generator().manual_seed(base_seed),
            restoration_guidance_scale=-1.0,
            restoration_ori_latent=restoration_ori_latent,
            return_dict=False,
        )
        prev_sample_disabled, pred_original_disabled = scheduler.step(
            model_output=model_output,
            old_pred_original_sample=old_pred_original_sample,
            timestep=timestep,
            timestep_back=timestep_back,
            sample=sample,
            generator=torch.Generator().manual_seed(base_seed),
            restoration_guidance_scale=0.0,
            restoration_ori_latent=restoration_ori_latent,
            return_dict=False,
        )
        prev_sample_guided, pred_original_guided = scheduler.step(
            model_output=model_output,
            old_pred_original_sample=old_pred_original_sample,
            timestep=timestep,
            timestep_back=timestep_back,
            sample=sample,
            generator=torch.Generator().manual_seed(base_seed),
            restoration_guidance_scale=1.0,
            restoration_ori_latent=restoration_ori_latent,
            return_dict=False,
        )

        self.assertTrue(torch.isfinite(prev_sample_base).all())
        self.assertTrue(torch.isfinite(prev_sample_disabled).all())
        self.assertTrue(torch.allclose(prev_sample_base, prev_sample_disabled))
        self.assertTrue(torch.allclose(pred_original_base, pred_original_disabled))
        self.assertTrue(torch.isfinite(prev_sample_guided).all())
        self.assertTrue(torch.isfinite(pred_original_guided).all())
        self.assertFalse(torch.allclose(pred_original_base, pred_original_guided))

    def test_vivid_transformer_sidecar_weight_shapes_match_reference(self):
        hf_config = _load_json(TRANSFORMER_CONFIG_PATH)
        config = CogVideoXConfig()
        config.update_model_arch(hf_config)

        with _meta_device():
            model = CogVideoXVividVRTransformer3DModel(config, hf_config=hf_config)

        expected_parts = {
            "connectors": (model.connectors.state_dict(), torch.load(CONNECTORS_PATH, map_location="cpu")),
            "control_feat_proj": (
                model.control_feat_proj.state_dict(),
                torch.load(CONTROL_FEAT_PROJ_PATH, map_location="cpu"),
            ),
            "control_patch_embed": (
                model.control_patch_embed.state_dict(),
                torch.load(CONTROL_PATCH_EMBED_PATH, map_location="cpu"),
            ),
        }

        for part_name, (current_state, expected_state) in expected_parts.items():
            with self.subTest(part_name=part_name):
                self.assertEqual(set(current_state.keys()), set(expected_state.keys()))
                for key, tensor in expected_state.items():
                    with self.subTest(part_name=part_name, key=key):
                        self.assertEqual(tuple(current_state[key].shape), tuple(tensor.shape))

    def test_vivid_controlnet_checkpoint_shapes_match_reference(self):
        config = _load_json(CONTROLNET_CONFIG_PATH)
        config.pop("_class_name", None)

        with _meta_device():
            model = CogVideoXVividVRControlNetModel(**config)

        current_state = model.state_dict()
        expected_state = load_file(CONTROLNET_WEIGHTS_PATH)

        self.assertEqual(set(current_state.keys()), set(expected_state.keys()))
        for key, tensor in expected_state.items():
            with self.subTest(key=key):
                self.assertEqual(tuple(current_state[key].shape), tuple(tensor.shape))

    def test_controlnet_and_transformer_forward_smoke(self):
        config = CogVideoXConfig()
        config.arch_config.num_attention_heads = 4
        config.arch_config.attention_head_dim = 8
        config.arch_config.num_layers = 4
        config.arch_config.in_channels = 4
        config.arch_config.out_channels = 4
        config.arch_config.text_embed_dim = 32
        config.arch_config.time_embed_dim = 32
        config.arch_config.sample_frames = 4
        config.arch_config.sample_height = 8
        config.arch_config.sample_width = 8
        config.arch_config.patch_size = 2
        config.arch_config.patch_size_t = 2
        config.arch_config.temporal_compression_ratio = 1
        config.arch_config.max_text_seq_length = 6
        config.arch_config.use_rotary_positional_embeddings = True
        config.arch_config.use_learned_positional_embeddings = False
        config.arch_config.__post_init__()

        controlnet_kwargs = {
            "num_attention_heads": config.arch_config.num_attention_heads,
            "attention_head_dim": config.arch_config.attention_head_dim,
            "in_channels": config.arch_config.in_channels,
            "out_channels": config.arch_config.out_channels,
            "flip_sin_to_cos": config.arch_config.flip_sin_to_cos,
            "freq_shift": config.arch_config.freq_shift,
            "time_embed_dim": config.arch_config.time_embed_dim,
            "ofs_embed_dim": config.arch_config.ofs_embed_dim,
            "text_embed_dim": config.arch_config.text_embed_dim,
            "num_layers": 2,
            "dropout": config.arch_config.dropout,
            "attention_bias": config.arch_config.attention_bias,
            "sample_width": config.arch_config.sample_width,
            "sample_height": config.arch_config.sample_height,
            "sample_frames": config.arch_config.sample_frames,
            "patch_size": config.arch_config.patch_size,
            "patch_size_t": config.arch_config.patch_size_t,
            "temporal_compression_ratio": config.arch_config.temporal_compression_ratio,
            "max_text_seq_length": config.arch_config.max_text_seq_length,
            "activation_fn": config.arch_config.activation_fn,
            "timestep_activation_fn": config.arch_config.timestep_activation_fn,
            "norm_elementwise_affine": config.arch_config.norm_elementwise_affine,
            "norm_eps": config.arch_config.norm_eps,
            "spatial_interpolation_scale": config.arch_config.spatial_interpolation_scale,
            "temporal_interpolation_scale": config.arch_config.temporal_interpolation_scale,
            "use_rotary_positional_embeddings": config.arch_config.use_rotary_positional_embeddings,
            "use_learned_positional_embeddings": config.arch_config.use_learned_positional_embeddings,
            "patch_bias": config.arch_config.patch_bias,
        }

        transformer = CogVideoXVividVRTransformer3DModel(config, hf_config={}).eval()
        controlnet = CogVideoXVividVRControlNetModel(**controlnet_kwargs).eval()

        batch_size = 2
        num_frames = 4
        noisy = torch.randn(batch_size, num_frames, 4, 8, 8)
        control = torch.randn(batch_size, num_frames, 4, 8, 8)
        encoder_hidden_states = torch.randn(batch_size, 6, 32)
        timestep = torch.tensor([10, 20], dtype=torch.long)

        with torch.no_grad():
            control_out = controlnet(
                hidden_states=noisy,
                encoder_hidden_states=encoder_hidden_states,
                control_states=control,
                timestep=timestep,
                return_dict=True,
            )
            output = transformer(
                hidden_states=torch.cat([noisy, control], dim=2),
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                control_hidden_states=control_out.sample,
                return_dict=True,
            )

        self.assertEqual(output.sample.shape, noisy.shape)
        self.assertEqual(output.sample.dtype, noisy.dtype)
        self.assertTrue(torch.isfinite(output.sample).all())


if __name__ == "__main__":
    unittest.main()
