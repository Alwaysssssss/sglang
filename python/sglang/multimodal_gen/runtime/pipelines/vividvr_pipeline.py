# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import torch
from diffusers.video_processor import VideoProcessor
from safetensors.torch import load_file
from torch import nn
from transformers import T5EncoderModel

from sglang.multimodal_gen.configs.models.dits.cogvideox import CogVideoXConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.vividvr import VividVRPipelineConfig
from sglang.multimodal_gen.configs.sample.vividvr import VividVRSamplingParams
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr import (
    CogVideoXVividVRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.cogvideox_vividvr_controlnet import (
    CogVideoXVividVRControlNetModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRBeforeDenoisingStage,
    VividVRDecodingStage,
    VividVRDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class _VividVRT5EncoderWrapper(nn.Module):
    def __init__(self, encoder: T5EncoderModel):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        # Original Vivid-VR calls T5 with input_ids only. Keep that behavior so
        # prompt embeddings stay numerically aligned with the reference pipeline.
        outputs = self.encoder(input_ids=input_ids, **kwargs)
        return BaseEncoderOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class VividVRPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "CogVideoXVividVRControlNetPipeline"
    is_video_pipeline = True
    pipeline_config_cls = VividVRPipelineConfig
    sampling_params_cls = VividVRSamplingParams

    _required_config_modules = [
        "tokenizer",
        "vae",
        "scheduler",
    ]

    def build_executor(self, server_args: ServerArgs):
        return SyncExecutor(server_args=server_args)

    def initialize_pipeline(self, server_args: ServerArgs):
        vivid_root = Path(
            server_args.component_paths.get(
                "vividvr",
                str(Path(self.model_path).resolve().parent / "Vivid-VR"),
            )
        )
        text_encoder_component_path = self._resolve_component_path(
            server_args,
            "text_encoder",
            "text_encoder",
        )
        transformer_component_path = self._resolve_component_path(
            server_args,
            "transformer",
            "transformer",
        )
        controlnet_dir = Path(
            server_args.component_paths.get("controlnet", str(vivid_root / "controlnet"))
        )
        text_encoder_dtype = PRECISION_TO_TYPE[
            server_args.pipeline_config.text_encoder_precisions[0]
        ]

        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_component_path,
            torch_dtype=text_encoder_dtype,
        )
        text_encoder = _VividVRT5EncoderWrapper(text_encoder).to(
            device=get_local_torch_device(),
            dtype=text_encoder_dtype,
        )
        text_encoder.eval()

        hf_config = get_diffusers_component_config(transformer_component_path)
        server_args.pipeline_config.dit_config.update_model_arch(hf_config)
        vividvr_config = CogVideoXConfig()
        vividvr_config.update_model_arch(hf_config)

        safetensors_list = resolve_transformer_safetensors_to_load(
            server_args,
            transformer_component_path,
        )
        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=hf_config,
            server_args=server_args,
            safetensors_list=safetensors_list,
            component_model_path=transformer_component_path,
            model_cls=CogVideoXVividVRTransformer3DModel,
            cls_name=CogVideoXVividVRTransformer3DModel.__name__,
        )

        transformer = maybe_load_fsdp_model(
            model_cls=CogVideoXVividVRTransformer3DModel,
            init_params={
                "config": vividvr_config,
                "hf_config": hf_config,
                "quant_config": quant_spec.runtime_quant_config,
            },
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(transformer)
        transformer.load_connectors(str(vivid_root / "connectors.pt"))
        transformer.load_control_feat_proj(str(vivid_root / "control_feat_proj.pt"))
        transformer.load_control_patch_embed(str(vivid_root / "control_patch_embed.pt"))
        transformer.patch_embed.use_positional_embeddings = False
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False
        transformer.config.use_rotary_positional_embeddings = True
        transformer.eval()

        controlnet_config = get_diffusers_component_config(str(controlnet_dir))
        controlnet_dtype = quant_spec.param_dtype or PRECISION_TO_TYPE[
            server_args.pipeline_config.dit_precision
        ]
        with set_default_torch_dtype(controlnet_dtype), torch.device("meta"):
            controlnet = CogVideoXVividVRControlNetModel(**controlnet_config)
        controlnet_state_dict = load_file(
            str(controlnet_dir / "diffusion_pytorch_model.safetensors"),
            device="cpu",
        )
        controlnet.load_state_dict(controlnet_state_dict, strict=True, assign=True)
        controlnet = controlnet.to(device=get_local_torch_device(), dtype=controlnet_dtype)
        controlnet.eval()

        self.add_module("text_encoder", text_encoder)
        self.add_module("transformer", transformer)
        self.add_module("controlnet", controlnet)

        vae = self.get_module("vae")
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        del server_args
        self.add_stages(
            [
                VividVRBeforeDenoisingStage(
                    text_encoder=self.get_module("text_encoder"),
                    tokenizer=self.get_module("tokenizer"),
                    vae=self.get_module("vae"),
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    video_processor=self.video_processor,
                ),
                VividVRDenoisingStage(
                    transformer=self.get_module("transformer"),
                    controlnet=self.get_module("controlnet"),
                    scheduler=self.get_module("scheduler"),
                ),
                VividVRDecodingStage(
                    vae=self.get_module("vae"),
                    video_processor=self.video_processor,
                ),
            ]
        )


EntryClass = VividVRPipeline
