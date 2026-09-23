# SPDX-License-Identifier: Apache-2.0
"""Native SGLang pipeline for VSR (video super-resolution / restoration).

One stage, because VSR has no text encoder, no scheduler and no sampling loop;
see ``stages/model_specific_stages/vsr.py`` for why that is deliberate rather
than unfinished.

Weights are loaded with the reference's three-way split -- VAE encoder from
``wan_root/vae``, DiT from ``<checkpoint>/transformer_ema``, decoder from
``<checkpoint>/vae_decoder_ema.pt`` -- instead of SGLang's usual
``-model-path`` + ``-transformer-path`` pair. The VideoEdit docs flag that
mismatch as a silent-wrong-weights hazard, and phase 1 loads the diffusers
classes directly anyway, so the paths keep their reference meaning:

    -model-path   the Stage-3 checkpoint directory
    -wan-root     the base Wan2.2-TI2V-5B-Diffusers directory

Because the checkpoint is not a diffusers ``model_index.json`` tree, the
pipeline is selected explicitly with ``-pipeline-class-name WanVSRPipeline``
rather than by auto-detection (``requirements.md`` §5.1).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
from sglang.multimodal_gen.configs.sample.vsr import WanVRSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vsr import (
    VSRRestoreStage,
    resolve_output_path,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

#: dtype name -> torch dtype for the `dtype` sampling parameter.
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class WanVSRPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanVSRPipeline"
    pipeline_config_cls = WanVSRPipelineConfig
    sampling_params_cls = WanVRSamplingParams
    is_video_pipeline = True

    # Nothing is loaded through the standard component registry: the DiT comes
    # from the checkpoint (not from `transformer/` under model_path) and there
    # is no text encoder or scheduler at all.
    _required_config_modules: ClassVar[list[str]] = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer

        wan_root = server_args.component_paths.get("wan_root")
        if not wan_root:
            raise ValueError(
                "VSR needs the base Wan2.2 TI2V-5B directory for the VAE encoder; "
                "pass it with --wan-root"
            )

        cfg = server_args.pipeline_config
        model_kwargs = {
            "checkpoint_dir": self.model_path,
            "wan_root": wan_root,
            "dtype": _DTYPES[cfg.precision],
            "vae_cpu_offload": bool(server_args.vae_cpu_offload),
            "dit_cpu_offload": bool(server_args.dit_cpu_offload),
            "dit_layerwise_offload": bool(server_args.dit_layerwise_offload),
            "dit_offload_prefetch_size": server_args.dit_offload_prefetch_size,
            "pin_cpu_memory": server_args.pin_cpu_memory,
            "cudnn_benchmark": cfg.cudnn_benchmark,
            "channels_last_3d": cfg.channels_last_3d,
            "compile_decoder": cfg.compile_decoder,
            "compile_encoder": cfg.compile_encoder,
            "decoder_implicit_padding": cfg.decoder_implicit_padding,
            "cache_dit_condition": cfg.cache_dit_condition,
            "tile_t": cfg.tile_t,
            "tile_h": cfg.tile_h,
            "tile_w": cfg.tile_w,
            "t_overlap": cfg.temporal_overlap,
            "s_overlap": cfg.spatial_overlap,
        }
        if cfg.tile_devices:
            if server_args.num_gpus != 1:
                raise ValueError("VSR tile replicas require one scheduler (num_gpus=1)")
            from sglang.multimodal_gen.runtime.vsr.parallel import ParallelVSRRestorer

            self.restorer = ParallelVSRRestorer(cfg.tile_devices, **model_kwargs)
        else:
            self.restorer = VSRRestorer.from_pretrained(device="cuda", **model_kwargs)
        # Deliberately *not* calling super().load_modules(): that implementation
        # exists to read a diffusers `model_index.json` and to pull components
        # from it (falling back to a Hub download). A VSR checkpoint is not a
        # diffusers tree, so it would fail before reaching anything useful.
        # Everything this pipeline needs has just been loaded above, and
        # `_required_config_modules` is empty.
        return dict(loaded_modules or {})

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stages([VSRRestoreStage(self.restorer, server_args.pipeline_config)])

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if self.executor is None:
            raise RuntimeError("WanVSRPipeline requires a pipeline executor")
        with self.executor.profile_execution(batch, dump_rank=0):
            result = self.executor.execute_with_profiling(
                self.stages, batch, server_args
            )
            return OutputBatch(
                output_file_paths=[
                    str(resolve_output_path(result.sampling_params).resolve())
                ],
                metrics=result.metrics,
            )


EntryClass = WanVSRPipeline
