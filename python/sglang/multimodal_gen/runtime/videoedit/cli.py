# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT,
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GenerationResult,
    prepare_request,
)
from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs
from sglang.multimodal_gen.runtime.videoedit.preprocess import (
    resolve_videoedit_num_frames,
)


def _add_common_repair_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--transformer-path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT)
    parser.add_argument("--video-input-path", required=True)
    parser.add_argument("--mask-input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-file-name")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--infer-len", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--dynamic-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-cfg-max-step", type=int, default=15)
    parser.add_argument("--dynamic-cfg-min", type=float, default=1.0)
    parser.add_argument("--bbox-padding", type=int, default=0)
    parser.add_argument("--dilate-px", type=int, default=15)
    parser.add_argument("--mask-scale", type=float, default=1.2)
    parser.add_argument("--feather-px", type=int, default=12)
    parser.add_argument("--adain-boundary-dilate", type=int, default=15)
    parser.add_argument("--enable-paste-back", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-crop-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-reference-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-intermediate-windows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-repaired-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vary-seed-by-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-mode", choices=["eager", "stream"], default="eager")
    parser.add_argument("--generator-device")
    parser.add_argument("--output-quality", default="default")
    parser.add_argument("--output-compression", type=int)
    parser.add_argument("--enable-teacache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-frame-interpolation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-interpolation-exp", type=int, default=1)
    parser.add_argument("--frame-interpolation-scale", type=float, default=1.0)
    parser.add_argument("--frame-interpolation-model-path")
    parser.add_argument("--enable-upscaling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upscaling-model-path")
    parser.add_argument("--upscaling-scale", type=int, default=4)
    parser.add_argument("--perf-dump-path")

    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--sp-degree", type=int)
    parser.add_argument("--ulysses-degree", type=int)
    parser.add_argument("--ring-degree", type=int)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--dp-degree", type=int, default=1)
    parser.add_argument("--hsdp-replicate-dim", type=int, default=1)
    parser.add_argument("--hsdp-shard-dim", type=int)
    parser.add_argument("--dist-timeout", type=int)
    parser.add_argument("--attention-backend")
    parser.add_argument("--attention-backend-config")
    parser.add_argument("--cache-dit-config")
    parser.add_argument("--dit-cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dit-layerwise-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dit-offload-prefetch-size", type=float, default=0.0)
    parser.add_argument("--text-encoder-cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--image-encoder-cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vae-cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-fsdp-inference", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pin-cpu-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-torch-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--disable-autocast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--scheduler-port", type=int)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoEdit-diffusers tools for SGLang")
    subparsers = parser.add_subparsers(dest="command", required=True)
    repair = subparsers.add_parser("repair", help="Run VideoEdit repair locally")
    _add_common_repair_args(repair)
    return parser


def _result_to_jsonable(result: Any) -> Any:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, list):
        return [_result_to_jsonable(item) for item in result]
    return result


def _server_args_kwargs(args: argparse.Namespace, component_paths: dict[str, str]) -> dict[str, Any]:
    kwargs = {
        "model_path": args.model_path,
        "backend": Backend.SGLANG,
        "component_paths": component_paths,
        "output_path": args.output_path,
        "tp_size": args.tp_size,
        "num_gpus": args.num_gpus,
        "dp_size": args.dp_size,
        "dp_degree": args.dp_degree,
        "hsdp_replicate_dim": args.hsdp_replicate_dim,
        "attention_backend": args.attention_backend,
        "attention_backend_config": args.attention_backend_config,
        "cache_dit_config": args.cache_dit_config,
        "dit_offload_prefetch_size": args.dit_offload_prefetch_size,
        "warmup_steps": args.warmup_steps,
        "trust_remote_code": args.trust_remote_code,
    }
    for name in (
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "hsdp_shard_dim",
        "dist_timeout",
        "dit_cpu_offload",
        "dit_layerwise_offload",
        "text_encoder_cpu_offload",
        "image_encoder_cpu_offload",
        "vae_cpu_offload",
        "use_fsdp_inference",
        "pin_cpu_memory",
        "enable_torch_compile",
        "warmup",
        "disable_autocast",
        "master_port",
        "scheduler_port",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    return kwargs


def _generate_videoedit_locally(
    generator: DiffGenerator,
    sampling_params: WanVideoEditSamplingParams,
) -> GenerationResult | None:
    req = prepare_request(
        server_args=generator.server_args,
        sampling_params=sampling_params,
    )
    output_batch = generator._send_to_scheduler_and_wait_for_response([req])
    if output_batch.error:
        raise RuntimeError(f"{output_batch.error}")
    if output_batch.output is None and output_batch.output_file_paths is None:
        return None

    output_file_path = (
        output_batch.output_file_paths[0] if output_batch.output_file_paths else None
    )
    return GenerationResult(
        prompt=req.prompt,
        size=(req.height, req.width, req.num_frames),
        peak_memory_mb=output_batch.peak_memory_mb,
        metrics=output_batch.metrics.to_dict() if output_batch.metrics else {},
        trajectory_latents=output_batch.trajectory_latents,
        trajectory_timesteps=output_batch.trajectory_timesteps,
        trajectory_decoded=output_batch.trajectory_decoded,
        output_file_path=output_file_path,
    )


def repair_cmd(args: argparse.Namespace) -> int:
    component_paths = {}
    if args.transformer_path:
        component_paths["transformer"] = args.transformer_path

    server_args = ServerArgs.from_kwargs(**_server_args_kwargs(args, component_paths))
    resolved_num_frames = resolve_videoedit_num_frames(
        args.num_frames,
        args.video_input_path,
        args.mask_input_path,
    )
    sampling_params = WanVideoEditSamplingParams.from_user_kwargs(
        server_args,
        request_id=generate_request_id(),
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        video_input_path=args.video_input_path,
        mask_input_path=args.mask_input_path,
        output_path=args.output_path,
        output_file_name=args.output_file_name,
        num_frames=resolved_num_frames,
        infer_len=args.infer_len,
        overlap=args.overlap,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        generator_device=args.generator_device,
        dtype=args.dtype,
        dynamic_cfg=args.dynamic_cfg,
        dynamic_cfg_max_step=args.dynamic_cfg_max_step,
        dynamic_cfg_min=args.dynamic_cfg_min,
        bbox_padding=args.bbox_padding,
        dilate_px=args.dilate_px,
        mask_scale=args.mask_scale,
        feather_px=args.feather_px,
        adain_boundary_dilate=args.adain_boundary_dilate,
        enable_paste_back=args.enable_paste_back,
        save_crop_only=args.save_crop_only,
        drop_reference_frame=args.drop_reference_frame,
        keep_intermediate_windows=args.keep_intermediate_windows,
        use_repaired_context=args.use_repaired_context,
        vary_seed_by_window=args.vary_seed_by_window,
        decode_mode=args.decode_mode,
        output_quality=args.output_quality,
        output_compression=args.output_compression,
        enable_teacache=args.enable_teacache,
        enable_frame_interpolation=args.enable_frame_interpolation,
        frame_interpolation_exp=args.frame_interpolation_exp,
        frame_interpolation_scale=args.frame_interpolation_scale,
        frame_interpolation_model_path=args.frame_interpolation_model_path,
        enable_upscaling=args.enable_upscaling,
        upscaling_model_path=args.upscaling_model_path,
        upscaling_scale=args.upscaling_scale,
        perf_dump_path=args.perf_dump_path,
    )
    with DiffGenerator.from_pretrained(
        model_path=server_args.model_path,
        server_args=server_args,
        local_mode=True,
    ) as generator:
        result = _generate_videoedit_locally(generator, sampling_params)
    print(json.dumps(_result_to_jsonable(result), ensure_ascii=False, indent=2))
    return 0 if result is not None else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "repair":
        return repair_cmd(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
