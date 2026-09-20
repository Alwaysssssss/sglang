# SPDX-License-Identifier: Apache-2.0
"""Standalone CLI for VSR restore.

Kept out of ``entrypoints/cli/main.py``, following the VideoEdit precedent
(``runtime/videoedit/cli.py``).

Design points from ``requirements.md`` §5.1:

* **Weight paths keep the reference's semantics.** The checkpoint is split three
  ways -- the VAE encoder comes from ``wan_root/vae``, the DiT from
  ``checkpoint_dir/transformer_ema``, and the decoder is a bare state dict. That
  does not fit SGLang's usual ``--model-path`` + ``--transformer-path`` split,
  and the VideoEdit docs call the mismatch a silent-wrong-weights hazard. Phase 1
  loads diffusers classes directly, so it does not re-shape the paths to fit.
* **Everything that affects the output is exposed**, so a reference run can be
  reproduced flag for flag. Only the reference's YAML ``--config`` is dropped;
  defaults live in the argument declarations.
* **Single file only.** Directory batching is deferred.
* **No audio**, matching the reference: the reference writes picture only and
  copies no audio track.

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.cli restore \\
        --checkpoint_dir ... --wan_root ... --input in.mp4 --output out.mp4 \\
        --target_resolution 3840x2160 --tile_h 320 --tile_w 640 --tile_t 33
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.geometry import (
    parse_resolution,
    resize_to_long_edge,
)
from sglang.multimodal_gen.runtime.vsr.stream import COLOR_REF_MODES, stream_restore
from sglang.multimodal_gen.runtime.vsr.video_io import probe_video

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _add_restore_args(parser: argparse.ArgumentParser) -> None:
    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "--input", required=True, help="Input video path (single file)"
    )
    io_group.add_argument("--output", required=True, help="Output video path")

    w_group = parser.add_argument_group("weights")
    w_group.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Stage-3 checkpoint dir (transformer_ema/ + vae_decoder_ema.pt)",
    )
    w_group.add_argument(
        "--wan_root",
        required=True,
        help="Base Wan2.2-TI2V-5B-Diffusers dir; only vae/ is read from it",
    )

    g_group = parser.add_argument_group("geometry")
    # Spelled H x W, matching the reference. Do not read it as W x H.
    g_group.add_argument(
        "--target_resolution",
        default=None,
        help="Exact target size as HxW (height first). Overrides --long_edge.",
    )
    g_group.add_argument(
        "--long_edge",
        type=int,
        default=3840,
        help="Resize so the long edge becomes this many pixels "
        "(aspect preserving). Ignored when --target_resolution is set.",
    )

    t_group = parser.add_argument_group("tiling (must match training)")
    t_group.add_argument(
        "--tile_t", type=int, default=33, help="Temporal tile size in frames"
    )
    t_group.add_argument("--tile_h", type=int, default=320)
    t_group.add_argument("--tile_w", type=int, default=640)
    t_group.add_argument("--temporal_overlap", type=int, default=5)
    t_group.add_argument("--spatial_overlap", type=int, default=32)

    c_group = parser.add_argument_group("colour")
    c_group.add_argument(
        "--color_ref",
        default="global",
        choices=list(COLOR_REF_MODES),
        help="global: one cheap pre-pass gives a fixed global mean/std; "
        "chunk: match each chunk against its own input; none: no correction",
    )
    c_group.add_argument(
        "--color_ref_samples",
        type=int,
        default=64,
        help="Frames sampled by the global pre-pass (0 = every frame)",
    )
    c_group.add_argument(
        "--no_color_correct", action="store_true", help="Alias for --color_ref none"
    )

    m_group = parser.add_argument_group("compute / output")
    m_group.add_argument("--device", default="cuda")
    m_group.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    m_group.add_argument(
        "--crf", type=int, default=5, help="x264 quality; lower is better"
    )
    m_group.add_argument(
        "--read_queue",
        type=int,
        default=2,
        help="Decoded chunks held in flight. This is the memory knob.",
    )
    m_group.add_argument(
        "--write_queue",
        type=int,
        default=4,
        help="Encoded uint8 batches held in flight",
    )
    m_group.add_argument(
        "--save_tiles_dir",
        default=None,
        help="Debug only: save each restored tile as a separate mp4",
    )

    p_group = parser.add_argument_group("execution path")
    p_group.add_argument(
        "--via-pipeline",
        action="store_true",
        help="Run through the registered SGLang pipeline "
        "(WanVSRPipeline + DiffGenerator) instead of calling the "
        "streaming core directly. Exercises the native integration; "
        "the direct path is the shorter one for debugging.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsr",
        description="VSR streaming restore (SGLang)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    restore = sub.add_parser("restore", help="Restore one video file")
    _add_restore_args(restore)
    return parser


def restore_cmd(args: argparse.Namespace) -> int:
    # Deferred: importing model pulls in diffusers, which is slow and only
    # needed once we actually run.
    from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer

    if args.no_color_correct:
        args.color_ref = "none"

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise SystemExit(f"input video not found: {input_path}")

    fps, total_frames, src_h, src_w = probe_video(input_path)
    if args.target_resolution:
        target_h, target_w = parse_resolution(args.target_resolution)
    else:
        target_h, target_w = resize_to_long_edge(src_h, src_w, args.long_edge)

    print(
        f"[vsr] {input_path}: {total_frames} frames, {src_w}x{src_h}, {fps:.1f} fps "
        f"-> target H={target_h} W={target_w}"
    )

    restorer = VSRRestorer.from_pretrained(
        checkpoint_dir=args.checkpoint_dir,
        wan_root=args.wan_root,
        device=args.device,
        dtype=DTYPES[args.dtype],
        tile_t=args.tile_t,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        t_overlap=args.temporal_overlap,
        s_overlap=args.spatial_overlap,
    )

    written = stream_restore(
        restorer,
        input_path,
        output_path,
        target_h=target_h,
        target_w=target_w,
        fps=fps,
        total_frames=total_frames,
        color_ref=args.color_ref,
        color_samples=args.color_ref_samples,
        crf=args.crf,
        read_queue=args.read_queue,
        write_queue=args.write_queue,
        save_tiles_dir=args.save_tiles_dir,
    )
    print(
        f"[vsr] done. {written} frames restored at {target_h}x{target_w} -> {output_path}"
    )
    return 0


def restore_via_pipeline(args: argparse.Namespace) -> int:
    """Drive the registered ``WanVSRPipeline`` through ``DiffGenerator``.

    This is the native path: registry -> ServerArgs -> scheduler -> stage. The
    direct path exists because for a file-to-file batch job the request plumbing
    adds nothing but ways to fail; this one exists because "natively integrated"
    has to mean runnable, not merely registered.
    """
    from sglang.multimodal_gen import DiffGenerator
    from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
    from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
    from sglang.multimodal_gen.configs.sample.vsr import WanVRSamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs

    server_args = ServerArgs.from_kwargs(
        model_path=args.checkpoint_dir,
        pipeline_class_name="WanVSRPipeline",
        pipeline_config=WanVSRPipelineConfig(precision=args.dtype),
        backend=Backend.SGLANG,
        # `wan_root` is a genuine weight location (the base VAE encoder lives
        # there), so it belongs in component_paths rather than in the sampling
        # params -- unlike VSR's geometry and tiling knobs, which live there
        # precisely so they cannot be mistaken for component paths.
        component_paths={"wan_root": args.wan_root},
        output_path=str(Path(args.output).parent),
        num_gpus=1,
        trust_remote_code=True,
    )

    params = WanVRSamplingParams.from_user_kwargs(
        server_args,
        request_id=generate_request_id(),
        video_input_path=args.input,
        output_path=args.output,
        target_resolution=args.target_resolution,
        long_edge=args.long_edge,
        tile_t=args.tile_t,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        temporal_overlap=args.temporal_overlap,
        spatial_overlap=args.spatial_overlap,
        color_ref="none" if args.no_color_correct else args.color_ref,
        color_ref_samples=args.color_ref_samples,
        dtype=args.dtype,
        crf=args.crf,
        read_queue=args.read_queue,
        write_queue=args.write_queue,
        save_tiles_dir=args.save_tiles_dir,
    )

    with DiffGenerator.from_pretrained(
        model_path=server_args.model_path, server_args=server_args, local_mode=True
    ) as generator:
        req = prepare_request(server_args=generator.server_args, sampling_params=params)
        output_batch = generator._send_to_scheduler_and_wait_for_response([req])
        if output_batch.error:
            raise RuntimeError(str(output_batch.error))

    print(f"[vsr] pipeline mode done -> {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "restore":
        return restore_via_pipeline(args) if args.via_pipeline else restore_cmd(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
