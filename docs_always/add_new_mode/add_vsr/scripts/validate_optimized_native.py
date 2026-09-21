# SPDX-License-Identifier: Apache-2.0
"""Validate optimized scheduler entry and intermediate tensors on a selected GPU."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

root = Path.cwd()
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-dir", type=Path, default=root / "output_results/vsr/optimization_round1"
)
parser.add_argument("--compile-decoder", action="store_true")
parser.add_argument("--compile-encoder", action="store_true")
parser.add_argument("--gpu-postprocess", action="store_true")
parser.add_argument("--decoder-implicit-padding", action="store_true")
parser.add_argument("--cache-dit-condition", action="store_true")
parser.add_argument("--skip-matrix-wait", action="store_true")
parser.add_argument("--gpu", type=int, default=6)
parser.add_argument("--no-cudnn-benchmark", action="store_true")
parser.add_argument("--compare-only", action="store_true")
parser.add_argument("--min-ssim", type=float, default=0.989)
parser.add_argument("--max-mse", type=float, default=36.0)
parser.add_argument("--max-mae", type=float, default=6.0)
options = parser.parse_args()
out = options.output_dir.resolve() / "native"
out.mkdir(parents=True, exist_ok=True)
while (
    not options.skip_matrix_wait
    and not (out.parent / "production/quality_results.json").exists()
):
    time.sleep(30)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(options.gpu), OMP_NUM_THREADS="8")
common = [
    "--input",
    str(root / "output_results/vsr/migration_20260920/input33.mp4"),
    "--output",
    str(out / "direct.mp4"),
    "--checkpoint_dir",
    "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
    "--wan_root",
    "/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
    "--target_resolution",
    "320x640",
    "--cudnn-benchmark",
    "--channels-last-3d",
]
if options.compile_decoder:
    common.append("--compile-decoder")
if options.compile_encoder:
    common.append("--compile-encoder")
if options.cache_dit_condition:
    common.append("--cache-dit-condition")
if options.decoder_implicit_padding:
    common.append("--decoder-implicit-padding")
if options.gpu_postprocess:
    common.append("--gpu-postprocess")
if options.no_cudnn_benchmark:
    common.remove("--cudnn-benchmark")
commands = [
    [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.vsr.verify.dump_candidate",
        "--vsr-repo",
        str(root.parent / "vsr"),
        "--dump-root",
        str(out / "dump"),
        "--dump-frames",
        "--dump-tiles",
        "all",
        "--dump-chunks",
        "--",
        *common,
    ],
    [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.vsr.cli",
        "restore",
        "--via-pipeline",
        *[
            str(out / "native.mp4") if arg == str(out / "direct.mp4") else arg
            for arg in common
        ],
    ],
]
if not options.compare_only:
    (out / "commands.json").write_text(json.dumps(commands, indent=2))
for i, args in enumerate([] if options.compare_only else commands):
    with (out / f"{i}.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
limits = {
    "window": 1e-7,
    "chunk_input": 1e-7,
    "latent": 0.0019,
    "velocity": 0.0067,
    "decoded": 0.048,
    "spatial_fused": 0.048,
}
reports = []
for name, limit in limits.items():
    refs = sorted(
        (
            root / "output_results/vsr/runtime210_validation/stages_reference" / name
        ).glob("*.pt")
    )
    cands = sorted((out / "dump" / name).glob("*.pt"))
    assert refs and len(refs) == len(cands), (name, len(refs), len(cands))
    for ref, cand in zip(refs, cands):
        a, b = (torch.load(p, map_location="cpu").float() for p in (ref, cand))
        assert a.shape == b.shape and b.isfinite().all()
        error = (a - b).abs().mean().item() / max(a.abs().max().item(), 1e-12)
        reports.append(
            {"stage": name, "rel_mean": error, "limit": limit, "pass": error <= limit}
        )
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check

structure = check(str(out / "direct.mp4"), str(out / "native.mp4"))
video = compare_videos(
    str(out / "direct.mp4"),
    str(out / "native.mp4"),
    min_ssim=options.min_ssim,
    max_mse=options.max_mse,
    max_mae=options.max_mae,
    allow_frame_count_delta=0,
    max_failed_frame_ratio=0,
)
original_video = compare_videos(
    str(root / "output_results/vsr/runtime210_validation/native_reference.mp4"),
    str(out / "native.mp4"),
    min_ssim=options.min_ssim,
    max_mse=options.max_mse,
    max_mae=options.max_mae,
    allow_frame_count_delta=0,
    max_failed_frame_ratio=0,
)
result = {
    "stages": reports,
    "native_structure": structure,
    "native_vs_direct": video,
    "native_vs_original": original_video,
    "native_gate_policy": f"User-authorized tolerance: SSIM>={options.min_ssim}, MSE<={options.max_mse}, MAE<={options.max_mae}; zero failed frames",
}
(out / "report.json").write_text(json.dumps(result, indent=2))
assert all(r["pass"] for r in reports), reports
assert structure["pass_structural"] and video["summary"]["pass_compare"]
assert original_video["summary"]["pass_compare"]
print("PASS stages and native", flush=True)
