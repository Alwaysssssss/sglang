# SPDX-License-Identifier: Apache-2.0
"""Run independent quality experiments on the selected GPUs."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "output_results/vsr/optimization_round1"
COMPILE_DECODER = False
COMPILE_ENCODER = False
GPU_POSTPROCESS = False
DECODER_IMPLICIT_PADDING = False
RESUME = False
QUALITY_LIMITS = (0.989, 36.0, 6.0)


def case_config(case):
    media = "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/zy_test"
    configs = {
        "B1": (f"{media}/val_input_480x832.mp4", "480x832", (0.993763, 3.47, 1.65)),
        "B2": (f"{media}/val_input_512x512.mp4", "512x512", (0.991861, 4.18, 1.82)),
        "B3": (f"{media}/val_input_768x1280.mp4", "768x1280", (0.987439, 7.88, 2.50)),
        "C1": (
            str(ROOT / "output_results/vsr/media/loop64.mp4"),
            "1920x1080",
            (0.991078, 4.18, 1.87),
        ),
        "C2": (
            str(ROOT / "output_results/vsr/media/loop200.mp4"),
            "1920x1080",
            (0.990769, 4.42, 1.88),
        ),
        "C3": (
            str(ROOT / "output_results/vsr/media/loop64.mp4"),
            "2160x3840",
            (0.990601, 2.05, 1.32),
        ),
        "A1": (
            str(ROOT.parent / "vsr/input/input.mp4"),
            "3840x2160",
            (0.993550, 1.33, 1.04),
        ),
        "A2": (
            str(ROOT.parent / "vsr/input/input.mp4"),
            "3840x2160",
            (0.993415, 1.32, 1.04),
        ),
        "A3": (
            str(ROOT.parent / "vsr/input/input.mp4"),
            "3840x2160",
            (0.993580, 1.43, 1.07),
        ),
    }
    return configs[case]


def run(job):
    gpu, mode, case = job
    out = OUT / f"{mode}_{case}"
    out.mkdir(parents=True, exist_ok=True)
    inp, shape, limits = case_config(case)
    if QUALITY_LIMITS is not None:
        limits = QUALITY_LIMITS
    args = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.vsr.verify.dump_candidate",
        "--vsr-repo",
        str(ROOT.parent / "vsr"),
        "--dump-root",
        str(out / "dump"),
        "--dump-frames",
        "--",
        "--input",
        inp,
        "--output",
        str(out / "output.mp4"),
        "--checkpoint_dir",
        "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300",
        "--wan_root",
        "/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers",
        "--target_resolution",
        shape,
        "--color_ref",
        {"A2": "chunk", "A3": "none"}.get(case, "global"),
        "--read_queue",
        "2",
    ]
    if mode in {"benchmark", "both"}:
        args.append("--cudnn-benchmark")
    if mode in {"layout", "both"}:
        args.append("--channels-last-3d")
    if COMPILE_DECODER:
        args.append("--compile-decoder")
    if COMPILE_ENCODER:
        args.append("--compile-encoder")
    if DECODER_IMPLICIT_PADDING:
        args.append("--decoder-implicit-padding")
    if GPU_POSTPROCESS:
        args.append("--gpu-postprocess")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="8")
    if RESUME and all(
        (out / name).exists()
        for name in ("frames.json", "mp4.json", "structural.json", "command.json")
    ):
        previous = json.loads((out / "command.json").read_text())
        if previous["argv"] == args:
            frame = json.loads((out / "frames.json").read_text())["summary"]
            video = json.loads((out / "mp4.json").read_text())["summary"]
            structure = json.loads((out / "structural.json").read_text())
            gates = {"min_ssim": limits[0], "max_mse": limits[1], "max_mae": limits[2]}
            if all(
                frame["thresholds"][k] == v and video["thresholds"][k] == v
                for k, v in gates.items()
            ):
                passed = (
                    frame["pass_compare"]
                    and video["pass_compare"]
                    and structure["pass_structural"]
                )
                print(mode, case, "resumed", passed, flush=True)
                return {
                    "mode": mode,
                    "case": case,
                    "gpu": previous["gpu"],
                    "quality_exit": 0 if passed else 1,
                }
    (out / "command.json").write_text(json.dumps({"argv": args, "gpu": gpu}, indent=2))
    with (out / "inference.log").open("w") as log:
        subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    args = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.vsr.verify.compare_frames",
        "--reference-dir",
        str(ROOT / f"output_results/vsr/dumps/EPS_{case}_swiftvr"),
        "--candidate-dir",
        str(out / "dump"),
        "--report-json",
        str(out / "frames.json"),
        "--min-ssim",
        str(limits[0]),
        "--max-mse",
        str(limits[1]),
        "--max-mae",
        str(limits[2]),
    ]
    with (out / "compare.log").open("w") as log:
        result = subprocess.run(
            args, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode == 0:
        from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
        from sglang.multimodal_gen.runtime.vsr.verify.structural_check import check

        reference = str(ROOT / f"output_results/vsr/EPS_{case}_swiftvr.mp4")
        structure = check(reference, str(out / "output.mp4"))
        (out / "structural.json").write_text(json.dumps(structure, indent=2))
        assert structure["pass_structural"], structure
        video = compare_videos(
            reference,
            str(out / "output.mp4"),
            min_ssim=QUALITY_LIMITS[0] if QUALITY_LIMITS else 0.97,
            max_mse=QUALITY_LIMITS[1] if QUALITY_LIMITS else 25,
            max_mae=QUALITY_LIMITS[2] if QUALITY_LIMITS else 2.5,
            allow_frame_count_delta=0,
            max_failed_frame_ratio=0,
        )
        (out / "mp4.json").write_text(json.dumps(video, indent=2))
        if not video["summary"]["pass_compare"]:
            print(mode, case, "mp4 quality failed", video["summary"], flush=True)
            result.returncode = 1
    print(mode, case, "quality exit", result.returncode, flush=True)
    return {"mode": mode, "case": case, "gpu": gpu, "quality_exit": result.returncode}


def run_group(group):
    gpu, cases = group
    return [run((gpu, "both", case)) for case in cases]


def run_pairs(group):
    gpu, pairs = group
    return [run((gpu, mode, case)) for mode, case in pairs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cases", default="A1,A2,A3,B1,B2,B3,C1,C2,C3")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--compile-decoder", action="store_true")
    parser.add_argument("--compile-encoder", action="store_true")
    parser.add_argument("--gpu-postprocess", action="store_true")
    parser.add_argument("--decoder-implicit-padding", action="store_true")
    parser.add_argument(
        "--gpus",
        default="6",
        help="Comma-separated validation GPU IDs; reserve GPU7 for timing",
    )
    parser.add_argument("--min-ssim", type=float, default=0.989)
    parser.add_argument("--max-mse", type=float, default=36.0)
    parser.add_argument("--max-mae", type=float, default=6.0)
    parser.add_argument(
        "--legacy-frame-gates",
        action="store_true",
        help="Reproduce the original per-case frame and mp4 gates",
    )
    options = parser.parse_args()
    QUALITY_LIMITS = (
        None
        if options.legacy_frame_gates
        else (options.min_ssim, options.max_mse, options.max_mae)
    )
    OUT = options.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    COMPILE_DECODER = options.compile_decoder
    COMPILE_ENCODER = options.compile_encoder
    GPU_POSTPROCESS = options.gpu_postprocess
    DECODER_IMPLICIT_PADDING = options.decoder_implicit_padding
    RESUME = options.resume
    gpus = [int(gpu) for gpu in options.gpus.split(",")]
    if options.full:
        OUT = OUT / "production"
        OUT.mkdir(exist_ok=True)
        cases = options.cases.split(",")
        jobs = [(gpu, cases[i :: len(gpus)]) for i, gpu in enumerate(gpus)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            groups = list(pool.map(run_group, jobs))
        results = [result for group in groups for result in group]
    else:
        pairs = [("benchmark", "B1"), ("layout", "B1"), ("both", "B1"), ("both", "C1")]
        jobs = [(gpu, pairs[i :: len(gpus)]) for i, gpu in enumerate(gpus)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            groups = list(pool.map(run_pairs, jobs))
        results = [result for group in groups for result in group]
    (OUT / "quality_results.json").write_text(json.dumps(results, indent=2))
    raise SystemExit(int(any(r["quality_exit"] for r in results)))
