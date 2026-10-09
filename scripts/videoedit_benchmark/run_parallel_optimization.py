#!/usr/bin/env python3
"""Run isolated VideoEdit conditioning/VAE parallelism candidates.

The runner uses only explicitly selected idle GPUs, executes configurations
serially, and destroys each container before starting the next one.  It never
restarts or reconfigures the online VideoEdit service.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from run import ROOT, gpu_state, write

VARIANTS = {
    "owner1": {"vae_sp": True, "mode": "both", "encode_degree": 1, "decode_degree": 1},
    "encode2": {"vae_sp": True, "mode": "both", "encode_degree": 2, "decode_degree": 1},
    "both2": {"vae_sp": True, "mode": "both", "encode_degree": 2, "decode_degree": 2},
}


def run(args):
    out = Path(args.out).resolve()
    out.relative_to(ROOT)
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2 or 7 in gpus:
        raise ValueError(
            "Select two distinct benchmark GPUs and do not use online GPU 7"
        )
    labels = args.variants.split(",")
    unknown = set(labels) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    if args.requests < 1:
        raise ValueError("--requests must be positive")

    out.mkdir(parents=True, exist_ok=False)
    with (ROOT / ".local/teacache-sp.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
            text=True,
        ).strip()
        sources = [
            ROOT / "python/sglang/multimodal_gen/configs/sample/videoedit_wan.py",
            ROOT / "python/sglang/multimodal_gen/runtime/managers/gpu_worker.py",
            ROOT
            / "python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py",
            ROOT
            / "python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/videoedit_wan.py",
            ROOT / "python/sglang/multimodal_gen/runtime/videoedit/distributed.py",
            ROOT / "python/sglang/multimodal_gen/runtime/videoedit/vae_spatial.py",
            ROOT
            / "python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py",
            Path(__file__).with_name("worker.py").resolve(),
            Path(__file__).resolve(),
        ]
        write(
            out / "manifest.json",
            {
                "image": args.image,
                "image_id": image_id,
                "gpus": gpus,
                "geometry": args.geometry,
                "started_at": time.time(),
                "source_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sources
                },
            },
        )

        for label in labels:
            target = out / label
            target.mkdir()
            for directory in ("home", "tmp", "cache"):
                (target / directory).mkdir()
            write(target / "gpu-before.json", gpu_state(gpus))
            candidate = VARIANTS[label]
            environment = {
                "SGLANG_VIDEOEDIT_PARALLEL_CONDITIONING": "1",
                "SGLANG_VIDEOEDIT_TEXT_OWNER": "0",
                "SGLANG_VIDEOEDIT_CLIP_OWNER": "1",
                "SGLANG_VIDEOEDIT_VAE_SP_MODE": candidate["mode"],
                "SGLANG_VIDEOEDIT_VAE_ENCODE_DEGREE": str(candidate["encode_degree"]),
                "SGLANG_VIDEOEDIT_VAE_DECODE_DEGREE": str(candidate["decode_degree"]),
                "SGLANG_VIDEOEDIT_RESIDENT_LAYERS": "40",
                "SGLANG_VIDEOEDIT_RELEASE_AFTER_DENOISE": "1",
                "SGLANG_VIDEOEDIT_ENCODER_CPU_MASTER": "1",
                "SGLANG_VIDEOEDIT_OVERLAP_PREFETCH": "0",
                "SGLANG_VIDEOEDIT_FUSED_QKV": "0",
                "SGLANG_VIDEOEDIT_USP_CHUNKS": "1",
                "SGLANG_VIDEOEDIT_ASYNC_STAGE_LOAD": "0",
                "SGLANG_VIDEOEDIT_PREFETCH_IMAGE_ENCODER": "0",
                "SGLANG_VIDEOEDIT_OFFLOAD_PINNED_MB": "0",
                "SGLANG_VIDEOEDIT_AUTO_RESIDENCY": "1",
                "SGLANG_VIDEOEDIT_ACTIVATION_RESERVE_MB": "4096",
                "SGLANG_VIDEOEDIT_FULL_FRAME_RESERVE_MB": "16384",
                "SGLANG_VIDEOEDIT_RESIDENCY_MARGIN_MB": "2048",
                "SGLANG_VIDEOEDIT_WEIGHT_BUDGET_MB": "0",
                "SGLANG_VIDEOEDIT_SHARED_CPU_DIR": "",
            }
            variant = {
                "id": label,
                "host_gpus": gpus,
                "num_gpus": 2,
                "sp_degree": 2,
                "ulysses_degree": 2,
                "ring_degree": 1,
                "dit_layerwise_offload": True,
                "dit_offload_prefetch_size": 4,
                "stage_resident": False,
                "vae_sp": candidate["vae_sp"],
                "environment": environment,
                "requests": [
                    {
                        "phase": "warmup" if index == 0 else f"timed{index}",
                        "geometry": args.geometry,
                    }
                    for index in range(args.requests)
                ],
            }
            write(target / "variant.json", variant)
            environment.update(
                {
                    "HOME": str(target / "home"),
                    "TMPDIR": str(target / "tmp"),
                    "XDG_CACHE_HOME": str(target / "cache"),
                    "HF_HOME": str(target / "cache/hf"),
                    "TRITON_CACHE_DIR": str(target / "cache/triton"),
                    "FLASHINFER_WORKSPACE_BASE": str(target / "cache/flashinfer"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1",
                    "HF_HUB_OFFLINE": "1",
                    "VIDEOEDIT_BENCH_UID": str(os.getuid()),
                    "VIDEOEDIT_BENCH_GID": str(os.getgid()),
                }
            )
            name = "ve-parallel-" + uuid.uuid4().hex[:12]
            command = [
                "docker",
                "run",
                "--user",
                "root",
                "--name",
                name,
                "--network",
                "none",
                "--shm-size",
                "16g",
                "--gpus",
                '"device=' + ",".join(map(str, gpus)) + '"',
                "-v",
                f"{ROOT}:{ROOT}",
                "-v",
                f"{args.model}:{args.model}:ro",
                "-w",
                str(ROOT),
            ]
            for key, value in environment.items():
                command += ["-e", f"{key}={value}"]
            command += [
                image_id,
                "python3",
                str(Path(__file__).with_name("worker.py").resolve()),
                "--out",
                str(target),
                "--model",
                args.model,
                "--geometry",
                args.geometry,
                "--request-timeout",
                str(args.request_timeout),
            ]
            write(target / "docker-command.json", command)
            print(f"START {label}: {name}", flush=True)
            with (target / "runner.log").open("w") as log:
                try:
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.container_timeout,
                    )
                    write(target / "exit.json", {"returncode": process.returncode})
                finally:
                    inspection = subprocess.run(
                        ["docker", "inspect", name], capture_output=True, text=True
                    )
                    (target / "container.json").write_text(inspection.stdout)
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
            write(target / "gpu-after.json", gpu_state(gpus))
            if process.returncode:
                raise RuntimeError(f"{label} failed; stopping without retry")
            print(f"END {label}", flush=True)
        write(
            out / "completion.json", {"status": "completed", "finished_at": time.time()}
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--variants", default="owner1,encode2,both2")
    parser.add_argument("--geometry", choices=("crop", "full"), default="crop")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--container-timeout", type=int, default=10000)
    parser.add_argument("--image", default="sglang-videoedit-dev-v2:l40s")
    parser.add_argument(
        "--model",
        default="/home/zhouhao6/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model",
    )
    run(parser.parse_args())
