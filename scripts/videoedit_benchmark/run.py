#!/usr/bin/env python3
"""Host-side serial matrix runner; never starts or changes a business service."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
MATRIX = Path(__file__).with_name("teacache_sp_matrix.json")


def ordered_variants(matrix):
    variants = matrix["variants"]
    for v in variants:
        gpus = v["host_gpus"]
        if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= {0, 1, 2, 3}:
            raise ValueError(f"Unsafe GPU selection: {gpus}")
        if not (len(gpus) == v["num_gpus"] == v["sp_degree"] == v["ulysses_degree"] * v["ring_degree"]):
            raise ValueError(f"Invalid SP configuration: {v['id']}")
    if len({v["id"] for v in variants}) != len(variants):
        raise ValueError("Duplicate variant IDs")
    def phase(v):
        if v.get("stage_resident"):
            return 3
        if v["num_gpus"] == 1:
            return 0
        if v["dit_offload_prefetch_size"] == 1:
            return 1
        return 2
    return sorted(variants, key=phase)


def gpu_state(gpus):
    raw = subprocess.check_output(["nvidia-smi", "--id=" + ",".join(map(str, gpus)),
                                   "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    rows = [line.split(", ") for line in raw.strip().splitlines()]
    if len(rows) != len(gpus):
        raise RuntimeError("Incomplete GPU query")
    if any(int(row[2]) > 64 for row in rows):
        raise RuntimeError(f"Test GPUs are not idle: {rows}")
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    selected_uuids = {row[1] for row in rows}
    if any(line.split(",")[0].strip() in selected_uuids for line in processes.splitlines()):
        raise RuntimeError("A compute process is already using a selected test GPU")
    return rows


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(args):
    matrix = json.loads(MATRIX.read_text())
    variants = ordered_variants(matrix)
    if args.variants:
        requested = args.variants.split(",")
        by_id = {v["id"]: v for v in variants}
        variants = [by_id[name] for name in requested]
    out = Path(args.out).resolve()
    out.relative_to(ROOT)
    selection = out / "selection.json"
    if selection.exists():
        keep = json.loads(selection.read_text())["keep"]
        if args.variants and any(v["id"] not in keep for v in variants):
            raise ValueError("Requested variants include cases removed from this run's selection")
        if not args.variants:
            by_id = {v["id"]: v for v in variants}
            variants = [by_id[name] for name in keep]
    if args.dry_run:
        print(json.dumps(variants, indent=2))
        return
    out.mkdir(parents=True, exist_ok=args.resume)
    # One lock across all output directories prevents competing benchmark jobs.
    with (ROOT / ".local/teacache-sp.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session_tag = str(int(time.time()))
        with (out / f"preflight-{session_tag}.log").open("w") as log:
            subprocess.run(["bash", str(MATRIX.with_name("preflight.sh"))], stdout=log, stderr=subprocess.STDOUT, check=True)
        image = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True).strip()
        if args.resume:
            previous = json.loads((out / "manifest.json").read_text())
            if previous["image"] != image or previous["geometry"] != args.geometry:
                raise ValueError("Resume requires the original image and geometry")
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            if previous["git_head"] != head:
                raise ValueError("Worktree HEAD changed since the original run")
            pending = []
            for variant in variants:
                target = out / variant["id"]
                if not target.exists():
                    pending.append(variant)
                    continue
                if not (target / "exit.json").exists() or not (target / "gpu-after.json").exists():
                    raise ValueError(f"Unfinished configuration needs inspection: {target}")
                if json.loads((target / "variant.json").read_text()) != variant:
                    raise ValueError(f"Configuration changed: {variant['id']}")
            variants = pending
        manifest_name = f"manifest-resume-{session_tag}.json" if args.resume else "manifest.json"
        write(out / manifest_name, dict(image=image, matrix=matrix, geometry=args.geometry,
              git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              meminfo=Path("/proc/meminfo").read_text(), loadavg=Path("/proc/loadavg").read_text(),
              thp=Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text(),
              cpu_count=os.cpu_count(), started_at=time.time(),
              harness_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in MATRIX.parent.rglob("*.py")}))
        # Preserve source snapshot metadata, including untracked test harness files.
        subprocess.run(["git", "diff", "--binary"], cwd=ROOT, stdout=(out / f"worktree-{session_tag}.diff").open("w"), check=True)
        for variant in variants:
            state = gpu_state(variant["host_gpus"])
            target = out / variant["id"]
            target.mkdir()
            write(target / "variant.json", variant)
            write(target / "gpu-before.json", state)
            name = "ve-teacache-bench-" + uuid.uuid4().hex[:12]
            cache = target / "cache"
            cache.mkdir()
            for sub in ("tmp", "home"):
                (target / sub).mkdir()
            command = ["docker", "run", "--user", "root", "--name", name, "--label", "videoedit.teacache-benchmark=true",
                       "--network", "none", "--shm-size", "16g",
                       "--gpus", '"device=' + ",".join(map(str, variant["host_gpus"])) + '"',
                       "-v", f"{ROOT}:{ROOT}", "-v", f"{args.model}:{args.model}:ro", "-w", str(ROOT)]
            for key, value in dict(HOME=str(target / "home"), TMPDIR=str(target / "tmp"),
                                   XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "hf"),
                                   FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
                                   TRITON_CACHE_DIR=str(cache / "triton"),
                                   VIDEOEDIT_BENCH_UID=str(os.getuid()),
                                   VIDEOEDIT_BENCH_GID=str(os.getgid()),
                                   PYTHONDONTWRITEBYTECODE="1").items():
                command.extend(["-e", f"{key}={value}"])
            command += [image, "python3", str(Path(__file__).with_name("worker.py")),
                        "--out", str(target), "--model", args.model, "--geometry", args.geometry,
                        "--request-timeout", str(args.request_timeout)]
            write(target / "docker-command.json", command)
            print(json.dumps(dict(variant=variant["id"], container=name, state="starting")), flush=True)
            with (target / "runner.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=1900 + 4 * (args.request_timeout + 240))
                    write(target / "exit.json", {"returncode": code})
                except subprocess.TimeoutExpired:
                    write(target / "exit.json", {"error": "configuration timeout"})
                except BaseException as exc:
                    write(target / "exit.json", {"error": type(exc).__name__})
                    raise
                finally:
                    inspect = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
                    (target / "container.json").write_text(inspect.stdout)
                    subprocess.run(["docker", "rm", "-f", name], check=True, stdout=log, stderr=log)
                    process.wait(timeout=30)
            # CUDA contexts may take a few seconds to disappear after container exit.
            for attempt in range(30):
                try:
                    after = gpu_state(variant["host_gpus"])
                    break
                except RuntimeError:
                    if attempt == 29:
                        raise
                    time.sleep(2)
            write(target / "gpu-after.json", after)
            subprocess.run(["python3", str(MATRIX.with_name("report.py")), str(out)], check=True)
            print(json.dumps(dict(variant=variant["id"], state="finished")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--image", default="sglang-videoedit-dev-v2:l40s")
    parser.add_argument("--model", default="/home/zhouhao6/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model")
    parser.add_argument("--variants", help="Comma-separated IDs, in execution order; default is the full matrix")
    parser.add_argument("--geometry", choices=("full", "crop"), default="full")
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip terminal configurations with verified GPU cleanup")
    run(parser.parse_args())
