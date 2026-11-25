#!/usr/bin/env python3
"""One configuration, one isolated container, four complete local requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ID = "4d187b56-62b7-4121-8d7b-7528150e35d9"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def historical_request(out):
    history = ROOT / ".local/videoedit-l40s.0123/dual"
    with sqlite3.connect(f"file:{history}/queue.sqlite3?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT request_json FROM tasks WHERE task_id=?", (HISTORICAL_ID,)).fetchone()
    request = json.loads(row[0])
    for key in list(request):
        if any(word in key for word in ("url", "minio", "bucket", "object_key")):
            request.pop(key)
    hashes = {}
    for kind, ext, field in [("video", "mov", "video_input_path"),
                             ("mask", "json", "mask_input_path"),
                             ("reference", "png", "reference_image_path")]:
        path = history / "inputs/normal" / f"{HISTORICAL_ID}_{kind}.{ext}"
        request[field] = str(path)
        hashes[kind] = {"path": str(path), "sha256": file_hash(path)}
    request.update(model="videoedit-normal", output_storage="local", enable_teacache=True,
                   teacache_thresh=0.3, num_inference_steps=40, seed=42, guidance_scale=5.0)
    write_json(out / "inputs.json", hashes)
    write_json(out / "request-template.json", request)
    return request


def command_for(variant, out, model):
    command = ["python3", str(ROOT / "scripts/videoedit_dual_service/disable_thp_exec.py"),
               "sglang", "serve", "--model-type", "diffusion", "--backend", "sglang",
               "--model-path", model, "--transformer-path", model + "/transformer",
               "--host", "127.0.0.1", "--port", "30000", "--master-port", "31105",
               "--scheduler-port", "31755", "--nccl-port", "31955", "--strict-ports", "true",
               "--scheduler-response-timeout", "-1", "--warmup", "false",
               "--attention-backend", "fa", "--pin-cpu-memory", "true",
               "--text-encoder-cpu-offload", "true", "--image-encoder-cpu-offload", "true",
               "--vae-cpu-offload", "true", "--output-path", str(out / "outputs"),
               "--input-save-path", str(out / "inputs")]
    values = {key: variant[key] for key in ("num_gpus", "sp_degree", "ulysses_degree", "ring_degree", "dit_layerwise_offload")}
    values.update(dit_cpu_offload=bool(variant.get("stage_resident")),
                  dit_offload_prefetch_size=variant.get("dit_offload_prefetch_size", 1),
                  vae_sp=bool(variant.get("vae_sp", False)))
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value).lower()])
    return command


def video_metadata(path):
    result = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                      "-count_frames", "-show_entries", "stream=width,height,nb_read_frames",
                                      "-of", "json", str(path)], text=True, timeout=120)
    return json.loads(result)["streams"][0]


def decoded_validation(path, out, phase):
    stream = video_metadata(path)
    frames = out / f"{phase}.framemd5"
    with frames.open("wb") as handle:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                        "-f", "framemd5", "-"], stdout=handle, check=True, timeout=180)
    # Compare decoded pixels, without FFmpeg version/timebase header differences.
    hashes = [line.split(",")[-1].strip() for line in frames.read_text().splitlines()
              if line and not line.startswith("#")]
    return dict(stream=stream, frame_hashes=hashes,
                decoded_sha256=hashlib.sha256("\n".join(hashes).encode()).hexdigest())


def run(args):
    out = Path(args.out).resolve()
    out.relative_to(ROOT)
    variant = json.loads((out / "variant.json").read_text())
    for sub in ("tmp", "inputs", "outputs", "cache", "perf_logs"):
        (out / sub).mkdir(exist_ok=True)
    request = historical_request(out)
    source = video_metadata(request["video_input_path"])
    write_json(out / "source-video.json", source)
    expected_frames = int(source["nb_read_frames"])
    if request.get("num_frames", -1) and request["num_frames"] > 0:
        expected_frames = min(expected_frames, request["num_frames"])
    env = dict(os.environ, TMPDIR=str(out / "tmp"),
               VIDEOEDIT_SP_BENCH_DIR=str(out), VIDEOEDIT_SP_BENCH_GEOMETRY=args.geometry,
               VIDEOEDIT_SP_BENCH_RESIDENT="1" if variant.get("stage_resident") else "0",
               PYTHONPATH=str(ROOT / "scripts/videoedit_benchmark/hooks") + ":" + str(ROOT / "python"),
               SGLANG_PERF_LOG_DIR=str(out / "perf_logs"),
               SGLANG_USE_RUNAI_MODEL_STREAMER="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               CUDA_VISIBLE_DEVICES=",".join(map(str, range(variant["num_gpus"]))),
               HF_HUB_OFFLINE="1", SGLANG_CACHE_DIT_ENABLED="0")
    command = command_for(variant, out, args.model)
    write_json(out / "command.json", command)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, payload=None):
        req = urllib.request.Request("http://127.0.0.1:30000" + path,
                                     data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=30) as response:
            return json.load(response)

    started = time.monotonic()
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"backend exited {process.returncode}; see server.log")
                try:
                    if api("/health").get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError("model loading")
                time.sleep(2)
            write_json(out / "load.json", {"seconds": time.monotonic() - started})
            sequence = variant.get("requests") or [
                {"phase": "warmup" if index == 0 else f"timed{index}", "geometry": args.geometry}
                for index in range(4)
            ]
            template = dict(request)
            for item in sequence:
                phase = item["phase"]
                request = dict(template)
                geometry = item["geometry"]
                if variant.get("requests") and geometry == "full":
                    request.update(chunk_bbox_mode="global", bbox_expand_scale=100.0)
                elif variant.get("requests") and geometry != "crop":
                    raise ValueError(f"Invalid mixed-request geometry: {geometry}")
                task = variant["id"] + "-" + phase
                output = out / "outputs" / f"{task}.mov"
                request.update(task_id=task, timeout=args.request_timeout, output_path=str(output),
                               perf_dump_path=str(out / f"{phase}-perf.json"))
                begin = time.monotonic()
                admission = api("/v1/videos/repairs", request)
                if admission.get("code") != 0:
                    raise RuntimeError(f"admission rejected: {admission}")
                print(json.dumps(dict(phase=phase, state="submitted")), flush=True)
                with (out / f"{phase}-samples.jsonl").open("w") as samples:
                    while True:
                        state = api("/v1/videos/" + task)
                        now = time.monotonic()
                        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu,power.draw",
                                                       "--format=csv,noheader,nounits"], text=True, timeout=10)
                        sample = {"elapsed_s": now - begin, "gpus": raw.strip().splitlines()}
                        for key, path in [("memory_bytes", "/sys/fs/cgroup/memory.current"),
                                          ("cpu_stat", "/sys/fs/cgroup/cpu.stat")]:
                            sample[key] = Path(path).read_text().strip()
                        samples.write(json.dumps(sample) + "\n")
                        samples.flush()
                        if state.get("status") in ("completed", "failed", "cancelled"):
                            break
                        if now - begin > args.request_timeout:
                            raise TimeoutError(f"request {task}")
                        if process.poll() is not None:
                            raise RuntimeError("backend exited during request")
                        time.sleep(1)
                record = dict(phase=phase, geometry=geometry, status=state.get("status"), response=state,
                              client_seconds=now - begin)
                write_json(out / f"{phase}-result.json", record)
                if state.get("status") != "completed":
                    raise RuntimeError(f"{phase}: {state.get('error')}")
                record["validation"] = decoded_validation(output, out, phase)
                write_json(out / f"{phase}-result.json", record)
                stream = record["validation"]["stream"]
                if (int(stream["nb_read_frames"]) != expected_frames or
                        (stream["width"], stream["height"]) != (source["width"], source["height"])):
                    raise RuntimeError(f"Invalid output geometry/frame count: {stream}, source={source}")
                if variant.get("requests"):
                    expected_shape = (1920, 1088, 49) if geometry == "full" else (544, 480, 49)
                    rank_latents = []
                    for rank in range(variant["num_gpus"]):
                        events = [json.loads(line) for line in (out / f"rank{rank}.jsonl").read_text().splitlines()]
                        rows = [e for e in events if e.get("output") == str(output)
                                and e.get("stage") == "VideoEditDenoisingStage"]
                        if len(rows) != 1 or rows[0]["status"] != "ok":
                            raise RuntimeError("Expected one successful denoising window per rank")
                        row = rows[0]
                        if (row["model_width"], row["model_height"], row["model_frames"]) != expected_shape:
                            raise RuntimeError(f"Unexpected actual model geometry: {row}")
                        if row.get("gpu_layers_after_stage") != [0]:
                            raise RuntimeError("Weight cache survived the stage boundary")
                        if variant.get("expect_adaptive"):
                            plan = row.get("residency_plan")
                            if not plan:
                                raise RuntimeError("Adaptive residency plan was not recorded")
                            residents = plan["resident_layers"]
                            if geometry == "crop" and residents != [40]:
                                raise RuntimeError(f"Small input did not recover full residency: {residents}")
                            if geometry == "full" and not all(0 <= n < 40 for n in residents):
                                raise RuntimeError(f"Full frame did not reduce residency: {residents}")
                        rank_latents.extend(e["sha256"] for e in events if e.get("output") == str(output) and e["kind"] == "latent")
                    if len(set(rank_latents)) != 1:
                        raise RuntimeError("Latent hashes differ between ranks")
                print(json.dumps(dict(phase=phase, geometry=geometry, state="completed", seconds=now-begin)), flush=True)
            write_json(out / "completion.json", {"status": "completed"})
        finally:
            # The host additionally destroys this dedicated container, including
            # detached worker descendants, before inspecting GPU release.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--geometry", choices=("full", "crop"), default="full")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=3600)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        write_json(Path(args.out) / "error.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        # The image needs root to read historical inputs, but artifacts belong
        # to the host user, including endpoint-created private output files.
        if "VIDEOEDIT_BENCH_UID" in os.environ:
            uid = int(os.environ["VIDEOEDIT_BENCH_UID"])
            gid = int(os.environ["VIDEOEDIT_BENCH_GID"])
            output_dir = Path(args.out).resolve()
            output_dir.relative_to(ROOT)
            for path in [*output_dir.rglob("*"), output_dir]:
                os.chown(path, uid, gid, follow_symlinks=False)
