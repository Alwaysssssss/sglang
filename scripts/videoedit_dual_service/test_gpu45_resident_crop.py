#!/usr/bin/env python3
"""Replay the cached small-crop case on 5403, serially; stop on any failure.

No cloud credentials, uploads, or business callbacks. One warmup, three timed
requests. Read per-rank synchronized DiT timings from the opt-in service probe.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / ".local/videoedit-l40s.45-u2-5403"
BASE = "http://127.0.0.1:5403"
SOURCE = "4d187b56-62b7-4121-8d7b-7528150e35d9"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=6100) as response:
        return json.load(response)


def payload(task):
    inputs = SERVICE / "dual/inputs/normal"
    paths = {"video_input_path": inputs / f"{SOURCE}_video.mov",
             "mask_input_path": inputs / f"{SOURCE}_mask.json",
             "reference_image_path": inputs / f"{SOURCE}_reference.png"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(
        task_id=task, timeout=6000, model="videoedit-normal",
        prompt="替换LOGO 视频画面自然流畅，植入后的元素与原视频运动、光影及镜头移动完全同步一致，时序稳定无闪烁。",
        negative_prompt=None, **{key: str(value) for key, value in paths.items()},
        callback_url=None, output_storage="local", num_frames=-1,
        ref_frame_idx=0, bridge_overlap=5, infer_len=49, overlap=5,
        num_inference_steps=40, guidance_scale=5.0, seed=42, dtype="bf16",
        dynamic_cfg=True, dynamic_cfg_max_step=15, dynamic_cfg_min=1.0,
        bbox_padding=0, bbox_expand_scale=0.3, dilate_px=8, mask_scale=1.0,
        feather_px=8, adain_boundary_dilate=0, crop_edge_feather=0,
        chunk_bbox_mode="tight", stabilize_mask_union=False,
        stabilize_mask_shape=False, stabilize_smooth_window=5,
        preserve_audio=True, enable_paste_back=True, save_crop_only=False,
        use_clip=True, clip_preprocess="diffuser", decode_mode="stream",
        enable_teacache=True, teacache_thresh=0.3, teacache_start_skipping=5,
        teacache_end_skipping=1.0, enable_frame_interpolation=False,
        enable_upscaling=False, output_quality="default")


def timings(task, metrics_dir="stage-resident-metrics"):
    records = []
    for rank in (0, 1):
        path = SERVICE / f"dual/{metrics_dir}/rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        matches = [row for row in rows if task in str(row.get("output", ""))
                   and row.get("stage") == "VideoEditDenoisingStage"]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one denoise window for rank {rank}, got {len(matches)}")
        row = matches[0]
        if row["status"] != "ok" or row.get("denoise_seconds") is None:
            raise RuntimeError(f"Denoising failed on rank {rank}: {row['status']}")
        if (row["model_width"], row["model_height"], row["model_frames"]) != (544, 480, 49):
            raise RuntimeError("Actual model input differs from 544x480x49")
        records.append(row)
    if records[0]["decisions"] != records[1]["decisions"]:
        raise RuntimeError("TeaCache decisions differ across ranks")
    return {"ranks": records,
            "dit_seconds": max(row["denoise_seconds"] for row in records),
            "stage_with_transfers_seconds": max(row["seconds"] for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefetch", type=int, choices=(1, 2, 4, 8),
                        help="Measure layerwise offload instead of stage residency")
    parser.add_argument("--overlap", action="store_true", help="Require the opt-in asynchronous prefetch path")
    parser.add_argument("--fused-qkv", action="store_true", help="Label the separately verified fused QKV experiment")
    parser.add_argument("--resident-layers", type=int, default=0)
    parser.add_argument("--release-after-denoise", action="store_true")
    cli = parser.parse_args()
    mode = f"prefetch{cli.prefetch}" if cli.prefetch else "resident"
    if cli.overlap:
        if cli.prefetch is None:
            parser.error("--overlap requires --prefetch")
        mode = "overlap-" + mode
    if cli.fused_qkv:
        mode = "fused-" + mode
    if cli.resident_layers:
        mode = f"resident{cli.resident_layers}-" + mode
    if cli.release_after_denoise:
        mode = "stage-cache-" + mode
    metrics_dir = f"prefetch{cli.prefetch}-metrics" if cli.prefetch else "stage-resident-metrics"
    run = SERVICE / f"{mode}-crop-tests" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        lines = (SERVICE / "dual/logs/normal.log").read_text().splitlines()
        startup = [line.split("server_args: ", 1)[1] for line in lines
                   if "server_args: " in line]
        if not startup:
            raise RuntimeError("No backend startup configuration found")
        args, _ = json.JSONDecoder().raw_decode(startup[-1])
        if cli.overlap and not any("overlap_prefetch=True" in line for line in lines):
            raise RuntimeError("Asynchronous overlap implementation is not enabled in the running service")
        if cli.resident_layers and not any(f"resident_layers={cli.resident_layers}" in line for line in lines):
            raise RuntimeError("Requested resident-layer cache is not enabled in the running service")
        expected = {"dit_layerwise_offload": cli.prefetch is not None,
                    "dit_cpu_offload": cli.prefetch is None,
                    "ulysses_degree": 2, "num_gpus": 2, "sp_degree": 2, "ring_degree": 1}
        if cli.prefetch:
            expected["dit_offload_prefetch_size"] = cli.prefetch
        if any(args.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"5403 has not started with the expected configuration: {expected}")
        health = request("GET", "/health")
        (run / "health.json").write_text(json.dumps(health, indent=2))
        if not health.get("backends", {}).get("normal"):
            raise RuntimeError("Normal backend is not healthy")
        if any(health.get("queue", {}).get(key, 0) for key in ("queued", "running", "dispatching", "cancelling")):
            raise RuntimeError("Service is busy; refusing to add benchmark traffic")
        for index in range(4):
            task = mode + "-crop-" + uuid.uuid4().hex
            item = {"task_id": task, "warmup": index == 0}
            results.append(item)
            print(f"Submitting {'warmup' if index == 0 else f'timed {index}'}: {task}", flush=True)
            state = request("POST", "/v1/videos/repairs", payload(task))
            deadline = time.monotonic() + 6100
            while state.get("status") not in {"completed", "failed", "cancelled"}:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Request still pending: {task}; no retry submitted")
                time.sleep(30)
                state = request("GET", f"/v1/videos/{task}")
            item["response"] = state
            if state["status"] != "completed":
                raise RuntimeError(f"Request failed; stopping without retry: {state.get('error')}")
            item.update(timings(task, metrics_dir))
            if cli.fused_qkv and not all(row.get("fused_qkv") for row in item["ranks"]):
                raise RuntimeError("Fused QKV was not enabled in both backend ranks")
            if cli.resident_layers and not all(row.get("resident_layers") == cli.resident_layers for row in item["ranks"]):
                raise RuntimeError("Resident layer count differs from the requested experiment")
            if cli.release_after_denoise and not all(
                    row.get("release_after_denoise") and row.get("gpu_layers_after_stage") == [0]
                    for row in item["ranks"]):
                raise RuntimeError("GPU weight cache was not released at the stage boundary")
            (run / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"DiT: {item['dit_seconds']:.4f}s", flush=True)
        median = statistics.median(row["dit_seconds"] for row in results[1:])
        summary = {"median_dit_seconds": median,
                   "median_full_dit_seconds": statistics.median(row["stage_with_transfers_seconds"] for row in results[1:]),
                   "mode": mode,
                   "median_inference_seconds": statistics.median(row["response"]["inference_time_s"] for row in results[1:]),
                   "median_execution_seconds": statistics.median(row["response"]["completed_at"] - row["response"]["started_at"] for row in results[1:]),
                   "historical_single_stage_seconds": 44.6995,
                   "historical_dual_layerwise_stage_seconds": 56.7991,
                   "note": "median_full_dit_seconds includes stage-boundary transfers and release. Historical logs are not identically instrumented. Legacy resident denoise_seconds excludes boundary transfers; stage-scoped layer caching includes loading and release in denoise_seconds."}
        (run / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (run / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        (run / "error.json").write_text(json.dumps({"error": str(exc), "no_retry": True}, indent=2))
        raise


if __name__ == "__main__":
    main()
