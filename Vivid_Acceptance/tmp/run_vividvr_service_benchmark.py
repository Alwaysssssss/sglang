from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos


DEFAULT_INPUT_VIDEO = Path(
    "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4"
)
DEFAULT_CAPTION_FILE = Path(
    "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt"
)
DEFAULT_REFERENCE_VIDEO = Path(
    "/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4"
)
DEFAULT_RESULT_DIR = Path(
    "/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark"
)
DEFAULT_INDICATOR_DIR = Path("/home/zhiheng/sglang/Vivid_Acceptance/indicator")
DEFAULT_RAW_DIR = Path("/home/zhiheng/sglang/Vivid_Acceptance/indicator/raw")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a VividVR service benchmark through curl with warmup excluded."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="VividVR")
    parser.add_argument("--task-prefix", default="vividvr-service-benchmark-long-130f-20step")
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTION_FILE)
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-temporal-process-frames", type=int, default=121)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--indicator-dir", type=Path, default=DEFAULT_INDICATOR_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args()


def run_curl_json(args: list[str]) -> dict[str, Any]:
    command = ["curl", "--silent", "--show-error", "--fail", "--noproxy", "*", *args]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def run_curl_download(args: list[str], output_path: Path) -> None:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail",
        "--noproxy",
        "*",
        *args,
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ffprobe_profile(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,bit_rate",
        "-show_entries",
        "format=bit_rate,duration,size",
        "-select_streams",
        "v:0",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload.get("streams", [{}])[0]
    fmt = payload.get("format", {})
    return {
        "path": str(path),
        "codec_name": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "level": stream.get("level"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
        "bit_rate": stream.get("bit_rate"),
        "format_bit_rate": fmt.get("bit_rate"),
        "duration": fmt.get("duration"),
        "size": fmt.get("size"),
    }


def make_payload(
    *,
    model: str,
    task_id: str,
    input_video: Path,
    caption_file: Path,
    reference_video: Path,
    num_inference_steps: int,
    seed: int,
    num_temporal_process_frames: int,
    output_path: Path,
    perf_dump_path: Path,
) -> dict[str, Any]:
    return {
        "model": model,
        "task_id": task_id,
        "video_input_path": str(input_video),
        "caption_file_path": str(caption_file),
        "reference_video_path": str(reference_video),
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "num_temporal_process_frames": num_temporal_process_frames,
        "output_path": str(output_path),
        "perf_dump_path": str(perf_dump_path),
    }


def submit_task(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    return run_curl_json(
        [
            "-X",
            "POST",
            f"{base_url}/v1/videos/repairs",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(payload, separators=(",", ":")),
        ]
    )


def poll_progress(
    *,
    base_url: str,
    task_id: str,
    poll_interval_seconds: float,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    status = "created"
    while status not in {"completed", "failed"}:
        time.sleep(poll_interval_seconds)
        progress = run_curl_json(["-X", "GET", f"{base_url}/v1/videos/{task_id}/progress"])
        status = progress["status"]
        entry = {
            "timestamp": utc_now(),
            "status": status,
            "progress": progress.get("progress"),
        }
        history.append(entry)
        print(
            f"[{entry['timestamp']}] task_id={task_id} status={entry['status']} "
            f"progress={entry['progress']}"
        )
    return history


def fetch_detail(base_url: str, task_id: str) -> dict[str, Any]:
    return run_curl_json(["-X", "GET", f"{base_url}/v1/videos/{task_id}"])


def download_content(base_url: str, task_id: str, output_path: Path) -> None:
    run_curl_download(["-X", "GET", f"{base_url}/v1/videos/{task_id}/content"], output_path)


def run_task(
    *,
    base_url: str,
    payload: dict[str, Any],
    poll_interval_seconds: float,
    raw_dir: Path,
    record_runtime: bool,
    download_path: Path | None = None,
) -> dict[str, Any]:
    task_id = payload["task_id"]
    created_path = raw_dir / f"{task_id}_created.json"
    progress_path = raw_dir / f"{task_id}_progress.json"
    detail_path = raw_dir / f"{task_id}_detail.json"

    print(f"[{utc_now()}] submitting task_id={task_id}")
    print(json.dumps(payload, indent=2))
    start_time = time.perf_counter()
    created = submit_task(base_url, payload)
    write_json(created_path, created)

    progress_history = poll_progress(
        base_url=base_url,
        task_id=task_id,
        poll_interval_seconds=poll_interval_seconds,
    )
    write_json(progress_path, progress_history)

    detail = fetch_detail(base_url, task_id)
    write_json(detail_path, detail)

    if detail["status"] != "completed":
        raise RuntimeError(json.dumps(detail, indent=2))

    if download_path is not None:
        download_content(base_url, task_id, download_path)

    result = {
        "created": created,
        "progress_history": progress_history,
        "detail": detail,
    }
    if record_runtime:
        result["total_runtime_seconds"] = time.perf_counter() - start_time
    return result


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.indicator_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    stamp = utc_stamp()
    warmup_task_id = f"{args.task_prefix}-{args.label}-warmup-{stamp}"
    formal_task_id = f"{args.task_prefix}-{args.label}-{stamp}"

    warmup_output_path = args.result_dir / f"{warmup_task_id}.mp4"
    warmup_perf_dump_path = args.indicator_dir / f"{warmup_task_id}_perf.json"
    formal_output_path = args.result_dir / f"{formal_task_id}.mp4"
    downloaded_output_path = args.result_dir / f"downloaded_{formal_task_id}.mp4"
    formal_perf_dump_path = args.indicator_dir / f"{formal_task_id}_perf.json"
    report_path = args.indicator_dir / f"{formal_task_id}.json"
    framewise_path = args.indicator_dir / f"{formal_task_id}_framewise_ssim.json"

    warmup_payload = make_payload(
        model=args.model,
        task_id=warmup_task_id,
        input_video=args.input_video,
        caption_file=args.caption_file,
        reference_video=args.reference_video,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        num_temporal_process_frames=args.num_temporal_process_frames,
        output_path=warmup_output_path,
        perf_dump_path=warmup_perf_dump_path,
    )
    formal_payload = make_payload(
        model=args.model,
        task_id=formal_task_id,
        input_video=args.input_video,
        caption_file=args.caption_file,
        reference_video=args.reference_video,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        num_temporal_process_frames=args.num_temporal_process_frames,
        output_path=formal_output_path,
        perf_dump_path=formal_perf_dump_path,
    )

    warmup_result: dict[str, Any] | None = None
    if args.skip_warmup:
        print(f"[{utc_now()}] skip warmup enabled")
    else:
        warmup_result = run_task(
            base_url=args.base_url,
            payload=warmup_payload,
            poll_interval_seconds=args.poll_interval_seconds,
            raw_dir=args.raw_dir,
            record_runtime=False,
        )
        print(f"[{utc_now()}] warmup completed for task_id={warmup_task_id}")

    formal_result = run_task(
        base_url=args.base_url,
        payload=formal_payload,
        poll_interval_seconds=args.poll_interval_seconds,
        raw_dir=args.raw_dir,
        record_runtime=True,
        download_path=downloaded_output_path,
    )
    print(f"[{utc_now()}] formal request completed for task_id={formal_task_id}")

    detail = formal_result["detail"]
    total_runtime_seconds = formal_result["total_runtime_seconds"]
    local_output_path = Path(detail.get("file_path") or formal_output_path)
    if not local_output_path.exists():
        raise FileNotFoundError(f"Local output missing: {local_output_path}")

    compare_report = compare_videos(str(args.reference_video), str(local_output_path))
    write_json(framewise_path, compare_report)

    perf_dump = None
    if formal_perf_dump_path.exists():
        perf_dump = json.loads(formal_perf_dump_path.read_text(encoding="utf-8"))

    summary = compare_report["summary"]
    metrics = {
        "benchmark_label": args.label,
        "task_id": formal_task_id,
        "warmup_excluded": not args.skip_warmup,
        "warmup_task_id": None if args.skip_warmup else warmup_task_id,
        "status": detail["status"],
        "service_url": args.base_url,
        "input_video": str(args.input_video),
        "caption_file": str(args.caption_file),
        "reference_video": str(args.reference_video),
        "local_output_video": str(local_output_path),
        "downloaded_output_video": str(downloaded_output_path),
        "requested_output_path": str(formal_output_path),
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "num_temporal_process_frames": args.num_temporal_process_frames,
        "total_runtime_seconds": total_runtime_seconds,
        "model_inference_runtime_seconds": detail.get("inference_time_s"),
        "peak_memory_mb": detail.get("peak_memory_mb"),
        "reference_profile": ffprobe_profile(args.reference_video),
        "service_output_profile": ffprobe_profile(local_output_path),
        "download_profile": ffprobe_profile(downloaded_output_path),
        "formal_request_payload": formal_payload,
        "formal_detail_response": detail,
        "formal_progress_history": formal_result["progress_history"],
        "perf_dump_path": str(formal_perf_dump_path) if formal_perf_dump_path.exists() else None,
        "perf_dump": perf_dump,
        **summary,
    }
    if warmup_result is not None:
        metrics["warmup_detail_response"] = warmup_result["detail"]
        metrics["warmup_progress_history"] = warmup_result["progress_history"]
        metrics["warmup_requested_output_path"] = str(warmup_output_path)
        metrics["warmup_perf_dump_path"] = (
            str(warmup_perf_dump_path) if warmup_perf_dump_path.exists() else None
        )

    write_json(report_path, metrics)
    print(f"[{utc_now()}] report -> {report_path}")
    print(json.dumps(metrics, indent=2))
    return 0 if summary["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
