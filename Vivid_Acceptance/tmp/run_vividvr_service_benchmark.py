from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
from sglang.multimodal_gen.tools.run_flowcut_vividvr_service_acceptance import (
    _FlowCutCallbackRecorder,
    _LocalFlowCutCallbackServer,
    _validate_final_callback_payload,
    submit_flowcut_task_with_retry,
)


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
        description="Run a VividVR FlowCut service benchmark with warmup excluded."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--task-prefix", default="vividvr-service-benchmark-long-130f-20step")
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTION_FILE)
    parser.add_argument("--reference-video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-temporal-process-frames", type=int, default=121)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--callback-host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=0)
    parser.add_argument("--final-callback-timeout-s", type=float, default=60.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--indicator-dir", type=Path, default=DEFAULT_INDICATOR_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ffprobe_profile(path: Path) -> dict[str, Any]:
    import subprocess

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
    callback_url: str,
    task_id: str,
    input_video: Path,
    caption_file: Path,
    num_inference_steps: int,
    seed: int,
    num_temporal_process_frames: int,
    output_path: Path,
    perf_dump_path: Path,
) -> dict[str, Any]:
    return {
        "taskId": task_id,
        "timeout": -1,
        "callbackUrl": callback_url,
        "video_input_path": str(input_video),
        "caption_file_path": str(caption_file),
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "num_temporal_process_frames": num_temporal_process_frames,
        "output_path": str(output_path),
        "perf_dump_path": str(perf_dump_path),
    }


def poll_progress(
    *,
    client: httpx.Client,
    base_url: str,
    task_id: str,
    poll_interval_seconds: float,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    status = "created"
    while status not in {"completed", "failed"}:
        time.sleep(poll_interval_seconds)
        response = client.get(f"{base_url}/v1/videos/{task_id}/progress", timeout=60.0)
        response.raise_for_status()
        progress = response.json()
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


def fetch_detail(client: httpx.Client, base_url: str, task_id: str) -> dict[str, Any]:
    response = client.get(f"{base_url}/v1/videos/{task_id}", timeout=60.0)
    response.raise_for_status()
    return response.json()


def download_content(
    client: httpx.Client, base_url: str, task_id: str, output_path: Path
) -> None:
    response = client.get(f"{base_url}/v1/videos/{task_id}/content", timeout=600.0)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _resolve_local_video_path(detail: dict[str, Any]) -> Path | None:
    for key in ("file_path", "url"):
        candidate = detail.get(key)
        if not candidate or not isinstance(candidate, str):
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    return None


def run_task(
    *,
    args: argparse.Namespace,
    base_url: str,
    task_id: str,
    input_video: Path,
    caption_file: Path,
    num_inference_steps: int,
    seed: int,
    num_temporal_process_frames: int,
    output_path: Path,
    perf_dump_path: Path,
    poll_interval_seconds: float,
    raw_dir: Path,
    record_runtime: bool,
    download_path: Path | None = None,
) -> dict[str, Any]:
    created_path = raw_dir / f"{task_id}_created.json"
    progress_path = raw_dir / f"{task_id}_progress.json"
    detail_path = raw_dir / f"{task_id}_detail.json"
    callback_path = raw_dir / f"{task_id}_callback.json"
    callback_log_path = raw_dir / f"{task_id}_callback.jsonl"

    callback_recorder = _FlowCutCallbackRecorder(str(callback_log_path))
    with _LocalFlowCutCallbackServer(
        host=args.callback_host,
        port=args.callback_port,
        task_id=task_id,
        recorder=callback_recorder,
    ) as callback_server:
        payload = make_payload(
            callback_url=callback_server.callback_url or "",
            task_id=task_id,
            input_video=input_video,
            caption_file=caption_file,
            num_inference_steps=num_inference_steps,
            seed=seed,
            num_temporal_process_frames=num_temporal_process_frames,
            output_path=output_path,
            perf_dump_path=perf_dump_path,
        )

        print(f"[{utc_now()}] submitting task_id={task_id}")
        print(json.dumps(payload, indent=2))
        start_time = time.perf_counter()
        with httpx.Client(follow_redirects=True, trust_env=False) as client:
            created = submit_flowcut_task_with_retry(
                client=client,
                base_url=base_url,
                payload=payload,
                submit_timeout_s=1800.0,
                retry_interval_seconds=30.0,
                max_submit_attempts=60,
            )
            write_json(created_path, created)

            progress_history = poll_progress(
                client=client,
                base_url=base_url,
                task_id=task_id,
                poll_interval_seconds=poll_interval_seconds,
            )
            write_json(progress_path, progress_history)

            detail = fetch_detail(client, base_url, task_id)
            write_json(detail_path, detail)

            if detail["status"] != "completed":
                raise RuntimeError(json.dumps(detail, indent=2))

            local_video_path = _resolve_local_video_path(detail)
            if download_path is not None:
                if local_video_path is not None:
                    if local_video_path.resolve() != download_path.resolve():
                        shutil.copy2(local_video_path, download_path)
                else:
                    download_content(client, base_url, task_id, download_path)

        final_callback = callback_recorder.wait_for_final(args.final_callback_timeout_s)
        _validate_final_callback_payload(final_callback)
        write_json(callback_path, final_callback)

        result = {
            "created": created,
            "progress_history": progress_history,
            "detail": detail,
            "final_callback": final_callback,
            "request_payload": payload,
            "local_video_path": (
                str(local_video_path) if local_video_path is not None else None
            ),
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

    warmup_result: dict[str, Any] | None = None
    if args.skip_warmup:
        print(f"[{utc_now()}] skip warmup enabled")
    else:
        warmup_result = run_task(
            args=args,
            base_url=args.base_url,
            task_id=warmup_task_id,
            input_video=args.input_video,
            caption_file=args.caption_file,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            num_temporal_process_frames=args.num_temporal_process_frames,
            output_path=warmup_output_path,
            perf_dump_path=warmup_perf_dump_path,
            poll_interval_seconds=args.poll_interval_seconds,
            raw_dir=args.raw_dir,
            record_runtime=False,
        )
        print(f"[{utc_now()}] warmup completed for task_id={warmup_task_id}")

    formal_result = run_task(
        args=args,
        base_url=args.base_url,
        task_id=formal_task_id,
        input_video=args.input_video,
        caption_file=args.caption_file,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        num_temporal_process_frames=args.num_temporal_process_frames,
        output_path=formal_output_path,
        perf_dump_path=formal_perf_dump_path,
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
        "formal_request_payload": formal_result["request_payload"],
        "formal_detail_response": detail,
        "formal_progress_history": formal_result["progress_history"],
        "formal_callback_response": formal_result["final_callback"],
        "perf_dump_path": str(formal_perf_dump_path) if formal_perf_dump_path.exists() else None,
        "perf_dump": perf_dump,
        **summary,
    }
    if warmup_result is not None:
        metrics["warmup_detail_response"] = warmup_result["detail"]
        metrics["warmup_progress_history"] = warmup_result["progress_history"]
        metrics["warmup_callback_response"] = warmup_result["final_callback"]
        metrics["warmup_request_payload"] = warmup_result["request_payload"]
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
