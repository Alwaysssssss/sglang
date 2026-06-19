from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx

from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos

BASE_URL = "http://127.0.0.1:31081"
INPUT_VIDEO = Path("/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4")
CAPTION_FILE = Path("/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt")
REFERENCE_VIDEO = Path(
    "/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4"
)
RESULT_DIR = Path("/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_acceptance")
INDICATOR_DIR = Path("/home/zhiheng/sglang/Vivid_Acceptance/indicator")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ffprobe_profile(path: Path) -> dict:
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


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    INDICATOR_DIR.mkdir(parents=True, exist_ok=True)

    task_id = f"vividvr-service-long-130f-20step-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    requested_output_path = RESULT_DIR / f"{task_id}.mp4"
    downloaded_output_path = RESULT_DIR / f"downloaded_{task_id}.mp4"
    report_path = INDICATOR_DIR / f"{task_id}.json"
    framewise_path = INDICATOR_DIR / f"{task_id}_framewise_ssim.json"

    payload = {
        "model": "VividVR",
        "task_id": task_id,
        "video_input_path": str(INPUT_VIDEO),
        "caption_file_path": str(CAPTION_FILE),
        "reference_video_path": str(REFERENCE_VIDEO),
        "num_inference_steps": 20,
        "seed": 42,
        "num_temporal_process_frames": 121,
        "output_path": str(requested_output_path),
    }

    print(f"[{utc_now()}] submitting task_id={task_id}")
    print(json.dumps(payload, indent=2))

    start_time = time.perf_counter()
    with httpx.Client(trust_env=False, timeout=60.0) as client:
        response = client.post(f"{BASE_URL}/v1/videos/repairs", json=payload)
        response.raise_for_status()
        created = response.json()
        print(f"[{utc_now()}] created job={created['id']} status={created['status']}")

        status = created["status"]
        progress = created.get("progress", 0)
        while status not in {"completed", "failed"}:
            time.sleep(15)
            progress_response = client.get(f"{BASE_URL}/v1/videos/{task_id}/progress")
            progress_response.raise_for_status()
            progress_body = progress_response.json()
            status = progress_body["status"]
            progress = progress_body.get("progress", progress)
            print(f"[{utc_now()}] status={status} progress={progress}")

        detail_response = client.get(f"{BASE_URL}/v1/videos/{task_id}")
        detail_response.raise_for_status()
        detail = detail_response.json()
        print(f"[{utc_now()}] final status={detail['status']}")
        if detail["status"] != "completed":
            raise RuntimeError(json.dumps(detail, indent=2))

        content_response = client.get(f"{BASE_URL}/v1/videos/{task_id}/content")
        content_response.raise_for_status()
        downloaded_output_path.write_bytes(content_response.content)
        print(f"[{utc_now()}] downloaded -> {downloaded_output_path}")

    total_runtime_seconds = time.perf_counter() - start_time
    local_output_path = Path(detail.get("file_path") or requested_output_path)
    if not local_output_path.exists():
        raise FileNotFoundError(f"Local output missing: {local_output_path}")

    print(f"[{utc_now()}] local output -> {local_output_path}")
    compare_report = compare_videos(str(REFERENCE_VIDEO), str(local_output_path))
    framewise_path.write_text(json.dumps(compare_report, indent=2), encoding="utf-8")

    summary = compare_report["summary"]
    metrics = {
        "task_id": task_id,
        "status": detail["status"],
        "service_url": BASE_URL,
        "input_video": str(INPUT_VIDEO),
        "caption_file": str(CAPTION_FILE),
        "reference_video": str(REFERENCE_VIDEO),
        "local_output_video": str(local_output_path),
        "downloaded_output_video": str(downloaded_output_path),
        "requested_output_path": str(requested_output_path),
        "num_inference_steps": 20,
        "seed": 42,
        "num_temporal_process_frames": 121,
        "total_runtime_seconds": total_runtime_seconds,
        "model_inference_runtime_seconds": detail.get("inference_time_s"),
        "peak_memory_mb": detail.get("peak_memory_mb"),
        "reference_profile": ffprobe_profile(REFERENCE_VIDEO),
        "service_output_profile": ffprobe_profile(local_output_path),
        "download_profile": ffprobe_profile(downloaded_output_path),
        **summary,
    }

    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] report -> {report_path}")
    print(json.dumps(metrics, indent=2))
    return 0 if summary["pass_compare"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
