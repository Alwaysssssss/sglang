"""Evaluate both forward windows before/after encoding, once inference completes."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
repo = Path(__file__).resolve().parents[1]
ref = int(os.environ.get("VE_TEST_REF", "0"))
assert ref in (0, 44)
work = repo / "outputs" / ("videoedit-forward-92f-40s-ref0-20260924" if ref == 0 else "videoedit-middle-92f-40s-ref44-20260924")
name = f"case0008_92f_40s_ref{ref}_tight"
spec = importlib.util.spec_from_file_location("ve_compare", repo / "python/sglang/multimodal_gen/runtime/videoedit/compare.py")
cmp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cmp
spec.loader.exec_module(cmp)

if "--wait" in sys.argv:
    deadline = time.monotonic() + 7200
    while True:
        paths = [work / f"{name}_{side}.log" for side in ("reference", "sglang")]
        logs = [p.read_text(errors="replace") if p.exists() else "" for p in paths]
        if "Saved crop" in logs[0] and "Shutdown complete" in logs[1]:
            break
        if any("Traceback (most recent call last)" in log for log in logs):
            raise RuntimeError("Inference failed; inspect the per-side logs")
        if time.monotonic() > deadline:
            raise TimeoutError("Inference did not finish within two hours")
        time.sleep(10)

def summarize(frames):
    return dict(compared_frames=len(frames),
                ssim_mean=float(np.mean([f["ssim"] for f in frames])),
                ssim_min=min(f["ssim"] for f in frames),
                mse_mean=float(np.mean([f["mse"] for f in frames])),
                mae_mean=float(np.mean([f["mae"] for f in frames])),
                failed_frames=[f["index"] for f in frames if not f["pass_frame"]])

report = {"configuration": dict(frames=92, steps=40, reference_frame=ref, strict=False,
                                  infer_len=49, overlap=5, seed=42, teacache=False),
          "raw": {}, "encoded": {}, "video_streams": {}}
layout = [(0, 1, 0, 1), (1, 5, 48, 1)] if ref == 0 else [(0, 1, 44, 1), (1, 5, 43, -1)]
for window, start, offset, direction in layout:
    a, b = [torch.load(work / "raw" / side / f"window{window}_rgb.pt", weights_only=False)
            for side in ("reference", "sglang")]
    assert a.shape == b.shape and len(a) == 49, (a.shape, b.shape)
    frames = []
    for index in range(start, 49):
        delta = a[index].astype(np.float32) - b[index].astype(np.float32)
        ssim, mse, mae = cmp._ssim(a[index], b[index]), float(np.mean(delta ** 2)), float(np.mean(np.abs(delta)))
        frames.append(dict(index=offset+direction*(index-start), ssim=ssim, mse=mse, mae=mae,
                           pass_frame=ssim >= .97 and mse <= 25 and mae <= 2.5))
    report["raw"][f"window{window}"] = dict(summary=summarize(frames), frames=frames)
    del a, b

for variant, suffix, threshold in [("crop", "_crop_only", .97), ("full", "", .98)]:
    paths = [work / side / f"{name}{suffix}.mp4" for side in ("reference", "sglang")]
    result = cmp.compare_videos(str(paths[0]), str(paths[1]), min_ssim=threshold,
                                max_mse=25, max_mae=2.5, allow_frame_count_delta=0,
                                max_failed_frame_ratio=0)
    result["windows"] = {"window0": summarize(result["frames"][:48] if ref == 0 else result["frames"][44:]),
                         "window1": summarize(result["frames"][48:] if ref == 0 else result["frames"][:44])}
    assert len(result["frames"]) == 92
    report["encoded"][variant] = result
    for side, path in zip(("reference", "sglang"), paths):
        streams = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))["streams"]
        report["video_streams"][f"{side}_{variant}"] = streams

def audio_packets(path):
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets", "-show_data_hash", "sha256", "-show_entries", "packet=pts_time,duration_time,data_hash", "-of", "json", str(path)]))["packets"]

source = audio_packets("/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008/video.mp4")
report["audio"] = {}
for side in ("reference", "sglang"):
    packets = audio_packets(work / side / f"{name}.mp4")
    report["audio"][side] = dict(packet_count=len(packets),
                                  source_prefix_hash_pts_equal=bool(packets) and packets == source[:len(packets)])

(work / "comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"raw": {k:v["summary"] for k,v in report["raw"].items()},
                  "encoded": {k:v["summary"] for k,v in report["encoded"].items()},
                  "audio": report["audio"]}, indent=2), flush=True)
