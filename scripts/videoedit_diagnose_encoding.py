"""Diagnostic: identical pre-encoding frames through the two finalizers."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

repo = Path(__file__).resolve().parents[1]
work = repo / "outputs/videoedit-root-cause-20260924"
out = work / "encoding_control"
out.mkdir(exist_ok=True)
sys.path.insert(0, str(work / "reference_snapshot"))
from utils import video_io_ffmpeg as native

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sg = load_module("sg_stream_io", repo / "python/sglang/multimodal_gen/runtime/videoedit/stream_io.py")
cmp = load_module("ve_compare", repo / "python/sglang/multimodal_gen/runtime/videoedit/compare.py")
source = "/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008/video.mp4"
# Resolve the exact source filename used by the validation launcher.
case = Path(source).parent
source = str(case / "src_video.mp4") if (case / "src_video.mp4").exists() else source
if len(sys.argv) > 1:
    source = sys.argv[1]
meta = native.probe_video(source)
profile = sg.encode_profile(meta, lossless=True)
parts = []
for label, windows in [("short", [(2, 5, 25)]), ("long", [(0, 1, 49), (1, 5, 47)])]:
    path = out / f"{label}.mp4"
    writer = sg.VideoWriterFfmpeg(str(path), 1472, 608, meta["fps"], profile=profile)
    try:
        for window, lo, hi in windows:
            frames = torch.load(work / "sglang" / f"w{window}/generated_rgb.pt", weights_only=False)
            for frame in frames[lo:hi]:
                writer.write(frame)
    finally:
        writer.close()
    parts.append((str(path), label == "short"))
native.finalize_clips(parts, str(out / "native.mp4"), meta, start_frame=0,
                      bitrate_scale=1472 * 608 / (1920 * 1080), work_dir=str(out))
sg.merge_clips(parts, str(out / "sglang.mp4"), meta["fps"], sg.encode_profile(meta))
report = cmp.compare_videos(str(out / "native.mp4"), str(out / "sglang.mp4"),
                            min_ssim=.97, max_mse=25, max_mae=2.5,
                            allow_frame_count_delta=0, max_failed_frame_ratio=0)
print(json.dumps(report["summary"], indent=2))
