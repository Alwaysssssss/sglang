import json
import subprocess
import sys
import time
from pathlib import Path

import torch

root = Path.cwd()
out = root / "output_results/vsr"
run = out / "migration_20260920"
py = sys.executable
repo = root.parent / "vsr"
ckpt = "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300"
wan = "/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers"
verify = root / "python/sglang/multimodal_gen/runtime/vsr/verify"


def call(args, log):
    with open(log, "w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)


for n in (1, 10, 33):
    dest = run / f"input{n}.mp4"
    call(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(repo / "input/input.mp4"),
            "-frames:v",
            str(n),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(dest),
        ],
        run / f"input{n}.log",
    )
results = []
for name, inp, size in [
    ("single", run / "input33.mp4", "320x640"),
    ("three", out / "media/loop64.mp4", "320x640"),
    ("pad", run / "input33.mp4", "336x656"),
    ("short", run / "input10.mp4", "320x640"),
    ("one", run / "input1.mp4", "320x640"),
]:
    print(time.strftime("%H:%M:%S"), name, "start", flush=True)
    for side in ["reference", "candidate"]:
        dump = run / f"{name}_{side}"
        tool = (
            [str(verify / "dump_baseline.py")]
            if side == "reference"
            else ["-m", "sglang.multimodal_gen.runtime.vsr.verify.dump_candidate"]
        )
        args = [
            py,
            *tool,
            "--vsr-repo",
            str(repo),
            "--dump-root",
            str(dump),
            "--dump-frames",
            "--dump-tiles",
            "all",
            "--dump-chunks",
            "--",
            "--input",
            str(inp),
            "--output",
            str(run / f"{name}_{side}.mp4"),
            "--checkpoint_dir",
            ckpt,
            "--wan_root",
            wan,
            "--target_resolution",
            size,
        ]
        (run / f"{name}_{side}_command.json").write_text(json.dumps(args, indent=2))
        call(args, run / f"{name}_{side}.log")
    ref = run / f"{name}_reference"
    cand = run / f"{name}_candidate"
    pairs = []
    for x in sorted(ref.rglob("*.pt")):
        y = cand / x.relative_to(ref)
        a = torch.load(x, map_location="cpu")
        b = torch.load(y, map_location="cpu")
        assert (
            a.dtype == b.dtype
            and a.shape == b.shape
            and torch.isfinite(a).all()
            and torch.equal(a, b)
        ), (name, x)
        pairs.append(str(x.relative_to(ref)))
    ma = json.loads((ref / "manifest.json").read_text())
    mb = json.loads((cand / "manifest.json").read_text())
    assert ma["shapes"] == mb["shapes"] and ma["color_stats"] == mb["color_stats"], name
    assert len(pairs) == len(list(cand.rglob("*.pt")))
    results.append(
        {
            "case": name,
            "equal_tensors": pairs,
            "frames": mb["frames_written"],
            "manifest_shapes": mb["shapes"],
            "color_stats_exact": True,
        }
    )
    (run / "edge_results.json").write_text(json.dumps(results, indent=2))
    print(name, "PASS", flush=True)
args = [
    py,
    "-m",
    "sglang.multimodal_gen.runtime.vsr.cli",
    "restore",
    "--via-pipeline",
    "--input",
    str(run / "input33.mp4"),
    "--output",
    str(run / "native.mp4"),
    "--checkpoint_dir",
    ckpt,
    "--wan_root",
    wan,
    "--target_resolution",
    "320x640",
]
(run / "native_command.json").write_text(json.dumps(args, indent=2))
call(args, run / "native.log")
assert (run / "native.mp4").read_bytes() == (run / "single_candidate.mp4").read_bytes()
print("NATIVE PASS; ALL EDGES PASS", flush=True)
