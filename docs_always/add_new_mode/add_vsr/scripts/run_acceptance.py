import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

root = Path.cwd()
out = root / "output_results/vsr"
run = out / "migration_20260920"
run.mkdir(exist_ok=True)
py = sys.executable
repo = root.parent / "vsr"
media = Path("/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/zy_test")
ckpt = "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300"
wan = "/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers"
configs = [
    ("B1", media / "val_input_480x832.mp4", "480x832", "global"),
    ("B2", media / "val_input_512x512.mp4", "512x512", "global"),
    ("B3", media / "val_input_768x1280.mp4", "768x1280", "global"),
    ("C1", out / "media/loop64.mp4", "1920x1080", "global"),
    ("C2", out / "media/loop200.mp4", "1920x1080", "global"),
    ("C3", out / "media/loop64.mp4", "2160x3840", "global"),
] + [
    (n, repo / "input/input.mp4", "3840x2160", c)
    for n, c in [("A1", "global"), ("A2", "chunk"), ("A3", "none")]
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd(args, log):
    with open(log, "w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)


results = []
for name, inp, target, color in configs:
    print(time.strftime("%H:%M:%S"), name, "start", flush=True)
    dump = run / name
    video = run / (name + ".mp4")
    args = [
        py,
        "-m",
        "sglang.multimodal_gen.runtime.vsr.verify.dump_candidate",
        "--vsr-repo",
        str(repo),
        "--dump-root",
        str(dump),
        "--dump-frames",
        "--",
        "--input",
        str(inp),
        "--output",
        str(video),
        "--checkpoint_dir",
        ckpt,
        "--wan_root",
        wan,
        "--target_resolution",
        target,
        "--color_ref",
        color,
        "--read_queue",
        "2",
    ]
    (run / (name + "_command.json")).write_text(json.dumps(args, indent=2))
    cmd(args, run / (name + ".log"))
    ref = out / "dumps" / ("EPS_" + name + "_sglang")
    a = sorted((ref / "retired").glob("*.pt"))
    b = sorted((dump / "retired").glob("*.pt"))
    assert a and len(a) == len(b), (name, "parts")
    counts = []
    for x, y in zip(a, b):
        tx = torch.load(x, map_location="cpu")
        ty = torch.load(y, map_location="cpu")
        assert tx.dtype == ty.dtype and tx.shape == ty.shape and torch.equal(tx, ty), (
            name,
            x,
            y,
        )
        counts.append(tx.shape[0])
    same = sha(video) == sha(out / ("EPS_" + name + "_sglang.mp4"))
    assert same, (name, "mp4 mismatch")
    # Exact identity to the measured same-environment reference transfers its
    # cross-environment metrics without recomputing lossy video comparisons.
    reports = {
        layer: json.loads((out / "reports" / f"EPS_{name}_{layer}.json").read_text())
        for layer in ["frames", "mp4", "structural"]
    }
    result = {
        "case": name,
        "frames": sum(counts),
        "parts": counts,
        "frames_bitwise_equal": True,
        "mp4_sha256": sha(video),
        "reference_mp4_sha256": sha(out / f"EPS_{name}_sglang.mp4"),
        "input_sha256": sha(inp),
        "reference_reports": reports,
    }
    results.append(result)
    (run / "results.json").write_text(json.dumps(results, indent=2))
    print(
        time.strftime("%H:%M:%S"),
        name,
        "PASS",
        sum(counts),
        "frames; exact frames and mp4",
        flush=True,
    )
print("ALL PASS", flush=True)
