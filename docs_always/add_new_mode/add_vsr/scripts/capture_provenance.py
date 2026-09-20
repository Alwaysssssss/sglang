import hashlib
import importlib.metadata as m
import json
import subprocess
import sys
from pathlib import Path

r = Path.cwd()
out = r / "output_results/vsr/migration_20260920"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(p):
    return {
        key: subprocess.check_output(["git", "-C", str(p), *args], text=True)
        for key, args in [
            ("commit", ["rev-parse", "HEAD"]),
            ("status", ["status", "--short"]),
            ("diff", ["diff"]),
        ]
    }


weights = [
    Path("/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers/vae"),
    Path(
        "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300/transformer_ema"
    ),
]
files = [p for d in weights for p in sorted(d.rglob("*")) if p.is_file()] + [
    Path(
        "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300/vae_decoder_ema.pt"
    )
]
sources = list((r / "python/sglang/multimodal_gen/runtime/vsr").rglob("*.py"))
sources += list((r / "test/registered/multimodal_gen/vsr").glob("*.py"))
sources += [
    r / "python/sglang" / name
    for name in [
        "utils.py",
        "multimodal_gen/registry.py",
        "multimodal_gen/configs/pipeline_configs/base.py",
        "multimodal_gen/configs/pipeline_configs/vsr.py",
        "multimodal_gen/configs/pipeline_configs/__init__.py",
        "multimodal_gen/configs/sample/vsr.py",
        "multimodal_gen/configs/sample/__init__.py",
        "multimodal_gen/runtime/pipelines/wan_vsr_pipeline.py",
        "multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vsr.py",
    ]
]
info = {
    "python": sys.executable,
    "python_version": sys.version,
    "attention_backend": "diffusers default torch SDPA",
    "versions": {
        n: m.version(n)
        for n in [
            "torch",
            "diffusers",
            "numpy",
            "decord",
            "imageio",
            "imageio-ffmpeg",
            "transformers",
            "accelerate",
            "opencv-python-headless",
        ]
    },
    "gpu": subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "7",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    ),
    "sglang": git(r),
    "reference": git(r.parent / "vsr"),
    "weights": {str(p): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files},
    "source_sha256": {str(p.relative_to(r)): sha(p) for p in sources},
}
(out / "provenance.json").write_text(json.dumps(info, indent=2))
