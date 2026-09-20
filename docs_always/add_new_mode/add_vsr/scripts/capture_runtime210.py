# SPDX-License-Identifier: Apache-2.0
"""Record the effective VSR runtime, including native extension locations."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[4]
report = {
    "python": sys.executable,
    "python_version": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "packages": {},
}
assert torch.__version__ == "2.10.0+cu126"
for name in [
    "torch",
    "torchvision",
    "diffusers",
    "decord",
    "numpy",
    "imageio",
    "imageio_ffmpeg",
    "transformers",
    "sgl_kernel",
    "sglang",
]:
    module = importlib.import_module(name)
    report["packages"][name] = {
        "version": getattr(module, "__version__", None),
        "file": module.__file__,
    }
report["gpu"] = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        "7",
        "--query-gpu=uuid,name,memory.used,utilization.gpu",
        "--format=csv",
    ],
    text=True,
)
report["dependency_paths"] = (
    (Path(sys.prefix) / "lib/python3.11/site-packages/vsr_runtime.pth")
    .read_text()
    .splitlines()
)
report["sglang_commit"] = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()
report["sglang_status"] = subprocess.check_output(
    ["git", "-C", str(root), "status", "--short"], text=True
)
wheel = (
    root
    / "output_results/vsr/kernel210-wheel/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl"
)
report["kernel_wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
report["reference_validation_provenance"] = str(
    root / "output_results/vsr/migration_20260920/provenance.json"
)
(root / "output_results/vsr/runtime210_validation/environment.json").write_text(
    json.dumps(report, indent=2)
)
print(report["torch"], report["packages"]["sgl_kernel"]["file"])
