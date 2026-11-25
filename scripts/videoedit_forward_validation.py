"""Run the frozen reference and current streaming implementation at 92f/40s/ref0."""
import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
ref = int(os.environ.get("VE_TEST_REF", "0"))
assert ref in (0, 44), "Only the two validated full-window layouts are supported"
work = repo / "outputs" / ("videoedit-forward-92f-40s-ref0-20260924" if ref == 0 else "videoedit-middle-92f-40s-ref44-20260924")
snapshot = repo / "outputs/videoedit-root-cause-20260924/reference_snapshot"
side = sys.argv[1]
assert side in ("reference", "sglang", "compare")
script = (repo / "scripts/videoedit_stream_validation.sh").read_text()
replacements = {
    'VE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"': f'VE_REPO="{repo}"',
    'VE_RUN_DIR="$VE_REPO/outputs/videoedit-stream-validation"': f'VE_RUN_DIR="{work}"',
    'VE_NATIVE=/mnt/shanhai-ai/liuh/VideoEdit-diffusers': f'VE_NATIVE="{snapshot}"',
    'VE_CASE="$VE_NATIVE/datas/edit_val_cases/0008"': 'VE_CASE=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008',
    '"$VE_NATIVE/ckpts/step_47500"': '"/mnt/shanhai-ai/liuh/VideoEdit-diffusers/ckpts/step_47500"',
    '/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python -B infer.py': f'/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python -B "{repo}/scripts/videoedit_forward_capture.py"',
    '"$VE_REPO/scripts/videoedit_existing_env.py"': '"$VE_REPO/scripts/videoedit_forward_capture.py"',
}
for old, new in replacements.items():
    assert old in script, old
    script = script.replace(old, new)
env = dict(os.environ, VE_FRAMES="92", VE_STEPS="40", VE_REF=str(ref),
           VE_MASTER_PORT="30305", VE_SCHEDULER_PORT="5865",
           VE_FORWARD_SIDE=side, VE_FORWARD_CAPTURE=str(work / "raw" / side),
           VE_FORWARD_REFERENCE=str(snapshot))
raise SystemExit(subprocess.call(["bash", "-c", script, "forward-validation", side], env=env))
