"""Launch immutable-in-memory validation commands with diagnostic-only hooks."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
work = root / "outputs/videoedit-root-cause-20260924"
side = sys.argv[1]
probe = side.endswith("_probe")
if probe:
    side = side.removesuffix("_probe")
strict = side == "sglang_strict"
if strict:
    side = "sglang"
same_runtime = side == "reference29"
if same_runtime:
    side = "reference"
assert side in ("reference", "sglang")
script = (root / "scripts/videoedit_stream_validation.sh").read_text()
replacements = {
    'VE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"': f'VE_REPO="{root}"',
    'VE_RUN_DIR="$VE_REPO/outputs/videoedit-stream-validation"': f'VE_RUN_DIR="{work}"',
    'VE_NATIVE=/mnt/shanhai-ai/liuh/VideoEdit-diffusers': f'VE_NATIVE="{work}/reference_snapshot"',
    'VE_CASE="$VE_NATIVE/datas/edit_val_cases/0008"': 'VE_CASE=/mnt/shanhai-ai/liuh/VideoEdit-diffusers/datas/edit_val_cases/0008',
    '"$VE_NATIVE/ckpts/step_47500"': '"/mnt/shanhai-ai/liuh/VideoEdit-diffusers/ckpts/step_47500"',
    '/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python -B infer.py':
        f'/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python -B "{root}/scripts/videoedit_diagnose_boundaries.py"',
    '"$VE_REPO/scripts/videoedit_existing_env.py"': '"$VE_REPO/scripts/videoedit_diagnose_boundaries.py"',
}
for old, new in replacements.items():
    assert old in script, old
    script = script.replace(old, new)
if same_runtime:
    script = script.replace(f'VE_RUN_DIR="{work}"', f'VE_RUN_DIR="{work}/same_runtime"')
    script = script.replace('/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python', '/opt/conda/bin/python')
    packages = "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages"
    script = 'export PYTHONPATH="/opt/conda/lib/python3.11:' + packages + ':' + packages + '/nvidia_cutlass_dsl/python_packages:/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/lib/python3.11/site-packages"\n' + script
env = dict(os.environ, VE_DIAG_SIDE="native" if side == "reference" else "sglang",
           VE_DIAG_DIR=str(work / ("reference29" if same_runtime else side)), VE_FRAMES="110", VE_STEPS="2", VE_REF="20",
           VE_MASTER_PORT="30205", VE_SCHEDULER_PORT="5765")
if strict:
    script = script.replace(f'VE_RUN_DIR="{work}"', f'VE_RUN_DIR="{work}/strict"')
    env.update(VE_DIAG_STRICT="1", VE_DIAG_DIR=str(work / "sglang_strict"))
if probe:
    tag = "reference29_probe" if same_runtime else "sglang_probe"
    env.update(VE_DIAG_PROBE="1", VE_DIAG_DIR=str(work / tag),
               VE_MASTER_PORT="30215", VE_SCHEDULER_PORT="5775")
    old = f'VE_RUN_DIR="{work}/same_runtime"' if same_runtime else f'VE_RUN_DIR="{work}"'
    script = script.replace(old, f'VE_RUN_DIR="{work}/{tag}_run"')
raise SystemExit(subprocess.call(["bash", "-c", script, "diagnose", side], env=env))
