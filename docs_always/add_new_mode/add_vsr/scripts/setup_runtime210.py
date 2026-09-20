# SPDX-License-Identifier: Apache-2.0
"""Create the VSR-specific runtime from existing, locally validated dependencies."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
output = root / "output_results/vsr"
env = output / "migration_env210"
paths = [
    output / "torch210-cu126",
    output / "kernel210-runtime",
    output / "migration_env/lib/python3.11/site-packages",
    root.parent / "env/sglang/lib/python3.11/site-packages",
    root / "python",
]
for path in paths:
    if not path.is_dir():
        raise FileNotFoundError(f"Required local dependency directory: {path}")
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("The existing native extensions require CPython 3.11")
subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
site = env / "lib/python3.11/site-packages"
(site / "vsr_runtime.pth").write_text("".join(str(path) + "\n" for path in paths))
print(env / "bin/python")
