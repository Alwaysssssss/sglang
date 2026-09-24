"""Diagnostic: compare pre-encoding RGB and locate changed input frames."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
repo = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ve_compare", repo / "python/sglang/multimodal_gen/runtime/videoedit/compare.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = repo / "outputs/videoedit-root-cause-20260924"
left, right = sys.argv[1:3]
for window, start, end in [(0, 1, 49), (1, 5, 47), (2, 5, 25)]:
    for file in ["input_rgb", "generated_rgb"]:
        paths = [root / side / f"w{window}" / f"{file}.pt" for side in [left, right]]
        if not all(p.exists() for p in paths):
            continue
        a, b = [torch.load(p, weights_only=False) for p in paths]
        changed = np.flatnonzero(np.any(a != b, axis=(1, 2, 3))).tolist()
        result = dict(window=window, file=file, changed_frames=changed)
        if file == "generated_rgb":
            scores = [module._ssim(a[i], b[i]) for i in range(start, end)]
            result.update(ssim_mean=float(np.mean(scores)), ssim_min=min(scores),
                          failed_local_frames=[start + i for i, score in enumerate(scores) if score < .97])
        print(json.dumps(result), flush=True)
