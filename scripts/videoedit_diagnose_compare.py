"""Compare already-captured model boundaries without rerunning inference."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/videoedit-root-cause-20260924")


def compare(a, b, name):
    if isinstance(a, dict) and not isinstance(b, dict) and set(a) == {"0"}:
        a = a["0"]
    if isinstance(b, dict) and not isinstance(a, dict) and set(b) == {"0"}:
        b = b["0"]
    if isinstance(a, dict):
        return {k: compare(a[k], b[k], name + "/" + k) for k in a.keys() & b.keys()}
    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b
    if not isinstance(a, torch.Tensor):
        return {"equal": a == b}
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape)}
    af, bf = a.float(), b.float()
    delta = af - bf
    result = {"shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
              "equal": torch.equal(a, b), "mae": delta.abs().mean().item(),
              "max_abs": delta.abs().max().item(), "mse": delta.square().mean().item()}
    if name.rsplit("/", 1)[-1] == "hidden_states":
        result["channels"] = {k: compare(a[:, lo:hi], b[:, lo:hi], k)
                              for k, lo, hi in [("noise", 0, 16), ("mask", 16, 20), ("condition", 20, 36)]}
    return result


report = {}
for index in range(3):
    a_dir = root / (sys.argv[2] if len(sys.argv) > 2 else "reference") / f"w{index}"
    b_dir = root / (sys.argv[3] if len(sys.argv) > 3 else "sglang") / f"w{index}"
    files = sorted(set(p.name for p in a_dir.glob("*.pt")) & set(p.name for p in b_dir.glob("*.pt")))
    report[f"w{index}"] = {}
    for name in files:
        try:
            a = torch.load(a_dir / name, map_location="cpu", weights_only=False)
            b = torch.load(b_dir / name, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError):
            continue  # A capture may still be writing.
        result = compare(a, b, name)
        report[f"w{index}"][name] = result
        print(json.dumps({"window": index, "file": name, "result": result}), flush=True)
        del a, b
print(json.dumps(report, indent=2))
