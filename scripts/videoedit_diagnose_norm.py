"""Replay the first-block norm rounding difference from diagnostic captures."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1] / "outputs/videoedit-root-cause-20260924"

def read(side, name):
    return torch.load(root / side / "w0" / f"probe_{name}.pt", weights_only=False)

checkpoint = "/mnt/shanhai-ai/liuh/VideoEdit-diffusers/ckpts/step_47500/transformer/diffusion_pytorch_model.safetensors"
with safe_open(checkpoint, framework="pt", device="cpu") as weights:
    table = weights.get_tensor("blocks.0.scale_shift_table")

# First 50 complete tokens from the captured first-block norm output.
norm = read("reference29_probe", "blocks.0.norm1")[:256000].reshape(1, 50, 5120)
temb = read("reference29_probe", "condition_embedder")["1"].reshape(1, 6, 5120)
modulation = table.float() + temb.float()
shift, scale = modulation[:, 0:1], modulation[:, 1:2]
actual = read("sglang_probe", "blocks.0.norm1")[:256000]
for label, value in [("native_fp32", norm), ("early_bf16", norm.bfloat16().float())]:
    expected = (value * (1 + scale) + shift).bfloat16().flatten()
    delta = (expected.float() - actual.float()).abs()
    equal = torch.equal(expected, actual)
    print(json.dumps(dict(case=label, equal=equal, mae=delta.mean().item(),
                          max_abs=delta.max().item(), different=int((expected != actual).sum()),
                          count=actual.numel())))
    assert equal == (label == "early_bf16"), "Captured rounding signature changed"
