# SPDX-License-Identifier: Apache-2.0
"""Fixed-latent decoder differential loop; no DiT or video IO."""

import json
import statistics
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from sglang.multimodal_gen.runtime.vsr.compile import compile_decoder
from sglang.multimodal_gen.runtime.vsr.model import VSRAutoencoder

out = Path("output_results/vsr/optimization_round2/precision")
out.mkdir(exist_ok=True)
vae = AutoencoderKLWan.from_pretrained(
    "/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers", subfolder="vae"
)
model = (
    VSRAutoencoder(
        vae,
        "/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300/vae_decoder_ema.pt",
    )
    .to("cuda", dtype=torch.bfloat16)
    .eval()
)
torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
torch.backends.cudnn.benchmark = True
p = Path("output_results/vsr/optimization_round2/debug/direct")
z = torch.load(p / "latent.pt", map_location="cuda") - torch.load(
    p / "velocity.pt", map_location="cuda"
)
eager = vae.decoder.forward
compile_decoder(vae)
fast = vae.decoder.forward
precise = torch.compile(
    eager, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
)
reports = []
with torch.no_grad():
    for name, forward in [("eager", eager), ("compiled", fast), ("precise", precise)]:
        vae.decoder.forward = forward
        for _ in range(4):
            y = model.decode(z)
        torch.cuda.synchronize()
        samples = []
        for _ in range(6):
            start = time.perf_counter()
            y = model.decode(z)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
        y = y.cpu().float()
        if name == "eager":
            reference = y
        report = {
            "mode": name,
            "median_s": statistics.median(samples),
            "rel_mean": float((y - reference).abs().mean() / reference.abs().max()),
            "max_abs": float((y - reference).abs().max()),
            "equal": torch.equal(y, reference),
        }
        reports.append(report)
        torch.save(y, out / f"{name}.pt")
        (out / "results.json").write_text(json.dumps(reports, indent=2))
        print(report, flush=True)
