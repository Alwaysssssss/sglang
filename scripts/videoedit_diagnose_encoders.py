"""Replay identical captured pixels through CLIP and both VAE implementations."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

root = Path(__file__).resolve().parents[1]
work = root / "outputs/videoedit-root-cause-20260924"
label = sys.argv[1]
out = work / label
out.mkdir(exist_ok=True)
base = "/mnt/shanhai-ai/shanhai-workspace/fanruidi/projects/VideoEdit-new/VideoEdit_diffusers/pretrain_models/Wan2.1-I2V-14B-480P-Diffusers"
torch.set_num_threads(4)

pixels = torch.load(work / "reference/w0/input_rgb.pt", weights_only=False)[0]
processor = CLIPImageProcessor.from_pretrained(base, subfolder="image_processor")
inputs = processor(images=Image.fromarray(pixels), return_tensors="pt")
torch.save(inputs["pixel_values"], out / "clip_pixels.pt")
model = CLIPVisionModel.from_pretrained(base, subfolder="image_encoder", torch_dtype=torch.float32).eval().to("cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = model(**inputs.to("cuda"), output_hidden_states=True).hidden_states[-2]
torch.save(image.cpu(), out / "clip.pt")
print(json.dumps({"torch": torch.__version__, "attention": model.config._attn_implementation,
                  "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn": torch.backends.cudnn.version()}), flush=True)
del model
torch.cuda.empty_cache()

sys.path.insert(0, str(work / "reference_snapshot"))
from models.autoencoder_kl_wan import AutoencoderKLWan as NativeVAE
from diffusers import AutoencoderKLWan as HFVAE

prepared = torch.load(work / "reference/w0/prepared.pt", weights_only=False)
masked = prepared["masked_video_tensor"].permute(1, 0, 2, 3).unsqueeze(0).to("cuda")
for name, cls in (("native", NativeVAE), ("hf", HFVAE)):
    vae = cls.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16).eval().to("cuda")
    vae.enable_tiling()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = vae.encode(masked).latent_dist.mode()
        mean = torch.tensor(vae.config.latents_mean,device="cuda",dtype=torch.bfloat16).view(1,-1,1,1,1)
        std = torch.tensor(vae.config.latents_std,device="cuda",dtype=torch.bfloat16).view(1,-1,1,1,1)
        z = (z - mean) / std
    torch.save(z.cpu(), out / f"vae_{name}.pt")
    print(f"saved {label}/{name}", flush=True)
    del vae,z
    torch.cuda.empty_cache()
