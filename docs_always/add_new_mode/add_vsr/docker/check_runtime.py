"""Fail fast on version drift; --gpu also checks the native extension and SDPA."""

import argparse
import importlib.metadata
import json
import subprocess

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()
    expected = {
        "torch": "2.10.0+cu126",
        "torchvision": "0.25.0+cu126",
        "triton": "3.6.0",
        "diffusers": "0.37.0",
        "transformers": "5.3.0",
        "numpy": "2.4.6",
        "imageio": "2.36.0",
        "imageio-ffmpeg": "0.6.0",
        "opencv-python-headless": "4.10.0.84",
        "decord": "0.6.0",
        "sglang-kernel": "0.4.1",
    }
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected or torch.version.cuda != "12.6":
        raise RuntimeError(f"Runtime drift: {versions}; CUDA={torch.version.cuda}")
    import imageio_ffmpeg
    from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: F401
    from sglang.multimodal_gen.runtime.entrypoints.openai import vsr_api  # noqa: F401
    from sglang.multimodal_gen.runtime.launch_server import launch_server  # noqa: F401
    from sglang.multimodal_gen.runtime.pipelines.wan_vsr_pipeline import (  # noqa: F401
        WanVSRPipeline,
    )

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], check=True)
    if args.gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != args.devices:
            raise RuntimeError(f"Expected exactly {args.devices} visible CUDA devices")
        import sgl_kernel  # noqa: F401

        for index in range(args.devices):
            with torch.cuda.device(index):
                q = torch.randn(
                    1, 2, 16, 64, device=f"cuda:{index}", dtype=torch.bfloat16
                )
                out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                if not torch.isfinite(out).all().item():
                    raise RuntimeError("SDPA check produced non-finite values")
                torch.cuda.synchronize()
                print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
    print(json.dumps(versions, indent=2))


if __name__ == "__main__":
    main()
