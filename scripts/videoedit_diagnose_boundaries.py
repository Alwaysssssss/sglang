"""Diagnostic-only boundary capture; production modules and environments unchanged.

Set VE_DIAG_SIDE=native|sglang and VE_DIAG_DIR, then pass the normal CLI arguments.
The native source is a frozen, repository-local snapshot. Spawn workers install
the same hooks when importing this launcher.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = os.environ["VE_DIAG_SIDE"]
OUT = Path(os.environ["VE_DIAG_DIR"])
SNAPSHOT = ROOT / "outputs/videoedit-root-cause-20260924/reference_snapshot"

if SIDE == "sglang":
    spec = importlib.util.spec_from_file_location(
        "existing_env", ROOT / "scripts/videoedit_existing_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    sys.path.insert(0, str(SNAPSHOT))

import numpy as np
import torch

window = -1
call = 0


def save(name, value):
    path = OUT / f"w{window}"
    path.mkdir(parents=True, exist_ok=True)

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cpu(v) for v in x]
        return x

    torch.save(cpu(value), path / f"{name}.pt")


def begin(frames, masks):
    global window, call
    window += 1
    call = 0
    save("input_rgb", np.stack([np.asarray(f) for f in frames]))
    save("input_masks", np.stack([np.asarray(f) for f in masks]))


def patch_transformer(cls):
    original = cls.forward

    def forward(self, *args, **kwargs):
        global call
        if os.environ.get("VE_DIAG_PROBE") == "1" and call == 0:
            def sample(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().flatten()[:262144].cpu()
                if isinstance(x, (tuple, list)):
                    return {str(i): sample(v) for i, v in enumerate(x) if v is not None}
                return x
            def hook(name):
                def capture(mod, inputs, output):
                    save("probe_" + name, sample(output))
                    if name == "blocks.0":
                        raise RuntimeError("Diagnostic first-block capture complete (intentional stop)")
                return capture
            for name, mod in self.named_modules():
                if name in ("patch_embedding", "condition_embedder", "rope", "rotary_emb", "blocks.0") or name.startswith("condition_embedder.") or name.startswith("blocks.0."):
                    mod.register_forward_hook(hook(name))
        if SIDE == "sglang" and os.environ.get("VE_DIAG_STRICT") == "1":
            self.strict_videoedit_math = True
            for block in self.blocks:
                block.strict_videoedit_math = True
        index = call
        call += 1
        save(f"dit{index}_input", {
            k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        })
        result = original(self, *args, **kwargs)
        save(f"dit{index}_output", result[0] if isinstance(result, tuple) else result)
        return result

    cls.forward = forward


if SIDE == "native":
    import infer
    from diffusers.video_processor import VideoProcessor

    original_prepare = infer.prepare_window_inputs

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        # These are the actual padded frames/masks delivered to the pipeline.
        frames = list(kwargs["resized_video"])
        valid = len(frames)
        for i in range(kwargs["window_len"] - valid):
            frames.append(frames[max(valid - 1 - i, 0)])
        begin(frames, result["window_masks"])
        save("prepared", {k: result[k] for k in ("masked_video_tensor", "cond_masks")})
        return result

    infer.prepare_window_inputs = prepare
    patch_transformer(infer.WanTransformer3DModel)
    original_post = infer.post_latents

    def post(vae, latents, *args, **kwargs):
        save("final_latents", latents)
        return original_post(vae, latents, *args, **kwargs)

    infer.post_latents = post
    original_rgb = VideoProcessor.postprocess_video

    def postprocess(self, *args, **kwargs):
        result = original_rgb(self, *args, **kwargs)
        save("generated_rgb", (result[0].clamp(0, 1) * 255).to(torch.uint8).movedim(1, -1).cpu().numpy())
        return result

    VideoProcessor.postprocess_video = postprocess
else:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import videoedit_wan as stages
    from sglang.multimodal_gen.runtime.models.dits.wan_videoedit import WanVideoEditTransformer3DModel

    original_validate = stages.VideoEditWindowValidationStage.forward

    def validate(self, batch, server_args):
        params = batch.sampling_params
        begin(params.runtime_window_frames, params.runtime_window_masks)
        return original_validate(self, batch, server_args)

    stages.VideoEditWindowValidationStage.forward = validate
    original_condition = stages.VideoEditConditionEncodingStage.forward

    def condition(self, batch, server_args):
        result = original_condition(self, batch, server_args)
        params = batch.sampling_params
        save("prepared", {"masked_video_tensor": params.runtime_masked_video_tensor,
                          "cond_masks": params.runtime_cond_masks})
        return result

    stages.VideoEditConditionEncodingStage.forward = condition
    original_decode = stages.VideoEditDecodingStage.forward

    def decode(self, batch, server_args):
        save("final_latents", batch.sampling_params.runtime_latents)
        result = original_decode(self, batch, server_args)
        save("generated_rgb", np.stack([np.asarray(f) for f in batch.sampling_params.runtime_window_output_frames]))
        return result

    stages.VideoEditDecodingStage.forward = decode
    patch_transformer(WanVideoEditTransformer3DModel)


if __name__ == "__main__":
    if SIDE == "native":
        infer.infer(infer.build_parser().parse_args())
    else:
        from sglang.multimodal_gen.runtime.videoedit.cli import main
        raise SystemExit(main())
