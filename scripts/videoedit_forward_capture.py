"""Forward validation launcher: save generated RGB before video encoding only."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDE = os.environ["VE_FORWARD_SIDE"]
OUT = Path(os.environ["VE_FORWARD_CAPTURE"])
window = 0

if SIDE == "reference":
    sys.path.insert(0, os.environ["VE_FORWARD_REFERENCE"])
else:
    spec = importlib.util.spec_from_file_location("existing_env", ROOT / "scripts/videoedit_existing_env.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)

import numpy as np
import torch

def save(frames):
    global window
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(frames, OUT / f"window{window}_rgb.pt")
    print(f"[forward-validation] saved pre-encoding RGB window={window} shape={frames.shape}", flush=True)
    window += 1

if SIDE == "reference":
    import infer
    from diffusers.video_processor import VideoProcessor
    original = VideoProcessor.postprocess_video
    def postprocess(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        save((result[0].clamp(0, 1) * 255).to(torch.uint8).movedim(1, -1).cpu().numpy())
        return result
    VideoProcessor.postprocess_video = postprocess
else:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import videoedit_wan as stages
    original = stages.VideoEditDecodingStage.forward
    def decode(self, batch, server_args):
        result = original(self, batch, server_args)
        save(np.stack([np.asarray(f) for f in batch.sampling_params.runtime_window_output_frames]))
        return result
    stages.VideoEditDecodingStage.forward = decode

if __name__ == "__main__":
    if SIDE == "reference":
        infer.infer(infer.build_parser().parse_args())
    else:
        from sglang.multimodal_gen.runtime.videoedit.cli import main
        raise SystemExit(main())
