# SPDX-License-Identifier: Apache-2.0
"""VSR (video super-resolution / restoration) runtime.

A port of the reference implementation in ``$VSR_REPO``. Phase 1 uses the
diffusers ``AutoencoderKLWan`` / ``WanTransformer3DModel`` classes directly, so
that only the surrounding orchestration is new and the numerics stay comparable
to the reference -- see ``docs_always/add_new_mode/add_vsr/requirements.md`` §1.

Module layout mirrors the reference's, so the two can be read side by side:

    geometry    resize / padding / tile positions      <- infer/utils/video_io.py, tiling.py
    blending    feather weights, accumulation           <- infer/utils/tiling.py, stream.py
    color       per-channel AdaIN correction            <- infer/utils/video_io.py, stream.py
    video_io    decord decode, imageio encode           <- infer/utils/video_io.py
    model       diffusers components, one restore step  <- infer/models/stage3.py
    stream      the streaming core                      <- infer/stream.py
"""

from sglang.multimodal_gen.runtime.vsr import blending, color, geometry, video_io

__all__ = ["blending", "color", "geometry", "video_io"]
