# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from .cogvlm2 import CogVLM2Captioner


def create_captioner(args) -> CogVLM2Captioner:
    if args.caption_backend != "cogvlm2":
        raise ValueError(f"Unsupported caption backend: {args.caption_backend}")
    return CogVLM2Captioner.from_args(args)
