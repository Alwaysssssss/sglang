# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class STARCogVideoXSRDecodingStage(DecodingStage):
    """STAR-specific decoding shell.

    Phase 3 keeps this stage intentionally thin so the pipeline wiring and stage
    boundary are fixed before phase 5 adds STAR's temporal window decoding
    details.
    """

    def decode_windows(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return super().decode(latents, server_args)

    def postprocess_output(
        self,
        frames: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return server_args.pipeline_config.post_decoding(frames, server_args)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        frames = self.decode_windows(latents, server_args)
        return self.postprocess_output(frames, server_args)
