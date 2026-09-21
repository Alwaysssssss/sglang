# SPDX-License-Identifier: Apache-2.0
"""Process-local experimental switches; deliberately outside production defaults.

Usage: experiment_runtime.py benchmark|layout|both <dump_candidate arguments>
"""

import sys

import torch
from sglang.multimodal_gen.runtime.vsr.model import VSRRestorer

mode = sys.argv.pop(1)
if mode not in {"benchmark", "layout", "both"}:
    raise ValueError(mode)
torch.backends.cudnn.benchmark = mode in {"benchmark", "both"}
original_init = VSRRestorer.__init__


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.cudnn_benchmark = mode in {"benchmark", "both"}
    if mode in {"layout", "both"}:
        torch.nn.utils.convert_conv3d_weight_memory_format(
            self.vae, torch.channels_last_3d
        )


VSRRestorer.__init__ = initialize
from sglang.multimodal_gen.runtime.vsr.verify.dump_candidate import main

raise SystemExit(main())
