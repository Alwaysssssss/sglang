# SPDX-License-Identifier: Apache-2.0
"""Observe the migrated CLI using the same sampling points as the reference."""

import sys
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.vsr.verify.dump_baseline import (
    _Dumper,
    _git_state,
    install_patches,
    parse_args,
)


def main():
    args = parse_args()
    dumper = _Dumper(
        args.dump_root, args.dump_frames, args.dump_tiles, args.dump_chunks
    )
    install_patches(dumper, args.vsr_repo, candidate=True)
    from sglang.multimodal_gen.runtime.vsr.cli import main as restore

    if "--via-pipeline" in args.passthrough:
        raise ValueError("Dump hooks are process-local; use the direct CLI for dumps")
    try:
        return restore(["restore", *args.passthrough])
    finally:
        dumper.manifest(
            {
                "candidate_git": _git_state(Path.cwd()),
                "candidate_argv": args.passthrough,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "python": sys.version.split()[0],
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
