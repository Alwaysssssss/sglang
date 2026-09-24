"""Run validation with existing packages only; no environment mutation.

The mounted workspace cannot unlock an already unlinked inode. The existing
system filelock 3.18 keeps lock files until after unlocking, unlike the version
in VE_PACKAGES. Select that one package explicitly, including spawned workers.
All model dependencies still resolve through the caller's VE_PACKAGES path.
"""

import importlib.util
import sys
from pathlib import Path


def load_existing_filelock():
    path = Path("/opt/conda/lib/python3.11/site-packages/filelock/__init__.py")
    spec = importlib.util.spec_from_file_location(
        "filelock", path, submodule_search_locations=[str(path.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load existing filelock: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["filelock"] = module
    spec.loader.exec_module(module)


load_existing_filelock()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-runtime"]:
        import os

        import torch
        from filelock import FileLock
        from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
        from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
            apply_flashinfer_rope_qk_inplace,
        )

        with FileLock(Path(os.environ["TMPDIR"]) / "runtime-check.lock"):
            pass
        x = torch.zeros(1, 2, 5120, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(LayerNormScaleShift(5120)(x, x, x), x)
        q = torch.zeros(1, 2, 40, 128, device="cuda", dtype=torch.bfloat16)
        cache = torch.ones(2, 128, device="cuda", dtype=torch.float32)
        rotated_q, rotated_k = apply_flashinfer_rope_qk_inplace(
            q, q.clone(), cache, is_neox=False
        )
        assert torch.equal(rotated_q, q) and torch.equal(rotated_k, q)
        print("PASS: file locking, fused LayerNorm, FlashInfer RoPE")
    else:
        from sglang.multimodal_gen.runtime.videoedit.cli import main

        raise SystemExit(main())
