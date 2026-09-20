# SPDX-License-Identifier: Apache-2.0
"""Run the reference VSR streaming CLI and dump its intermediate tensors.

The reference repository is read-only, so this is a pure observation layer: it
imports ``infer`` from ``$VSR_REPO``, wraps the functions the streaming path
calls, and then hands control to the reference CLI's own ``main()``. The
reference's control flow is never modified -- only its return values are
copied on the way past.

Dump points follow ``docs_always/add_new_mode/add_vsr/requirements.md`` §4.1.3:

    1. decoded / resized / padded input and the per-tile window
    2. VAE-encoded latent
    3. DiT velocity and VAE-decoded pixels
    4. spatial fusion, temporal fusion and tail commit
    5. colour correction and the pre-encode uint8 frames

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.dump_baseline \\
        --dump-root DIR [--dump-frames] [--dump-tiles none|first|all] [--dump-chunks] \\
        -- <the reference CLI's own arguments>

Everything after ``--`` is passed through verbatim, so the dumped run and the
plain reference run are driven by an identical command line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

#: Human-readable names for the dump points, keyed by directory name.
DUMP_POINTS = {
    "chunk_input": "1. decoded/resized/padded input chunk",
    "window": "1. per-tile window (after replicate pad)",
    "latent": "2. VAE-encoded latent (normalised)",
    "velocity": "3. DiT velocity",
    "decoded": "3. VAE-decoded tile",
    "spatial_fused": "4a. spatial fusion result (per chunk)",
    "retired": "4b. retired frames after temporal blend, pre-encode uint8",
    "color_stats": "5. colour reference vs measured per-channel stats",
}


class _Dumper:
    """Collects dumped tensors and writes the manifest.

    All hooks run on the main thread -- the reference's reader and writer
    workers never call any of the wrapped functions -- so no locking is needed.
    """

    def __init__(
        self, root: Path, dump_frames: bool, dump_tiles: str, dump_chunks: bool
    ):
        self.root = root
        self.dump_frames = dump_frames
        self.dump_tiles = dump_tiles
        self.dump_chunks = dump_chunks
        self.tile_calls = 0
        #: Index of the tile currently being restored; set by the
        #: ``_restore_window`` hook and read by the hooks nested inside it.
        self.current_tile = 0
        self.chunk_calls = 0
        self.frame_parts = 0
        self.frames_written = 0
        self.shapes: dict[str, Any] = {}
        self.color_stats: list[dict[str, Any]] = []
        for name in DUMP_POINTS:
            (root / name).mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------
    def want_tile(self, idx: int) -> bool:
        """Whether per-tile tensors should be written for tile ``idx``."""
        return self.dump_tiles == "all" or (self.dump_tiles == "first" and idx == 0)

    def _record(self, name: str, t: torch.Tensor) -> None:
        entry = self.shapes.setdefault(
            name,
            {
                "shape": None,
                "dtype": None,
                "count": 0,
                "distinct_shapes": [],
                "devices": [],
                "finite": True,
            },
        )
        entry["finite"] = entry["finite"] and bool(torch.isfinite(t).all())
        if str(t.device) not in entry["devices"]:
            entry["devices"].append(str(t.device))
        shape = list(t.shape)
        entry["shape"] = shape  # the most recent write
        entry["dtype"] = str(t.dtype)
        entry["count"] += 1
        if not entry["distinct_shapes"] or entry["distinct_shapes"][-1] != shape:
            entry["distinct_shapes"].append(shape)

    def save(self, name: str, t: torch.Tensor, tag: str) -> None:
        self._record(name, t)
        torch.save(t.detach().to("cpu").clone(), self.root / name / f"{tag}.pt")

    def manifest(self, extra: dict[str, Any]) -> None:
        payload = {
            "dump_points": DUMP_POINTS,
            "shapes": self.shapes,
            "frames_written": self.frames_written,
            "color_stats": self.color_stats,
            "config": {
                "dump_frames": self.dump_frames,
                "dump_tiles": self.dump_tiles,
                "dump_chunks": self.dump_chunks,
            },
            **extra,
        }
        (self.root / "manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _git_state(repo: Path) -> dict[str, str]:
    """Source version of the reference, per requirements §3."""
    import subprocess

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        except (
            OSError,
            subprocess.SubprocessError,
        ) as exc:  # pragma: no cover - diagnostic only
            return f"<unavailable: {exc}>"

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
        "describe": run("describe", "--always", "--dirty"),
    }


def install_patches(
    dumper: _Dumper, vsr_repo: Path, *, candidate: bool = False
) -> None:
    """Wrap the reference's streaming-path functions with dumping hooks."""
    if candidate:
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.vsr import model, stream

        stage3 = SimpleNamespace(
            Stage3VAE=model.VSRAutoencoder, Stage3Pipeline=model.VSRRestorer
        )
        window_method = "restore_window"
    else:
        sys.path.insert(0, str(vsr_repo))
        from infer import stream
        from infer.models import stage3

        window_method = "_restore_window"

    # --- 2 & 3: latent and decoded pixels ---------------------------------
    orig_encode = stage3.Stage3VAE.encode
    orig_decode = stage3.Stage3VAE.decode

    def encode(self, frames):
        out = orig_encode(self, frames)
        if dumper.want_tile(dumper.current_tile):
            dumper.save("latent", out, f"tile_{dumper.current_tile:05d}")
        return out

    def decode(self, latent):
        out = orig_decode(self, latent)
        if dumper.want_tile(dumper.current_tile):
            dumper.save("decoded", out, f"tile_{dumper.current_tile:05d}")
        return out

    stage3.Stage3VAE.encode = encode
    stage3.Stage3VAE.decode = decode

    # --- 1 & 3: per-tile window and DiT velocity ---------------------------
    orig_restore_window = getattr(stage3.Stage3Pipeline, window_method)

    def restore_window(self, window):
        dumper.current_tile = dumper.tile_calls
        dumper.tile_calls += 1
        if dumper.want_tile(dumper.current_tile):
            dumper.save("window", window, f"tile_{dumper.current_tile:05d}")
        return orig_restore_window(self, window)

    setattr(stage3.Stage3Pipeline, window_method, restore_window)

    # The DiT is called as ``self.dit(...)``, so wrapping the *instance*
    # attribute is enough: nn.Module.__call__ resolves ``self.forward``
    # through the instance dict, and our closure already holds the bound
    # original.
    orig_from_pretrained = stage3.Stage3Pipeline.from_pretrained

    def from_pretrained(cls, *args, **kwargs):
        pipeline = orig_from_pretrained.__func__(cls, *args, **kwargs)
        orig_dit_forward = pipeline.dit.forward

        def dit_forward(*a, **kw):
            out = orig_dit_forward(*a, **kw)
            if dumper.want_tile(dumper.current_tile):
                velocity = out[0] if isinstance(out, (tuple, list)) else out
                dumper.save("velocity", velocity, f"tile_{dumper.current_tile:05d}")
            return out

        pipeline.dit.forward = dit_forward
        return pipeline

    stage3.Stage3Pipeline.from_pretrained = classmethod(from_pretrained)

    # --- 1 & 4a: chunk input and spatial fusion result ---------------------
    orig_tiled = stream.tiled_restore_rect

    def tiled_restore_rect(frames, *args, **kwargs):
        idx = dumper.chunk_calls
        if dumper.dump_chunks:
            dumper.save("chunk_input", frames, f"chunk_{idx:03d}")
        out = orig_tiled(frames, *args, **kwargs)
        if dumper.dump_chunks:
            dumper.save("spatial_fused", out, f"chunk_{idx:03d}")
        dumper.chunk_calls += 1
        return out

    stream.tiled_restore_rect = tiled_restore_rect

    # --- 5: colour correction ---------------------------------------------
    def _stats(v: torch.Tensor) -> dict[str, list[float]]:
        v = v.float()
        return {
            "mean": v.mean(dim=(0, 2, 3, 4)).tolist(),
            "std": v.std(dim=(0, 2, 3, 4)).tolist(),
        }

    orig_to_stats = stream.match_color_to_stats

    def match_color_to_stats(out, ref_mean, ref_std, *a, **kw):
        res = orig_to_stats(out, ref_mean, ref_std, *a, **kw)
        dumper.color_stats.append(
            {
                "mode": "global",
                "chunk": len(dumper.color_stats),
                "ref_mean": ref_mean.flatten().tolist(),
                "ref_std": ref_std.flatten().tolist(),
                "in": _stats(out),
                "out": _stats(res),
            }
        )
        return res

    orig_match_color = stream.match_color

    def match_color(out, ref, *a, **kw):
        res = orig_match_color(out, ref, *a, **kw)
        dumper.color_stats.append(
            {
                "mode": "chunk",
                "chunk": len(dumper.color_stats),
                "in": _stats(out),
                "ref": _stats(ref),
                "out": _stats(res),
            }
        )
        return res

    stream.match_color_to_stats = match_color_to_stats
    stream.match_color = match_color

    # --- 4b & 5: pre-encode uint8 frames -----------------------------------
    orig_to_uint8 = stream.to_uint8_hwc

    def to_uint8_hwc(video):
        arr = orig_to_uint8(video)
        if dumper.dump_frames:
            t = torch.from_numpy(arr)
            part = dumper.root / "retired" / f"part_{dumper.frame_parts:03d}.pt"
            torch.save(t, part)
            dumper.frame_parts += 1
            dumper.frames_written += int(t.shape[0])
            dumper._record("retired", t)
        return arr

    stream.to_uint8_hwc = to_uint8_hwc


def _load_reference_cli(vsr_repo: Path):
    """Import the reference CLI module without executing it."""
    import importlib.util

    path = vsr_repo / "run_inference_stream.py"
    spec = importlib.util.spec_from_file_location("_vsr_reference_cli", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load reference CLI from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        cut = raw.index("--")
        ours, passthrough = raw[:cut], raw[cut + 1 :]
    else:
        ours, passthrough = [], raw

    parser = argparse.ArgumentParser(
        description="Run the reference VSR streaming CLI with intermediate dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vsr-repo",
        default=os.environ.get("VSR_REPO"),
        help="Reference repository root (read-only)",
    )
    parser.add_argument("--dump-root", required=True, help="Where to write dumps")
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="Dump the pre-encode uint8 frames (§4.2.1)",
    )
    parser.add_argument(
        "--dump-tiles",
        choices=["none", "first", "all"],
        default="none",
        help="Dump per-tile tensors: window / latent / velocity / decoded",
    )
    parser.add_argument(
        "--dump-chunks",
        action="store_true",
        help="Dump full chunk tensors; large, only for small geometries",
    )
    args = parser.parse_args(ours)

    if not args.vsr_repo:
        parser.error("--vsr-repo is required (or set VSR_REPO)")
    args.vsr_repo = Path(args.vsr_repo).resolve()
    args.dump_root = Path(args.dump_root).resolve()
    args.passthrough = passthrough
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dump_root.mkdir(parents=True, exist_ok=True)

    dumper = _Dumper(
        args.dump_root, args.dump_frames, args.dump_tiles, args.dump_chunks
    )
    install_patches(dumper, args.vsr_repo)

    cli = _load_reference_cli(args.vsr_repo)
    try:
        cli.main(args.passthrough)
    finally:
        dumper.manifest(
            {
                "reference_repo": str(args.vsr_repo),
                "reference_git": _git_state(args.vsr_repo),
                "reference_argv": args.passthrough,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "python": sys.version.split()[0],
            }
        )
    print(
        f"[dump] wrote {dumper.frames_written} frames and manifest to {args.dump_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
