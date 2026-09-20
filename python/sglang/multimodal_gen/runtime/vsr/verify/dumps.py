# SPDX-License-Identifier: Apache-2.0
"""Frame-dump helpers with no SGLang dependency.

The acceptance tools have to run in *both* environments: the reference side
lives in a conda env that has no ``sglang`` installed, so anything the
producers import must be importable there too. This module deliberately depends
on nothing but ``torch`` -- keep it that way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import torch


def list_parts(dump_dir) -> List[Path]:
    """The ``retired/part_*.pt`` files of a dump, in write order."""
    part_dir = Path(dump_dir) / "retired"
    parts = sorted(part_dir.glob("part_*.pt"))
    if not parts:
        raise FileNotFoundError(
            f"no frame dumps under {part_dir}; run dump_baseline.py with --dump-frames"
        )
    return parts


def load_frames(dump_dir) -> torch.Tensor:
    """Concatenate a dump's frame parts into ``[T, H, W, C]`` uint8."""
    frames = [torch.load(p, map_location="cpu") for p in list_parts(dump_dir)]
    return torch.cat(frames, dim=0)


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
