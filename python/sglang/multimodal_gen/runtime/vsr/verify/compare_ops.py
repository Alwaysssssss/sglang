# SPDX-License-Identifier: Apache-2.0
"""Compare the ported pure functions against the reference, function by function.

The geometry, blending and colour helpers are pure functions, so their
correctness is not inferred from matching end-to-end output -- it is checked
directly, on the same inputs, against the reference implementation. That covers
``requirements.md`` §4.1.3 sampling point 1 (decode/resize/pad/tile positions)
and the weight arithmetic of point 4, and it needs no GPU at all.

Structural agreement (shape, dtype) is required exactly; values are required to
be bit-identical, because these are the same operations on the same inputs and
any difference at all means the port changed something.

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.compare_ops [--vsr-repo DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch


class Case:
    """One function pair and the inputs to try them on."""

    def __init__(self, name: str, ours: Callable, theirs: Callable, inputs: List[tuple]):
        self.name = name
        self.ours = ours
        self.theirs = theirs
        self.inputs = inputs


def _flatten(x) -> torch.Tensor:
    """Normalise a result to a tensor for comparison (lists, tuples, numpy)."""
    import numpy as np

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).reshape(-1)
    if isinstance(x, (list, tuple)):
        parts = [_flatten(p) for p in x]
        return torch.cat(parts) if parts else torch.empty(0)
    return torch.tensor([float(x)], dtype=torch.float64)


def build_cases(ref_tiling, ref_video_io, ref_stream, ours) -> List[Case]:
    geometry, blending, color, video_io = (
        ours["geometry"], ours["blending"], ours["color"], ours["video_io"]
    )
    torch.manual_seed(0)

    cases: List[Case] = []

    # --- resolution parsing / long-edge -----------------------------------
    cases.append(Case(
        "parse_resolution",
        geometry.parse_resolution,
        ref_video_io.parse_resolution,
        [("3840x2160",), ("1080x1920",), ("320X640",), ("2x2",)],
    ))
    cases.append(Case(
        "resize_to_long_edge",
        geometry.resize_to_long_edge,
        ref_video_io.resize_to_long_edge,
        [(1918, 1080, 3840), (480, 832, 3840), (1918, 1080, 1920),
         (1080, 1918, 3840), (3, 5, 64), (1918, 1080, 7)],
    ))

    # --- resize ------------------------------------------------------------
    frames = {
        "f_small": torch.randn(1, 3, 5, 24, 40),
        "f_time_chunk": torch.randn(1, 3, 9, 16, 32),   # exercises the 4-frame chunking
        "f_noop": torch.randn(1, 3, 3, 32, 32),
    }
    cases.append(Case(
        "resize_video",
        geometry.resize_video,
        ref_video_io.resize_video,
        [(frames["f_small"], 12, 20), (frames["f_time_chunk"], 8, 16),
         (frames["f_noop"], 32, 32), (frames["f_small"], 48, 80)],
    ))

    # --- temporal reflect padding -----------------------------------------
    cases.append(Case(
        "reflect_pad_time",
        geometry.reflect_pad_time,
        ref_video_io.reflect_pad_time,
        [(torch.randn(1, 3, 5, 4, 4), 13), (torch.randn(1, 3, 1, 4, 4), 7),
         (torch.randn(1, 3, 4, 4, 4), 2), (torch.randn(1, 3, 9, 4, 4), 9)],
    ))

    # --- tile positions ----------------------------------------------------
    # T=64 with tile 33/overlap 5 is the triple-coverage case from §7-4 and must
    # be in here; the tail-shift rule shows up as an overlap larger than asked.
    pos_inputs = [
        (100, 33, 5), (64, 33, 5), (53, 33, 5), (33, 33, 5), (10, 33, 5),
        (1, 33, 5), (62, 33, 5), (200, 33, 5), (320, 320, 32), (1088, 320, 32),
        (512, 640, 32), (2176, 640, 32), (37, 7, 3), (36, 7, 3),
    ]
    cases.append(Case(
        "compute_tile_positions",
        geometry.compute_tile_positions,
        ref_tiling.compute_tile_positions,
        pos_inputs,
    ))

    # --- spatial feather weights ------------------------------------------
    axis_inputs = [
        (32, 5, True, True), (32, 5, False, True), (32, 0, True, True),
        (5, 5, True, True), (3, 8, True, True), (33, 30, True, False),
    ]
    cases.append(Case(
        "axis_weight",
        lambda *a: blending.axis_weight(a[0], a[1], a[2], a[3], dtype=torch.float64),
        lambda *a: ref_tiling._axis_weight(
            a[0], a[1], a[2], a[3], device=torch.device("cpu"), dtype=torch.float64),
        axis_inputs,
    ))

    mask_inputs = [
        (4, 6, 8, 2, 2, (True, True), (False, True), (True, False)),
        (33, 20, 40, 5, 32, (False, False), (False, True), (True, False)),
        (3, 3, 3, 1, 1, (True, True), (True, True), (True, True)),
    ]
    cases.append(Case(
        "build_blend_mask_3d",
        lambda t, h, w, to, so, rt, rh, rw: blending.build_blend_mask_3d(
            t, h, w, to, so, rt, rh, rw, dtype=torch.float64),
        lambda t, h, w, to, so, rt, rh, rw, **kw: ref_tiling.build_blend_mask_3d(
            t, h, w, to, so, rt, rh, rw,
            device=torch.device("cpu"), dtype=torch.float64),
        mask_inputs,
    ))

    # --- temporal feather weights -----------------------------------------
    # ov == 1 is the guarded case: no ramp, or the frame is zeroed from both
    # sides by a lone linspace(0, 1, 1) == 0.0.
    tw_inputs = [
        (33, 5, 5), (33, 0, 30), (33, 30, 0), (33, 1, 1), (33, 1, 0),
        (33, 0, 1), (10, 10, 10), (33, 0, 0), (33, 32, 32),
    ]
    cases.append(Case(
        "temporal_weight",
        lambda t, a, b: blending.temporal_weight(t, a, b, torch.float64),
        lambda t, a, b: ref_stream._temporal_weight(t, a, b, torch.float64),
        tw_inputs,
    ))

    # --- spatial tiling + blending driver ----------------------------------
    # A deterministic stand-in for the model call keeps this on CPU: both sides
    # drive the *same* function, so any difference is in the tiling, padding,
    # weighting or normalisation -- never in the model.
    def _fake_restore(w):
        return (w * 0.5 + 1.0) * (1.0 + 0.01 * w)

    tile_inputs = [
        # (frames, tile_t, tile_h, tile_w, t_overlap, s_overlap)
        (torch.randn(1, 3, 9, 20, 30), 5, 8, 10, 2, 3),
        (torch.randn(1, 3, 4, 16, 16), 4, 16, 16, 0, 0),
        (torch.randn(1, 3, 3, 10, 12), 7, 12, 14, 3, 4),   # both axes padded
        (torch.randn(1, 3, 6, 8, 8), 3, 5, 5, 1, 1),
    ]
    cases.append(Case(
        "tiled_restore_rect",
        lambda f, tt, th, tw, to, so: blending.tiled_restore_rect(
            f, _fake_restore, tile_t=tt, tile_h=th, tile_w=tw,
            t_overlap=to, s_overlap=so, show_progress=False),
        lambda f, tt, th, tw, to, so: ref_tiling.tiled_restore_rect(
            f, _fake_restore, tile_t=tt, tile_h=th, tile_w=tw,
            t_overlap=to, s_overlap=so, show_progress=False),
        tile_inputs,
    ))

    # --- colour -------------------------------------------------------------
    vol = torch.randn(1, 3, 4, 6, 8) * 0.5
    ref = torch.randn(1, 3, 4, 6, 8) * 0.3 - 0.1
    cases.append(Case("color_stats[0]", lambda v: color.color_stats(v)[0], lambda v: ref_video_io.color_stats(v)[0], [(vol,)]))
    cases.append(Case("color_stats[1]", lambda v: color.color_stats(v)[1], lambda v: ref_video_io.color_stats(v)[1], [(vol,)]))
    cases.append(Case(
        "match_color",
        lambda a, b: color.match_color(a, b),
        lambda a, b: ref_video_io.match_color(a, b),
        [(vol, ref), (ref, vol)],
    ))
    rm, rs = ref_video_io.color_stats(ref)
    cases.append(Case(
        "match_color_to_stats",
        lambda o, m, s: color.match_color_to_stats(o, m, s),
        lambda o, m, s: ref_video_io.match_color_to_stats(o, m, s),
        [(vol, rm, rs)],
    ))

    # --- quantisation -------------------------------------------------------
    # 0.6 -> 0 under truncation, 1 under rounding: the two differ on ~half of
    # all pixels, which is why this is compared bit-for-bit.
    q = torch.tensor([[[[[-1.0, -0.5, 0.0, 0.2, 0.6, 0.999, 1.0]]]]])
    cases.append(Case(
        "to_uint8_hwc",
        video_io.to_uint8_hwc,
        ref_video_io.to_uint8_hwc,
        [(q,), (torch.rand(1, 3, 2, 5, 7) * 2 - 1,)],
    ))

    return cases


def run(cases: List[Case]) -> Tuple[List[dict], int]:
    results: List[dict] = []
    failures = 0

    for case in cases:
        for n, args in enumerate(case.inputs):
            try:
                got = _flatten(case.ours(*args))
                want = _flatten(case.theirs(*args))
            except Exception as exc:  # pragma: no cover - surfaces as a failure row
                results.append({"case": case.name, "input": n, "ok": False,
                                "detail": f"{type(exc).__name__}: {exc}"})
                failures += 1
                continue

            if got.shape != want.shape:
                results.append({"case": case.name, "input": n, "ok": False,
                                "detail": f"shape {tuple(got.shape)} vs {tuple(want.shape)}"})
                failures += 1
                continue

            max_abs = float((got - want).abs().max()) if got.numel() else 0.0
            dtype_ok = got.numel() == want.numel()
            ok = max_abs == 0.0 and dtype_ok
            if not ok:
                failures += 1
            results.append({"case": case.name, "input": n, "ok": ok, "max_abs": max_abs,
                            "detail": "" if ok else f"max_abs={max_abs:g}"})

    return results, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ported ops against the reference.")
    parser.add_argument("--vsr-repo", default=os.environ.get("VSR_REPO"))
    args = parser.parse_args(argv)
    if not args.vsr_repo:
        parser.error("--vsr-repo is required (or set VSR_REPO)")

    sys.path.insert(0, str(Path(args.vsr_repo).resolve()))
    import infer.stream as ref_stream
    import infer.utils.tiling as ref_tiling
    import infer.utils.video_io as ref_video_io

    from sglang.multimodal_gen.runtime.vsr import blending, color, geometry, video_io

    ours = {"geometry": geometry, "blending": blending, "color": color, "video_io": video_io}
    cases = build_cases(ref_tiling, ref_video_io, ref_stream, ours)
    results, failures = run(cases)

    by_case: dict = {}
    for r in results:
        entry = by_case.setdefault(r["case"], {"n": 0, "bad": 0, "worst": 0.0, "detail": ""})
        entry["n"] += 1
        if not r["ok"]:
            entry["bad"] += 1
            entry["detail"] = r["detail"]
        entry["worst"] = max(entry["worst"], r.get("max_abs", 0.0))

    for name, e in by_case.items():
        mark = "PASS" if e["bad"] == 0 else "FAIL"
        extra = f" worst_max_abs={e['worst']:g}" if e["worst"] else " (bit-identical)"
        fail = f"  [{e['bad']}/{e['n']} bad: {e['detail']}]" if e["bad"] else ""
        print(f"  [{mark}] {name}: {e['n']} inputs{extra}{fail}")

    total = len(results)
    print(f"\n{total - failures}/{total} comparisons bit-identical")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
