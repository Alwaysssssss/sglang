# SPDX-License-Identifier: Apache-2.0
"""Re-encode a frame dump with the reference writer, for the §4.2.3 check.

``requirements.md`` §4.2.3 asks one question: does the encoder turn identical
frames into identical bytes? If it does, then a difference between two mp4s
implies a difference between the frames that produced them, which is what makes
the mp4 threshold in §4.2.2 interpretable.

The writer is imported from the reference repo rather than reimplemented, so
the codec, pixel format, CRF and ``macro_block_size`` are exactly the ones the
reference run used -- otherwise this would be measuring the writer instead of
the encoder.

Run it once per environment and compare the outputs byte-for-byte.

Usage:
    python -m sglang.multimodal_gen.runtime.vsr.verify.encode_frames \\
        --dump-dir DIR --output OUT.mp4 --fps 25 --crf 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dumps import load_frames, md5  # noqa: E402  (works when run as a plain script)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-encode a frame dump with the reference writer.")
    parser.add_argument("--dump-dir", required=True, help="Dump produced by dump_baseline.py --dump-frames")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--crf", type=int, default=5)
    parser.add_argument("--vsr-repo", default=os.environ.get("VSR_REPO"),
                        help="Reference repository root, for its writer (or set VSR_REPO)")
    args = parser.parse_args(argv)

    if not args.vsr_repo:
        parser.error("--vsr-repo is required (or set VSR_REPO)")
    sys.path.insert(0, str(Path(args.vsr_repo).resolve()))
    from infer.utils.video_io import write_video  # the reference's own writer

    frames = load_frames(Path(args.dump_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out), frames.numpy(), fps=args.fps, crf=args.crf)

    print(f"encoded {frames.shape[0]} frames -> {out}")
    print(f"md5 {md5(out)}  size {out.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
