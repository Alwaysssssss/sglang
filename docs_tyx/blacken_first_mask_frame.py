#!/usr/bin/env python
"""Set the first frame of a mask video to black.

Default:
  input : /mnt/nas/tyx/tmp/mask_1080_acc.mp4
  output: /mnt/nas/tyx/tmp/mask_1080_acc_first_black.mp4

You can override paths with --input and --output.
"""

from __future__ import print_function

import argparse
import os
import subprocess


DEFAULT_INPUT = "/mnt/nas/tyx/tmp/mask_1080_acc.mp4"
DEFAULT_OUTPUT = "/mnt/nas/tyx/tmp/mask_1080_acc_first_black.mp4"


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode a mask video with frame 0 forced to all black."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input mask video path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output mask video path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crf", default="18")
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise IOError("Input mask video does not exist: %s" % args.input)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Draw a full-frame black box only on frame index 0. Frame count and timing
    # are kept unchanged; audio, if present, is copied.
    run([
        "ffmpeg",
        "-y" if args.overwrite else "-n",
        "-i", args.input,
        "-vf", "drawbox=enable=eq(n\\,0):x=0:y=0:w=iw:h=ih:color=black:t=fill",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", str(args.preset),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        args.output,
    ])
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
