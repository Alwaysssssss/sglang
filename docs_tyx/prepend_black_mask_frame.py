#!/usr/bin/env python
"""Prepend one all-black frame to a mask video.

Default:
  input : /mnt/nas/tyx/tmp/mask_1080_acc.mp4
  output: /mnt/nas/tyx/tmp/mask_1080_acc_with_black_first.mp4

You can override paths with --input and --output.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
from fractions import Fraction


DEFAULT_INPUT = "/mnt/nas/tyx/tmp/mask_1080_acc.mp4"
DEFAULT_OUTPUT = "/mnt/nas/tyx/tmp/mask_1080_acc_with_black_first.mp4"


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def run_capture(cmd):
    print(" ".join(cmd))
    return subprocess.check_output(cmd)


def probe_video(video_path):
    output = run_capture([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json",
        video_path,
    ])
    if not isinstance(output, str):
        output = output.decode("utf-8")
    streams = json.loads(output).get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found in %s" % video_path)
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1"
    if fps in ("0/0", "0"):
        fps = stream.get("r_frame_rate") or "25/1"
    Fraction(fps)
    return width, height, fps


def main():
    parser = argparse.ArgumentParser(
        description="Prepend exactly one all-black frame to a mask video."
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

    width, height, fps = probe_video(args.input)
    black_source = "color=c=black:s=%dx%d:r=%s" % (width, height, fps)
    filter_complex = (
        "[0:v]trim=end_frame=1,setpts=PTS-STARTPTS,format=yuv420p[black];"
        "[1:v]scale=%d:%d,setsar=1,format=yuv420p,fps=%s,setpts=PTS-STARTPTS[mask];"
        "[black][mask]concat=n=2:v=1:a=0[v]"
    ) % (width, height, fps)

    run([
        "ffmpeg",
        "-y" if args.overwrite else "-n",
        "-f", "lavfi",
        "-i", black_source,
        "-i", args.input,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-crf", str(args.crf),
        "-preset", str(args.preset),
        "-pix_fmt", "yuv420p",
        args.output,
    ])
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
