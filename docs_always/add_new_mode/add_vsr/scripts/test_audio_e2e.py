# SPDX-License-Identifier: Apache-2.0
"""Real-model audio acceptance for direct/native CLI and a running VSR server."""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--wan-root")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=1.7",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.7",
            "-itsoffset",
            "0.3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.8",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-c:a:0",
            "aac",
            "-c:a:1",
            "pcm_s16le",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=zho",
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            str(source),
        ],
        check=True,
    )
    records = []

    def inspect(path, audio_count):
        streams = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["streams"]
        video = streams[0]
        audio = [s for s in streams if s["codec_type"] == "audio"]
        assert int(video["nb_frames"]) == 17, streams
        assert (video["height"], video["width"]) == (64, 64)
        assert len(audio) == audio_count, streams
        if audio_count:
            assert [s["codec_name"] for s in audio] == ["aac", "aac"]
            assert [s["tags"]["language"] for s in audio] == ["eng", "zho"]
            assert [s["disposition"]["default"] for s in audio] == [0, 1]
            assert 0.24 < float(audio[1]["start_time"]) < 0.34
        pixels = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "framemd5",
                "-",
            ]
        )
        record = {
            "path": str(path),
            "audio_tracks": len(audio),
            "video_framemd5_sha256": hashlib.sha256(pixels).hexdigest(),
            "streams": streams,
        }
        records.append(record)
        (
            root / ("api_results.json" if args.base_url else "cli_results.json")
        ).write_text(json.dumps(records, indent=2))
        print(f"PASS {path.name}: {len(audio)} audio tracks", flush=True)

    if not args.base_url:
        if not args.checkpoint_dir or not args.wan_root:
            parser.error("CLI tests need --checkpoint-dir and --wan-root")
        for name, extra, count in [
            ("cli_audio", [], 2),
            ("cli_silent", ["--no-preserve-audio"], 0),
            ("native_audio", ["--via-pipeline"], 2),
        ]:
            output = root / f"{name}.mp4"
            print(f"START {name}", flush=True)
            with (root / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "sglang.multimodal_gen.runtime.vsr.cli",
                        "restore",
                        "--checkpoint_dir",
                        args.checkpoint_dir,
                        "--wan_root",
                        args.wan_root,
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--target_resolution",
                        "64x64",
                        "--tile_t",
                        "17",
                        "--tile_h",
                        "64",
                        "--tile_w",
                        "64",
                        "--temporal_overlap",
                        "0",
                        "--spatial_overlap",
                        "0",
                        "--color_ref",
                        "none",
                        *extra,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            inspect(output, count)
        assert (
            records[0]["video_framemd5_sha256"] == records[1]["video_framemd5_sha256"]
        )
        return

    opener = build_opener(ProxyHandler({}))
    callbacks = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(source.read_bytes())

        def do_POST(self):
            callbacks.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    local_url = f"http://127.0.0.1:{http.server_port}"

    def call(path, data=None):
        request = Request(
            args.base_url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=30) as response:
            return json.load(response)

    try:
        for name, extra, count in [
            ("api_local", {"video_input_path": str(source)}, 2),
            ("api_url", {"videoUrl": local_url + "/source.mkv"}, 2),
            (
                "api_silent",
                {"video_input_path": str(source), "preserve_audio": False},
                0,
            ),
        ]:
            job_id = f"{name}_{time.time_ns()}"
            result = call(
                "/v1/videos/restorations",
                {
                    "taskId": job_id,
                    "target_resolution": "64x64",
                    "tile_t": 17,
                    "tile_h": 64,
                    "tile_w": 64,
                    "temporal_overlap": 0,
                    "spatial_overlap": 0,
                    "color_ref": "none",
                    "callbackUrl": local_url,
                    **extra,
                },
            )
            assert result["code"] == 0, result
            print(f"START {name}: {job_id}", flush=True)
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                job = call(f"/v1/videos/{job_id}")
                if job["status"] in ("completed", "failed"):
                    break
                time.sleep(2)
            assert job["status"] == "completed", job
            output = root / f"{name}.mp4"
            with opener.open(
                args.base_url + f"/v1/videos/{job_id}/content", timeout=30
            ) as response:
                output.write_bytes(response.read())
            inspect(output, count)
            deadline = time.monotonic() + 30
            while (
                not any(c.get("id") == job_id for c in callbacks)
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            assert any(
                c.get("id") == job_id and c.get("status") == "completed"
                for c in callbacks
            ), callbacks
        assert len({r["video_framemd5_sha256"] for r in records}) == 1
        (root / "callbacks.json").write_text(json.dumps(callbacks, indent=2))
    finally:
        http.shutdown()
        http.server_close()


if __name__ == "__main__":
    main()
