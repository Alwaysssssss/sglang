import json
from pathlib import Path

import pytest
import sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark as benchmark_tool
from sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark import (
    CaptionSidecarBenchmarkResult,
    benchmark_caption_sidecar,
    main,
    parse_args,
)


def test_parse_args_supports_explicit_paths_and_defaults():
    args = parse_args(
        [
            "--video-path",
            "/tmp/input.mp4",
            "--baseline-caption-path",
            "/tmp/baseline.txt",
        ]
    )

    assert args.video_path == "/tmp/input.mp4"
    assert args.baseline_caption_path == "/tmp/baseline.txt"
    assert args.sidecar_base_url == "http://127.0.0.1:31200"
    assert args.sidecar_timeout_s == 1800.0
    assert args.num_temporal_process_frames == 121
    assert args.tile_size == 128
    assert args.tile_stride == 64
    assert args.work_dir == Path("Vivid_Acceptance/caption_sidecar_benchmark")
    assert args.manifest_path is None
    assert args.output_caption_path is None
    assert args.metrics_json_path is None


def test_benchmark_caption_sidecar_builds_manifest_requests_sidecar_and_writes_metrics(
    tmp_path, monkeypatch
):
    calls = {}
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("caption 0\ncaption 1\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    output_caption_path = tmp_path / "generated.txt"
    metrics_json_path = tmp_path / "metrics.json"

    class FakeManifest:
        expected_caption_count = 2
        clips = [object(), object()]

        def write_json(self, path):
            calls["manifest_write_path"] = str(path)
            Path(path).write_text('{"version": 1}\n', encoding="utf-8")

    async def fake_request_vividvr_caption_sidecar(
        *,
        config,
        manifest_path,
        output_caption_path,
        expected_caption_count,
    ):
        calls["request"] = {
            "base_url": config.base_url,
            "timeout_s": config.timeout_s,
            "manifest_path": manifest_path,
            "output_caption_path": output_caption_path,
            "expected_caption_count": expected_caption_count,
        }
        Path(output_caption_path).write_text(
            "caption 0\ncaption 1\n",
            encoding="utf-8",
        )
        return type(
            "FakeBridgeResult",
            (),
            {
                "caption_file_path": output_caption_path,
                "caption_count": 2,
                "mode": "parallel",
                "worker_count": 2,
                "fallback_used": False,
                "request_id": "req-123",
                "total_clip_count": 2,
                "assigned_clip_indices_by_worker": {"0": [0], "1": [1]},
                "timing": {
                    "read_seconds": 0.4,
                    "write_seconds": 0.01,
                    "total_seconds": 1.2,
                    "worker_batches": [],
                },
            },
        )()

    def fake_build_manifest_for_video_path(**kwargs):
        calls["build_manifest"] = kwargs
        return FakeManifest()

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "build_vividvr_caption_manifest_for_video_path",
        fake_build_manifest_for_video_path,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "request_vividvr_caption_sidecar",
        fake_request_vividvr_caption_sidecar,
    )

    result = benchmark_caption_sidecar(
        video_path=str(tmp_path / "input.mp4"),
        baseline_caption_path=str(baseline_path),
        sidecar_base_url="http://127.0.0.1:31200",
        sidecar_timeout_s=30.0,
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
        manifest_path=str(manifest_path),
        output_caption_path=str(output_caption_path),
        metrics_json_path=str(metrics_json_path),
    )

    assert calls["build_manifest"] == {
        "video_path": str(tmp_path / "input.mp4"),
        "num_temporal_process_frames": 121,
        "tile_size": 128,
        "tile_stride": 64,
    }
    assert calls["manifest_write_path"] == str(manifest_path)
    assert calls["request"] == {
        "base_url": "http://127.0.0.1:31200",
        "timeout_s": 30.0,
        "manifest_path": str(manifest_path),
        "output_caption_path": str(output_caption_path),
        "expected_caption_count": 2,
    }
    assert result.captions_match is True
    assert result.first_mismatch_index is None
    assert result.generated_caption_count == 2
    assert result.baseline_caption_count == 2
    assert result.sidecar_mode == "parallel"
    assert result.sidecar_worker_count == 2
    assert result.sidecar_request_id == "req-123"
    assert result.sidecar_timing == {
        "read_seconds": 0.4,
        "write_seconds": 0.01,
        "total_seconds": 1.2,
        "worker_batches": [],
    }
    assert metrics_json_path.is_file()
    metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    assert metrics["captions_match"] is True
    assert metrics["expected_caption_count"] == 2
    assert metrics["sidecar_assigned_clip_indices_by_worker"] == {
        "0": [0],
        "1": [1],
    }
    assert metrics["sidecar_timing"]["total_seconds"] == 1.2


def test_benchmark_caption_sidecar_reports_exact_line_mismatch(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("caption 0\ncaption 1\n", encoding="utf-8")

    class FakeManifest:
        expected_caption_count = 2
        clips = [object(), object()]

        def write_json(self, path):
            Path(path).write_text('{"version": 1}\n', encoding="utf-8")

    async def fake_request_vividvr_caption_sidecar(
        *,
        config,
        manifest_path,
        output_caption_path,
        expected_caption_count,
    ):
        Path(output_caption_path).write_text(
            "caption 0\ncaption changed\n",
            encoding="utf-8",
        )
        return type(
            "FakeBridgeResult",
            (),
            {
                "caption_file_path": output_caption_path,
                "caption_count": 2,
                "mode": None,
                "worker_count": None,
                "fallback_used": None,
                "request_id": None,
                "total_clip_count": None,
                "assigned_clip_indices_by_worker": None,
                "timing": None,
            },
        )()

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "build_vividvr_caption_manifest_for_video_path",
        lambda **kwargs: FakeManifest(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "request_vividvr_caption_sidecar",
        fake_request_vividvr_caption_sidecar,
    )

    result = benchmark_caption_sidecar(
        video_path=str(tmp_path / "input.mp4"),
        baseline_caption_path=str(baseline_path),
        sidecar_base_url="http://127.0.0.1:31200",
        sidecar_timeout_s=30.0,
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
        manifest_path=str(tmp_path / "manifest.json"),
        output_caption_path=str(tmp_path / "generated.txt"),
    )

    assert result.captions_match is False
    assert result.first_mismatch_index == 1
    assert result.generated_caption_count == 2
    assert result.baseline_caption_count == 2


def test_benchmark_caption_sidecar_detects_raw_file_mismatch_even_when_lines_normalize(
    tmp_path, monkeypatch
):
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("caption 0\ncaption 1\n", encoding="utf-8")

    class FakeManifest:
        expected_caption_count = 2
        clips = [object(), object()]

        def write_json(self, path):
            Path(path).write_text('{"version": 1}\n', encoding="utf-8")

    async def fake_request_vividvr_caption_sidecar(
        *,
        config,
        manifest_path,
        output_caption_path,
        expected_caption_count,
    ):
        Path(output_caption_path).write_text(
            "caption 0\ncaption 1 \n",
            encoding="utf-8",
        )
        return type(
            "FakeBridgeResult",
            (),
            {
                "caption_file_path": output_caption_path,
                "caption_count": 2,
                "mode": None,
                "worker_count": None,
                "fallback_used": None,
                "request_id": None,
                "total_clip_count": None,
                "assigned_clip_indices_by_worker": None,
                "timing": None,
            },
        )()

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "build_vividvr_caption_manifest_for_video_path",
        lambda **kwargs: FakeManifest(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar_benchmark."
        "request_vividvr_caption_sidecar",
        fake_request_vividvr_caption_sidecar,
    )

    result = benchmark_caption_sidecar(
        video_path=str(tmp_path / "input.mp4"),
        baseline_caption_path=str(baseline_path),
        sidecar_base_url="http://127.0.0.1:31200",
        sidecar_timeout_s=30.0,
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
        manifest_path=str(tmp_path / "manifest.json"),
        output_caption_path=str(tmp_path / "generated.txt"),
    )

    assert result.captions_match is False
    assert result.first_mismatch_index is None
    assert result.generated_caption_count == 2
    assert result.baseline_caption_count == 2


def test_benchmark_main_exits_nonzero_when_captions_mismatch(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_tool,
        "benchmark_caption_sidecar",
        lambda **kwargs: CaptionSidecarBenchmarkResult(
            video_path="/tmp/input.mp4",
            manifest_path="/tmp/manifest.json",
            output_caption_path="/tmp/output.txt",
            baseline_caption_path="/tmp/baseline.txt",
            expected_caption_count=2,
            generated_caption_count=2,
            baseline_caption_count=2,
            captions_match=False,
            first_mismatch_index=1,
            elapsed_seconds=1.0,
            sidecar_mode="parallel",
            sidecar_worker_count=2,
            sidecar_fallback_used=False,
            sidecar_request_id="req-1",
            sidecar_total_clip_count=2,
            sidecar_assigned_clip_indices_by_worker={"0": [0], "1": [1]},
            sidecar_timing=None,
        ),
    )

    with pytest.raises(SystemExit, match="does not exactly match"):
        main(
            [
                "--video-path",
                "/tmp/input.mp4",
                "--baseline-caption-path",
                "/tmp/baseline.txt",
            ]
        )

    captured = capsys.readouterr()
    assert '"captions_match": false' in captured.out


def test_benchmark_main_exits_nonzero_when_parallel_request_falls_back(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_tool,
        "benchmark_caption_sidecar",
        lambda **kwargs: CaptionSidecarBenchmarkResult(
            video_path="/tmp/input.mp4",
            manifest_path="/tmp/manifest.json",
            output_caption_path="/tmp/output.txt",
            baseline_caption_path="/tmp/baseline.txt",
            expected_caption_count=2,
            generated_caption_count=2,
            baseline_caption_count=2,
            captions_match=True,
            first_mismatch_index=None,
            elapsed_seconds=1.0,
            sidecar_mode="serial",
            sidecar_worker_count=2,
            sidecar_fallback_used=True,
            sidecar_request_id="req-2",
            sidecar_total_clip_count=2,
            sidecar_assigned_clip_indices_by_worker={"0": [0], "1": [1]},
            sidecar_timing=None,
        ),
    )

    with pytest.raises(SystemExit, match="fell back to serial mode"):
        main(
            [
                "--video-path",
                "/tmp/input.mp4",
                "--baseline-caption-path",
                "/tmp/baseline.txt",
            ]
        )

    captured = capsys.readouterr()
    assert '"sidecar_fallback_used": true' in captured.out


def test_benchmark_main_exits_nonzero_when_response_is_not_parallel(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_tool,
        "benchmark_caption_sidecar",
        lambda **kwargs: CaptionSidecarBenchmarkResult(
            video_path="/tmp/input.mp4",
            manifest_path="/tmp/manifest.json",
            output_caption_path="/tmp/output.txt",
            baseline_caption_path="/tmp/baseline.txt",
            expected_caption_count=2,
            generated_caption_count=2,
            baseline_caption_count=2,
            captions_match=True,
            first_mismatch_index=None,
            elapsed_seconds=1.0,
            sidecar_mode="serial",
            sidecar_worker_count=2,
            sidecar_fallback_used=None,
            sidecar_request_id="req-3",
            sidecar_total_clip_count=2,
            sidecar_assigned_clip_indices_by_worker={"0": [0], "1": [1]},
            sidecar_timing=None,
        ),
    )

    with pytest.raises(SystemExit, match="did not remain on the parallel path"):
        main(
            [
                "--video-path",
                "/tmp/input.mp4",
                "--baseline-caption-path",
                "/tmp/baseline.txt",
            ]
        )

    captured = capsys.readouterr()
    assert '"sidecar_mode": "serial"' in captured.out


def test_benchmark_main_exits_nonzero_when_only_one_worker_gets_all_clips(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_tool,
        "benchmark_caption_sidecar",
        lambda **kwargs: CaptionSidecarBenchmarkResult(
            video_path="/tmp/input.mp4",
            manifest_path="/tmp/manifest.json",
            output_caption_path="/tmp/output.txt",
            baseline_caption_path="/tmp/baseline.txt",
            expected_caption_count=2,
            generated_caption_count=2,
            baseline_caption_count=2,
            captions_match=True,
            first_mismatch_index=None,
            elapsed_seconds=1.0,
            sidecar_mode="parallel",
            sidecar_worker_count=2,
            sidecar_fallback_used=False,
            sidecar_request_id="req-4",
            sidecar_total_clip_count=2,
            sidecar_assigned_clip_indices_by_worker={"0": [0, 1], "1": []},
            sidecar_timing=None,
        ),
    )

    with pytest.raises(
        SystemExit,
        match="did not exercise both workers on the dual-clip request",
    ):
        main(
            [
                "--video-path",
                "/tmp/input.mp4",
                "--baseline-caption-path",
                "/tmp/baseline.txt",
            ]
        )

    captured = capsys.readouterr()
    assert '"sidecar_assigned_clip_indices_by_worker": {' in captured.out
