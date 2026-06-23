import argparse
import os
import sys
from concurrent.futures import Future
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

from fastapi.testclient import TestClient
import pytest
import sglang.multimodal_gen.tools.run_vividvr_caption_sidecar as sidecar_tool
import torch
from sglang.multimodal_gen.runtime.vividvr.caption_sidecar_runtime import (
    CaptionClipResult,
    CaptionWorkerBatchResult,
)

from sglang.multimodal_gen.tools.run_vividvr_caption_sidecar import (
    CaptionSidecarState,
    _ensure_python_dev_headers_for_sidecar,
    _build_fallback_handler,
    _generate_captions,
    create_app,
    main,
    parse_args,
)


def test_sidecar_writes_captions_in_manifest_order(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 2,
          "clips": [
            {
              "clip_index": 0,
              "start_frame": 0,
              "end_frame": 9,
              "original_num_frames": 9,
              "padded_num_frames": 9,
              "tile_count": 1,
              "tiles": [
                {
                  "tile_index": 0,
                  "t_start": 0,
                  "t_end": 9,
                  "h_start": 0,
                  "h_end": 64,
                  "w_start": 0,
                  "w_end": 64
                }
              ]
            },
            {
              "clip_index": 1,
              "start_frame": 3,
              "end_frame": 9,
              "original_num_frames": 6,
              "padded_num_frames": 9,
              "tile_count": 2,
              "tiles": [
                {
                  "tile_index": 0,
                  "t_start": 0,
                  "t_end": 9,
                  "h_start": 0,
                  "h_end": 32,
                  "w_start": 0,
                  "w_end": 32
                },
                {
                  "tile_index": 1,
                  "t_start": 0,
                  "t_end": 9,
                  "h_start": 32,
                  "h_end": 64,
                  "w_start": 32,
                  "w_end": 64
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").arange(
            9 * 3 * 64 * 64, dtype=__import__("torch").float32
        ).reshape(9, 3, 64, 64), 24.0),
    )

    seen_shapes = []

    class FakeCaptioner:
        def to(self, device):
            return self

        def __call__(self, video, fps=None):
            seen_shapes.append(tuple(video.shape))
            assert fps == 24.0
            return f"frames={video.shape[0]} start={float(video[0, 0, 0, 0])}"

    state = CaptionSidecarState(captioner=FakeCaptioner(), device="cpu")
    app = create_app(state)
    client = TestClient(app)

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["caption_count"] == 2
    assert seen_shapes == [(9, 3, 64, 64), (9, 3, 64, 64)]
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "frames=9 start=0.0",
        "frames=9 start=36864.0",
    ]


def test_sidecar_python_header_helper_uses_fallback_include_dir(tmp_path, monkeypatch):
    include_dir = (
        tmp_path / "tmp_py310dev" / "extracted" / "usr" / "include" / "python3.10"
    )
    include_dir.mkdir(parents=True)
    (include_dir / "Python.h").write_text("/* test header */\n", encoding="utf-8")

    multiarch_dir = (
        tmp_path
        / "tmp_py310dev"
        / "extracted"
        / "usr"
        / "include"
        / "x86_64-linux-gnu"
        / "python3.10"
    )
    multiarch_dir.mkdir(parents=True)
    (multiarch_dir / "pyconfig.h").write_text("/* test pyconfig */\n", encoding="utf-8")

    def fake_get_config_var(name):
        if name == "INCLUDEPY":
            return "/missing/python3.10"
        if name == "MULTIARCH":
            return "x86_64-linux-gnu"
        return None

    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.sysconfig.get_config_var",
        fake_get_config_var,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.sysconfig.get_path",
        lambda name: None,
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.Path.home",
        lambda: Path(tmp_path),
    )
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.delenv("C_INCLUDE_PATH", raising=False)

    resolved = _ensure_python_dev_headers_for_sidecar()

    assert resolved == include_dir
    assert os.environ["CPATH"] == f"{include_dir.parent}{os.pathsep}{include_dir}"
    assert os.environ["C_INCLUDE_PATH"] == (
        f"{include_dir.parent}{os.pathsep}{include_dir}"
    )


def test_sidecar_parallel_controller_path_includes_metrics(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 3,
          "clips": [
            {
              "clip_index": 0,
              "start_frame": 0,
              "end_frame": 9,
              "original_num_frames": 9,
              "padded_num_frames": 9,
              "tile_count": 1,
              "tiles": []
            },
            {
              "clip_index": 1,
              "start_frame": 0,
              "end_frame": 9,
              "original_num_frames": 9,
              "padded_num_frames": 9,
              "tile_count": 1,
              "tiles": []
            },
            {
              "clip_index": 2,
              "start_frame": 0,
              "end_frame": 9,
              "original_num_frames": 9,
              "padded_num_frames": 9,
              "tile_count": 1,
              "tiles": []
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.uuid.uuid4",
        lambda: type("FakeUUID", (), {"hex": "req-123"})(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").zeros(9, 3, 8, 8), 24.0),
    )

    class FakeExecutor:
        def submit(self, fn, worker_index, jobs):
            future = Future()
            if worker_index == 0:
                clip_results = [
                    CaptionClipResult(
                        clip_index=jobs[1].clip_index,
                        caption=f"clip-{jobs[1].clip_index}",
                        worker_index=worker_index,
                    ),
                    CaptionClipResult(
                        clip_index=jobs[0].clip_index,
                        caption=f"clip-{jobs[0].clip_index}",
                        worker_index=worker_index,
                    ),
                ]
            else:
                clip_results = [
                    CaptionClipResult(
                        clip_index=jobs[0].clip_index,
                        caption=f"clip-{jobs[0].clip_index}",
                        worker_index=worker_index,
                    ),
                ]
            future.set_result(
                CaptionWorkerBatchResult(
                    worker_index=worker_index,
                    clip_results=clip_results,
                )
            )
            return future

    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(
        captioner=FakeCaptioner(),
        device="cpu",
        worker_count=2,
        executors=(FakeExecutor(), FakeExecutor()),
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "caption_file_path": str(output_path),
        "caption_count": 3,
        "manifest_path": str(manifest_path),
        "mode": "parallel",
        "worker_count": 2,
        "fallback_used": False,
        "request_id": "req-123",
        "total_clip_count": 3,
        "assigned_clip_indices_by_worker": {
            "0": [0, 2],
            "1": [1],
        },
        "timing": payload["timing"],
    }
    assert payload["timing"]["read_seconds"] is not None
    assert payload["timing"]["write_seconds"] is not None
    assert payload["timing"]["total_seconds"] is not None
    assert len(payload["timing"]["worker_batches"]) == 2
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "clip-0",
        "clip-1",
        "clip-2",
    ]


def test_sidecar_parallel_merge_writes_clip_index_order(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 4,
          "clips": [
            {"clip_index": 0, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []},
            {"clip_index": 1, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []},
            {"clip_index": 2, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []},
            {"clip_index": 3, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []}
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").zeros(9, 3, 8, 8), 24.0),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._collect_parallel_worker_results",
        lambda state, worker_jobs_by_worker: [
            CaptionWorkerBatchResult(
                worker_index=1,
                clip_results=[
                    CaptionClipResult(
                        clip_index=3,
                        caption="clip-3",
                        worker_index=1,
                    ),
                    CaptionClipResult(
                        clip_index=1,
                        caption="clip-1",
                        worker_index=1,
                    ),
                ],
            ),
            CaptionWorkerBatchResult(
                worker_index=0,
                clip_results=[
                    CaptionClipResult(
                        clip_index=2,
                        caption="clip-2",
                        worker_index=0,
                    ),
                    CaptionClipResult(
                        clip_index=0,
                        caption="clip-0",
                        worker_index=0,
                    ),
                ],
            ),
        ],
    )

    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(captioner=FakeCaptioner(), device="cpu", worker_count=2)
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 4,
        },
    )

    assert response.status_code == 200
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "clip-0",
        "clip-1",
        "clip-2",
        "clip-3",
    ]


def test_sidecar_parallel_merge_rejects_duplicate_clip_indices(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 2,
          "clips": [
            {"clip_index": 0, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []},
            {"clip_index": 1, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []}
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").zeros(9, 3, 8, 8), 24.0),
    )

    class FakeExecutor:
        def submit(self, fn, worker_index, jobs):
            future = Future()
            future.set_result(
                CaptionWorkerBatchResult(
                    worker_index=worker_index,
                    clip_results=[
                        CaptionClipResult(
                            clip_index=0,
                            caption=f"dup-{worker_index}",
                            worker_index=worker_index,
                        )
                    ],
                )
            )
            return future

    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(
        captioner=FakeCaptioner(),
        device="cpu",
        worker_count=2,
        executors=(FakeExecutor(), FakeExecutor()),
        allow_serial_fallback=False,
    )
    client = TestClient(create_app(state), raise_server_exceptions=False)

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 2,
        },
    )

    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert not output_path.exists()


def test_sidecar_parallel_failure_falls_back_to_serial_when_enabled(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "captions.txt"
    manifest_path.write_text(
        """
        {
          "version": 1,
          "video_path": "/tmp/in.mp4",
          "fps": 24.0,
          "num_frames": 9,
          "height": 64,
          "width": 64,
          "num_temporal_process_frames": 9,
          "tile_size": 128,
          "tile_stride": 64,
          "expected_caption_count": 2,
          "clips": [
            {"clip_index": 0, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []},
            {"clip_index": 1, "start_frame": 0, "end_frame": 9, "original_num_frames": 9, "padded_num_frames": 9, "tile_count": 1, "tiles": []}
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar.uuid.uuid4",
        lambda: type("FakeUUID", (), {"hex": "req-fallback"})(),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._load_video_tensor",
        lambda video_path: (__import__("torch").zeros(9, 3, 8, 8), 24.0),
    )
    monkeypatch.setattr(
        "sglang.multimodal_gen.tools.run_vividvr_caption_sidecar._caption_manifest_serial",
        lambda state, manifest: (["serial-0", "serial-1"], 0.1, []),
    )

    class FakeExecutor:
        def __init__(self, should_fail):
            self.should_fail = should_fail

        def submit(self, fn, worker_index, jobs):
            future = Future()
            if self.should_fail:
                future.set_exception(RuntimeError("worker boom"))
            else:
                future.set_result(
                    CaptionWorkerBatchResult(
                        worker_index=worker_index,
                        clip_results=[
                            CaptionClipResult(
                                clip_index=job.clip_index,
                                caption=f"worker-{worker_index}-{job.clip_index}",
                                worker_index=worker_index,
                            )
                            for job in jobs
                        ],
                    )
                )
            return future

        def shutdown(self, wait=False, cancel_futures=True):
            return None

    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(
        captioner=FakeCaptioner(),
        device="cpu",
        worker_count=2,
        allow_serial_fallback=True,
        executors=(FakeExecutor(False), FakeExecutor(True)),
    )
    client = TestClient(create_app(state))

    response = client.post(
        "/v1/vividvr/captions",
        json={
            "manifest_path": str(manifest_path),
            "output_caption_path": str(output_path),
            "expected_caption_count": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "serial"
    assert response.json()["fallback_used"] is True
    assert response.json()["worker_count"] == 2
    assert response.json()["request_id"] == "req-fallback"
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "serial-0",
        "serial-1",
    ]


def test_generate_captions_controller_error_does_not_fall_back_to_serial(monkeypatch):
    serial_calls = {"count": 0}

    manifest = type(
        "FakeManifest",
        (),
        {
            "clips": [type("Clip", (), {"clip_index": 0})(), type("Clip", (), {"clip_index": 1})()],
        },
    )()

    monkeypatch.setattr(
        sidecar_tool,
        "_caption_manifest_parallel",
        lambda state, manifest, metrics: (_ for _ in ()).throw(
            ValueError("controller misconfiguration")
        ),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_caption_manifest_serial",
        lambda state, manifest: (
            serial_calls.__setitem__("count", serial_calls["count"] + 1) or
            ["serial-0", "serial-1"],
            0.1,
            [],
        ),
    )
    monkeypatch.setattr(sidecar_tool, "_ensure_parallel_executors", lambda state: None)

    state = CaptionSidecarState(
        captioner=object(),
        device="cpu",
        worker_count=2,
        allow_serial_fallback=True,
    )

    with pytest.raises(ValueError, match="controller misconfiguration"):
        _generate_captions(state, manifest)

    assert serial_calls["count"] == 0


def test_generate_captions_restarts_parallel_workers_before_serial_fallback(
    monkeypatch,
):
    shutdown_calls = {"count": 0, "wait": []}
    ensure_calls = {"count": 0}
    release_calls = {"count": 0}

    manifest = type(
        "FakeManifest",
        (),
        {
            "clips": [type("Clip", (), {"clip_index": 0})(), type("Clip", (), {"clip_index": 1})()],
        },
    )()

    monkeypatch.setattr(
        sidecar_tool,
        "_caption_manifest_parallel",
        lambda state, manifest, metrics: (_ for _ in ()).throw(
            sidecar_tool.ParallelCaptionWorkerError("worker oom")
        ),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_caption_manifest_serial",
        lambda state, manifest: (["serial-0", "serial-1"], 0.1, []),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_shutdown_parallel_executors",
        lambda state, wait=False: (
            shutdown_calls.__setitem__("count", shutdown_calls["count"] + 1),
            shutdown_calls["wait"].append(wait),
            setattr(state, "executors", None),
        ),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_ensure_parallel_executors",
        lambda state: ensure_calls.__setitem__("count", ensure_calls["count"] + 1),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_release_serial_captioner",
        lambda state: (
            release_calls.__setitem__("count", release_calls["count"] + 1),
            setattr(state, "captioner", None),
        ),
    )
    monkeypatch.setattr(
        sidecar_tool.uuid,
        "uuid4",
        lambda: type("FakeUUID", (), {"hex": "req-recover"})(),
    )

    state = CaptionSidecarState(
        captioner=object(),
        device="cpu",
        worker_count=2,
        allow_serial_fallback=True,
        executors=("executor-0", "executor-1"),
    )

    captions, metadata = _generate_captions(state, manifest)

    assert captions == ["serial-0", "serial-1"]
    assert metadata.mode == "serial"
    assert metadata.fallback_used is True
    assert shutdown_calls == {"count": 1, "wait": [True]}
    assert ensure_calls["count"] == 2
    assert release_calls["count"] == 1
    assert state.captioner is None


def test_caption_manifest_serial_releases_cuda_memory_after_success(monkeypatch):
    release_calls = []

    class FakeCaptioner:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

        def __call__(self, video, fps=None):
            return "serial-caption"

    fake_captioner = FakeCaptioner()
    monkeypatch.setattr(
        sidecar_tool,
        "_load_video_tensor",
        lambda video_path: (torch.zeros(2, 3, 4, 4), 24.0),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_get_serial_captioner",
        lambda state: fake_captioner,
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_release_cuda_memory",
        lambda device: release_calls.append(device),
    )

    manifest = type(
        "FakeManifest",
        (),
        {
            "video_path": "/tmp/in.mp4",
            "fps": 24.0,
            "clips": [
                type(
                    "Clip",
                    (),
                    {
                        "clip_index": 0,
                        "start_frame": 0,
                        "end_frame": 2,
                        "padded_num_frames": 2,
                    },
                )(),
            ],
        },
    )()

    captions, _, worker_batches = sidecar_tool._caption_manifest_serial(
        CaptionSidecarState(device="cuda:0"),
        manifest,
    )

    assert captions == ["serial-caption"]
    assert len(worker_batches) == 1
    assert fake_captioner.devices == ["cuda:0", torch.device("cpu")]
    assert release_calls == ["cuda:0"]


def test_run_worker_caption_job_releases_cuda_memory_after_success(monkeypatch):
    release_calls = []

    class FakeCaptioner:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(device)
            return self

        def __call__(self, video, fps=None):
            return f"fps={fps}"

    monkeypatch.setattr(
        sidecar_tool,
        "_WORKER_STATE",
        type(
            "FakeWorkerState",
            (),
            {
                "captioner": FakeCaptioner(),
                "device": "cuda:1",
            },
        )(),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_release_cuda_memory",
        lambda device: release_calls.append(device),
    )

    result = sidecar_tool._run_worker_caption_job(
        worker_index=1,
        clip_jobs=[
            sidecar_tool.CaptionWorkerClipJob(
                clip_index=3,
                video=torch.zeros(2, 3, 4, 4),
                fps=12.5,
            )
        ],
    )

    assert result.worker_index == 1
    assert [clip.caption for clip in result.clip_results] == ["fps=12.5"]
    assert sidecar_tool._WORKER_STATE.captioner.devices == [
        "cuda:1",
        torch.device("cpu"),
    ]
    assert release_calls == ["cuda:1"]


def test_fallback_http_handler_returns_400_for_request_validation_errors():
    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(captioner=FakeCaptioner(), device="cpu")
    handler_cls = _build_fallback_handler(state)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(server.server_address[0], server.server_address[1])
        connection.request(
            "POST",
            "/v1/vividvr/captions",
            body=b'{"manifest_path": 123}',
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read().decode("utf-8")
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.status == HTTPStatus.BAD_REQUEST
    assert "expected_caption_count" in payload


def test_fallback_http_handler_returns_400_for_invalid_content_length():
    class FakeCaptioner:
        def to(self, device):
            return self

    state = CaptionSidecarState(captioner=FakeCaptioner(), device="cpu")
    handler_cls = _build_fallback_handler(state)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(server.server_address[0], server.server_address[1])
        connection.putrequest("POST", "/v1/vividvr/captions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "abc")
        connection.endheaders()
        response = connection.getresponse()
        payload = response.read().decode("utf-8")
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.status == HTTPStatus.BAD_REQUEST
    assert "Content-Length" in payload


def test_parse_args_supports_parallel_worker_flags():
    args = parse_args(
        [
            "--parallel-workers",
            "2",
            "--worker-devices",
            "cuda:0,cuda:1",
            "--disable-serial-fallback",
        ]
    )

    assert args.parallel_workers == 2
    assert args.worker_devices == ("cuda:0", "cuda:1")
    assert args.allow_serial_fallback is False


def test_parse_args_rejects_legacy_vividvr_root_flag():
    with pytest.raises(SystemExit):
        parse_args(["--vividvr-root", "/tmp/legacy"])


def test_load_caption_backend_factory_falls_back_to_local_package(
    monkeypatch, tmp_path
):
    original_import_module = sidecar_tool.importlib.import_module
    package_root = (
        tmp_path
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "runtime"
        / "vividvr"
        / "caption_sidecar_backend"
    )
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        "from .captioner import create_captioner\n",
        encoding="utf-8",
    )
    (package_root / "captioner.py").write_text(
        "def create_captioner(args):\n"
        "    return {\n"
        "        'caption_backend': args.caption_backend,\n"
        "        'cogvlm2_ckpt_path': args.cogvlm2_ckpt_path,\n"
        "    }\n",
        encoding="utf-8",
    )

    def fake_import_module(name):
        if name == (
            "sglang.multimodal_gen.runtime.vividvr."
            "caption_sidecar_backend.captioner"
        ):
            raise ModuleNotFoundError("No module named 'pybase64'")
        return original_import_module(name)

    monkeypatch.setattr(sidecar_tool.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        sidecar_tool,
        "__file__",
        str(
            tmp_path
            / "python"
            / "sglang"
            / "multimodal_gen"
            / "tools"
            / "run_vividvr_caption_sidecar.py"
        ),
        raising=False,
    )
    sys.modules.pop("vividvr_caption_sidecar_backend", None)

    factory = sidecar_tool._load_caption_backend_factory()

    assert factory(
        argparse.Namespace(
            caption_backend="cogvlm2",
            cogvlm2_ckpt_path="/tmp/ckpt",
        )
    ) == {
        "caption_backend": "cogvlm2",
        "cogvlm2_ckpt_path": "/tmp/ckpt",
    }


def test_sidecar_uses_local_captioner_factory(monkeypatch):
    calls = {}

    class FakeCaptioner:
        def to(self, device):
            return self

    def fake_build_cogvlm2_captioner(cogvlm2_ckpt_path):
        calls["cogvlm2_ckpt_path"] = cogvlm2_ckpt_path
        return FakeCaptioner()

    monkeypatch.setattr(
        sidecar_tool,
        "_build_cogvlm2_captioner",
        fake_build_cogvlm2_captioner,
        raising=False,
    )
    state = CaptionSidecarState(
        captioner=None,
        device="cpu",
        worker_count=1,
        worker_devices=("cpu",),
        cogvlm2_ckpt_path="/tmp/cogvlm2",
    )

    captioner = sidecar_tool._get_serial_captioner(state)

    assert isinstance(captioner, FakeCaptioner)
    assert calls["cogvlm2_ckpt_path"] == "/tmp/cogvlm2"
    assert not hasattr(state, "vividvr_root")


def test_load_video_tensor_prefers_decord(monkeypatch):
    set_bridge_calls = []
    fake_batch = torch.arange(2 * 4 * 5 * 3, dtype=torch.uint8).reshape(2, 4, 5, 3)

    class FakeVideoReader:
        def __init__(self, uri, num_threads):
            assert uri == "/tmp/video.mp4"
            assert num_threads == 1

        def __len__(self):
            return 2

        def get_batch(self, indices):
            assert indices == [0, 1]
            return fake_batch

        def get_avg_fps(self):
            return 23.976

    fake_decord = type(
        "FakeDecord",
        (),
        {
            "bridge": type(
                "Bridge",
                (),
                {"set_bridge": staticmethod(lambda name: set_bridge_calls.append(name))},
            )(),
            "VideoReader": FakeVideoReader,
        },
    )()

    monkeypatch.setitem(sys.modules, "decord", fake_decord)
    monkeypatch.setattr(
        sidecar_tool,
        "_load_video_tensor_cv2",
        lambda video_path: (_ for _ in ()).throw(
            AssertionError("OpenCV fallback should not be used")
        ),
    )

    tensor, fps = sidecar_tool._load_video_tensor("/tmp/video.mp4")

    assert set_bridge_calls == ["torch"]
    assert fps == pytest.approx(23.976)
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (2, 3, 4, 5)
    assert torch.equal(
        tensor[:, :, 0, 0],
        fake_batch[:, 0, 0, :].float().div(255.0),
    )


def test_load_video_tensor_falls_back_to_cv2_when_decord_decode_fails(monkeypatch):
    monkeypatch.setattr(
        sidecar_tool,
        "_load_video_tensor_decord",
        lambda video_path: (_ for _ in ()).throw(RuntimeError("decord boom")),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_load_video_tensor_cv2",
        lambda video_path: (torch.ones(1, 3, 2, 2), 12.5),
    )

    tensor, fps = sidecar_tool._load_video_tensor("/tmp/video.mp4")

    assert fps == pytest.approx(12.5)
    assert tuple(tensor.shape) == (1, 3, 2, 2)


def test_main_parallel_mode_does_not_eagerly_load_serial_captioner(monkeypatch):
    captured = {}
    load_calls = {"count": 0}
    executor_calls = {"count": 0}
    shutdown_calls = {"count": 0}

    def fake_create_app(state):
        captured["state"] = state
        return object()

    monkeypatch.setattr(
        sidecar_tool,
        "parse_args",
        lambda argv=None: argparse.Namespace(
            host="127.0.0.1",
            port=31200,
            cogvlm2_ckpt_path="/tmp/ckpt",
            device="cuda",
            parallel_workers=2,
            worker_devices=("cuda:0", "cuda:1"),
            allow_serial_fallback=True,
        ),
    )
    monkeypatch.setattr(sidecar_tool, "_ensure_python_dev_headers_for_sidecar", lambda: None)
    monkeypatch.setattr(
        sidecar_tool,
        "_build_cogvlm2_captioner",
        lambda args: load_calls.__setitem__("count", load_calls["count"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_create_parallel_executors",
        lambda **kwargs: executor_calls.__setitem__("count", executor_calls["count"] + 1)
        or ("executor-0", "executor-1"),
    )
    monkeypatch.setattr(
        sidecar_tool,
        "_shutdown_parallel_executors",
        lambda state: shutdown_calls.__setitem__("count", shutdown_calls["count"] + 1),
    )
    monkeypatch.setattr(sidecar_tool, "create_app", fake_create_app)
    monkeypatch.setattr(
        sidecar_tool.uvicorn,
        "run",
        lambda app, host, port: captured.update(
            {
                "app": app,
                "host": host,
                "port": port,
            }
        ),
    )

    main()

    assert load_calls["count"] == 0
    assert executor_calls["count"] == 1
    assert shutdown_calls["count"] == 1
    assert captured["state"].captioner is None
    assert captured["state"].worker_count == 2
    assert captured["state"].worker_devices == ("cuda:0", "cuda:1")
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 31200
