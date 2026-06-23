import os
from pathlib import Path

from fastapi.testclient import TestClient

from sglang.multimodal_gen.tools.run_vividvr_caption_sidecar import (
    CaptionSidecarState,
    _ensure_python_dev_headers_for_sidecar,
    create_app,
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
