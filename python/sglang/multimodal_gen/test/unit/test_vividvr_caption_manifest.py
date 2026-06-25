import sglang.multimodal_gen.runtime.vividvr.caption_manifest as caption_manifest
from sglang.multimodal_gen.runtime.vividvr.caption_manifest import (
    VividVRCaptionManifest,
    build_vividvr_caption_manifest_for_video_path,
    build_vividvr_caption_manifest_from_video_info,
)


def test_manifest_counts_temporal_clips_and_spatial_tiles():
    manifest = build_vividvr_caption_manifest_from_video_info(
        video_path="/tmp/input.mp4",
        fps=24.0,
        num_frames=130,
        height=720,
        width=1280,
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
    )

    assert isinstance(manifest, VividVRCaptionManifest)
    assert manifest.video_path == "/tmp/input.mp4"
    assert manifest.num_frames == 130
    assert manifest.num_temporal_process_frames == 121
    assert len(manifest.clips) == 2
    assert manifest.expected_caption_count == len(manifest.clips)
    assert sum(clip.tile_count for clip in manifest.clips) > manifest.expected_caption_count
    assert manifest.clips[0].clip_index == 0
    assert manifest.clips[1].clip_index == 1
    assert manifest.clips[0].tiles[0].tile_index == 0


def test_manifest_round_trips_json(tmp_path):
    manifest = build_vividvr_caption_manifest_from_video_info(
        video_path="/tmp/input.mp4",
        fps=24.0,
        num_frames=9,
        height=64,
        width=64,
        num_temporal_process_frames=9,
        tile_size=128,
        tile_stride=64,
    )
    path = tmp_path / "manifest.json"

    manifest.write_json(path)
    loaded = VividVRCaptionManifest.read_json(path)

    assert loaded == manifest
    assert loaded.expected_caption_count == 1


def test_manifest_for_video_path_uses_metadata_probe_and_counts_four_clips(
    monkeypatch,
):
    monkeypatch.setattr(
        caption_manifest,
        "probe_vividvr_caption_video_metadata",
        lambda video_path: {
            "fps": 25.0,
            "num_frames": 301,
            "height": 720,
            "width": 960,
        },
    )

    manifest = build_vividvr_caption_manifest_for_video_path(
        video_path="/tmp/input_301f.mp4",
        num_temporal_process_frames=121,
        tile_size=128,
        tile_stride=64,
    )

    assert manifest.video_path == "/tmp/input_301f.mp4"
    assert manifest.num_frames == 301
    assert manifest.expected_caption_count == 4
    assert [(clip.start_frame, clip.end_frame) for clip in manifest.clips] == [
        (0, 121),
        (60, 181),
        (120, 241),
        (180, 301),
    ]
