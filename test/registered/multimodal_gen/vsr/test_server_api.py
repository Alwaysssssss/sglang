# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from sglang.multimodal_gen.configs.pipeline_configs.vsr import WanVSRPipelineConfig
from sglang.multimodal_gen.runtime.entrypoints.openai import vsr_api
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.vsr.blending import tiled_restore_rect


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"video_input_path": "/a", "videoUrl": "http://localhost/a"},
        {"video_input_path": "/a", "taskId": "../escape"},
        {"video_input_path": "/a", "target_resolution": "321x640"},
        {"video_input_path": "/a", "read_queue": 0},
        {"video_input_path": "/a", "prompt": "unused"},
    ],
)
def test_invalid_contract(payload):
    with pytest.raises(ValueError):
        vsr_api.VideoRestorationRequest.model_validate(payload)


def test_aliases_and_tiling_validation():
    req = vsr_api.VideoRestorationRequest.model_validate(
        {
            "videoUrl": "http://localhost/a.mp4",
            "taskId": "my-task",
            "callbackUrl": "http://localhost/callback",
            "tile_h": 320,
        }
    )
    assert req.task_id == "my-task"
    assert req.preserve_audio is True
    assert vsr_api.VideoRestorationRequest.model_validate(
        {"video_input_path": "/a", "preserve_audio": False}
    ).preserve_audio is False
    vsr_api._validate_tiling(req, WanVSRPipelineConfig())
    req.spatial_overlap = 320
    with pytest.raises(ValueError):
        vsr_api._validate_tiling(req, WanVSRPipelineConfig())


def test_admission_duplicate_and_recovery(monkeypatch, tmp_path):
    cfg = WanVSRPipelineConfig()
    server = SimpleNamespace(pipeline_config=cfg, output_path=str(tmp_path))
    store = AsyncDictStore()
    monkeypatch.setattr(vsr_api, "get_global_server_args", lambda: server)
    monkeypatch.setattr(vsr_api, "VIDEO_STORE", store)
    monkeypatch.setattr(vsr_api, "_VSR_SEMAPHORE", asyncio.Semaphore(1))
    monkeypatch.setattr(vsr_api, "probe_video", lambda _: (25, 53, 320, 640))
    monkeypatch.setattr(
        vsr_api.WanVRSamplingParams,
        "from_user_kwargs",
        lambda server, **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(vsr_api, "prepare_request", lambda **kw: kw["sampling_params"])
    registered = []

    async def register(*args, **kwargs):
        registered.append((args, kwargs))

    monkeypatch.setattr(vsr_api, "_create_registered_video_task", register)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"input")
    app = FastAPI()
    app.include_router(vsr_api.router)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            url = "/v1/videos/restorations"
            invalid = await client.post(url, json={"video_input_path": "/missing"})
            assert invalid.json()["code"] == 1
            payload = {"taskId": "first", "video_input_path": str(source)}
            success = await client.post(url, json=payload)
            assert success.json() == {
                "code": 0,
                "message": "Task submitted",
                "id": "first",
            }
            busy = await client.post(url, json={**payload, "taskId": "second"})
            assert busy.json()["code"] == 2
            assert len(registered) == 1
            vsr_api._VSR_SEMAPHORE.release()
            duplicate = await client.post(url, json=payload)
            assert duplicate.json()["code"] == 1
            assert not vsr_api._VSR_SEMAPHORE.locked()
            success = await client.post(
                url, json={**payload, "taskId": "second", "preserve_audio": False}
            )
            assert success.json()["code"] == 0
            assert registered[0][0][3].preserve_audio is True
            assert registered[1][0][3].preserve_audio is False
            vsr_api._VSR_SEMAPHORE.release()

    asyncio.run(scenario())


def test_interrupt_stops_before_next_tile():
    import torch

    calls = []

    def check():
        if calls:
            raise TimeoutError("stop")

    def restore(x):
        calls.append(1)
        return x

    with pytest.raises(TimeoutError):
        tiled_restore_rect(
            torch.zeros(1, 3, 1, 4, 8),
            restore,
            tile_t=1,
            tile_h=4,
            tile_w=4,
            t_overlap=0,
            s_overlap=0,
            check_interrupt=check,
            show_progress=False,
        )
    assert len(calls) == 1


def test_native_pipeline_returns_existing_file(tmp_path):
    from contextlib import nullcontext

    from sglang.multimodal_gen.runtime.pipelines.wan_vsr_pipeline import WanVSRPipeline

    path = tmp_path / "result.mp4"
    path.write_bytes(b"video")
    result = SimpleNamespace(
        sampling_params=SimpleNamespace(
            output_path=str(tmp_path), output_file_name="result.mp4"
        ),
        metrics=object(),
    )
    executor = SimpleNamespace(
        profile_execution=lambda *a, **k: nullcontext(),
        execute_with_profiling=lambda *a: result,
    )
    pipeline = SimpleNamespace(executor=executor, stages=[])
    output = WanVSRPipeline.forward(pipeline, result, None)
    assert output.output_file_paths == [str(path)]
    assert (
        output.output is None
    )  # HTTP must not encode the already-written video again.
    assert output.metrics is result.metrics


def test_cancel_unblocks_full_reader_queue():
    import queue
    import threading

    from sglang.multimodal_gen.runtime.vsr.stream import _put_read_item

    q = queue.Queue(maxsize=1)
    q.put("pending")
    stopped = threading.Event()
    results = []
    worker = threading.Thread(
        target=lambda: results.append(_put_read_item(q, "next", stopped))
    )
    worker.start()
    stopped.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert results == [False]
    assert q.get_nowait() == "pending"


@pytest.mark.parametrize("scheduler_count", [1, 2])
def test_pipeline_tile_worker_configuration(monkeypatch, scheduler_count):
    from sglang.multimodal_gen.runtime.pipelines.wan_vsr_pipeline import WanVSRPipeline
    from sglang.multimodal_gen.runtime.vsr import parallel

    captured = []
    restorer = object()

    def factory(devices, **kwargs):
        captured.append((devices, kwargs))
        return restorer

    monkeypatch.setattr(parallel, "ParallelVSRRestorer", factory)
    config = WanVSRPipelineConfig(
        tile_devices=["cuda:0", "cuda:1"], compile_encoder=True
    )
    args = SimpleNamespace(
        pipeline_config=config,
        num_gpus=scheduler_count,
        component_paths={"wan_root": "/fake/wan"},
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        dit_offload_prefetch_size=0.0,
        pin_cpu_memory=True,
    )
    pipeline = SimpleNamespace(model_path="/fake/checkpoint")
    if scheduler_count == 2:
        with pytest.raises(ValueError, match="one scheduler"):
            WanVSRPipeline.load_modules(pipeline, args)
        assert not captured
    else:
        assert WanVSRPipeline.load_modules(pipeline, args) == {}
        assert pipeline.restorer is restorer
        assert captured[0][0] == ["cuda:0", "cuda:1"]
        assert captured[0][1]["compile_encoder"] is True
        assert captured[0][1]["checkpoint_dir"] == "/fake/checkpoint"
