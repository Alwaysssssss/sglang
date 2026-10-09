from types import SimpleNamespace
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.configs.sample.videoedit_wan import (
    DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT,
    WanVideoEditSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline import (
    WanVideoEditPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    videoedit_wan,
)
from sglang.multimodal_gen.runtime.videoedit.vae_spatial import spatial_tiles
from sglang.multimodal_gen.runtime.videoedit.distributed import (
    broadcast_tensor_from_owner,
)


def _broadcast_failure_worker(rank, rendezvous, failure):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        group = SimpleNamespace(
            world_size=2,
            rank_in_group=rank,
            ranks=[0, 1],
            device=torch.device("cpu"),
            cpu_group=dist.group.WORLD,
            device_group=dist.group.WORLD,
        )
        value = torch.tensor([1.0, 2.0]) if rank == 0 else None
        original_empty = torch.empty
        original_to = torch.Tensor.to

        def fail_empty(*args, **kwargs):
            if (
                rank == 1
                and failure == "allocation"
                and kwargs.get("dtype") == torch.float32
            ):
                raise RuntimeError("injected allocation failure")
            return original_empty(*args, **kwargs)

        def fail_to(tensor, *args, **kwargs):
            if rank == 0 and failure == "copy" and tensor is value:
                raise RuntimeError("injected copy failure")
            return original_to(tensor, *args, **kwargs)

        if failure == "missing":
            value = None
        with patch.object(torch, "empty", fail_empty), patch.object(
            torch.Tensor, "to", fail_to
        ):
            with pytest.raises(
                RuntimeError, match="produced no tensor|payload preparation failed"
            ):
                broadcast_tensor_from_owner(group, value, 0, "test")

        # A failed request must leave every rank able to enter the next request.
        expected = torch.tensor([3.0, 4.0])
        result = broadcast_tensor_from_owner(
            group, expected if rank == 0 else None, 0, "recovery"
        )
        assert torch.equal(result, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["missing", "allocation", "copy"])
def test_broadcast_failure_is_shared_and_next_request_recovers(tmp_path, failure):
    torch.multiprocessing.start_processes(
        _broadcast_failure_worker,
        args=((tmp_path / "rendezvous").as_uri(), failure),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failure", [False, True])
def test_service_preparation_returns_unused_encoder_memory(failure):
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()

    def precompute(_):
        # Like startup text encoding, the device allocation is temporary while
        # the persistent embedding belongs on CPU.
        temporary = torch.empty(16 * 1024**2, dtype=torch.float32, device="cuda")
        del temporary
        if failure:
            raise RuntimeError("precompute failed")

    pipeline = SimpleNamespace(
        get_module=lambda _: SimpleNamespace(layerwise_offload_managers=[]),
        _startup_conditioning=SimpleNamespace(precompute_default_negative=precompute),
        server_args=SimpleNamespace(),
    )
    if failure:
        with pytest.raises(RuntimeError, match="precompute failed"):
            WanVideoEditPipeline.prepare_service(pipeline)
    else:
        WanVideoEditPipeline.prepare_service(pipeline)
    assert torch.cuda.memory_reserved() <= before


class _FakeTextStage:
    def __init__(self):
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        self.cached_negative_text = DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT
        self.cached_negative_embeds = torch.tensor([[[-1.0]]])
        self.positive_encode_calls = 0
        self.negative_encode_calls = 1
        self.calls = []

    def encode_texts_to_cpu(self, texts, server_args):
        del server_args
        self.calls.append(list(texts))
        return [torch.tensor([[[float(len(text))]]]) for text in texts]


class _FakeImageStage:
    def __init__(self):
        self.calls = 0

    def forward(self, batch, server_args):
        del server_args
        self.calls += 1
        batch.sampling_params.runtime_image_embeds = torch.tensor(
            [[[float(self.calls)]]]
        )
        return batch


def _server_args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_precision="fp32"),
        disable_autocast=True,
    )


def _stage(monkeypatch):
    monkeypatch.setattr(videoedit_wan, "_conditioning_group", lambda: None)
    monkeypatch.setattr(
        videoedit_wan, "get_local_torch_device", lambda: torch.device("cpu")
    )
    stage = object.__new__(videoedit_wan.VideoEditConditioningEncodingStage)
    stage.text_stage = _FakeTextStage()
    stage.image_stage = _FakeImageStage()
    return stage


def _batch(**kwargs):
    params = WanVideoEditSamplingParams(
        prompt=kwargs.pop("prompt", "a cat"),
        num_frames=1,
        infer_len=1,
        overlap=0,
        **kwargs,
    )
    return SimpleNamespace(sampling_params=params)


def test_request_text_is_reused_across_windows(monkeypatch):
    stage = _stage(monkeypatch)
    batch = _batch()

    stage.forward(batch, _server_args())
    first_prompt = batch.sampling_params.runtime_request_prompt_embeds
    batch.sampling_params.runtime_prompt_embeds = None
    batch.sampling_params.runtime_negative_prompt_embeds = None
    batch.sampling_params.runtime_image_embeds = None
    stage.forward(batch, _server_args())

    assert stage.text_stage.calls == [["a cat"]]
    assert stage.text_stage.positive_encode_calls == 1
    assert stage.text_stage.negative_encode_calls == 1
    assert stage.image_stage.calls == 2
    assert batch.sampling_params.runtime_request_prompt_embeds is first_prompt
    assert torch.equal(
        batch.sampling_params.runtime_request_negative_prompt_embeds,
        stage.text_stage.cached_negative_embeds,
    )


def test_negative_cache_updates_only_when_cfg_consumes_it(monkeypatch):
    stage = _stage(monkeypatch)
    changed = _batch(prompt="first", negative_prompt="new negative")
    stage.forward(changed, _server_args())
    assert stage.text_stage.calls == [["first", "new negative"]]
    assert stage.text_stage.cached_negative_text == "new negative"
    assert stage.text_stage.positive_encode_calls == 1
    assert stage.text_stage.negative_encode_calls == 2

    unchanged = _batch(prompt="second", negative_prompt="new negative")
    stage.forward(unchanged, _server_args())
    assert stage.text_stage.calls[-1] == ["second"]
    assert stage.text_stage.positive_encode_calls == 2
    assert stage.text_stage.negative_encode_calls == 2

    no_cfg = _batch(
        prompt="third", negative_prompt="ignored until cfg", guidance_scale=1.0
    )
    stage.forward(no_cfg, _server_args())
    assert stage.text_stage.calls[-1] == ["third"]
    assert stage.text_stage.cached_negative_text == "new negative"
    assert no_cfg.sampling_params.runtime_request_negative_prompt_embeds is None
    assert stage.text_stage.positive_encode_calls == 3
    assert stage.text_stage.negative_encode_calls == 2


def test_default_negative_is_precomputed_once_before_requests(monkeypatch):
    stage = _stage(monkeypatch)
    stage.text_stage.cached_negative_text = None
    stage.text_stage.cached_negative_embeds = None
    stage.text_stage.negative_encode_calls = 0

    stage.precompute_default_negative(_server_args())

    assert stage.text_stage.calls == [[DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT]]
    assert stage.text_stage.positive_encode_calls == 0
    assert stage.text_stage.negative_encode_calls == 1
    assert stage.text_stage.cached_negative_text == DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT
    assert stage.text_stage.cached_negative_embeds.device.type == "cpu"


def test_window_reset_preserves_request_conditioning():
    params = WanVideoEditSamplingParams(num_frames=1, infer_len=1, overlap=0)
    params.runtime_request_prompt_embeds = torch.ones(1)
    params.runtime_request_negative_prompt_embeds = torch.zeros(1)
    params.runtime_prompt_embeds = torch.ones(1)

    params.reset_window_runtime(SimpleNamespace(window_index=1))
    assert params.runtime_request_prompt_embeds is not None
    assert params.runtime_request_negative_prompt_embeds is not None
    assert params.runtime_prompt_embeds is None

    params.reset_request_runtime()
    assert params.runtime_request_prompt_embeds is None
    assert params.runtime_request_negative_prompt_embeds is None


def test_spatial_tiles_are_stable_row_major():
    tiles = spatial_tiles(height=7, width=8, stride_height=3, stride_width=4)
    assert [tile.index for tile in tiles] == list(range(6))
    assert [(tile.top, tile.left) for tile in tiles] == [
        (0, 0),
        (0, 4),
        (3, 0),
        (3, 4),
        (6, 0),
        (6, 4),
    ]
