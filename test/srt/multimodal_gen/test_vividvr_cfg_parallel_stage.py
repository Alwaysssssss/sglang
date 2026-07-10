from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    vividvr,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.vividvr import (
    VividVRDenoisingStage,
    _resolve_vividvr_parallel_mode,
)


@pytest.mark.parametrize(
    ("mode", "enable_cfg_parallel", "sp_degree", "expected"),
    [
        ("auto", False, 1, "single"),
        ("auto", False, 2, "sp"),
        ("auto", True, 1, "cfg"),
        ("auto", True, 2, "cfg_sp"),
        ("single", False, 1, "single"),
        ("sp", False, 2, "sp"),
        ("cfg", True, 1, "cfg"),
        ("cfg_sp", True, 2, "cfg_sp"),
    ],
)
def test_vividvr_parallel_mode_resolves_valid_combinations(
    mode,
    enable_cfg_parallel,
    sp_degree,
    expected,
):
    server_args = SimpleNamespace(
        vividvr_parallel_mode=mode,
        enable_cfg_parallel=enable_cfg_parallel,
        sp_degree=sp_degree,
    )
    assert _resolve_vividvr_parallel_mode(server_args) == expected


@pytest.mark.parametrize(
    ("mode", "enable_cfg_parallel", "sp_degree"),
    [
        ("single", True, 1),
        ("single", False, 2),
        ("sp", True, 2),
        ("sp", False, 1),
        ("cfg", False, 1),
        ("cfg", True, 2),
        ("cfg_sp", False, 2),
        ("cfg_sp", True, 1),
    ],
)
def test_vividvr_parallel_mode_rejects_mismatched_flags(
    mode,
    enable_cfg_parallel,
    sp_degree,
):
    server_args = SimpleNamespace(
        vividvr_parallel_mode=mode,
        enable_cfg_parallel=enable_cfg_parallel,
        sp_degree=sp_degree,
    )
    with pytest.raises(ValueError, match="vividvr_parallel_mode"):
        _resolve_vividvr_parallel_mode(server_args)


def test_vividvr_denoising_stage_declares_cfg_parallel(monkeypatch):
    monkeypatch.setattr(
        vividvr,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_cfg_parallel=True),
    )
    stage = object.__new__(VividVRDenoisingStage)

    assert stage.parallelism_type is StageParallelismType.CFG_PARALLEL


def test_vividvr_denoising_stage_preserves_replicated_default(monkeypatch):
    monkeypatch.setattr(
        vividvr,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_cfg_parallel=False),
    )
    stage = object.__new__(VividVRDenoisingStage)

    assert stage.parallelism_type is StageParallelismType.REPLICATED


def test_vividvr_select_prompt_branch_uses_cond_on_cfg_rank_zero():
    prompt_embeds = torch.tensor([[[1.0]], [[2.0]]])
    negative_prompt_embeds = torch.tensor([[[-1.0]], [[-2.0]]])

    selected, branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds,
        negative_prompt_embeds,
        slice(1, 2),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=0,
    )

    assert branch == "cond"
    torch.testing.assert_close(selected, prompt_embeds[1:2])


def test_vividvr_select_prompt_branch_uses_uncond_on_cfg_rank_one():
    prompt_embeds = torch.tensor([[[1.0]], [[2.0]]])
    negative_prompt_embeds = torch.tensor([[[-1.0]], [[-2.0]]])

    selected, branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds,
        negative_prompt_embeds,
        slice(1, 2),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=1,
    )

    assert branch == "uncond"
    torch.testing.assert_close(selected, negative_prompt_embeds[1:2])


def test_vividvr_select_prompt_branch_preserves_serial_batch_order():
    prompt_embeds = torch.tensor([[[2.0]]])
    negative_prompt_embeds = torch.tensor([[[-2.0]]])

    selected, branch = VividVRDenoisingStage._select_cfg_prompt_embeds(
        prompt_embeds,
        negative_prompt_embeds,
        slice(0, 1),
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
        cfg_rank=0,
    )

    assert branch == "serial"
    torch.testing.assert_close(selected, torch.tensor([[[-2.0]], [[2.0]]]))


def test_vividvr_select_model_input_preserves_serial_batch_two():
    tensor = torch.arange(2.0).reshape(1, 1, 2)

    selected = VividVRDenoisingStage._select_cfg_model_input(
        tensor,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
    )

    assert selected.shape[0] == 2
    torch.testing.assert_close(selected, torch.cat([tensor, tensor], dim=0))


def test_vividvr_select_model_input_keeps_single_branch_for_cfg_parallel():
    tensor = torch.arange(2.0).reshape(1, 1, 2)

    selected = VividVRDenoisingStage._select_cfg_model_input(
        tensor,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
    )

    assert selected.shape[0] == 1
    torch.testing.assert_close(selected, tensor)


def test_vividvr_combine_noise_preserves_serial_cfg_formula():
    uncond = torch.tensor([1.0, 2.0])
    cond = torch.tensor([3.0, 5.0])
    noise_pred = torch.stack([uncond, cond], dim=0)

    combined = VividVRDenoisingStage._combine_cfg_noise_pred(
        noise_pred,
        guidance_scale=2.5,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=False,
        cfg_rank=0,
        all_reduce_fn=lambda partial: partial,
    )

    torch.testing.assert_close(combined, (uncond + 2.5 * (cond - uncond)).unsqueeze(0))


@pytest.mark.parametrize(
    ("cfg_rank", "expected"),
    [
        (0, torch.tensor([7.5, 12.5])),
        (1, torch.tensor([-1.5, -3.0])),
    ],
)
def test_vividvr_combine_noise_uses_cfg_parallel_partial_formula(cfg_rank, expected):
    noise_pred = torch.tensor([3.0, 5.0]) if cfg_rank == 0 else torch.tensor([1.0, 2.0])

    combined = VividVRDenoisingStage._combine_cfg_noise_pred(
        noise_pred,
        guidance_scale=2.5,
        do_classifier_free_guidance=True,
        enable_cfg_parallel=True,
        cfg_rank=cfg_rank,
        all_reduce_fn=lambda partial: partial,
    )

    torch.testing.assert_close(combined, expected)
