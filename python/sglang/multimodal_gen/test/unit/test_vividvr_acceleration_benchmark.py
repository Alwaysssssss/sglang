import json
import os
import sysconfig
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from sglang.multimodal_gen.tools import (
    run_vividvr_acceleration_benchmark as benchmark_module,
)
from sglang.multimodal_gen.tools.run_vividvr_acceleration_benchmark import (
    ALL_SCHEMES,
    SCHEMES,
    VAE_ENCODE_SP_TREATMENTS,
    VAE_SP_TREATMENTS,
    VIVIDVR_STAGE_NAMES,
    BenchmarkCleanupError,
    BenchmarkConfig,
    BenchmarkConfigError,
    BenchmarkDataError,
    BenchmarkRunner,
    FlowCutRequestExecutor,
    GpuMemorySampler,
    RunRole,
    SchemeStatus,
    TmuxBenchmarkLifecycle,
    TmuxManager,
    _config_cli_arguments,
    _config_from_args,
    _download_result,
    atomic_write_json,
    build_dry_run_report,
    build_request_payload,
    build_service_command,
    build_service_environment,
    build_unsupported_record,
    compute_config_fingerprint,
    compute_derived_metrics,
    compute_vae_encode_sp_derived_metrics,
    compute_vae_sp_derived_metrics,
    load_historical_controls,
    parse_args,
    quality_not_worse_than_control,
    run_preflight,
    summarize_perf,
    validate_effective_config,
    verify_historical_controls_unchanged,
)


def make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        repo_root=tmp_path,
        python_executable=tmp_path / ".venv/bin/python",
        model_path=tmp_path / "model",
        vividvr_path=tmp_path / "vividvr",
        input_video=tmp_path / "input.mp4",
        caption_file=tmp_path / "captions.txt",
        reference_video=tmp_path / "reference.mp4",
        output_root=tmp_path / "outputs",
        gpu_ids=(4, 5, 6, 7),
    )


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_scheme_registry_has_fixed_order_and_capabilities():
    assert list(SCHEMES) == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
        "R99",
        "R100",
    ]
    assert {key for key, value in SCHEMES.items() if not value.executable} == {
        "R7",
        "R8",
        "R9",
    }
    assert all(
        SCHEMES[key].status is SchemeStatus.UNSUPPORTED for key in ("R7", "R8", "R9")
    )
    assert all(SCHEMES[key].unsupported_reason for key in ("R7", "R8", "R9"))


def test_vae_sp_treatments_do_not_expand_default_run_all_matrix():
    assert list(VAE_SP_TREATMENTS) == [
        "R99_VAE_SP",
        "R100_VAE_SP",
        "R101_VAE_SP4",
    ]
    assert list(SCHEMES)[-2:] == ["R99", "R100"]
    assert "R99_VAE_SP" not in SCHEMES
    assert "R101_VAE_SP4" not in SCHEMES
    assert ALL_SCHEMES["R99_VAE_SP"].controls == ("R99",)
    assert ALL_SCHEMES["R100_VAE_SP"].controls == ("R100",)
    r101 = ALL_SCHEMES["R101_VAE_SP4"]
    assert r101.controls == ("R4",)
    assert r101.gpu_count == 4
    assert r101.parallel_mode == "sp"
    assert r101.sp_degree == 4
    assert r101.compile_enabled is True
    assert r101.modulation_fusion is True
    assert r101.vae_sp is True
    assert r101.cfg_parallel is False
    assert r101.expected_effective_backend == "fa_sp"


def test_vae_encode_sp_treatments_do_not_expand_default_run_all_matrix():
    assert list(VAE_ENCODE_SP_TREATMENTS) == [
        "R99_VAE_ENCODE_SP",
        "R100_VAE_ENCODE_SP",
        "R101_VAE_ENCODE_SP4",
    ]
    assert all(scheme_id not in SCHEMES for scheme_id in VAE_ENCODE_SP_TREATMENTS)
    assert ALL_SCHEMES["R99_VAE_ENCODE_SP"].controls == ("R99_VAE_SP",)
    assert ALL_SCHEMES["R100_VAE_ENCODE_SP"].controls == ("R100_VAE_SP",)
    r101 = ALL_SCHEMES["R101_VAE_ENCODE_SP4"]
    assert r101.controls == ("R101_VAE_SP4",)
    assert r101.vae_sp is True
    assert r101.vae_encode_sp is True


def test_r0_vae_sp_clean_treatments_are_isolated_and_have_no_extra_acceleration(
    tmp_path: Path,
):
    treatments = benchmark_module.R0_VAE_SP_TREATMENTS
    assert list(treatments) == ["R0_VAE_SP2", "R0_VAE_SP4"]
    assert all(scheme_id not in SCHEMES for scheme_id in treatments)

    for scheme_id, gpu_count in (("R0_VAE_SP2", 2), ("R0_VAE_SP4", 4)):
        scheme = ALL_SCHEMES[scheme_id]
        assert scheme.gpu_count == gpu_count
        assert scheme.backend == "sdpa"
        assert scheme.parallel_mode == "sp"
        assert scheme.sp_degree == gpu_count
        assert scheme.compile_enabled is False
        assert scheme.modulation_fusion is False
        assert scheme.cfg_parallel is False
        assert scheme.vae_sp is True
        assert scheme.vae_encode_sp is True
        assert scheme.controls == ("R0",)
        assert scheme.expected_effective_backend == "sdpa_sp"

        command = build_service_command(scheme, make_config(tmp_path))
        assert option_value(command, "--attention-backend") == "sdpa"
        assert "--vae-sp" in command
        assert "--vae-encode-sp" in command
        assert "--enable-torch-compile" not in command
        assert "--enable-cogvideox-modulation-fusion" not in command
        assert "--enable-cfg-parallel" not in command


@pytest.mark.parametrize(
    ("treatment_id", "control_id"),
    [
        ("R99_VAE_ENCODE_SP", "R99_VAE_SP"),
        ("R100_VAE_ENCODE_SP", "R100_VAE_SP"),
        ("R101_VAE_ENCODE_SP4", "R101_VAE_SP4"),
    ],
)
def test_vae_encode_sp_treatment_adds_only_encode_flag(
    tmp_path: Path, treatment_id: str, control_id: str
):
    treatment = build_service_command(ALL_SCHEMES[treatment_id], make_config(tmp_path))
    control = build_service_command(ALL_SCHEMES[control_id], make_config(tmp_path))
    assert treatment == control + ["--vae-encode-sp"]
    assert "--vae-sp" in treatment


@pytest.mark.parametrize("scheme_id", ["R99_VAE_SP", "R100_VAE_SP"])
def test_vae_sp_treatment_adds_only_vae_sp_to_control_command(
    tmp_path: Path, scheme_id: str
):
    treatment = ALL_SCHEMES[scheme_id]
    control = SCHEMES[treatment.controls[0]]
    treatment_command = build_service_command(treatment, make_config(tmp_path))
    control_command = build_service_command(control, make_config(tmp_path))
    assert treatment_command == control_command + ["--vae-sp"]


def test_r101_vae_sp4_command_keeps_fusion_and_disables_cfg(tmp_path: Path):
    command = build_service_command(ALL_SCHEMES["R101_VAE_SP4"], make_config(tmp_path))

    assert option_value(command, "--num-gpus") == "4"
    assert option_value(command, "--sp-degree") == "4"
    assert option_value(command, "--ulysses-degree") == "4"
    assert option_value(command, "--vividvr-parallel-mode") == "sp"
    assert "--enable-torch-compile" in command
    assert "--enable-cogvideox-modulation-fusion" in command
    assert "--vae-sp" in command
    assert "--enable-cfg-parallel" not in command


def test_vae_sp_formal_defaults_follow_mock_test_service_contract():
    config = BenchmarkConfig()
    assert config.model_path == Path("/home/zhiheng/ckpts/CogVideoX1.5-5B")
    assert config.vividvr_path == Path("/home/zhiheng/ckpts/Vivid-VR")
    assert config.service_port == 31221
    assert config.caption_port == 31200
    assert config.callback_port == 39090
    assert config.s3_port == 4566
    assert config.s3_bucket == "flowcut"

    r99 = build_service_command(ALL_SCHEMES["R99_VAE_SP"], config)
    assert r99[r99.index("--model-path") + 1] == str(config.model_path)
    assert r99[r99.index("--component-paths.vividvr") + 1] == str(config.vividvr_path)
    assert "--vividvr-caption-bridge" in r99
    assert "--vae-sp" in r99


def test_vae_sp_formal_request_keeps_flowcut_contract(tmp_path: Path):
    config = replace(BenchmarkConfig(), output_root=tmp_path)
    payload = build_request_payload(
        config,
        role=RunRole.FORMAL,
        task_id="r99-vae-sp-formal",
        callback_url="http://127.0.0.1:39090/tasks/r99/callback",
        output_path=tmp_path / "service-output.mp4",
        perf_path=tmp_path / "perf.json",
    )
    assert payload["num_inference_steps"] == 20
    assert payload["seed"] == 42
    assert payload["num_temporal_process_frames"] == 121
    assert payload["callbackUrl"].startswith("http://127.0.0.1:39090/")
    assert payload["minioConfig"]["endpoint"] == "127.0.0.1:4566"
    assert "caption_file_path" not in payload
    assert "prompt_file_path" not in payload


@pytest.mark.parametrize(
    ("scheme_id", "gpus", "backend", "sp", "mode", "compile_enabled"),
    [
        ("R0", 1, "sdpa", 1, "single", False),
        ("R1", 1, "fa", 1, "single", False),
        ("R2", 1, "fa", 1, "single", True),
        ("R3", 2, "fa", 2, "sp", True),
        ("R4", 4, "fa", 4, "sp", True),
        ("R5", 4, "fa", 2, "cfg_sp", True),
        ("R6", 1, "fa", 1, "single", True),
        ("R99", 2, "fa", 2, "sp", True),
        ("R100", 4, "fa", 2, "cfg_sp", True),
    ],
)
def test_service_command_maps_fixed_topology(
    tmp_path: Path,
    scheme_id: str,
    gpus: int,
    backend: str,
    sp: int,
    mode: str,
    compile_enabled: bool,
):
    command = build_service_command(SCHEMES[scheme_id], make_config(tmp_path))

    assert command[:2] == [
        str(tmp_path / ".venv/bin/sglang"),
        "serve",
    ]
    assert option_value(command, "--num-gpus") == str(gpus)
    assert option_value(command, "--attention-backend") == backend
    assert option_value(command, "--sp-degree") == str(sp)
    assert option_value(command, "--ulysses-degree") == str(sp)
    assert option_value(command, "--ring-degree") == "1"
    assert option_value(command, "--vividvr-parallel-mode") == mode
    assert ("--enable-torch-compile" in command) is compile_enabled
    assert ("--enable-cfg-parallel" in command) is (mode == "cfg_sp")


@pytest.mark.parametrize("scheme_id", ["R6", "R99", "R100"])
def test_modulation_fusion_schemes_use_only_verified_fusion(
    tmp_path: Path, scheme_id: str
):
    command = build_service_command(SCHEMES[scheme_id], make_config(tmp_path))

    assert "--enable-cogvideox-modulation-fusion" in command
    assert option_value(command, "--cogvideox-modulation-fusion-targets") == (
        "transformer,controlnet"
    )
    assert "--enable-cogvideox-qkv-fusion" not in command
    assert "--enable-cogvideox-qk-norm-fusion" not in command
    assert "--enable-cogvideox-qk-norm-rope-fusion" not in command


def test_r100_command_enables_cfg_sp_compile_and_modulation(tmp_path: Path):
    command = build_service_command(SCHEMES["R100"], make_config(tmp_path))
    assert "--enable-cfg-parallel" in command
    assert option_value(command, "--sp-degree") == "2"
    assert "--enable-torch-compile" in command
    assert "--enable-cogvideox-modulation-fusion" in command


def test_distributed_environment_uses_selected_gpus_and_global_context(tmp_path: Path):
    config = make_config(tmp_path)
    environment = build_service_environment(SCHEMES["R3"], config)

    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert environment["SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE"] == ("eager_global")
    assert "SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE" not in (
        build_service_environment(SCHEMES["R2"], config)
    )


def test_compile_environment_injects_existing_python_dev_headers(
    monkeypatch, tmp_path: Path
):
    include_dir = tmp_path / "python-dev" / "usr" / "include" / "python3.10"
    include_dir.mkdir(parents=True)
    (include_dir / "Python.h").write_text("/* test header */\n", encoding="utf-8")
    multiarch = sysconfig.get_config_var("MULTIARCH")
    assert multiarch
    multiarch_dir = include_dir.parent / multiarch / include_dir.name
    multiarch_dir.mkdir(parents=True)
    (multiarch_dir / "pyconfig.h").write_text(
        "/* test multiarch config */\n", encoding="utf-8"
    )
    monkeypatch.setenv("SGLANG_PYTHON_DEV_INCLUDE", str(include_dir))
    monkeypatch.setenv("CPATH", "/existing/cpath")
    monkeypatch.setenv("C_INCLUDE_PATH", "/existing/c-include")

    environment = build_service_environment(SCHEMES["R2"], make_config(tmp_path))

    expected_prefix = os.pathsep.join((str(include_dir.parent), str(include_dir)))
    assert environment["CPATH"] == os.pathsep.join((expected_prefix, "/existing/cpath"))
    assert environment["C_INCLUDE_PATH"] == os.pathsep.join(
        (expected_prefix, "/existing/c-include")
    )


def test_unsupported_scheme_cannot_build_service_command(tmp_path: Path):
    with pytest.raises(BenchmarkConfigError, match="R8.*unsupported"):
        build_service_command(SCHEMES["R8"], make_config(tmp_path))


def make_perf_fixture(
    *,
    requested_backend: str = "fa",
    effective_backend: str = "fa_sp",
    mode: str = "sp",
    sp_world_size: int = 2,
    cfg_enabled: bool = False,
    compile_enabled: bool = True,
    modulation_fusion: bool = False,
    vae_sp: bool = False,
    vae_encode_sp: bool = False,
) -> dict:
    stage_ms = {
        "VividVRInputValidationStage": 100.0,
        "VividVRPromptPreparationStage": 100.0,
        "VividVRTemporalWindowPlanningStage": 100.0,
        "VividVRLongClipPreparationStage": 200.0,
        "VividVRTimestepPreparationStage": 100.0,
        "VividVRMultiClipDenoisingStage": 8000.0,
        "VividVRMultiClipDecodeTrimStage": 100.0,
        "VividVRTemporalStitchPostprocessStage": 100.0,
    }
    perf = {
        "total_duration_ms": 10000.0,
        "steps": [
            {"name": name, "duration_ms": duration_ms}
            for name, duration_ms in stage_ms.items()
        ],
        "denoise_steps_ms": [
            {"step": 0, "duration_ms": 600.0},
            {"step": 1, "duration_ms": 300.0},
            {"step": 2, "duration_ms": 400.0},
            {"step": 3, "duration_ms": 500.0},
        ],
        "meta": {
            "vividvr_debug": {
                "num_clips": 2,
                "attention_backend_requested": requested_backend,
                "attention_backend_transformer": effective_backend,
                "attention_backend_controlnet": effective_backend,
                "vividvr_parallel_mode": mode,
                "sp_world_size": sp_world_size,
                "cfg_parallel_enabled": cfg_enabled,
                "torch_compile_requested": compile_enabled,
                "torch_compile_transformer": compile_enabled,
                "torch_compile_controlnet": compile_enabled,
                "modulation_fusion_requested": modulation_fusion,
                "modulation_fusion_transformer": (
                    "sglang_modulation_fused_ops" if modulation_fusion else None
                ),
                "modulation_fusion_controlnet": (
                    "sglang_modulation_fused_ops" if modulation_fusion else None
                ),
            }
        },
    }
    if vae_sp:
        base_count, remainder = divmod(15, sp_world_size)
        local_tile_counts = [
            base_count + (rank < remainder) for rank in range(sp_world_size)
        ]
        perf["meta"]["vividvr_debug"].update(
            {
                "vae_sp_requested": True,
                "vae_sp_effective": True,
                "vae_sp_fallback_reason": "effective",
                "vae_sp_world_size": sp_world_size,
                "vae_sp_group_type": "sp",
                "vae_total_tiles": 15,
                "vae_local_tiles_per_rank": local_tile_counts,
                "vae_tile_decode_seconds": 1.25,
                "vae_tile_gather_seconds": 0.25,
                "vae_tile_merge_seconds": 0.1,
                "vae_decode_seconds": 1.6,
            }
        )
    if vae_encode_sp:
        base_count, remainder = divmod(32, sp_world_size)
        local_tile_counts = [
            base_count + (rank < remainder) for rank in range(sp_world_size)
        ]
        clip_count = perf["meta"]["vividvr_debug"]["num_clips"]
        clip_total = 32 // clip_count
        clip_base, clip_remainder = divmod(clip_total, sp_world_size)
        clip_local_counts = [
            clip_base + (rank < clip_remainder) for rank in range(sp_world_size)
        ]
        clip_stats = {
            "vae_encode_sp_requested": True,
            "vae_encode_sp_effective": True,
            "vae_encode_sp_fallback_reason": "effective",
            "vae_encode_sp_world_size": sp_world_size,
            "vae_encode_sp_group_type": "sp",
            "vae_encode_total_tiles": clip_total,
            "vae_encode_local_tiles_per_rank": clip_local_counts,
            "vae_encode_tile_compute_seconds": 1.0,
            "vae_encode_tile_gather_seconds": 0.2,
            "vae_encode_tile_merge_seconds": 0.1,
            "vae_encode_seconds": 1.3,
        }
        perf["meta"]["vividvr_debug"].update(
            {
                "vae_encode_sp_requested": True,
                "vae_encode_sp_effective": True,
                "vae_encode_sp_fallback_reason": "effective",
                "vae_encode_sp_world_size": sp_world_size,
                "vae_encode_sp_group_type": "sp",
                "vae_encode_total_tiles": 32,
                "vae_encode_local_tiles_per_rank": local_tile_counts,
                "vae_encode_tile_compute_seconds": 2.0,
                "vae_encode_tile_gather_seconds": 0.4,
                "vae_encode_tile_merge_seconds": 0.2,
                "vae_encode_seconds": 2.6,
                "vae_encode_sp_clips": [dict(clip_stats) for _ in range(clip_count)],
            }
        )
    return perf


def test_summarize_perf_computes_table_metrics():
    summary = summarize_perf(make_perf_fixture())

    assert tuple(summary.stage_seconds) == VIVIDVR_STAGE_NAMES
    assert summary.model_inference_runtime_seconds == 10.0
    assert summary.denoising_runtime_seconds == 8.0
    assert summary.unclassified_seconds == pytest.approx(1.2)
    assert summary.denoise_fraction == pytest.approx(0.8)
    assert summary.mean_step_seconds == pytest.approx(0.45)
    assert summary.steady_step_median_seconds == pytest.approx(0.4)
    assert summary.temporal_clip_count == 2
    assert summary.inference_step_count == 4


def test_summarize_perf_rejects_missing_or_duplicate_stages():
    perf = make_perf_fixture()
    perf["steps"].pop()
    with pytest.raises(BenchmarkDataError, match="missing stages"):
        summarize_perf(perf)

    perf = make_perf_fixture()
    perf["steps"].append(perf["steps"][0])
    with pytest.raises(BenchmarkDataError, match="duplicate stage"):
        summarize_perf(perf)


def test_validate_effective_config_accepts_distributed_compile_and_fusion():
    validate_effective_config(SCHEMES["R99"], make_perf_fixture(modulation_fusion=True))


def test_validate_effective_config_rejects_wrong_sp_backend():
    with pytest.raises(BenchmarkDataError, match="effective backend"):
        validate_effective_config(
            SCHEMES["R3"], make_perf_fixture(effective_backend="fa")
        )


def test_validate_effective_config_rejects_compile_not_applied():
    perf = make_perf_fixture()
    perf["meta"]["vividvr_debug"]["torch_compile_controlnet"] = False
    with pytest.raises(BenchmarkDataError, match="torch.compile"):
        validate_effective_config(SCHEMES["R3"], perf)


def test_validate_effective_config_requires_effective_vae_sp_for_treatment():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True)
    validated = validate_effective_config(ALL_SCHEMES["R99_VAE_SP"], perf)
    assert validated["vae_sp_effective"] is True
    assert validated["vae_sp_world_size"] == 2


def test_validate_effective_config_accepts_vae_sp4_treatment():
    perf = make_perf_fixture(
        sp_world_size=4,
        modulation_fusion=True,
        vae_sp=True,
    )
    validated = validate_effective_config(ALL_SCHEMES["R101_VAE_SP4"], perf)
    assert validated["parallel_mode"] == "sp"
    assert validated["cfg_parallel_enabled"] is False
    assert validated["vae_sp_world_size"] == 4
    assert validated["vae_local_tiles_per_rank"] == [4, 4, 4, 3]


def test_validate_effective_config_rejects_vae_sp_silent_fallback():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True)
    perf["meta"]["vividvr_debug"]["vae_sp_effective"] = False
    perf["meta"]["vividvr_debug"]["vae_sp_fallback_reason"] = "sp_world_size_one"
    with pytest.raises(BenchmarkDataError, match="VAE SP expected effective"):
        validate_effective_config(ALL_SCHEMES["R99_VAE_SP"], perf)


def test_validate_effective_config_requires_effective_vae_encode_sp():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True, vae_encode_sp=True)
    validated = validate_effective_config(ALL_SCHEMES["R99_VAE_ENCODE_SP"], perf)
    assert validated["vae_encode_sp_effective"] is True
    assert validated["vae_encode_sp_world_size"] == 2
    assert validated["vae_encode_local_tiles_per_rank"] == [16, 16]
    assert len(validated["vae_encode_sp_clips"]) == 2


def test_validate_effective_config_rejects_vae_encode_sp_silent_fallback():
    perf = make_perf_fixture(modulation_fusion=True, vae_sp=True, vae_encode_sp=True)
    perf["meta"]["vividvr_debug"]["vae_encode_sp_effective"] = False
    perf["meta"]["vividvr_debug"]["vae_encode_sp_fallback_reason"] = "sp_world_size_one"
    with pytest.raises(BenchmarkDataError, match="VAE encode SP expected effective"):
        validate_effective_config(ALL_SCHEMES["R99_VAE_ENCODE_SP"], perf)


def formal_record(seconds: float, *, quality_passed: bool = True) -> dict:
    return {
        "status": "succeeded",
        "timings": {"model_inference_runtime_seconds": seconds},
        "quality": {"pass_compare": quality_passed},
    }


def test_compute_derived_metrics_uses_r0_and_declared_control():
    records = {
        "R0": formal_record(20.0),
        "R1": formal_record(10.0),
        "R2": formal_record(8.0),
    }
    derived = compute_derived_metrics(SCHEMES["R2"], records)

    assert derived["cumulative_speedup_vs_r0"] == pytest.approx(2.5)
    assert derived["control_scheme_id"] == "R1"
    assert derived["incremental_speedup"] == pytest.approx(1.25)
    assert derived["gpu_seconds"] == pytest.approx(8.0)
    assert derived["resource_efficiency_vs_r0"] == pytest.approx(2.5)


def test_r100_derived_metrics_select_fastest_quality_passing_control():
    records = {
        "R0": formal_record(20.0),
        "R4": formal_record(5.0),
        "R5": formal_record(4.0, quality_passed=False),
        "R100": formal_record(3.0),
    }
    derived = compute_derived_metrics(SCHEMES["R100"], records)

    assert derived["control_scheme_id"] == "R4"
    assert derived["incremental_speedup"] == pytest.approx(5.0 / 3.0)
    assert derived["gpu_seconds"] == pytest.approx(12.0)


def write_formal_record(
    path: Path,
    *,
    scheme_id: str = "R99",
    status: str = "succeeded",
    total: float,
    model: float,
    decode_trim: float,
    quality_passed: bool,
    ssim_mean: float = 0.99,
    ssim_min: float = 0.98,
    failed_frame_ratio: float = 0.0,
) -> dict:
    record = {
        "schema_version": 1,
        "batch_id": "historical-batch",
        "run_role": "formal",
        "scheme": {
            "scheme_id": scheme_id,
            "gpu_count": 2 if scheme_id == "R99" else 4,
        },
        "status": status,
        "timings": {
            "total_runtime_seconds": total,
            "model_inference_runtime_seconds": model,
            "stage_seconds": {"VividVRMultiClipDecodeTrimStage": decode_trim},
        },
        "quality": {
            "pass_compare": quality_passed,
            "ssim_mean": ssim_mean,
            "ssim_min": ssim_min,
            "failed_frame_ratio": failed_frame_ratio,
        },
    }
    atomic_write_json(path, record)
    return record


def formal_record_with_stage(*, total: float, model: float, decode_trim: float) -> dict:
    return {
        "status": "succeeded",
        "timings": {
            "total_runtime_seconds": total,
            "model_inference_runtime_seconds": model,
            "stage_seconds": {"VividVRMultiClipDecodeTrimStage": decode_trim},
        },
        "quality": {
            "pass_compare": True,
            "ssim_mean": 0.995,
            "ssim_min": 0.985,
            "failed_frame_ratio": 0.0,
        },
    }


def write_complete_encode_control(
    tmp_path: Path,
    *,
    treatment_id: str = "R99_VAE_ENCODE_SP",
    status: str = "quality_failed",
) -> Path:
    treatment = ALL_SCHEMES[treatment_id]
    control_id = treatment.controls[0]
    control = ALL_SCHEMES[control_id]
    config = BenchmarkConfig()
    control_dir = tmp_path / "encode-control"
    record = {
        "schema_version": 1,
        "batch_id": "historical-encode-control",
        "run_role": "formal",
        "scheme": benchmark_module._scheme_payload(control),
        "status": status,
        "inputs": {
            "input_video": str(config.input_video.resolve()),
            "caption_file": str(config.caption_file.resolve()),
            "reference_video": str(config.reference_video.resolve()),
            "num_frames": 130,
            "temporal_process_frames": 121,
            "inference_steps": 20,
            "seed": 42,
            "guidance_scale": 6.0,
            "restoration_guidance_scale": -1.0,
            "upscale": 1.0,
            "dtype": "bfloat16",
        },
        "runtime": {
            "requested_backend": control.backend,
            "effective_backend": control.expected_effective_backend,
            "parallel_mode": control.parallel_mode,
            "sp_world_size": control.sp_degree,
            "cfg_parallel_enabled": control.cfg_parallel,
            "torch_compile_applied": control.compile_enabled,
            "modulation_fusion_applied": control.modulation_fusion,
            "vae_sp_requested": True,
            "vae_sp_effective": True,
            "vae_sp_fallback_reason": "effective",
            "vae_sp_world_size": control.sp_degree,
            "vae_sp_group_type": "sp",
        },
        "timings": {
            "total_runtime_seconds": 120.0,
            "model_inference_runtime_seconds": 110.0,
            "stage_seconds": {
                "VividVRLongClipPreparationStage": 30.0,
                "VividVRMultiClipDenoisingStage": 60.0,
                "VividVRMultiClipDecodeTrimStage": 20.0,
            },
        },
        "quality": {
            "pass_compare": False,
            "ssim_mean": 0.99,
            "ssim_min": 0.98,
            "failed_frame_ratio": 0.0,
        },
    }
    atomic_write_json(control_dir / "records" / f"{control_id}_formal.json", record)
    return control_dir


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        (("scheme", "scheme_id"), "wrong"),
        (("scheme", "parallel_mode"), "cfg_sp"),
        (("inputs", "seed"), 41),
        (("inputs", "caption_file"), "/tmp/wrong.txt"),
        (("runtime", "vae_sp_effective"), False),
        (("runtime", "vae_encode_sp_effective"), True),
        (("timings", "stage_seconds", "VividVRLongClipPreparationStage"), None),
        (("timings", "stage_seconds", "VividVRMultiClipDenoisingStage"), None),
        (("timings", "stage_seconds", "VividVRMultiClipDecodeTrimStage"), None),
    ],
)
def test_encode_historical_control_rejects_identity_drift(
    tmp_path: Path, field_path: tuple[str, ...], bad_value
):
    control_dir = write_complete_encode_control(tmp_path)
    path = control_dir / "records/R99_VAE_SP_formal.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    target = record
    for key in field_path[:-1]:
        target = target[key]
    if bad_value is None:
        target.pop(field_path[-1])
    else:
        target[field_path[-1]] = bad_value
    atomic_write_json(path, record)
    with pytest.raises(BenchmarkDataError):
        load_historical_controls(control_dir, ALL_SCHEMES["R99_VAE_ENCODE_SP"])


def test_historical_control_snapshot_detects_content_change(tmp_path: Path):
    control_dir = write_complete_encode_control(tmp_path)
    controls = load_historical_controls(control_dir, ALL_SCHEMES["R99_VAE_ENCODE_SP"])
    path = control_dir / "records/R99_VAE_SP_formal.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(BenchmarkDataError, match="historical control changed"):
        verify_historical_controls_unchanged(controls)


def test_historical_control_snapshot_detects_mtime_only_change(tmp_path: Path):
    control_dir = write_complete_encode_control(tmp_path)
    controls = load_historical_controls(control_dir, ALL_SCHEMES["R99_VAE_ENCODE_SP"])
    path = control_dir / "records/R99_VAE_SP_formal.json"
    original = path.stat().st_mtime_ns
    os.utime(path, ns=(path.stat().st_atime_ns, original + 1_000_000))
    with pytest.raises(BenchmarkDataError, match="historical control changed"):
        verify_historical_controls_unchanged(controls)


def encode_perf_record(
    *,
    total: float,
    model: float,
    preparation: float,
    denoise: float,
    decode_trim: float,
) -> dict:
    record = formal_record_with_stage(total=total, model=model, decode_trim=decode_trim)
    record["timings"]["stage_seconds"].update(
        {
            "VividVRLongClipPreparationStage": preparation,
            "VividVRMultiClipDenoisingStage": denoise,
        }
    )
    return record


@pytest.mark.parametrize(
    ("scheme_id", "prep_speedup", "denoise_ratio", "decode_ratio", "passed"),
    [
        ("R99_VAE_ENCODE_SP", 1.5, 0.03, 0.03, True),
        ("R100_VAE_ENCODE_SP", 1.5, 0.030001, 0.03, False),
        ("R101_VAE_ENCODE_SP4", 2.5, 0.03, 0.030001, False),
        ("R101_VAE_ENCODE_SP4", 2.499, 0.0, 0.0, False),
    ],
)
def test_compute_vae_encode_sp_performance_gates(
    scheme_id: str,
    prep_speedup: float,
    denoise_ratio: float,
    decode_ratio: float,
    passed: bool,
):
    scheme = ALL_SCHEMES[scheme_id]
    control = encode_perf_record(
        total=120.0,
        model=110.0,
        preparation=30.0,
        denoise=60.0,
        decode_trim=20.0,
    )
    control["scheme"] = {"scheme_id": scheme.controls[0]}
    control["_control_record_snapshot"] = {
        "path": "/tmp/control.json",
        "sha256": "abc",
        "mtime_ns": 123,
    }
    treatment = encode_perf_record(
        total=100.0,
        model=100.0,
        preparation=30.0 / prep_speedup,
        denoise=60.0 * (1.0 + denoise_ratio),
        decode_trim=20.0 * (1.0 + decode_ratio),
    )
    derived = compute_vae_encode_sp_derived_metrics(scheme, treatment, control)
    assert derived["long_clip_preparation_speedup"] == pytest.approx(prep_speedup)
    assert derived["long_clip_preparation_gate"] is (
        prep_speedup >= (2.5 if scheme.sp_degree == 4 else 1.5)
    )
    assert derived["model_inference_improved"] is True
    assert derived["denoise_regression_ratio"] == pytest.approx(denoise_ratio)
    assert derived["decode_trim_regression_ratio"] == pytest.approx(decode_ratio)
    assert derived["performance_gates_passed"] is passed
    assert derived["control_record_sha256"] == "abc"


def test_load_historical_control_and_compute_vae_sp_speedups(tmp_path: Path):
    control_dir = tmp_path / "control"
    write_formal_record(
        control_dir / "records/R99_formal.json",
        total=551.119,
        model=544.321,
        decode_trim=100.274,
        quality_passed=True,
    )
    controls = load_historical_controls(control_dir, ALL_SCHEMES["R99_VAE_SP"])
    treatment = formal_record_with_stage(total=500.0, model=493.0, decode_trim=50.0)
    derived = compute_vae_sp_derived_metrics(
        ALL_SCHEMES["R99_VAE_SP"], treatment, controls["R99"]
    )
    assert derived["control_scheme_id"] == "R99"
    assert derived["decode_trim_speedup"] == pytest.approx(100.274 / 50.0)
    assert derived["model_inference_speedup"] == pytest.approx(544.321 / 493.0)
    assert derived["total_runtime_speedup"] == pytest.approx(551.119 / 500.0)


def test_load_historical_r100_accepts_recorded_quality_failed_control(
    tmp_path: Path,
):
    control_dir = tmp_path / "control"
    write_formal_record(
        control_dir / "records/R100_formal.json",
        scheme_id="R100",
        status="quality_failed",
        total=370.881,
        model=365.067,
        decode_trim=101.786,
        quality_passed=False,
        ssim_mean=0.9846193275671117,
        ssim_min=0.978691848628344,
        failed_frame_ratio=2 / 130,
    )
    controls = load_historical_controls(control_dir, ALL_SCHEMES["R100_VAE_SP"])
    assert controls["R100"]["status"] == "quality_failed"


@pytest.mark.parametrize(
    ("ssim_mean", "ssim_min", "failed_frame_ratio", "expected"),
    [
        (0.99, 0.98, 0.0, True),
        (0.989, 0.98, 0.0, False),
        (0.99, 0.979, 0.0, False),
        (0.99, 0.98, 0.01, False),
    ],
)
def test_quality_not_worse_than_control(
    ssim_mean: float,
    ssim_min: float,
    failed_frame_ratio: float,
    expected: bool,
):
    treatment = formal_record_with_stage(total=1.0, model=1.0, decode_trim=1.0)
    treatment["quality"].update(
        {
            "ssim_mean": ssim_mean,
            "ssim_min": ssim_min,
            "failed_frame_ratio": failed_frame_ratio,
        }
    )
    control = formal_record_with_stage(total=1.0, model=1.0, decode_trim=1.0)
    control["quality"].update(
        {"ssim_mean": 0.99, "ssim_min": 0.98, "failed_frame_ratio": 0.0}
    )
    assert quality_not_worse_than_control(treatment, control) is expected


def test_unsupported_record_contains_every_table_section(tmp_path: Path):
    record = build_unsupported_record(
        SCHEMES["R8"], make_config(tmp_path), batch_id="batch"
    )

    assert record["status"] == "unsupported"
    assert record["capability"]["reason"] == SCHEMES["R8"].unsupported_reason
    assert set(record) >= {
        "schema_version",
        "batch_id",
        "scheme",
        "capability",
        "inputs",
        "runtime",
        "timings",
        "gpu_memory",
        "quality",
        "derived",
        "artifacts",
        "reproducibility",
    }
    assert tuple(record["timings"]["stage_seconds"]) == VIVIDVR_STAGE_NAMES
    assert record["timings"]["sp_communication_seconds"] is None
    assert record["timings"]["sp_communication_reason"] == "not_profiled"


def test_atomic_write_json_replaces_complete_payload(tmp_path: Path):
    path = tmp_path / "nested/result.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2, "complete": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "value": 2,
        "complete": True,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_gpu_sampler_aggregates_per_device_peaks():
    sampler = GpuMemorySampler(
        [0, 1],
        sample_provider=iter(
            [
                {0: 1000.0, 1: 900.0},
                {0: 1100.0, 1: 1200.0},
            ]
        ),
        sampling_backend="fixture",
    )
    assert sampler.sample_once()
    assert sampler.sample_once()
    assert not sampler.sample_once()

    result = sampler.result()
    assert result["per_gpu_peak_mib"] == {"0": 1100.0, "1": 1200.0}
    assert result["max_single_gpu_peak_mib"] == 1200.0
    assert result["max_single_gpu_peak_gib"] == pytest.approx(1200.0 / 1024.0)
    assert result["sample_count"] == 2
    assert result["sampling_backend"] == "fixture"


class FakeCommandRunner:
    def __init__(self, *, stdout: str = ""):
        self.calls: list[list[str]] = []
        self.stdout = stdout

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        return CompletedProcess(command, 0, stdout=self.stdout, stderr="")


def test_tmux_manager_only_kills_owned_sessions(tmp_path: Path):
    fake = FakeCommandRunner()
    manager = TmuxManager(batch_id="batch", ownership_dir=tmp_path, command_runner=fake)

    manager.stop("vividvr_accel_batch_R0_service")
    assert fake.calls == []

    foreign = tmp_path / "vividvr_accel_batch_R1_service.json"
    foreign.write_text(
        '{"batch_id":"someone-else","session":"vividvr_accel_batch_R1_service"}',
        encoding="utf-8",
    )
    manager.stop("vividvr_accel_batch_R1_service")
    assert fake.calls == []
    assert foreign.exists()


def test_tmux_manager_starts_and_stops_owned_session(tmp_path: Path):
    fake = FakeCommandRunner()
    manager = TmuxManager(batch_id="batch", ownership_dir=tmp_path, command_runner=fake)
    log_path = tmp_path / "logs/service.log"

    manager.start(
        "vividvr_accel_batch_R0_service",
        ["python", "server.py", "--value", "two words"],
        log_path,
        environment={"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert fake.calls[0][:4] == [
        "tmux",
        "new-session",
        "-d",
        "-s",
    ]
    owner = tmp_path / "vividvr_accel_batch_R0_service.json"
    assert owner.exists()

    manager.stop("vividvr_accel_batch_R0_service")
    assert fake.calls[-1] == [
        "tmux",
        "kill-session",
        "-t",
        "vividvr_accel_batch_R0_service",
    ]
    assert not owner.exists()


def make_existing_config(tmp_path: Path) -> BenchmarkConfig:
    config = make_config(tmp_path)
    for directory in (config.model_path, config.vividvr_path):
        directory.mkdir(parents=True)
    for file_path in (
        config.python_executable,
        config.python_executable.parent / "sglang",
        config.python_executable.parent / "moto_server",
        config.input_video,
        config.caption_file,
        config.reference_video,
    ):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    return config


def test_preflight_read_only_mode_checks_paths_without_runtime_commands(
    tmp_path: Path,
):
    config = make_existing_config(tmp_path)
    fake = FakeCommandRunner()

    result = run_preflight(
        config,
        check_runtime_resources=False,
        command_runner=fake,
        which=lambda executable: f"/usr/bin/{executable}",
    )

    assert result["ok"] is True
    assert result["runtime_resources_checked"] is False
    assert fake.calls == []


def test_preflight_rejects_missing_required_path(tmp_path: Path):
    config = make_existing_config(tmp_path)
    config.reference_video.unlink()

    with pytest.raises(BenchmarkConfigError, match="reference_video"):
        run_preflight(
            config,
            check_runtime_resources=False,
            which=lambda executable: f"/usr/bin/{executable}",
        )


def test_preflight_rejects_busy_port_before_service_start(tmp_path: Path):
    config = make_existing_config(tmp_path)

    with pytest.raises(BenchmarkConfigError, match="occupied ports.*31221"):
        run_preflight(
            config,
            check_runtime_resources=True,
            command_runner=FakeCommandRunner(),
            which=lambda executable: f"/usr/bin/{executable}",
            port_checker=lambda host, port: port != 31221,
            gpu_process_checker=lambda gpu_ids: {},
        )


def test_preflight_rejects_unknown_gpu_processes(tmp_path: Path):
    config = make_existing_config(tmp_path)

    with pytest.raises(BenchmarkConfigError, match="GPU 2.*pid 991"):
        run_preflight(
            config,
            check_runtime_resources=True,
            command_runner=FakeCommandRunner(),
            which=lambda executable: f"/usr/bin/{executable}",
            port_checker=lambda host, port: True,
            gpu_process_checker=lambda gpu_ids: {
                2: [{"pid": 991, "process_name": "foreign.py"}]
            },
        )


def test_preflight_allows_existing_processes_when_explicitly_enabled_and_idle(
    tmp_path: Path,
):
    config = make_existing_config(tmp_path)
    config = BenchmarkConfig(
        **{
            **config.__dict__,
            "allow_idle_gpu_processes": True,
        }
    )

    result = run_preflight(
        config,
        check_runtime_resources=True,
        command_runner=FakeCommandRunner(),
        which=lambda executable: f"/usr/bin/{executable}",
        port_checker=lambda host, port: True,
        gpu_process_checker=lambda gpu_ids: {
            6: [{"pid": 991, "process_name": "resident.py"}]
        },
        gpu_utilization_checker=lambda gpu_ids: {gpu_id: 0 for gpu_id in gpu_ids},
    )

    assert result["gpu_process_policy"] == "allow_existing_when_idle"
    assert result["gpu_utilization_percent"] == {4: 0, 5: 0, 6: 0, 7: 0}
    assert result["gpu_processes"][6][0]["pid"] == 991


def test_preflight_rejects_existing_processes_when_any_selected_gpu_is_active(
    tmp_path: Path,
):
    config = make_existing_config(tmp_path)
    config = BenchmarkConfig(
        **{
            **config.__dict__,
            "allow_idle_gpu_processes": True,
        }
    )

    with pytest.raises(BenchmarkConfigError, match="GPU 6 utilization is 1%"):
        run_preflight(
            config,
            check_runtime_resources=True,
            command_runner=FakeCommandRunner(),
            which=lambda executable: f"/usr/bin/{executable}",
            port_checker=lambda host, port: True,
            gpu_process_checker=lambda gpu_ids: {
                6: [{"pid": 991, "process_name": "resident.py"}]
            },
            gpu_utilization_checker=lambda gpu_ids: {
                gpu_id: 1 if gpu_id == 6 else 0 for gpu_id in gpu_ids
            },
        )


def test_allow_idle_gpu_processes_flag_is_propagated_to_detached_batch():
    args = parse_args(["run-all", "--allow-idle-gpu-processes"])

    config = _config_from_args(args)

    assert config.allow_idle_gpu_processes is True
    assert "--allow-idle-gpu-processes" in _config_cli_arguments(config)


def test_s3_bucket_creation_explicitly_disables_proxy(monkeypatch, tmp_path: Path):
    import boto3

    captured: dict[str, object] = {}

    class FakeS3Client:
        def create_bucket(self, *, Bucket: str):
            captured["bucket"] = Bucket

    def fake_client(service_name: str, **kwargs):
        captured["service_name"] = service_name
        captured["client_kwargs"] = kwargs
        return FakeS3Client()

    monkeypatch.setattr(boto3, "client", fake_client)
    lifecycle = TmuxBenchmarkLifecycle(make_config(tmp_path), "test-batch")

    lifecycle._create_s3_bucket()

    assert captured["bucket"] == "flowcut"
    assert captured["client_kwargs"]["config"].proxies == {}


def test_download_result_uses_authenticated_s3_without_proxy(
    monkeypatch, tmp_path: Path
):
    import boto3

    captured: dict[str, object] = {}

    class FakeS3Client:
        def download_file(self, bucket: str, key: str, filename: str):
            captured["download"] = (bucket, key, Path(filename).name)
            Path(filename).write_bytes(b"private-moto-object")

    def fake_client(service_name: str, **kwargs):
        captured["service_name"] = service_name
        captured["client_kwargs"] = kwargs
        return FakeS3Client()

    monkeypatch.setattr(boto3, "client", fake_client)
    destination = tmp_path / "result.mp4"

    _download_result(
        "http://127.0.0.1:4566/flowcut/acceleration-benchmark/run.mp4",
        destination,
    )

    assert destination.read_bytes() == b"private-moto-object"
    assert captured["service_name"] == "s3"
    assert captured["download"] == (
        "flowcut",
        "acceleration-benchmark/run.mp4",
        "result.mp4.partial",
    )
    client_kwargs = captured["client_kwargs"]
    assert client_kwargs["endpoint_url"] == "http://127.0.0.1:4566"
    assert client_kwargs["aws_access_key_id"] == "test"
    assert client_kwargs["aws_secret_access_key"] == "test"
    assert client_kwargs["region_name"] == "us-east-1"
    assert client_kwargs["config"].proxies == {}


class FakeLifecycle:
    def __init__(self, *, fail_stop_scheme: str | None = None):
        self.events: list[tuple[str, str | None]] = []
        self.fail_stop_scheme = fail_stop_scheme

    def start_shared(self):
        self.events.append(("start_shared", None))

    def stop_shared(self):
        self.events.append(("stop_shared", None))

    def start_scheme(self, scheme):
        self.events.append(("start_scheme", scheme.scheme_id))

    def stop_scheme(self, scheme):
        self.events.append(("stop_scheme", scheme.scheme_id))
        if scheme.scheme_id == self.fail_stop_scheme:
            raise RuntimeError("cannot stop owned service")


class FakeRequestExecutor:
    def __init__(self, failures: set[tuple[str, str]] | None = None):
        self.order: list[tuple[str, str]] = []
        self.failures = failures or set()

    def __call__(self, scheme, role, *, batch_id, fingerprint):
        key = (scheme.scheme_id, role.value)
        self.order.append(key)
        if key in self.failures:
            return {
                "status": "failed",
                "failure": {"type": "FixtureFailure", "message": "failed"},
            }
        return {
            "status": "succeeded",
            "run_role": role.value,
            "timings": {
                "model_inference_runtime_seconds": float(100 - len(self.order))
            },
            "quality": {"pass_compare": True if role is RunRole.FORMAL else None},
        }


def make_runner(
    tmp_path: Path,
    *,
    lifecycle: FakeLifecycle | None = None,
    executor: FakeRequestExecutor | None = None,
    resume: bool = False,
) -> tuple[BenchmarkRunner, FakeLifecycle, FakeRequestExecutor]:
    config = make_config(tmp_path)
    actual_lifecycle = lifecycle or FakeLifecycle()
    actual_executor = executor or FakeRequestExecutor()
    runner = BenchmarkRunner(
        config=config,
        batch_id="batch",
        lifecycle=actual_lifecycle,
        request_executor=actual_executor,
        resume=resume,
    )
    return runner, actual_lifecycle, actual_executor


def test_runner_warms_only_compile_schemes_then_runs_formal(tmp_path: Path):
    runner, lifecycle, executor = make_runner(tmp_path)

    result = runner.run([SCHEMES["R0"], SCHEMES["R2"]])

    assert executor.order == [
        ("R0", "formal"),
        ("R2", "warmup"),
        ("R2", "formal"),
    ]
    assert lifecycle.events == [
        ("start_shared", None),
        ("start_scheme", "R0"),
        ("stop_scheme", "R0"),
        ("start_scheme", "R2"),
        ("stop_scheme", "R2"),
        ("stop_shared", None),
    ]
    assert result["status"] == "completed"
    warmup_record = json.loads(
        (tmp_path / "outputs/batch/records/R2_warmup.json").read_text(encoding="utf-8")
    )
    formal_record = json.loads(
        (tmp_path / "outputs/batch/records/R2_formal.json").read_text(encoding="utf-8")
    )
    assert warmup_record["inputs"]["inference_steps"] == 1
    assert formal_record["inputs"]["inference_steps"] == 20


def test_runner_skips_formal_after_warmup_failure_and_continues(tmp_path: Path):
    executor = FakeRequestExecutor({("R2", "warmup")})
    runner, _, _ = make_runner(tmp_path, executor=executor)

    result = runner.run([SCHEMES["R2"], SCHEMES["R3"]])

    assert executor.order == [
        ("R2", "warmup"),
        ("R3", "warmup"),
        ("R3", "formal"),
    ]
    assert result["schemes"]["R2"]["status"] == "failed"
    assert result["schemes"]["R3"]["status"] == "succeeded"
    failed_record = json.loads(
        (tmp_path / "outputs/batch/records/R2_warmup.json").read_text(encoding="utf-8")
    )
    assert {
        "schema_version",
        "scheme",
        "capability",
        "inputs",
        "runtime",
        "timings",
        "gpu_memory",
        "quality",
        "artifacts",
        "derived",
        "reproducibility",
    }.issubset(failed_record)
    assert tuple(failed_record["timings"]["stage_seconds"]) == VIVIDVR_STAGE_NAMES


def test_runner_records_unsupported_without_starting_service(tmp_path: Path):
    runner, lifecycle, executor = make_runner(tmp_path)

    result = runner.run([SCHEMES["R8"]])

    assert executor.order == []
    assert lifecycle.events == [
        ("start_shared", None),
        ("stop_shared", None),
    ]
    assert result["schemes"]["R8"]["status"] == "unsupported"
    assert (tmp_path / "outputs/batch/records/R8_unsupported.json").exists()


def test_runner_aborts_on_owned_service_cleanup_failure(tmp_path: Path):
    lifecycle = FakeLifecycle(fail_stop_scheme="R2")
    runner, _, _ = make_runner(tmp_path, lifecycle=lifecycle)

    with pytest.raises(BenchmarkCleanupError, match="R2"):
        runner.run([SCHEMES["R2"], SCHEMES["R3"]])

    assert ("start_scheme", "R3") not in lifecycle.events
    assert lifecycle.events[-1] == ("stop_shared", None)


def test_resume_reruns_warmup_when_only_previous_warmup_succeeded(
    tmp_path: Path,
):
    runner, _, first_executor = make_runner(tmp_path)
    first_executor.failures.add(("R2", "formal"))
    runner.run([SCHEMES["R2"]])

    resumed, _, executor = make_runner(tmp_path, resume=True)
    resumed.run([SCHEMES["R2"]])

    assert executor.order == [("R2", "warmup"), ("R2", "formal")]


def test_resume_skips_completed_formal_with_matching_fingerprint(tmp_path: Path):
    runner, _, _ = make_runner(tmp_path)
    runner.run([SCHEMES["R0"]])

    resumed, lifecycle, executor = make_runner(tmp_path, resume=True)
    result = resumed.run([SCHEMES["R0"]])

    assert executor.order == []
    assert ("start_scheme", "R0") not in lifecycle.events
    assert result["schemes"]["R0"]["status"] == "resumed"


def test_resume_skips_completed_quality_failed_formal_with_matching_fingerprint(
    tmp_path: Path,
):
    runner, _, _ = make_runner(tmp_path)
    fingerprint = compute_config_fingerprint(runner.config, SCHEMES["R0"])
    failed_quality = runner._stamp_record(
        {
            "status": "quality_failed",
            "quality": {"pass_compare": False},
        },
        SCHEMES["R0"],
        RunRole.FORMAL,
        fingerprint,
    )
    atomic_write_json(
        runner._record_path(SCHEMES["R0"], RunRole.FORMAL), failed_quality
    )

    resumed, lifecycle, executor = make_runner(tmp_path, resume=True)
    result = resumed.run([SCHEMES["R0"]])

    assert executor.order == []
    assert ("start_scheme", "R0") not in lifecycle.events
    assert result["schemes"]["R0"]["status"] == "resumed"


def test_config_fingerprint_changes_with_scheme_or_input_metadata(tmp_path: Path):
    config = make_existing_config(tmp_path)
    first = compute_config_fingerprint(config, SCHEMES["R2"])
    assert first != compute_config_fingerprint(config, SCHEMES["R3"])

    config.input_video.write_bytes(b"changed")
    assert first != compute_config_fingerprint(config, SCHEMES["R2"])


def test_request_payload_uses_caption_bridge_and_fixed_workload(tmp_path: Path):
    config = make_config(tmp_path)
    payload = build_request_payload(
        config,
        role=RunRole.FORMAL,
        task_id="batch-R2-formal",
        callback_url="http://127.0.0.1:39090/tasks/batch-R2-formal/callback",
        output_path=config.output_root / "result.mp4",
        perf_path=config.output_root / "perf.json",
    )

    assert payload["video_input_path"] == str(config.input_video)
    assert payload["num_inference_steps"] == 20
    assert payload["num_temporal_process_frames"] == 121
    assert payload["seed"] == 42
    assert payload["guidance_scale"] == 6.0
    assert payload["restoration_guidance_scale"] == -1.0
    assert payload["upscale"] == 1.0
    assert payload["outputObjectKey"] == "acceleration-benchmark/batch-R2-formal"
    assert payload["minioConfig"]["endpoint"] == "127.0.0.1:4566"
    assert "caption_file_path" not in payload
    assert "prompt" not in payload


def test_warmup_request_payload_uses_one_step(tmp_path: Path):
    config = make_config(tmp_path)

    payload = build_request_payload(
        config,
        role=RunRole.WARMUP,
        task_id="batch-R2-warmup",
        callback_url="http://127.0.0.1:39090/tasks/batch-R2-warmup/callback",
        output_path=config.output_root / "warmup.mp4",
        perf_path=config.output_root / "warmup-perf.json",
    )

    assert payload["num_inference_steps"] == 1


def test_warmup_request_uses_one_step(monkeypatch, tmp_path: Path):
    from sglang.multimodal_gen.tools import (
        run_flowcut_vividvr_service_acceptance as acceptance_module,
    )

    captured_payload: dict[str, object] = {}

    class StopAfterSubmit(Exception):
        pass

    class FakeGpuMemorySampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            return {}

    def capture_submit(*, payload, **_kwargs):
        captured_payload.update(payload)
        raise StopAfterSubmit

    monkeypatch.setattr(benchmark_module, "GpuMemorySampler", FakeGpuMemorySampler)
    monkeypatch.setattr(
        acceptance_module,
        "submit_flowcut_task_with_retry",
        capture_submit,
    )
    executor = FlowCutRequestExecutor(make_config(tmp_path))

    with pytest.raises(StopAfterSubmit):
        executor(
            SCHEMES["R2"],
            RunRole.WARMUP,
            batch_id="batch",
            fingerprint="fingerprint",
        )

    assert captured_payload["num_inference_steps"] == 1


def test_request_payload_rejects_path_outside_output_root(tmp_path: Path):
    config = make_config(tmp_path)

    with pytest.raises(BenchmarkConfigError, match="output_root"):
        build_request_payload(
            config,
            role=RunRole.FORMAL,
            task_id="task",
            callback_url="http://127.0.0.1:39090/callback",
            output_path=tmp_path.parent / "outside.mp4",
            perf_path=tmp_path / "outputs/perf.json",
        )


def test_dry_run_reports_fixed_matrix_without_runtime_commands(tmp_path: Path):
    config = make_existing_config(tmp_path)

    report = build_dry_run_report(config, list(SCHEMES.values()))

    assert report["scheme_count"] == 12
    assert report["preflight"]["runtime_resources_checked"] is False
    assert [item["scheme"]["scheme_id"] for item in report["schemes"]] == list(SCHEMES)
    assert report["schemes"][0]["requests"] == ["formal"]
    assert report["schemes"][2]["requests"] == ["warmup", "formal"]
    assert report["schemes"][7]["requests"] == []


def test_run_one_cli_requires_a_registered_scheme():
    args = parse_args(["run-one", "--scheme", "R99"])
    assert args.command == "run-one"
    assert args.scheme == "R99"

    with pytest.raises(SystemExit):
        parse_args(["run-one", "--scheme", "unknown"])
