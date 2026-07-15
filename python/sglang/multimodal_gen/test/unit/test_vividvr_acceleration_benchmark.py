from pathlib import Path
from subprocess import CompletedProcess

import pytest

from sglang.multimodal_gen.tools.run_vividvr_acceleration_benchmark import (
    BenchmarkDataError,
    BenchmarkConfig,
    BenchmarkConfigError,
    GpuMemorySampler,
    SCHEMES,
    SchemeStatus,
    TmuxManager,
    VIVIDVR_STAGE_NAMES,
    atomic_write_json,
    build_unsupported_record,
    build_service_command,
    build_service_environment,
    compute_derived_metrics,
    run_preflight,
    summarize_perf,
    validate_effective_config,
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
    assert {
        key for key, value in SCHEMES.items() if not value.executable
    } == {"R7", "R8", "R9"}
    assert all(
        SCHEMES[key].status is SchemeStatus.UNSUPPORTED
        for key in ("R7", "R8", "R9")
    )
    assert all(SCHEMES[key].unsupported_reason for key in ("R7", "R8", "R9"))


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
    assert environment["SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE"] == (
        "eager_global"
    )
    assert "SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE" not in (
        build_service_environment(SCHEMES["R2"], config)
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
    return {
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
                    True if modulation_fusion else None
                ),
                "modulation_fusion_controlnet": (
                    True if modulation_fusion else None
                ),
            }
        },
    }


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
    validate_effective_config(
        SCHEMES["R99"], make_perf_fixture(modulation_fusion=True)
    )


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

    import json

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
    manager = TmuxManager(
        batch_id="batch", ownership_dir=tmp_path, command_runner=fake
    )

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
    manager = TmuxManager(
        batch_id="batch", ownership_dir=tmp_path, command_runner=fake
    )
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
