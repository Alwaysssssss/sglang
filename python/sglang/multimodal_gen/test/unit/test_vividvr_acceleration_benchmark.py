import json
import os
import sysconfig
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from sglang.multimodal_gen.tools import (
    run_vividvr_acceleration_benchmark as benchmark_module,
)
from sglang.multimodal_gen.tools.run_vividvr_acceleration_benchmark import (
    BenchmarkDataError,
    BenchmarkCleanupError,
    BenchmarkConfig,
    BenchmarkConfigError,
    BenchmarkRunner,
    FlowCutRequestExecutor,
    GpuMemorySampler,
    RunRole,
    SCHEMES,
    SchemeStatus,
    TmuxBenchmarkLifecycle,
    TmuxManager,
    VIVIDVR_STAGE_NAMES,
    _config_cli_arguments,
    _config_from_args,
    _download_result,
    atomic_write_json,
    build_dry_run_report,
    build_request_payload,
    build_unsupported_record,
    build_service_command,
    build_service_environment,
    compute_derived_metrics,
    compute_config_fingerprint,
    parse_args,
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
    assert environment["CPATH"] == os.pathsep.join(
        (expected_prefix, "/existing/cpath")
    )
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
                "model_inference_runtime_seconds": float(
                    100 - len(self.order)
                )
            },
            "quality": {
                "pass_compare": True if role is RunRole.FORMAL else None
            },
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
        (tmp_path / "outputs/batch/records/R2_warmup.json").read_text(
            encoding="utf-8"
        )
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

    monkeypatch.setattr(
        benchmark_module, "GpuMemorySampler", FakeGpuMemorySampler
    )
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
    assert [item["scheme"]["scheme_id"] for item in report["schemes"]] == list(
        SCHEMES
    )
    assert report["schemes"][0]["requests"] == ["formal"]
    assert report["schemes"][2]["requests"] == ["warmup", "formal"]
    assert report["schemes"][7]["requests"] == []


def test_run_one_cli_requires_a_registered_scheme():
    args = parse_args(["run-one", "--scheme", "R99"])
    assert args.command == "run-one"
    assert args.scheme == "R99"

    with pytest.raises(SystemExit):
        parse_args(["run-one", "--scheme", "unknown"])
