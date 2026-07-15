from pathlib import Path

import pytest

from sglang.multimodal_gen.tools.run_vividvr_acceleration_benchmark import (
    BenchmarkConfig,
    BenchmarkConfigError,
    SCHEMES,
    SchemeStatus,
    build_service_command,
    build_service_environment,
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
