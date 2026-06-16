import os
import subprocess
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).with_name("start_qwen36_27b_agent.sh")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def test_agent_start_dry_run_uses_256k_context_and_memory_based_concurrency(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    fake_python = bin_dir / "python3"

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    _write_executable(fake_python, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "nvidia-smi",
        """\
        #!/usr/bin/env bash
        if [[ "$*" == *"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu"* ]]; then
          printf '0, NVIDIA A100-SXM4-80GB, 81920, 0, 81920, 0\\n'
          printf '1, NVIDIA A100-SXM4-80GB, 81920, 0, 81920, 0\\n'
          printf '2, NVIDIA A100-SXM4-80GB, 81920, 0, 81920, 0\\n'
          printf '3, NVIDIA A100-SXM4-80GB, 81920, 0, 81920, 0\\n'
          exit 0
        fi
        exit 1
        """,
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ROOT_DIR": str(root_dir),
        "SGLANG_PY": str(fake_python),
        "MODEL_PATH": str(model_dir),
        "MODEL_SIZE_MIB": "53012",
        "ALLOW_EMPTY_API_KEY": "1",
        "DRY_RUN": "1",
        "WAIT_FOR_READY": "0",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "CONTEXT_LENGTH=262144" in output
    assert "MAX_OUTPUT_TOKENS=128000" in output
    assert "MEMORY_TARGET_FRACTION=0.90" in output
    assert "MAX_RUNNING_REQUESTS=8" in output
    assert "--context-length 262144" in output
    assert "--mem-fraction-static 0.900" in output
    assert "--max-running-requests 8" in output
    assert "--log-requests" in output
    assert "--log-requests-level 1" in output
    assert "--log-requests-format json" in output


def test_agent_start_dry_run_respects_existing_gpu_usage(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    fake_python = bin_dir / "python3"

    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    _write_executable(fake_python, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "nvidia-smi",
        """\
        #!/usr/bin/env bash
        if [[ "$*" == *"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu"* ]]; then
          printf '0, NVIDIA A100-SXM4-80GB, 81920, 56000, 25920, 0\\n'
          printf '1, NVIDIA A100-SXM4-80GB, 81920, 56000, 25920, 0\\n'
          printf '2, NVIDIA A100-SXM4-80GB, 81920, 56000, 25920, 0\\n'
          printf '3, NVIDIA A100-SXM4-80GB, 81920, 56000, 25920, 0\\n'
          exit 0
        fi
        exit 1
        """,
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ROOT_DIR": str(root_dir),
        "SGLANG_PY": str(fake_python),
        "MODEL_PATH": str(model_dir),
        "MODEL_SIZE_MIB": "53012",
        "ALLOW_EMPTY_API_KEY": "1",
        "DRY_RUN": "1",
        "WAIT_FOR_READY": "0",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "MAX_RUNNING_REQUESTS=1" in output
    assert "--mem-fraction-static 0.216" in output
    assert "--max-running-requests 1" in output
