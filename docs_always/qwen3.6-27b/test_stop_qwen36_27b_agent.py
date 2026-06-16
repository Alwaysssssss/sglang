import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("stop_qwen36_27b_agent.sh")


def _start_fake_sglang(model_path: Path, served_model_name: str = "qwen3.6-27b"):
    argv0 = (
        "python -m sglang.launch_server "
        f"--model-path {model_path} "
        f"--served-model-name {served_model_name}"
    )
    return subprocess.Popen(
        ["bash", "-c", f"exec -a {shlex.quote(argv0)} sleep 1000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_exit(process: subprocess.Popen, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.05)
    return process.poll() is not None


def _cleanup_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    if _wait_for_exit(process, timeout=1.0):
        return
    process.kill()
    process.wait(timeout=2)


def test_agent_stop_stops_matching_pid_file_process_and_removes_pid_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    pid_file = log_dir / "qwen36_27b_agent.pid"
    process = _start_fake_sglang(model_dir)

    try:
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=SCRIPT.parents[2],
            env={
                **os.environ,
                "ROOT_DIR": str(root_dir),
                "MODEL_PATH": str(model_dir),
                "LOG_DIR": str(log_dir),
                "PID_FILE": str(pid_file),
                "STOP_TIMEOUT_SECONDS": "2",
                "SGLANG_PORT": "39876",
            },
            text=True,
            capture_output=True,
            timeout=10,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert f"Stopping PID {process.pid}" in output
        assert "Stop command completed" in output
        assert not pid_file.exists()
        assert _wait_for_exit(process)
    finally:
        _cleanup_process(process)


def test_agent_stop_refuses_non_matching_pid_and_keeps_pid_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    pid_file = log_dir / "qwen36_27b_agent.pid"
    process = subprocess.Popen(
        ["sleep", "1000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=SCRIPT.parents[2],
            env={
                **os.environ,
                "ROOT_DIR": str(root_dir),
                "MODEL_PATH": str(model_dir),
                "LOG_DIR": str(log_dir),
                "PID_FILE": str(pid_file),
                "STOP_TIMEOUT_SECONDS": "1",
                "SGLANG_PORT": "39877",
            },
            text=True,
            capture_output=True,
            timeout=10,
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert f"Refusing to stop PID {process.pid}" in output
        assert pid_file.exists()
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=2)
