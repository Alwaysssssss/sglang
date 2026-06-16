import os
import subprocess
import textwrap
import time
from pathlib import Path


SCRIPT = Path(__file__).with_name("setup_cloudflare_named_tunnel.sh")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(0o755)


def test_reuses_existing_tunnel_id_without_systemctl(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"
    config_file = tmp_path / "config.yml"
    run_dir = tmp_path / "run"
    api_key_file = tmp_path / "api_key"
    credentials_file = tmp_path / "existing.json"
    api_key_file.write_text("test-key", encoding="utf-8")
    credentials_file.write_text("{}", encoding="utf-8")

    _write_executable(
        bin_dir / "curl",
        """\
        #!/usr/bin/env bash
        exit 0
        """,
    )
    _write_executable(
        bin_dir / "systemctl",
        f"""\
        #!/usr/bin/env bash
        echo "systemctl $*" >> {calls}
        exit 99
        """,
    )
    _write_executable(
        bin_dir / "cloudflared",
        f"""\
        #!/usr/bin/env bash
        echo "cloudflared $*" >> {calls}
        if [[ "$*" == "tunnel list --output json" ]]; then
          printf '[]\\n'
          exit 0
        fi
        if [[ "$*" == tunnel\\ create* ]]; then
          echo "unexpected create" >&2
          exit 17
        fi
        pidfile=""
        logfile=""
        previous=""
        for arg in "$@"; do
          if [[ "$previous" == "--pidfile" ]]; then pidfile="$arg"; fi
          if [[ "$previous" == "--logfile" ]]; then logfile="$arg"; fi
          previous="$arg"
        done
        if [[ -n "$pidfile" ]]; then
          mkdir -p "$(dirname "$pidfile")"
          echo "$$" > "$pidfile"
        fi
        if [[ -n "$logfile" ]]; then
          mkdir -p "$(dirname "$logfile")"
          echo "fake cloudflared log" > "$logfile"
        fi
        exit 0
        """,
    )

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TUNNEL_ID": "396a68e0-e58b-45bb-9364-0e594a58fa03",
        "CREDENTIALS_FILE": str(credentials_file),
        "CONFIG_FILE": str(config_file),
        "RUN_DIR": str(run_dir),
        "API_KEY_FILE": str(api_key_file),
        "PUBLIC_HOSTNAME": "mgtvqwen36-apiexample.com",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    config = config_file.read_text(encoding="utf-8")
    assert "tunnel: 396a68e0-e58b-45bb-9364-0e594a58fa03" in config
    assert f"credentials-file: {credentials_file}" in config
    for _ in range(20):
        logged_calls = calls.read_text(encoding="utf-8")
        if f"--pidfile {run_dir}/cloudflared.pid" in logged_calls:
            break
        time.sleep(0.05)
    assert "cloudflared tunnel create" not in logged_calls
    assert "systemctl" not in logged_calls
    assert f"--config {config_file}" in logged_calls
    assert f"--pidfile {run_dir}/cloudflared.pid" in logged_calls
