#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhiheng/sglang"
PYTHON="$ROOT/.venv/bin/python"
SGLANG_BIN="$ROOT/.venv/bin/sglang"
BENCH_PY="$ROOT/Vivid_Acceptance/tmp/run_vividvr_service_benchmark.py"
LOG_DIR="$ROOT/Vivid_Acceptance/logs"
RESULT_DIR="$ROOT/Vivid_Acceptance/result_videos/service_benchmark"
INDICATOR_DIR="$ROOT/Vivid_Acceptance/indicator"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

usage() {
  cat <<'EOF'
Usage:
  run_vividvr_service_ablation.sh LABEL [LABEL...]

Supported labels:
  single_gpu_sdpa_no_compile
  single_gpu_fa_no_compile
  single_gpu_sdpa_compile
  single_gpu_fa_compile
  dual_gpu_sdpa_deferred_no_compile
  dual_gpu_fa_deferred_no_compile
  dual_gpu_sdpa_eager_no_compile
  dual_gpu_fa_eager_no_compile
  dual_gpu_sdpa_deferred_compile
  dual_gpu_fa_deferred_compile
  dual_gpu_sdpa_eager_compile
  dual_gpu_fa_eager_compile
EOF
}

utc_stamp() {
  date -u +%Y%m%dT%H%M%SZ
}

wait_for_health() {
  local base_url="$1"
  local timeout_s="${2:-900}"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if curl --silent --show-error --fail --noproxy '*' "${base_url}/health" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed="$((now - start_ts))"
    if [[ "$elapsed" -ge "$timeout_s" ]]; then
      echo "Timed out waiting for ${base_url}/health after ${timeout_s}s" >&2
      return 1
    fi
    sleep 5
  done
}

wait_for_tmux_session_exit() {
  local session_name="$1"
  while tmux has-session -t "$session_name" 2>/dev/null; do
    sleep 15
  done
}

kill_session_if_exists() {
  local session_name="$1"
  tmux kill-session -t "$session_name" 2>/dev/null || true
}

find_latest_report() {
  local label="$1"
  python - "$label" "$INDICATOR_DIR" <<'PY'
import glob
import sys
from pathlib import Path

label = sys.argv[1]
indicator_dir = Path(sys.argv[2])
pattern = str(indicator_dir / f"vividvr-service-benchmark-long-130f-20step-{label}-*.json")
candidates = [
    Path(path)
    for path in glob.glob(pattern)
    if not path.endswith("_perf.json") and not path.endswith("_framewise_ssim.json")
]
if not candidates:
    sys.exit(1)
print(max(candidates, key=lambda path: path.stat().st_mtime_ns))
PY
}

run_label() {
  local label="$1"
  local gpus host port master_port scheduler_port attention_backend compile_flag
  local sp_degree ulysses_degree ring_degree tp_size
  local context_mode=""
  local cuda_visible_devices=""
  local extra_env=""
  local session_suffix service_session client_session base_url
  local serve_log client_log stamp
  local single_gpu_device single_port master_port_single scheduler_port_single

  single_gpu_device="${SGLANG_VIVIDVR_SINGLE_GPU_DEVICE:-0}"
  single_port="${SGLANG_VIVIDVR_SINGLE_PORT:-31190}"
  master_port_single="${SGLANG_VIVIDVR_SINGLE_MASTER_PORT:-30190}"
  scheduler_port_single="${SGLANG_VIVIDVR_SINGLE_SCHEDULER_PORT:-56190}"

  case "$label" in
    single_gpu_sdpa_no_compile)
      cuda_visible_devices="${single_gpu_device}"
      port="${single_port}"
      master_port="${master_port_single}"
      scheduler_port="${scheduler_port_single}"
      gpus=1
      tp_size=1
      sp_degree=1
      ulysses_degree=1
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag=""
      ;;
    single_gpu_fa_no_compile)
      cuda_visible_devices="${single_gpu_device}"
      port="${single_port}"
      master_port="${master_port_single}"
      scheduler_port="${scheduler_port_single}"
      gpus=1
      tp_size=1
      sp_degree=1
      ulysses_degree=1
      ring_degree=1
      attention_backend="fa"
      compile_flag=""
      ;;
    single_gpu_sdpa_compile)
      cuda_visible_devices="${single_gpu_device}"
      port="${single_port}"
      master_port="${master_port_single}"
      scheduler_port="${scheduler_port_single}"
      gpus=1
      tp_size=1
      sp_degree=1
      ulysses_degree=1
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag="--enable-torch-compile"
      ;;
    single_gpu_fa_compile)
      cuda_visible_devices="${single_gpu_device}"
      port="${single_port}"
      master_port="${master_port_single}"
      scheduler_port="${scheduler_port_single}"
      gpus=1
      tp_size=1
      sp_degree=1
      ulysses_degree=1
      ring_degree=1
      attention_backend="fa"
      compile_flag="--enable-torch-compile"
      ;;
    dual_gpu_sdpa_deferred_no_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag=""
      context_mode="deferred_global"
      ;;
    dual_gpu_fa_deferred_no_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="fa"
      compile_flag=""
      context_mode="deferred_global"
      ;;
    dual_gpu_sdpa_eager_no_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag=""
      context_mode="eager_global"
      ;;
    dual_gpu_fa_eager_no_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="fa"
      compile_flag=""
      context_mode="eager_global"
      ;;
    dual_gpu_sdpa_deferred_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag="--enable-torch-compile"
      context_mode="deferred_global"
      ;;
    dual_gpu_fa_deferred_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="fa"
      compile_flag="--enable-torch-compile"
      context_mode="deferred_global"
      ;;
    dual_gpu_sdpa_eager_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="torch_sdpa"
      compile_flag="--enable-torch-compile"
      context_mode="eager_global"
      ;;
    dual_gpu_fa_eager_compile)
      cuda_visible_devices="0,1"
      port=31191
      master_port=30191
      scheduler_port=56191
      gpus=2
      tp_size=1
      sp_degree=2
      ulysses_degree=2
      ring_degree=1
      attention_backend="fa"
      compile_flag="--enable-torch-compile"
      context_mode="eager_global"
      ;;
    *)
      echo "Unsupported label: $label" >&2
      usage >&2
      return 1
      ;;
  esac

  session_suffix="${label//[^a-zA-Z0-9]/_}"
  service_session="vividvr_serve_${session_suffix}"
  client_session="vividvr_bench_${session_suffix}"
  base_url="http://127.0.0.1:${port}"
  stamp="$(utc_stamp)"
  serve_log="$LOG_DIR/${service_session}_${stamp}.log"
  client_log="$LOG_DIR/${client_session}_${stamp}.log"

  kill_session_if_exists "$service_session"
  kill_session_if_exists "$client_session"

  extra_env="export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1"
  if [[ -n "$context_mode" ]]; then
    extra_env="${extra_env} && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=${context_mode}"
  fi

  echo "[$(utc_stamp)] starting service for ${label} at ${base_url}"
  tmux new-session -d -s "$service_session" \
    "cd ${ROOT} && mkdir -p Vivid_Acceptance/logs && ${extra_env} && CUDA_VISIBLE_DEVICES=${cuda_visible_devices} ${SGLANG_BIN} serve --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR --attention-backend ${attention_backend} --num-gpus ${gpus} --tp-size ${tp_size} --sp-degree ${sp_degree} --ulysses-degree ${ulysses_degree} --ring-degree ${ring_degree} ${compile_flag} --dist-timeout 3600 --host 127.0.0.1 --port ${port} --master-port ${master_port} --scheduler-port ${scheduler_port} --strict-ports --output-path ${RESULT_DIR} --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt 2>&1 | tee ${serve_log}"

  wait_for_health "$base_url" 1200
  echo "[$(utc_stamp)] service ready for ${label}"

  tmux new-session -d -s "$client_session" \
    "cd ${ROOT} && export PYTHONPATH=python && export PYTHONUNBUFFERED=1 && ${PYTHON} ${BENCH_PY} --base-url ${base_url} --label ${label} --poll-interval-seconds 15 2>&1 | tee ${client_log}"
  echo "[$(utc_stamp)] benchmark client started for ${label}"
  echo "  service: tmux attach -r -t ${service_session}"
  echo "  client : tmux attach -r -t ${client_session}"

  wait_for_tmux_session_exit "$client_session"
  echo "[$(utc_stamp)] benchmark client finished for ${label}"
  if ! latest_report="$(find_latest_report "$label")"; then
    echo "No benchmark report found for ${label} after client exit" >&2
    return 1
  fi
  echo "[$(utc_stamp)] latest report for ${label}: ${latest_report}"

  kill_session_if_exists "$service_session"
}

main() {
  if [[ "$#" -lt 1 ]]; then
    usage >&2
    exit 1
  fi

  for label in "$@"; do
    run_label "$label"
  done
}

main "$@"
