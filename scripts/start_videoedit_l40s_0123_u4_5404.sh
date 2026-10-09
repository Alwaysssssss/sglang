#!/usr/bin/env bash
# Dedicated addition: never replace/restart an existing container.
# SP4 normal+DMD service on physical GPUs 0-3, host port 5404.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SERVICE_DIR="$REPO_DIR/.local/videoedit-l40s.0123-5404"
export IMAGE_NAME=sglang-videoedit-dev-v2:l40s
export CONTAINER_NAME=videoedit_l40s_0123_u4_5404
export HOST_PORT=5404 CONTAINER_PORT=30000
export HOST_GPUS=0,1,2,3 CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3
export HOST_REPO_DIR="$REPO_DIR"
export PROJECT_ROOT="$(dirname -- "$REPO_DIR")"
export CONTAINER_REPO_DIR=/sgl-workspace/sglang
export WORKDIR_IN_CONTAINER="$CONTAINER_REPO_DIR"
export DUAL_SERVICE_DIR_HOST="$SCRIPT_DIR/videoedit_dual_service"
export DUAL_SERVICE_CONFIG_HOST="$DUAL_SERVICE_DIR_HOST/config.l40s.0123.u4.5404.env"
export DUAL_SERVICE_CONFIG_CONTAINER="$CONTAINER_REPO_DIR/scripts/videoedit_dual_service/config.l40s.0123.u4.5404.env"
export INPUT_SAVE_DIR="$SERVICE_DIR/dual/inputs"
export VIDEOEDIT_OUTPUT_DIR="$SERVICE_DIR/outputs"
export VIDEOEDIT_REQUEST_LOG_DIR="$SERVICE_DIR/request-logs"
export VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES=false
export CACHE_DIR="$SERVICE_DIR/cache"
export FLASHINFER_WORKSPACE_BASE="$CACHE_DIR/flashinfer"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export CONTAINER_MEMORY=0g CONTAINER_MEMORY_SWAP=0g
export VIDEOEDIT_QUEUE_CAPACITY=1 RUN_AS_USER=root
export REFUSE_EXISTING_CONTAINER=1 RECREATE=0 RESTART_EXISTING=0

# Complete all read-only checks before the generic launcher creates directories.
docker ps --format '{{.Names}} {{.Status}} {{.Ports}}'
if [[ -n "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")" ]]; then
  echo "Dedicated container already exists; refusing to replace/restart it." >&2
  exit 1
fi
expected_image=sha256:5a738194cb3b16f3dcc7ab3760500ac660ac9743f93076398082068c6f9d2fe6
actual_image="$(docker image inspect "$IMAGE_NAME" --format '{{.Id}}')"
if [[ "$actual_image" != "$expected_image" ]]; then
  echo "Image differs from the measured benchmark image; inspect before deployment." >&2
  exit 1
fi
python3 - <<'PY'
import socket
import subprocess

with socket.socket() as probe:
    probe.bind(('0.0.0.0', 5404))
raw = subprocess.check_output([
    'nvidia-smi', '--id=0,1,2,3', '--query-gpu=index,uuid,name,memory.used',
    '--format=csv,noheader,nounits'], text=True)
rows = [[v.strip() for v in line.split(',')] for line in raw.splitlines()]
if len(rows) != 4 or {r[0] for r in rows} != {'0', '1', '2', '3'}:
    raise RuntimeError('Unable to confirm physical GPUs 0-3')
if any(float(r[3]) > 64 for r in rows):
    raise RuntimeError(f'GPU 0-3 is occupied: {rows}')
processes = subprocess.check_output([
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
uuids = {r[1] for r in rows}
if any(line.split(',')[0].strip() in uuids for line in processes.splitlines()):
    raise RuntimeError('A compute process already uses GPU 0-3; refusing to launch')
print(raw, end='')
print('Preflight passed: port 5404 and GPUs 0-3 are free.')
PY
exec bash "$SCRIPT_DIR/start_videoedit_container.sh"
