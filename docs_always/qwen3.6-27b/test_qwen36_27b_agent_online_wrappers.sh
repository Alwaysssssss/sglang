#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_WRAPPER="${SCRIPT_DIR}/start_qwen36_27b_agent_online.sh"
STOP_WRAPPER="${SCRIPT_DIR}/stop_qwen36_27b_agent_online.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_env() {
  local env_file="$1"
  local name="$2"
  local expected="$3"

  grep -Fx "${name}=${expected}" "$env_file" >/dev/null || {
    echo "Captured environment:" >&2
    cat "$env_file" >&2
    fail "expected ${name}=${expected}"
  }
}

assert_no_env() {
  local env_file="$1"
  local name="$2"

  if grep -E "^${name}=" "$env_file" >/dev/null; then
    echo "Captured environment:" >&2
    cat "$env_file" >&2
    fail "did not expect ${name} to be exported"
  fi
}

assert_args() {
  local args_file="$1"
  shift
  local expected_file

  expected_file="$(mktemp)"
  printf '%s\n' "$@" > "$expected_file"
  if ! diff -u "$expected_file" "$args_file"; then
    rm -f "$expected_file"
    fail "argv mismatch"
  fi
  rm -f "$expected_file"
}

make_fake_script() {
  local path="$1"

  cat > "$path" <<'FAKE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
env | sort > "${CAPTURE_ENV:?CAPTURE_ENV is required}"
printf '%s\n' "$@" > "${CAPTURE_ARGS:?CAPTURE_ARGS is required}"
FAKE_SCRIPT
  chmod +x "$path"
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

fake_start="${tmp_dir}/fake_start.sh"
fake_stop="${tmp_dir}/fake_stop.sh"
make_fake_script "$fake_start"
make_fake_script "$fake_stop"

capture_env="${tmp_dir}/start.env"
capture_args="${tmp_dir}/start.args"
env -i \
  PATH="$PATH" \
  CAPTURE_ENV="$capture_env" \
  CAPTURE_ARGS="$capture_args" \
  QWEN36_AGENT_START_SCRIPT="$fake_start" \
  "$START_WRAPPER" --print-only smoke

assert_env "$capture_env" ROOT_DIR "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang"
assert_env "$capture_env" CONTEXT_LENGTH "262144"
assert_env "$capture_env" MAX_RUNNING_REQUESTS "4"
assert_env "$capture_env" MEMORY_TARGET_FRACTION "0.85"
assert_env "$capture_env" MAX_OUTPUT_TOKENS "32768"
assert_env "$capture_env" LOG_REQUESTS "0"
assert_env "$capture_env" EXPORT_METRICS_TO_FILE "0"
assert_env "$capture_env" ENABLE_REQUEST_TIME_STATS_LOGGING "0"
assert_env "$capture_env" LOG_DIR "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b_agent_online"
assert_env "$capture_env" PID_FILE "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b_agent_online/qwen36_27b_agent_online.pid"
assert_no_env "$capture_env" MAX_TOTAL_TOKENS
assert_args "$capture_args" --print-only smoke

capture_env="${tmp_dir}/start_override.env"
capture_args="${tmp_dir}/start_override.args"
env -i \
  PATH="$PATH" \
  CAPTURE_ENV="$capture_env" \
  CAPTURE_ARGS="$capture_args" \
  QWEN36_AGENT_START_SCRIPT="$fake_start" \
  CONTEXT_LENGTH="131072" \
  MAX_RUNNING_REQUESTS="2" \
  MEMORY_TARGET_FRACTION="0.75" \
  LOG_REQUESTS="1" \
  EXPORT_METRICS_TO_FILE="1" \
  "$START_WRAPPER"

assert_env "$capture_env" CONTEXT_LENGTH "131072"
assert_env "$capture_env" MAX_RUNNING_REQUESTS "2"
assert_env "$capture_env" MEMORY_TARGET_FRACTION "0.75"
assert_env "$capture_env" LOG_REQUESTS "1"
assert_env "$capture_env" EXPORT_METRICS_TO_FILE "1"
assert_args "$capture_args"

capture_env="${tmp_dir}/stop.env"
capture_args="${tmp_dir}/stop.args"
env -i \
  PATH="$PATH" \
  CAPTURE_ENV="$capture_env" \
  CAPTURE_ARGS="$capture_args" \
  QWEN36_AGENT_STOP_SCRIPT="$fake_stop" \
  "$STOP_WRAPPER" --force-port-check

assert_env "$capture_env" ROOT_DIR "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang"
assert_env "$capture_env" SGLANG_PORT "30000"
assert_env "$capture_env" SERVED_MODEL_NAME "qwen3.6-27b"
assert_env "$capture_env" LOG_DIR "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b_agent_online"
assert_env "$capture_env" PID_FILE "/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b_agent_online/qwen36_27b_agent_online.pid"
assert_args "$capture_args" --force-port-check

echo "qwen36 agent online wrapper tests passed"
