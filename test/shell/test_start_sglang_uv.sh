#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/start_sglang_uv.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

make_fake_env() {
  local env_dir="$1"
  mkdir -p "$env_dir/bin"
  cat > "$env_dir/bin/python" <<'PYEOF'
#!/usr/bin/env bash
echo fake python
PYEOF
  chmod +x "$env_dir/bin/python"
}

make_fake_uv() {
  local bin_dir="$1"
  mkdir -p "$bin_dir"
  cat > "$bin_dir/uv" <<'UVEOF'
#!/usr/bin/env bash
printf '%q\n' "$@" > "${UV_CAPTURE:?missing UV_CAPTURE}"
UVEOF
  chmod +x "$bin_dir/uv"
}

run_script() {
  local out_file="$1"
  shift
  PATH="$TMP_DIR/bin:$PATH" UV_ENV_DIR="$TMP_DIR/env" UV_CAPTURE="$TMP_DIR/uv_args" "$SCRIPT" "$@" >"$out_file" 2>&1
}

run_script_ok() {
  local out_file="$1"
  shift
  if ! run_script "$out_file" "$@"; then
    cat "$out_file" >&2
    fail "script exited non-zero"
  fi
}

make_fake_env "$TMP_DIR/env"
make_fake_uv "$TMP_DIR/bin"

out="$TMP_DIR/default.out"
run_script_ok "$out"

if grep -q -- '--model-path\|sglang.launch_server\|Missing required --model-path' "$out"; then
  cat "$out" >&2
  fail "default startup must not require or launch a model"
fi

expected=$'run\n--directory\n'$REPO_ROOT$'/python\n--active\nbash'
actual="$(cat "$TMP_DIR/uv_args")"
[[ "$actual" == "$expected" ]] || fail "unexpected uv args. expected [$expected], got [$actual]"

grep -q "UV_PROJECT_ENVIRONMENT: $TMP_DIR/env" "$out" || fail "environment path was not reported"
grep -q "Starting uv shell" "$out" || fail "startup message should describe environment shell"

out="$TMP_DIR/command.out"
run_script_ok "$out" -- python -c 'print(123)'
expected=$'run\n--directory\n'$REPO_ROOT$'/python\n--active\npython\n-c\nprint\(123\)'
actual="$(cat "$TMP_DIR/uv_args")"
[[ "$actual" == "$expected" ]] || fail "unexpected command uv args. expected [$expected], got [$actual]"

echo "start_sglang_uv shell tests passed"
