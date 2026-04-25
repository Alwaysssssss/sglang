#!/usr/bin/env bash

set -euo pipefail

sudo chown -R "$(whoami)" /sgl-workspace/sglang
sudo chown -R "$(whoami)" ~/.codex ~/.claude 2>/dev/null || true

mkdir -p "$HOME/.codex"

CONFIG_FILE="$HOME/.codex/config.toml"
PROJECT_HEADER="[projects.'/sgl-workspace/sglang']"

if [ ! -f "$CONFIG_FILE" ]; then
  cat >"$CONFIG_FILE" <<'EOF'
[projects.'/sgl-workspace/sglang']
trust_level = "trusted"
EOF
elif ! grep -Fq "$PROJECT_HEADER" "$CONFIG_FILE"; then
  cat >>"$CONFIG_FILE" <<'EOF'

[projects.'/sgl-workspace/sglang']
trust_level = "trusted"
EOF
fi

if command -v codex >/dev/null 2>&1; then
  codex --version || true
fi

if command -v claude >/dev/null 2>&1; then
  claude --version || true
fi

# Point HuggingFace cache to the shared model directory
grep -q 'HF_HOME' ~/.zshrc 2>/dev/null || {
  echo 'export HF_HOME=/models/huggingface_cache' >> ~/.zshrc
  echo 'export TRANSFORMERS_CACHE=/models/huggingface_cache' >> ~/.zshrc
}
