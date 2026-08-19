#!/usr/bin/env bash
#
# 向导——逐步引导用户完成人工流程。
# 由 /wizard 技能生成。
#
# “STAGES”标记上方的内容都是向导库：不要手动编辑。
# 请在标记下方编写每一步的阶段。

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────
# 向导库——顺畅而一致的用户体验。所有向导中的内容完全相同。
# ──────────────────────────────────────────────────────────────────────────

if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
  BOLD=$(tput bold); DIM=$(tput dim); RESET=$(tput sgr0)
  BLUE=$(tput setaf 4); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); RED=$(tput setaf 1)
else
  BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; YELLOW=""; RED=""
fi

# 作者在阶段区域顶部设置以下两个值。
TOTAL_STAGES=0
TOTAL_MINUTES=0

_STAGE_INDEX=0
_MINUTES_ELAPSED=0
ENV_FILE="${ENV_FILE:-.env}"
WRITTEN_ENV=()    # 本次运行写入 ENV_FILE 的键
WRITTEN_SECRET=() # 本次运行设置的密钥名称
SKIPPED=()        # 无法完成的事项（例如缺少 gh）

# _clear——清空终端，使屏幕只显示当前步骤。输出并非终端时不执行，
# 从而保持管道日志可读。
_clear() {
  [[ -t 1 ]] || return 0
  if command -v tput >/dev/null 2>&1; then tput clear; else printf '\033[2J\033[3J\033[H'; fi
}

# banner "标题"——开场画面：说明向导的作用和所需时间。
banner() {
  _clear
  printf '\n%s%s  %s%s\n' "$BOLD" "$BLUE" "$1" "$RESET"
  printf '%s  共 %s 个阶段 · 约 %s 分钟%s\n\n' \
    "$DIM" "$TOTAL_STAGES" "$TOTAL_MINUTES" "$RESET"
  printf '%s  你负责操作浏览器；本向导会准确说明每一步，并采集你复制回来的值。\n' "$DIM"
  printf '  可随时按 Ctrl-C 停止，之后重新运行；\n'
  printf '  已保存的值会被自动记住。%s\n' "$RESET"
  pause "准备开始了吗？"
}

# stage "名称" <分钟数>——清屏后显示阶段名称、进度和剩余时间。
# 清屏可确保屏幕只保留当前步骤。
stage() {
  _clear
  _STAGE_INDEX=$((_STAGE_INDEX + 1))
  local remaining=$((TOTAL_MINUTES - _MINUTES_ELAPSED))
  (( remaining < 0 )) && remaining=0
  _MINUTES_ELAPSED=$((_MINUTES_ELAPSED + ${2:-0}))
  printf '\n%s%s▸ 阶段 %s/%s · %s%s  %s（约剩 %s 分钟）%s\n' \
    "$BOLD" "$BLUE" "$_STAGE_INDEX" "$TOTAL_STAGES" "$1" "$RESET" "$DIM" "$remaining" "$RESET"
}

# say "..."——普通说明行。
say()  { printf '  %s\n' "$1"; }
# step "..."——用户在浏览器中执行的操作。
step() { printf '  %s•%s %s\n' "$BLUE" "$RESET" "$1"; }
note() { printf '  %s%s%s\n' "$DIM" "$1" "$RESET"; }
warn() { printf '  %s⚠ %s%s\n' "$YELLOW" "$1" "$RESET"; }

# open_url URL——在用户的浏览器中打开，支持包括 WSL 在内的多平台。
open_url() {
  local url="$1"
  printf '  %s↗ 正在打开%s %s\n' "$GREEN" "$RESET" "$url"
  { if   command -v wslview     >/dev/null 2>&1; then wslview "$url"
    elif command -v explorer.exe >/dev/null 2>&1; then explorer.exe "$url"
    elif command -v xdg-open    >/dev/null 2>&1; then xdg-open "$url"
    elif command -v open        >/dev/null 2>&1; then open "$url"
    else warn "无法打开浏览器——请手动访问：$url"; fi
  } >/dev/null 2>&1 || warn "无法打开浏览器——请手动访问：$url"
}

# pause "消息"——等待用户确认已完成人工操作。
pause() {
  printf '  %s%s%s ' "$DIM" "${1:-按回车键继续}" "$RESET"
  read -r _ || true
}

# confirm "问题"——是/否确认关卡；回答“是”时返回成功。
confirm() {
  local reply=""
  printf '  %s? %s [y/N，默认否] ' "$YELLOW" "$1"
  read -r reply || true
  [[ "$reply" =~ ^[Yy] ]]
}

# _existing KEY——KEY 在 ENV_FILE 中的当前值（如有）。
_existing() {
  [[ -f "$ENV_FILE" ]] || return 1
  local line; line=$(grep -E "^${1}=" "$ENV_FILE" | tail -n1) || return 1
  printf '%s' "${line#*=}"
}

# ask KEY "提示"——将值读取到 $KEY。重新运行时使用现有 .env 值作为默认值
# （按回车保留）。输入可见，适用于非密钥。
ask() {
  local key="$1" prompt="$2" current input
  current=$(_existing "$key" || true)
  if [[ -n "$current" ]]; then
    printf '  %s%s%s %s[按回车保留当前值]%s ' "$BOLD" "$prompt" "$RESET" "$DIM" "$RESET"
  else
    printf '  %s%s%s ' "$BOLD" "$prompt" "$RESET"
  fi
  read -r input || true
  [[ -z "$input" && -n "$current" ]] && input="$current"
  printf -v "$key" '%s' "$input"
}

# ask_secret KEY "提示"——与 ask 相同，但隐藏输入。
ask_secret() {
  local key="$1" prompt="$2" current input
  current=$(_existing "$key" || true)
  if [[ -n "$current" ]]; then
    printf '  %s%s%s %s[按回车保留当前值]%s ' "$BOLD" "$prompt" "$RESET" "$DIM" "$RESET"
  else
    printf '  %s%s%s ' "$BOLD" "$prompt" "$RESET"
  fi
  read -rs input || true
  printf '\n'
  [[ -z "$input" && -n "$current" ]] && input="$current"
  printf -v "$key" '%s' "$input"
}

# write_env KEY VALUE——在 ENV_FILE 中插入或更新 KEY=VALUE
# （不存在时创建，存在时替换原行）。操作幂等。
write_env() {
  local key="$1" value="$2" tmp
  touch "$ENV_FILE"
  tmp=$(mktemp)
  grep -vE "^${key}=" "$ENV_FILE" > "$tmp" || true
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
  WRITTEN_ENV+=("$key")
  printf '  %s✓ 已写入%s %s → %s\n' "$GREEN" "$RESET" "$key" "$ENV_FILE"
}

# set_secret NAME VALUE——通过 gh 设置 GitHub Actions 仓库密钥。
# 如果 gh 不可用或未认证，则显示并记录警告。
set_secret() {
  local name="$1" value="$2"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    if printf '%s' "$value" | gh secret set "$name" >/dev/null 2>&1; then
      WRITTEN_SECRET+=("$name")
      printf '  %s✓ 已设置%s GitHub 密钥 %s\n' "$GREEN" "$RESET" "$name"
      return
    fi
  fi
  SKIPPED+=("GitHub 密钥 $name（请手动设置：gh secret set $name）")
  warn "已跳过 GitHub 密钥 $name——gh 尚未就绪，请稍后设置"
}

# set_var NAME VALUE——设置 GitHub Actions 仓库变量（非密钥）。
set_var() {
  local name="$1" value="$2"
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    if gh variable set "$name" --body "$value" >/dev/null 2>&1; then
      printf '  %s✓ 已设置%s GitHub 变量 %s\n' "$GREEN" "$RESET" "$name"
      return
    fi
  fi
  SKIPPED+=("GitHub 变量 $name")
  warn "已跳过 GitHub 变量 $name——gh 尚未就绪，请稍后设置"
}

# finish——清屏后汇总所有已配置内容。
finish() {
  _clear
  printf '\n%s%s  ✓ 配置完成%s\n' "$BOLD" "$GREEN" "$RESET"
  (( ${#WRITTEN_ENV[@]} ))    && note "已向 $ENV_FILE 写入 ${#WRITTEN_ENV[@]} 个值：${WRITTEN_ENV[*]}"
  (( ${#WRITTEN_SECRET[@]} )) && note "已设置 ${#WRITTEN_SECRET[@]} 个 GitHub 密钥：${WRITTEN_SECRET[*]}"
  if (( ${#SKIPPED[@]} )); then
    printf '\n'; warn "仍需手动完成："
    for s in "${SKIPPED[@]}"; do note "  - $s"; done
  fi
  printf '\n'
}

# ──────────────────────────────────────────────────────────────────────────
# STAGES——在此区域编写内容。用户执行的每一步对应一个 stage()。
# 替换下方示例，并根据实际阶段设置两个总数。
# ──────────────────────────────────────────────────────────────────────────

TOTAL_STAGES=1
TOTAL_MINUTES=5

banner "Stripe 配置"

# ── 示例阶段：请替换为实际步骤 ─────────────────────────────────────────────
stage "Stripe — API 密钥" 5
say "我们将获取 Stripe 测试密钥，并为本地开发和 CI 保存它们。"
open_url "https://dashboard.stripe.com/test/apikeys"
step "在 API 密钥页面复制可发布密钥（以 pk_test_ 开头）。"
ask STRIPE_PUBLISHABLE_KEY "粘贴可发布密钥："
step "在密钥一行点击“显示测试密钥”，然后复制。"
ask_secret STRIPE_SECRET_KEY "粘贴密钥："
write_env STRIPE_PUBLISHABLE_KEY "$STRIPE_PUBLISHABLE_KEY"
write_env STRIPE_SECRET_KEY "$STRIPE_SECRET_KEY"
set_secret STRIPE_SECRET_KEY "$STRIPE_SECRET_KEY"   # CI 需要此值
# ──────────────────────────────────────────────────────────────────────────

finish
