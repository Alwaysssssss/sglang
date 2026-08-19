#!/usr/bin/env bash
# 人机协作复现循环。
# 复制此文件，编辑下方步骤，然后运行。
# 智能体运行脚本；用户在自己的终端中按提示操作。
#
# 用法：
#   bash hitl-loop.template.sh
#
# 两个辅助函数：
#   step "<说明>"             → 显示说明并等待回车
#   capture VAR "<问题>"      → 显示问题，并将回答读入 VAR
#
# 最后以 KEY=VALUE 形式打印捕获的值，供智能体解析。

set -euo pipefail

step() {
  printf '\n>>> %s\n' "$1"
  read -r -p "    [完成后按回车] " _
}

capture() {
  local var="$1" question="$2" answer
  printf '\n>>> %s\n' "$question"
  read -r -p "    > " answer
  printf -v "$var" '%s' "$answer"
}

# --- 在下方编辑 --------------------------------------------------------

step "打开 http://localhost:3000 并登录应用。"

capture ERRORED "点击“导出”按钮。是否抛出错误？(y/n)"

capture ERROR_MSG "粘贴错误消息（没有则填写“无”）："

# --- 在上方编辑 --------------------------------------------------------

printf '\n--- 已捕获 ---\n'
printf 'ERRORED=%s\n' "$ERRORED"
printf 'ERROR_MSG=%s\n' "$ERROR_MSG"
