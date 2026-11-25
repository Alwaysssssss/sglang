#!/usr/bin/env bash
# 清除 videoedit_l40s 队列中 status='failed' 的任务记录，并查看服务状态。
# 用法：bash docs_handoff/clear_sqlite.sh
# 仅删除失败记录，保留视频文件及其他状态的任务，无需重启模型。
# 若任务卡在 cancelling，须先确认后端任务已不存在，再单独处理。
set -euo pipefail

docker exec -i videoedit_l40s python3 - <<'PY'
import sqlite3

db = "/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/queue.sqlite3"
with sqlite3.connect(f"file:{db}?mode=rw", uri=True, timeout=30) as conn:
    result = conn.execute("DELETE FROM tasks WHERE status = 'failed'")
    deleted = result.rowcount
print(f"已清除 {deleted} 条失败任务")
PY

curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:5402/health
printf '\n'
