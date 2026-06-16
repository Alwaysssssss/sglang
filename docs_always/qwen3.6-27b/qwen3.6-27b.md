# Feature: 基于 SGLang 部署 Qwen3.6-27B 在线服务

本文是 `Qwen/Qwen3.6-27B` 在当前机器上的上线 runbook。目标不是只把进程拉起，而是完成可复现的四卡 128K 上下文服务部署，并用 OpenAI 兼容接口验证真实请求、流式输出、长上下文、并发和稳定性。

## 0. 当前结论

- 目标模型已存在：`/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B`，约 `52G`。
- 当前机器可见 4 张 `NVIDIA A100-SXM4-80GB`，启动方案按 `tp=4` 固定。
- 必须使用仓库虚拟环境：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3`。
- 不要使用系统默认 `python` 启动。当前默认环境的 `sglang==0.5.2` 与 `sgl-kernel==0.3.16.post2` 不匹配，会在导入阶段报 `ImportError: cannot import name 'sgl_per_token_group_quant_fp8'`。
- 本轮上线 SGLang 监听 `127.0.0.1:30000`，Nginx 监听 `0.0.0.0:18080` 并反代到 SGLang。
- 外部客户端 API Base URL 为 `http://<server-ip>:18080/v1`；当前机器已验证 `http://10.119.16.70:18080/v1`。
- 客户端统一使用 `OPENAI_BASE_URL` 表示 OpenAI 兼容 Base URL。
- 当前机器 GPU 0-2 同时有既有 vLLM 服务占用约 `20.9GB/GPU`。本次启动通过 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` 允许 SGLang 在确认显存余量充足后继续启动。
- API key 存放在 `/etc/sglang/qwen36_openai_api_key`，文件权限为 `600`。客户端通过 `Authorization: Bearer <key>` 访问 Nginx 或本机 SGLang。

## 1. 目标和验收标准

### 目标

- 以 SGLang 启动 `Qwen/Qwen3.6-27B` 文本 chat 服务。
- 固定 OpenAI 兼容模型名：`qwen3.6-27b`。
- 固定上下文长度：`131072` tokens。
- 固定四卡 Tensor Parallel：`--tensor-parallel-size 4`。
- 通过 `/health`、`/v1/models`、`/v1/chat/completions` 验证服务可用。
- 至少覆盖非流式、流式、长上下文和并发请求。

### 非目标

- 不在本文中接入视觉/视频请求。当前模型目录带 `vision_config`，SGLang 日志中可能出现 `visual.* not found in params_dict`，只要文本接口验收通过，该日志不阻塞本文交付。
- 不修改 SGLang 代码。
- 不下载或转换模型权重。
- 不在没有 API key 和反向代理策略时对公网开放。

### 通过标准

- `curl http://<server-ip>:18080/health` 返回 HTTP 200。
- `GET http://<server-ip>:18080/v1/models` 返回模型 id `qwen3.6-27b`。
- `POST http://<server-ip>:18080/v1/chat/completions` 非流式 chat 返回非空 `choices[0].message.content`。
- `POST http://<server-ip>:18080/v1/chat/completions` 流式 chat 能持续返回 `data:` 分片并以 `[DONE]` 结束。
- 长上下文请求不触发 context length 错误，并返回可读输出。
- 并发请求全部 HTTP 200，且响应内容非空。
- `nvidia-smi` 显示 4 张 GPU 都有目标服务进程占用。

## 2. 环境基线

| 项目 | 当前值 |
| --- | --- |
| 仓库 | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang` |
| Python | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` |
| SGLang | `0.0.0.dev11299+gfc0dd7885.d20260506` |
| PyTorch | `2.9.1` |
| CUDA | `12.8` |
| Transformers | `5.3.0` |
| FlashInfer | `flashinfer-python==0.6.7.post2` |
| GPU | 4 x `NVIDIA A100-SXM4-80GB` |
| 模型路径 | `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` |
| 模型大小 | 约 `52G` |
| 文本最大位置 | `text_config.max_position_embeddings=262144` |
| 上线上下文 | `131072` tokens |
| SGLang 本机入口 | `http://127.0.0.1:30000/v1` |
| Nginx 入口 | `http://<server-ip>:18080/v1`，本机验证用 `http://127.0.0.1:18080/v1` |
| 当前已验证外部入口 | `http://10.119.16.70:18080/v1` |
| API key 文件 | `/etc/sglang/qwen36_openai_api_key` |

环境确认命令：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3 - <<'PY'
import importlib.metadata as md
import sys
import torch

print("python", sys.executable)
for pkg in ["sglang", "torch", "transformers", "flashinfer-python"]:
    print(pkg, md.version(pkg))
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
du -sh /mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
```

## 3. 服务设计

### 架构

```text
client
  -> Nginx HTTP entrypoint on 0.0.0.0:18080
  -> SGLang HTTP server on 127.0.0.1:30000
  -> Qwen/Qwen3.6-27B local weights
  -> 4 x A100 tensor parallel workers
```

SGLang 不直接监听公网地址；外部访问统一走 Nginx。当前 Nginx 是 HTTP 入口，跨机器或公网使用时应在上层补 TLS。

### 核心参数

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `--tensor-parallel-size` | `4` | 充分使用 4 张 A100，降低单卡权重压力 |
| `--context-length` | `131072` | 本轮 128K 上下文验收目标 |
| `--max-total-tokens` | `1048576` | 支撑多个请求共享 token pool；历史日志已验证可完成显存分配 |
| `--max-running-requests` | `8` | 多用户初始并发上限 |
| `--chunked-prefill-size` | `8192` | 长上下文分块 prefill，避免单次 prefill 过大 |
| `--max-prefill-tokens` | `16384` | 控制 prefill 峰值 |
| `--dtype` | `bfloat16` | A100 原生支持，匹配模型 dtype |
| `--attention-backend` | `flashinfer` | 当前虚拟环境已安装 FlashInfer |
| `--served-model-name` | `qwen3.6-27b` | OpenAI 兼容调用固定模型名 |

## 4. 启动、停止和重启

### 4.0 脚本入口

本文启动、停止和验收命令已整理为独立脚本：

- `docs_always/qwen3.6-27b/start_qwen36_27b.sh`：启动四卡 128K SGLang 服务，默认读取 `/etc/sglang/qwen36_openai_api_key`，并轮询 `/health`。
- `docs_always/qwen3.6-27b/stop_qwen36_27b.sh`：按 pid 文件和端口定位服务，只停止匹配 `Qwen3___6-27B` 或 `qwen3.6-27b` 的 SGLang 进程。
- `docs_always/qwen3.6-27b/check_long_context.py`：构造默认 100K token 请求，调用 OpenAI 兼容 chat 接口并校验返回；默认从 `OPENAI_BASE_URL` 读取 Base URL，未设置时兼容旧 `BASE_URL`。
- `docs_always/qwen3.6-27b/verify_qwen36_27b.py`：完整验收 Nginx 外部入口，覆盖 `/health`、`/v1/models`、错误 key、非流式、流式、并发和 100K token 长上下文；默认从 `OPENAI_BASE_URL` 读取 Base URL，未设置时兼容旧 `BASE_URL`。
- `docs_always/qwen3.6-27b/nginx_qwen36_27b.conf`：已加载到 `/etc/nginx/conf.d/qwen36_27b.conf` 的 Nginx 反代配置副本。

快速使用：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

# 启动 SGLang
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  docs_always/qwen3.6-27b/start_qwen36_27b.sh

# 完整验收，走 Nginx 入口
export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url "$OPENAI_BASE_URL"

# 停止 SGLang
docs_always/qwen3.6-27b/stop_qwen36_27b.sh
```

### 4.1 启动前检查

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export SGLANG_PY=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3
export MODEL_PATH=/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT=30000
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"

test -x "$SGLANG_PY"
test -d "$MODEL_PATH"
lsof -nP -iTCP:${SGLANG_PORT} -sTCP:LISTEN || true
nvidia-smi
```

如果 `lsof` 显示 `30000` 被占用，先确认旧服务是否属于本任务。只停止明确属于旧 SGLang/Qwen3.6 的进程：

```bash
lsof -nP -iTCP:30000 -sTCP:LISTEN
# 示例：kill <PID>
```

### 4.2 启动命令

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export SGLANG_PY=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3
export MODEL_PATH=/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
export SGLANG_HOST=127.0.0.1
export SGLANG_PORT=30000
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"

mkdir -p logs/qwen36_27b
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_FILE="/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b/qwen36_27b_tp4_128k_${STAMP}.log"

cat > /tmp/start_qwen36_27b.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 OPENAI_API_KEY="$OPENAI_API_KEY" \
  "$SGLANG_PY" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$SGLANG_HOST" \
  --port "$SGLANG_PORT" \
  --served-model-name qwen3.6-27b \
  --tensor-parallel-size 4 \
  --context-length 131072 \
  --max-total-tokens 1048576 \
  --max-running-requests 8 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 16384 \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --sampling-backend flashinfer \
  --api-key "$OPENAI_API_KEY" \
  --disable-piecewise-cuda-graph
SH
chmod +x /tmp/start_qwen36_27b.sh

setsid /tmp/start_qwen36_27b.sh > "$LOG_FILE" 2>&1 < /dev/null &

echo $! > logs/qwen36_27b/qwen36_27b.pid
echo "$LOG_FILE"
```

说明：

- 如果 `OPENAI_API_KEY` 未设置，启动脚本会读取 `/etc/sglang/qwen36_openai_api_key`。
- 启动脚本默认拒绝 `EMPTY` key；只有显式设置 `ALLOW_EMPTY_API_KEY=1` 时才允许本地测试模式。
- 如果 GPU 0-2 已有其它服务占用导致 SGLang 报 `The memory capacity is unbalanced`，确认剩余显存足够后再使用 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 docs_always/qwen3.6-27b/start_qwen36_27b.sh` 启动。
- 启动脚本写入日志时会将 `api_key` 和命令行 `--api-key` 脱敏为 `<redacted>`，避免 SGLang 启动参数日志泄露密钥。
- 在 Codex/自动化环境中，普通 `nohup ... &` 可能会随工具进程组退出；已验证的后台方式是 `setsid ... < /dev/null &`。
- 历史日志显示模型加载和 128K 显存分配可完成，失败点主要是端口占用：`[Errno 98] error while attempting to bind on address ('127.0.0.1', 30000): address already in use`。

### 4.3 等待服务就绪

```bash
LOG_FILE=$(ls -t logs/qwen36_27b/qwen36_27b_tp4_128k_*.log | head -n 1)
tail -f "$LOG_FILE"
```

看到以下日志后开始请求验收：

```text
Application startup complete.
```

也可以轮询健康检查：

```bash
for i in $(seq 1 240); do
  if curl --noproxy '*' -fsS \
    -H "Authorization: Bearer $(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)" \
    http://127.0.0.1:30000/health >/dev/null; then
    echo "ready"
    break
  fi
  sleep 2
done
```

### 4.4 停止服务

优先使用 pid 文件：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

if [ -f logs/qwen36_27b/qwen36_27b.pid ]; then
  kill "$(cat logs/qwen36_27b/qwen36_27b.pid)"
fi
```

如果 pid 文件不存在，用端口定位：

```bash
lsof -nP -iTCP:30000 -sTCP:LISTEN
# 确认是目标 SGLang 进程后再 kill <PID>
```

## 5. 端到端验收

以下命令默认服务在 `127.0.0.1:30000`。如要走 Nginx，把 URL 改成 `http://127.0.0.1:18080/...`。API key 从 `/etc/sglang/qwen36_openai_api_key` 读取：

```bash
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"
```

### 5.0 一键验收脚本

优先使用完整验收脚本，默认禁用环境代理，避免本机请求被 `HTTP_PROXY` 转发：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://127.0.0.1:18080/v1
```

也可以只设置 OpenAI 兼容客户端环境变量，脚本会默认读取 `OPENAI_BASE_URL`：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

预期输出包含：

```text
PASS health http=200
PASS models id=qwen3.6-27b max_model_len=131072
PASS bad_key http=401
PASS chat ...
PASS stream ...
PASS concurrency requests=8
PASS long_context ... prompt_tokens>=100000
PASS all requested checks
```

### 5.1 健康检查和模型列表

```bash
curl --noproxy '*' -i \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  http://127.0.0.1:30000/health

curl --noproxy '*' -sS \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  http://127.0.0.1:30000/v1/models
```

预期：

- `/health` 返回 HTTP 200。
- `/v1/models` 包含 `qwen3.6-27b`。

### 5.2 非流式 chat

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "你是助手，都用中文回答"},
      {"role": "user", "content": "用三句话介绍诗人李白。"}
    ],
    "max_tokens": 256,
    "temperature": 0
  }'
```

预期：HTTP 200，`choices[0].message.content` 非空。

### 5.3 流式 chat

```bash
curl --noproxy '*' -N http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "李白是谁。"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

预期：持续返回 `data:` 分片，最后返回 `data: [DONE]`。

### 5.4 长上下文

使用本地 tokenizer 构造接近 100K tokens 的输入，验证不是只跑短 prompt：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export SGLANG_PY=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3
export MODEL_PATH=/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"

"$SGLANG_PY" - <<'PY'
import os
import requests
from transformers import AutoTokenizer

model_path = os.environ["MODEL_PATH"]
api_key = os.environ["OPENAI_API_KEY"]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

unit = "这是长上下文验收片段。请只记住最后的问题。"
target_tokens = 100_000
pieces = []
while True:
    pieces.append(unit)
    text = "\n".join(pieces)
    if len(tokenizer.encode(text)) >= target_tokens:
        break

prompt = text + "\n\n最后的问题：请回答“长上下文验收通过”，不要输出其它内容。"
resp = requests.post(
    "http://127.0.0.1:30000/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "qwen3.6-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0,
    },
    timeout=600,
)
print("status", resp.status_code)
print(resp.text[:1000])
resp.raise_for_status()
content = resp.json()["choices"][0]["message"]["content"]
assert content.strip(), "empty response"
PY
```

预期：HTTP 200，返回内容非空，不出现 context length exceeded。

### 5.5 并发

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export SGLANG_PY=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3
export OPENAI_API_KEY="$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)"

"$SGLANG_PY" - <<'PY'
import concurrent.futures
import os
import requests

api_key = os.environ["OPENAI_API_KEY"]

def one(i: int):
    resp = requests.post(
        "http://127.0.0.1:30000/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "qwen3.6-27b",
            "messages": [{"role": "user", "content": f"请用一句话回答并发验收编号 {i}。"}],
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=300,
    )
    text = resp.text
    if resp.status_code != 200:
        return i, resp.status_code, text[:300]
    content = resp.json()["choices"][0]["message"]["content"]
    return i, resp.status_code, content[:120]

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(one, range(8)))

for row in results:
    print(row)
assert all(status == 200 and content for _, status, content in results)
PY
```

预期：8 个请求均 HTTP 200，内容非空。

## 6. Nginx 入口策略

当前环境已安装 `nginx/1.18.0 (Ubuntu)`，配置文件为 `/etc/nginx/conf.d/qwen36_27b.conf`。已验证 Nginx 监听 `*:18080` 并反代到 `127.0.0.1:30000`。

- SGLang 仍只监听 `127.0.0.1:30000`。
- Nginx 监听 `18080`，外部调用 `http://<server-ip>:18080/v1`。
- Nginx 保留 `Authorization` header，由 SGLang 执行 API key 校验。
- 反代关闭 buffering，支持流式输出和 100K token 级长上下文请求。
- 当前入口是 HTTP；若跨机器或公网使用，需要在上层补 TLS 或改为 HTTPS server 块。

### 6.1 对外 API 接口

外部客户端只使用 Nginx 入口，不直接访问 SGLang 内部端口。当前机器已验证的对外地址：

```text
http://10.119.16.70:18080/v1
```

接口清单：

| 方法 | 外部 URL | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `GET` | `http://<server-ip>:18080/health` | 健康检查可不带 key | 返回 HTTP 200 表示 Nginx 到 SGLang 链路可用。 |
| `GET` | `http://<server-ip>:18080/v1/models` | `Authorization: Bearer <key>` | 返回 OpenAI 兼容模型列表，目标模型 id 为 `qwen3.6-27b`。 |
| `POST` | `http://<server-ip>:18080/v1/chat/completions` | `Authorization: Bearer <key>` | OpenAI 兼容 chat completion，支持非流式和 `stream=true`。 |

客户端固定参数：

- `base_url`: `http://10.119.16.70:18080/v1`，其它机器按实际 `<server-ip>` 替换。
- `model`: `qwen3.6-27b`。
- `Authorization`: `Bearer <OPENAI_API_KEY>`。
- 最大上下文：`131072` tokens。
- 环境变量：统一使用 `OPENAI_BASE_URL`。

模型列表：

```bash
export OPENAI_BASE_URL=http://10.119.16.70:18080/v1
export OPENAI_API_KEY=<从管理员获取的 key>

curl --noproxy '*' -sS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

非流式 chat：

```bash
curl --noproxy '*' -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "请用一句话介绍 Qwen3.6-27B。"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

流式 chat：

```bash
curl --noproxy '*' -N "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "请用两句话说明流式输出。"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

Python OpenAI SDK：

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://10.119.16.70:18080/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "请用一句话介绍李白。"}],
    max_tokens=128,
    temperature=0,
)
print(resp.choices[0].message.content)
```

外部入口完整验收：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

docs_always/qwen3.6-27b/verify_qwen36_27b.py \
  --base-url http://10.119.16.70:18080/v1
```

当前 Nginx 配置：

```nginx
limit_req_zone $binary_remote_addr zone=qwen36_limit:10m rate=2r/s;

upstream qwen36_sglang {
    server 127.0.0.1:30000;
    keepalive 32;
}

server {
    listen 18080;
    server_name _;

    client_max_body_size 128m;
    proxy_connect_timeout 60s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    send_timeout 900s;

    location / {
        limit_req zone=qwen36_limit burst=16 nodelay;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Authorization $http_authorization;

        proxy_buffering off;
        proxy_request_buffering off;
        proxy_pass http://qwen36_sglang;
    }
}
```

加载配置：

```bash
nginx -t
nginx -s reload || nginx
```

## 7. 排障记录

### 7.1 默认 Python 导入失败

现象：

```text
ImportError: cannot import name 'sgl_per_token_group_quant_fp8' from 'sgl_kernel'
```

根因：系统默认 Python 的 `sglang` 与 `sgl-kernel` 版本不匹配。

处理：始终使用 `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` 启动。

### 7.2 端口占用

现象：

```text
[Errno 98] error while attempting to bind on address ('127.0.0.1', 30000): address already in use
```

根因：旧服务或其它进程占用了 `30000`。历史日志中模型加载、KV cache、Mamba cache、CUDA graph capture 都已完成，失败发生在 Uvicorn bind 阶段。

处理：

```bash
lsof -nP -iTCP:30000 -sTCP:LISTEN
```

确认进程归属后停止旧进程，或把 `SGLANG_PORT` 和客户端请求端口一起改成新的空闲端口。

### 7.3 `visual.* not found in params_dict`

现象：日志中出现多行 `Parameter visual.* not found in params_dict`。

处理：本文只验收文本 chat。只要 `/v1/chat/completions` 的非流式、流式、长上下文和并发测试通过，该日志不阻塞文本服务上线。视觉/多模态能力需要单独方案和单独验收。

### 7.4 Python requests 走环境代理导致 502

现象：

```text
HTTP_PROXY=http://localhost:10909
HTTPS_PROXY=http://localhost:10909
ALL_PROXY=http://localhost:10909
长上下文脚本请求 http://127.0.0.1:18080/v1 返回 502，但 Nginx access/error log 无该请求记录。
```

根因：Python `requests` 默认信任环境代理变量，本机请求被发到代理而不是 Nginx。

处理：`check_long_context.py` 和 `verify_qwen36_27b.py` 默认设置 `requests.Session().trust_env = False`。如确实需要使用环境代理，显式传 `--trust-env-proxy`。

### 7.5 TP 显存不均衡检查失败

现象：

```text
RuntimeError: The memory capacity is unbalanced. Some GPUs may be occupied by other processes.
pre_model_load_memory=57.4520263671875, local_gpu_memory=78.008056640625
```

根因：当前机器 GPU 0-2 上已有 vLLM 服务占用约 `20.9GB/GPU`，GPU 3 基本空闲。SGLang 的 TP 显存均衡检查会把这种状态视为风险并中止。

处理：不要停止不属于本任务的 vLLM 进程。先用 `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits` 确认剩余显存满足本服务需求；本次启动前 GPU 0-2 约 `60.3GB` free，启动后仍有约 `11.5GB/GPU` free。确认后使用：

```bash
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  docs_always/qwen3.6-27b/start_qwen36_27b.sh
```

### 7.6 外部客户端 TCP 连接超时

现象：

```text
curl: (28) Failed to connect to 10.119.16.70 port 18080 after 75002 ms: Couldn't connect to server
```

含义：失败发生在 TCP connect 阶段，请求还没有到达 HTTP/Nginx/SGLang 层。它不是模型名、API key、JSON payload 或 `stream=true` 的问题；这些问题会表现为 HTTP `401`、`404`、`422`、`502` 或已有连接后的读超时。

服务端确认命令：

```bash
ss -ltnp | awk 'NR==1 || /:18080|:30000/'

curl --noproxy '*' -v --connect-timeout 5 \
  -H "Authorization: Bearer $(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)" \
  http://10.119.16.70:18080/v1/models
```

当前服务端证据：Nginx 监听 `0.0.0.0:18080`，SGLang 监听 `127.0.0.1:30000`；服务端本机访问 `http://10.119.16.70:18080/v1/models` 返回 HTTP 200。

客户端排查：

```bash
export OPENAI_BASE_URL=http://10.119.16.70:18080/v1

nc -vz 10.119.16.70 18080
curl -v --connect-timeout 5 "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

如果客户端仍然 connect 超时，说明客户端机器到 `10.119.16.70:18080` 没有可用网络路径，常见原因是没有接入同一内网/VPN、云安全组或防火墙未放通、当前服务运行在容器/内网命名空间但没有映射到客户端可达的宿主机地址。临时方案是通过可达的 SSH 跳板做本地转发：

```bash
ssh -N -L 18080:127.0.0.1:18080 <user>@<reachable-host>

export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
curl -N "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

## 8. 本次部署记录

| 项目 | 结果 |
| --- | --- |
| 启动时间 UTC | `2026-06-12T14:42:10Z` |
| 验收时间 UTC | `2026-06-12T14:46:49Z` |
| PID | `1898437` |
| 日志文件 | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b/qwen36_27b_tp4_128k_20260612T144210Z.log` |
| 启动命令 | `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 docs_always/qwen3.6-27b/start_qwen36_27b.sh` |
| SGLang 监听 | `127.0.0.1:30000` |
| Nginx 监听 | `0.0.0.0:18080` |
| 外部 OpenAI Base URL | `OPENAI_BASE_URL=http://10.119.16.70:18080/v1` |
| 模型名 | `qwen3.6-27b` |
| API key | `/etc/sglang/qwen36_openai_api_key`，权限 `600` |
| Nginx `/health` | HTTP 200 |
| Nginx `/v1/models` | 返回 `qwen3.6-27b`，`max_model_len=131072` |
| Nginx 非流式 chat | HTTP 200，内容非空，`completion_tokens=96` |
| 错误 key | HTTP 401 |
| 流式 chat | 返回 `99` 个 `data:` 分片并以 `data: [DONE]` 结束 |
| Nginx 长上下文 | HTTP 200，`measured_prompt_tokens_before_chat_template=100002`，`prompt_tokens=100024`，`elapsed_sec=11.48` |
| 并发 | 8/8 请求 HTTP 200，响应内容非空 |
| GPU | 4 张 A100 均被服务进程占用；GPU 0-2 同时有既有 vLLM 服务，验收后显存约 `69.8GB/69.5GB/69.5GB/48.3GB` used |
| 显存不均衡处理 | GPU 0-2 启动前已有约 `20.9GB/GPU` 占用，已用 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0` 启动并完成验收 |
| 日志密钥脱敏 | `api_key='<redacted>'`，未发现明文 key |
| 限制 | 当前 Nginx 入口是 HTTP；跨机器或公网使用前需补 TLS |

进程和端口确认：

```text
PID 1898437:
/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3 -m sglang.launch_server ... --host 127.0.0.1 --port 30000 ... --api-key <from /etc/sglang/qwen36_openai_api_key>

LISTEN:
python3 1898437 root TCP 127.0.0.1:30000
nginx 1569400 root TCP *:18080
```

## 9. 交付记录模板

上线完成后记录：

```text
启动时间 UTC:
启动命令:
日志文件:
PID:
端口:
模型:
GPU:
/health 结果:
/v1/models 结果:
非流式测试:
流式测试:
长上下文测试:
并发测试:
已知限制:
```
