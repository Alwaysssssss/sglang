# Qwen3.6-27B 对外部署 需求

本文是 Qwen3.6-27B 对外上线的需求基线。完整操作步骤、排障记录和本次部署结果见 `docs_always/qwen3.6-27b/qwen3.6-27b.md`。

本文档是对外上线基线，固定以下配置：

| 项目 | 固定值 |
| --- | --- |
| 模型名 | `qwen3.6-27b` |
| 模型路径 | 默认 `/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B` |
| Tensor Parallel | `tp=4` |
| 上下文长度 | `128K`，即 `131072` tokens |
| API key 环境变量 | `OPENAI_API_KEY` |
| API Base URL 环境变量 | `OPENAI_BASE_URL` |
| 对外入口 | Nginx |
| 外网 API Base URL | `http://<server-ip>:18080/v1`；当前机器已验证 `http://10.119.16.70:18080/v1` |
| SGLang 监听 | 默认 `127.0.0.1:30000`，只给本机 Nginx 反代 |
| 多用户 | SGLang 并发队列 + Nginx 限流；可选 Nginx per-user bearer key |

## 当前运行状态

| 项目 | 当前值 |
| --- | --- |
| 启动时间 UTC | `2026-06-12T14:42:10Z` |
| 验收时间 UTC | `2026-06-12T14:46:49Z` |
| PID | `1898437` |
| SGLang 后端 | `127.0.0.1:30000` |
| Nginx 对外入口 | `0.0.0.0:18080` |
| 外部 Base URL | `OPENAI_BASE_URL=http://10.119.16.70:18080/v1` |
| 完整验收 | `/health`、`/v1/models`、错误 key、非流式、流式、8 并发、100K token 长上下文均通过 |
| 日志 | `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/logs/qwen36_27b/qwen36_27b_tp4_128k_20260612T144210Z.log` |

## 交付要求

- 使用本仓库虚拟环境 `/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3` 启动，避免系统 Python 的 `sglang`/`sgl-kernel` 版本不匹配。
- 服务端固定 `--served-model-name qwen3.6-27b`，客户端只能使用该 OpenAI 兼容模型名。
- SGLang 只监听 `127.0.0.1:30000`；对外访问统一通过 Nginx `0.0.0.0:18080`。
- API key 从 `/etc/sglang/qwen36_openai_api_key` 或 `OPENAI_API_KEY` 读取；禁止无鉴权对外开放。
- 需要提供可复用启动、停止、Nginx 配置、长上下文检查和完整验收脚本。
- 需要完成 `/health`、`/v1/models`、非流式 chat、流式 chat、错误 key、100K token 长上下文和 8 并发验收。

## 外网 API 接口

外部客户端统一访问 Nginx，不直接访问 `127.0.0.1:30000`。当前已验证入口：

```text
http://10.119.16.70:18080/v1
```

对外提供的接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查，完整 URL 为 `http://<server-ip>:18080/health`。 |
| `GET` | `/v1/models` | OpenAI 兼容模型列表，返回模型 id `qwen3.6-27b` 和 `max_model_len=131072`。 |
| `POST` | `/v1/chat/completions` | OpenAI 兼容 chat completion，支持非流式和 `stream=true` 流式输出。 |

业务接口必须带 API key：

```http
Authorization: Bearer <OPENAI_API_KEY>
```

客户端统一使用 `OPENAI_BASE_URL` 表示 OpenAI 兼容 Base URL。

非流式调用示例：

```bash
export OPENAI_BASE_URL=http://10.119.16.70:18080/v1
export OPENAI_API_KEY=<从管理员获取的 key>

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

流式调用示例：

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

Python OpenAI SDK 示例：

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

外网入口验收命令：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export OPENAI_BASE_URL=http://10.119.16.70:18080/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

## 当前交付件

- `start_qwen36_27b.sh`：启动四卡 128K SGLang 服务并等待 ready。
- `stop_qwen36_27b.sh`：按 pid 文件和端口安全停止目标服务。
- `nginx_qwen36_27b.conf`：Nginx 反代配置副本，当前已与 `/etc/nginx/conf.d/qwen36_27b.conf` 一致。
- `check_long_context.py`：单独长上下文验收脚本。
- `verify_qwen36_27b.py`：完整外部入口验收脚本。
- `qwen3.6-27b.md`：具体步骤、运行参数、排障和本次部署记录。
