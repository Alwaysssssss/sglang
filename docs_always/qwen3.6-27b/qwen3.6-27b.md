# Feature: 基于 SGLang 部署 Qwen3.6-27B 在线服务

本文是 `Qwen/Qwen3.6-27B` 在当前机器上的上线 runbook。目标不是只把进程拉起，而是完成可复现的四卡 256K 上下文服务部署，并用 OpenAI 兼容接口验证真实请求、流式输出、长上下文、并发和稳定性。

> 适用范围说明：本文保留普通 256K 在线服务的历史 runbook。Qwen3.6-27B agent 的 256K 上下文、默认并发 4、基础日志、reasoning 展示治理和显存优化方案，见 `docs_always/qwen3.6-27b/vibe/opt_start_qwen36_27b_agent.md`；SGLang 内部链路索引见 `docs_always/qwen3.6-27b/docs/README.md`。

## 0. 当前结论

- 目标模型已存在：`/mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B`，约 `52G`。
- 当前机器可见 4 张 `NVIDIA A100-SXM4-80GB`，启动方案按 `tp=4` 固定。
- 必须使用仓库虚拟环境：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/bin/python3`。
- 不要使用系统默认 `python` 启动。当前默认环境的 `sglang==0.5.2` 与 `sgl-kernel==0.3.16.post2` 不匹配，会在导入阶段报 `ImportError: cannot import name 'sgl_per_token_group_quant_fp8'`。
- 本轮上线 SGLang 监听 `127.0.0.1:30000`，Nginx 监听 `0.0.0.0:18080` 并反代到 SGLang。
- 外部客户端 API Base URL 为 `http://<server-ip>:18080/v1`；当前机器已验证 `http://10.119.16.70:18080/v1`。
- 客户端统一使用 `OPENAI_BASE_URL` 表示 OpenAI 兼容 Base URL。
- API key 存放在 `/etc/sglang/qwen36_openai_api_key`，文件权限为 `600`。客户端通过 `Authorization: Bearer <key>` 访问 Nginx 或本机 SGLang。

## 1. 目标和验收标准

### 目标

- 以 SGLang 启动 `Qwen/Qwen3.6-27B` 文本 chat 服务。
- 固定 OpenAI 兼容模型名：`qwen3.6-27b`。
- 固定上下文长度：`256K` tokens。
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
| 上线上下文 | `256K` tokens |
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
| `--context-length` | `256K` | 本轮 256K 上下文验收目标 |
| `--max-total-tokens` | `1048576` | 支撑多个请求共享 token pool；历史日志已验证可完成显存分配 |
| `--max-running-requests` | `8` | 多用户初始并发上限 |
| `--chunked-prefill-size` | `8192` | 长上下文分块 prefill，避免单次 prefill 过大 |
| `--max-prefill-tokens` | `16384` | 控制 prefill 峰值 |
| `--dtype` | `bfloat16` | A100 原生支持，匹配模型 dtype |
| `--attention-backend` | `flashinfer` | 当前虚拟环境已安装 FlashInfer |
| `--served-model-name` | `qwen3.6-27b` | OpenAI 兼容调用固定模型名 |
