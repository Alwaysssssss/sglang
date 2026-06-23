# Vivid-VR 服务对接现状与 Caption 兼容下一步交接

日期：`2026-06-22 UTC`

## 1. 当前项目状态

当前 `Vivid-VR` 在 `/home/zhiheng/sglang` 中仍按原生 `sglang.multimodal_gen` 集成路线推进，不依赖原版 `/home/zhiheng/Vivid-VR` 的运行时代码。原版仓库仍允许作为外部资源来源，包括 checkpoint、输入视频、`prompt.txt`、caption sidecar 和 reference 视频。

当前稳定基线仍然是：

- `Phase C` 单 clip 语义基线。
- `Phase D` 长视频 `clip split / merge / temporal orchestration` 语义基线。
- `Phase E` 默认性能配置：
  - 单卡：`single_gpu_fa_compile`
  - 双卡：`dual_gpu_fa_eager_compile`

双卡默认正式配置仍固定为：

- `--attention-backend fa`
- `SP=2`
- `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`
- `--enable-torch-compile`

## 2. 本轮服务侧完成情况

### 2.1 FlowCut 兼容入口

当前已经增加并验收过 FlowCut 兼容入口：

- `POST /v1/videos/repairs/flowcut`

该入口保持 `/v1/videos/repairs` 原有行为不变，FlowCut 专用语义包括：

- `code=0`：接单成功，异步执行。
- `code=1`：业务失败，以 HTTP 200 JSON 返回。
- `code=2`：队列满，以 HTTP 200 JSON 返回，调用方只重试提交，不轮询旧任务。
- `timeout=-1`：Vivid-VR 服务侧不对长推理设置超时。
- `callbackUrl` 支持 `running`、`succeeded`、`failed` 状态。
- `progress` 已通过 callback 上报中间进度；当前服务状态查询仍主要依赖任务状态和已有 progress 字段。

相关代码入口：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py`

详细历史交接见：

- `/home/zhiheng/sglang/docs_xzh/hand_over/flowcut_vividvr_service_compat_handover_20260622.md`

### 2.2 404 问题定位与短期处理

此前服务日志中出现过旧 `taskId` 的 `/progress` 和详情查询返回 `404`。根因不是任务执行失败，而是：

- `VIDEO_STORE` 目前是进程内存状态。
- 服务重启后旧进程接过的任务不会存在于新进程内存里。
- 客户端如果继续轮询旧 `taskId`，新进程只能返回 `404`。

当前短期处理已经完成：

- 验收脚本只在 `code=0` 接单后轮询。
- `code=2` 只重试提交，不轮询未接单任务。
- 轮询阶段遇到 `404` 时，明确按“服务可能已重启或当前进程未接单”处理。
- 服务端日志补充了任务接单和 missing task 信息。

长期如果需要跨重启查询历史任务，应把任务状态落到持久化存储；当前尚未实现。

### 2.3 文件存储与 MinIO 现状

默认情况下，Vivid-VR 服务使用服务机器本地文件系统：

- 输入本地路径由服务机器读取。
- `video_url` 会下载到服务机器本地临时或 `--input-save-path` 指定目录。
- 结果优先写到请求里的 `output_path`。
- 如果 `output_path` 是目录，则写为 `${taskId}.mp4`。
- 如果没有显式指定，则使用服务端 `--output-path` 或临时路径。

`minioConfig` 只在请求提供时启用：

- 服务端用 `boto3` 按 S3 兼容协议上传生成结果。
- 上传对象 key 当前按 `outputs/{job_id}.mp4` 组织。
- 成功 callback 中返回 `result_url`。
- FlowCut 分支的 MinIO 上传不会删除本地输出文件。

注意：MinIO 不是 Python 库，而是 S3 兼容对象存储服务；当前代码使用的是 `boto3` 客户端。

## 3. 对外暴露端口现状

当前 `sglang.multimodal_gen` 的服务代码支持通过 `--host` 暴露 HTTP API：

- `--host 127.0.0.1`：只允许本机访问。
- `--host 0.0.0.0`：监听所有 IPv4 网卡，其他机器可通过服务机器 IP 和端口访问，前提是网络路径放行。
- `--host <SERVER_LAN_IP>`：只监听指定网卡 IP。

运行命令和远程请求方法已更新到：

- `/home/zhiheng/sglang/docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

本轮实测当前服务器 IP：

- `10.119.16.10`

当前双卡服务在服务器本机的状态：

- tmux session：`vividvr_serve_dual_default`
- 启动参数：`--host 0.0.0.0 --port 31191`
- 日志显示：`Uvicorn running on http://0.0.0.0:31191`
- 服务器本机访问成功：
  - `curl http://127.0.0.1:31191/health` 返回 `{"status":"ok"}`
  - `curl http://10.119.16.10:31191/health` 返回 `{"status":"ok"}`

Mac 主机直接访问：

```bash
curl --noproxy '*' --silent --show-error --fail http://10.119.16.10:31191/health
```

当前表现为超时：

```text
curl: (28) Failed to connect to 10.119.16.10 port 31191 after 75003 ms
```

结论：

- Vivid-VR 服务进程与 `--host 0.0.0.0` 绑定没有问题。
- `BASE_URL=http://10.119.16.10:31191` 是正确的远程地址。
- Mac 到服务器 `31191` 的网络路径被防火墙、安全组、机房 ACL、VPN 或网络策略拦截。
- 因为 SSH 能通，所以短期推荐用 SSH 隧道。

Mac 上推荐短期命令：

```bash
ssh -N -L 31191:127.0.0.1:31191 <USER>@10.119.16.10
```

然后在 Mac 另一个终端使用：

```bash
export BASE_URL=http://127.0.0.1:31191
curl --noproxy '*' --silent --show-error --fail "${BASE_URL}/health"
```

如果 Mac 本地 `31191` 被占用：

```bash
ssh -N -L 41191:127.0.0.1:31191 <USER>@10.119.16.10
export BASE_URL=http://127.0.0.1:41191
```

## 4. 当前仍需注意的问题

### 4.1 Caption 环境不兼容是下一步主任务

当前最重要的下一步任务是解决原版 caption 模型环境不兼容问题。

已知现状：

- 原版 `/home/zhiheng/Vivid-VR` 的 caption 模型目前只能在原版环境 `/home/zhiheng/Vivid-VR/.venv` 中稳定正确产出。
- 在 `/home/zhiheng/sglang/.venv` 中运行 caption，会因 `transformers` 等依赖版本差异导致输出异常。
- 不能为了 caption 直接破坏 `sglang/.venv` 的主推理依赖，否则可能影响 Phase C/D/E 已验收语义。

当前安全做法仍然是：

- 新视频先用 `/home/zhiheng/Vivid-VR/.venv/bin/python` 跑原版 caption。
- 把每个 temporal clip 的 caption 按顺序保存为 sidecar 文本。
- `sglang` Vivid-VR 推理只消费 sidecar caption，不在主推理环境中 live 跑 CogVLM2。

下一步建议优先做“caption 桥接路径”，而不是直接改 `sglang` 全局依赖：

- 桥接进程使用 `/home/zhiheng/Vivid-VR/.venv/bin/python`。
- `sglang` 侧通过 CLI、JSON 文件、HTTP 本地服务或子进程协议请求 caption。
- 桥接输出统一落成 sidecar caption 文件。
- 推理主链仍走 `sglang` 原生 Vivid-VR pipeline，避免运行时依赖原版推理代码。

验收目标建议：

- 在一个此前没有 sidecar 的新视频上，桥接能稳定生成每个 temporal clip 的 caption。
- 生成的 sidecar 行数、顺序与长视频 clip 切分一致。
- 同一输入重复运行 caption 结果稳定或差异可解释。
- `sglang` 推理读取桥接生成的 sidecar 后，仍通过 Phase C/D/E 相关轻量回归。
- 不修改或降级 `/home/zhiheng/sglang/.venv` 中会影响主推理的核心依赖。

### 4.2 远程端口访问仍依赖基础设施放行

`--host 0.0.0.0` 只表示服务监听外部网卡，不等价于所有机器都能访问。真正远程访问还依赖：

- 服务器防火墙。
- 云安全组或机房 ACL。
- VPN / 内网路由。
- Docker / Kubernetes 端口映射。

如果短期只需要 Mac 发请求，SSH 隧道足够；如果要让其他服务长期调用，需要运维侧放行 `31190` 或 `31191`。

### 4.3 任务状态仍是内存态

当前 `/v1/videos/{taskId}` 和 `/progress` 依赖进程内 `VIDEO_STORE`。服务重启后旧任务查询会返回 `404`。这不是模型失败，而是状态未持久化。

如需生产级任务查询，需要后续引入持久化任务状态或外部状态服务。

## 5. 建议下一轮执行顺序

1. 先阅读本交接文档、`flowcut_vividvr_service_compat_handover_20260622.md` 和 `phase_e_default_configs_and_serve_followups_handover_20260622.md`。
2. 定位原版 caption 入口、依赖版本和当前 `sglang/.venv` 下异常的最小复现。
3. 不改主推理依赖，先设计 caption bridge 的输入输出契约。
4. 实现桥接生成 sidecar caption 的最小路径。
5. 用新视频验证 sidecar 行数、顺序和推理消费路径。
6. 跑相关轻量回归，必要时再做一次 tmux 长推理验收。

## 6. 一句话交接

当前 Vivid-VR 服务侧已经具备 FlowCut 兼容入口、本地/MinIO 输出、`--host 0.0.0.0` 对外监听和远程调用文档；当前真正的下一步主任务是解决原版 caption 只能在 `/home/zhiheng/Vivid-VR/.venv` 中稳定产出的问题，推荐通过独立 caption bridge 生成 sidecar，而不是破坏 `sglang/.venv` 的主推理依赖。
