# VividVR 视频超分与修复：基础使用文档

本文档说明如何在本机使用 SGLang 原生集成的 VividVR：准备 `uv` 环境和模型文件、通过 CLI 执行离线推理，以及启动本地 FlowCut 服务并提交任务。

适用范围：当前已验收的 VividVR 原生推理链。它包含视频修复/超分、长视频 temporal clip 切分与拼接、caption sidecar 和颜色修复等环节；推理运行时不依赖原版 Vivid-VR 仓库中的 Python 代码。

## 1. 使用方式概览

| 方式 | 适用场景 | 入口 |
| --- | --- | --- |
| 离线 CLI | 本地单次推理、调试、复现基准 | `python/sglang/multimodal_gen/tools/run_vividvr_inference.py` |
| 本地 HTTP 服务 | 通过异步 API 提交、查询和取消任务 | `sglang serve` + caption sidecar |

当前推荐的服务默认配置是双卡 `dual_gpu_fa_eager_compile`。单卡 `single_gpu_fa_compile` 也保留为正式配置，适合资源较少的本地运行。

## 2. 环境与目录约定

### 2.1 不使用 Docker

当前仓库没有经过 VividVR 正式验收的专用 Docker 镜像。请使用 `uv` 创建和维护 Python 环境；不要把仓库中的通用 Dockerfile 视为可直接部署 VividVR 的运行镜像。

| 用途 | 环境 | Python |
| --- | --- | --- |
| 主推理、CLI、`sglang serve` | `/home/zhiheng/sglang/.venv` | Python 3.10 |
| caption sidecar | `/home/zhiheng/sglang/.venv-vividvr-caption` | 独立 Python 3.10 环境 |

主推理与 caption sidecar 必须使用各自环境；不要为了运行 caption 而替换或降级主推理环境依赖。

### 2.2 首次创建主推理环境

以下命令在仓库根目录执行。它会在固定路径创建 Python 3.10 虚拟环境，并以 editable 方式安装当前源码。

```bash
cd /home/zhiheng/sglang
uv venv --seed --python python3.10 /home/zhiheng/sglang/.venv
uv pip install --python /home/zhiheng/sglang/.venv/bin/python --upgrade pip setuptools wheel
uv pip install --python /home/zhiheng/sglang/.venv/bin/python -e "python[diffusion]" --prerelease=allow
```

首次启动 caption 服务前，再创建它的独立环境：

```bash
cd /home/zhiheng/sglang
bash python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh
```

该脚本会创建 `/home/zhiheng/sglang/.venv-vividvr-caption` 并安装 caption 所需依赖。创建过程可能需要下载依赖；主服务启动不需要原版仓库作为 Python 运行时依赖。

### 2.3 模型与输入文件

下表列出本地运行必须准备的静态资源。可将资源保存在其他路径，但命令中的变量必须相应修改。

| 资源 | 默认位置 | 用途 |
| --- | --- | --- |
| CogVideoX 基座模型 | `/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B` | 主模型基础权重 |
| VividVR 组件权重 | `/home/zhiheng/Vivid-VR/ckpts/Vivid-VR` | VividVR transformer/controlnet 等组件 |
| CogVLM2 caption 权重 | `/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption` | caption sidecar 自动生成逐 clip caption |
| 输入视频 | 用户提供的本地视频文件 | 待修复/超分视频 |
| prompt 文件 | `/home/zhiheng/Vivid-VR/input/720p/prompt.txt` | CLI 的 prompt-file 模式默认 prompt |
| caption 文件（可选） | 每个视频对应的文本文件 | CLI 长视频/公平对比；一行对应一个 temporal clip |

为避免在命令中重复书写路径，建议先设置：

```bash
export REPO_ROOT=/home/zhiheng/sglang
export VIVIDVR_ASSETS=/home/zhiheng/Vivid-VR
export COGVIDEOX_CKPT="$VIVIDVR_ASSETS/ckpts/CogVideoX1.5-5B"
export VIVIDVR_CKPT="$VIVIDVR_ASSETS/ckpts/Vivid-VR"
export COGVLM2_CKPT="$VIVIDVR_ASSETS/ckpts/cogvlm2-llama3-caption"
export PROMPT_FILE="$VIVIDVR_ASSETS/input/720p/prompt.txt"
```

开始前可检查路径：

```bash
test -d "$COGVIDEOX_CKPT"
test -d "$VIVIDVR_CKPT"
test -d "$COGVLM2_CKPT"
test -f "$PROMPT_FILE"
```

## 3. 离线 CLI 推理

CLI 不会启动 HTTP 服务。它直接调用 SGLang 内的原生 VividVR pipeline，默认输出视频到 `Vivid_Acceptance/result_videos/`、报告 JSON 到 `Vivid_Acceptance/indicator/`。

### 3.1 单卡最小示例

下面以 prompt-file 模式运行单个本地视频。长时间推理必须在 `tmux` 中启动，以便查看进度且不会因终端断开而中断。

```bash
tmux new-session -d -s vividvr_cli_single \
  'cd /home/zhiheng/sglang && \
   export PYTHONPATH=python && \
   CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/python \
   python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
     --input-video /absolute/path/to/input.mp4 \
     --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
     --cogvideox-ckpt-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
     --vividvr-ckpt-path /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
     --output-dir Vivid_Acceptance/result_videos \
     --report-dir Vivid_Acceptance/indicator \
     --artifact-prefix vividvr_local_single \
     --num-inference-steps 20 \
     --num-temporal-process-frames 121 \
     --upscale 1.0 \
     --seed 42 \
     --num-gpus 1 --tp-size 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1 \
     --attention-backend fa \
     --enable-torch-compile \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_cli_single_$(date -u +%Y%m%dT%H%M%SZ).log'
```

只读查看任务：

```bash
tmux attach -r -t vividvr_cli_single
```

### 3.2 caption-file 模式

对长视频或需要固定 caption 的场景，增加 `--caption-file`。caption 文件必须按 temporal clip 顺序逐行保存；提供该参数后，CLI 不再使用 prompt-file 模式。

```bash
--caption-file /absolute/path/to/input_caption.txt
```

`--reference-video` 仅用于离线 benchmark 的逐帧对比，不是普通推理必填参数。

### 3.3 关键参数

| 参数 | 默认/推荐值 | 说明 |
| --- | --- | --- |
| `--num-inference-steps` | 普通推理默认 `50`；日常性能口径 `20` | 去噪步数 |
| `--num-temporal-process-frames` | `121` | temporal clip 长度，必须满足 `(value - 1) % 8 == 0` |
| `--upscale` | `1.0` | 输入预缩放语义：`0` 将短边缩到 1024，`1` 保持分辨率，其他正数按倍率 resize |
| `--attention-backend` | `fa` | 当前正式默认 attention backend |
| `--enable-torch-compile` | 启用 | 当前正式默认开启 |
| `--dtype` | `bf16` | 可选 `bf16`、`fp16`、`fp32` |

注意：`upscale` 是进入模型前的输入预缩放参数，不是额外的后处理超分开关。

## 4. 启动本地 FlowCut 服务

服务由两个本地进程组成，必须按顺序启动：

```text
调用方 curl / 本地程序
        │
        ▼
VividVR 主服务（127.0.0.1:31191，双卡默认）
        │
        ▼
caption sidecar（127.0.0.1:31200）
```

主服务对外 API 为 `POST /v1/videos/repairs/flowcut`，但本文件的示例均只绑定 `127.0.0.1`，不涉及外部 IP 或端口暴露。

### 4.1 端口分配

| 服务/用途 | 双卡默认 | 单卡可选 |
| --- | ---: | ---: |
| caption sidecar | `31200` | `31200` |
| VividVR HTTP 服务 | `31191` | `31190` |
| 分布式 master port | `30191` | `30190` |
| scheduler port | `56191` | `56190` |

如端口冲突，可整体更换一组端口；caption sidecar URL、主服务端口、master port 与 scheduler port 必须保持一致。

### 4.2 启动 caption sidecar

双卡默认服务的 caption sidecar 使用两个 worker，分别绑定两张 GPU：

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python \
   python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py \
     --host 127.0.0.1 --port 31200 \
     --parallel-workers 2 --worker-devices cuda:0,cuda:1 \
     --cogvlm2-ckpt-path /home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'
```

确认健康状态：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

预期返回：

```json
{"status":"ok"}
```

### 4.3 启动双卡默认主服务

这是当前本地服务的默认命令。`eager_global` 是双卡 SP 已验收的 full global control context 语义，不能省略或换成历史实验模式。

```bash
tmux new-session -d -s vividvr_serve_dual \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   export PYTHONUNBUFFERED=1 && \
   export PYTHONPATH=python && \
   export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
   CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
     --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
     --model-id VividVR \
     --pipeline-class-name CogVideoXVividVRControlNetPipeline \
     --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
     --attention-backend fa \
     --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
     --enable-torch-compile \
     --dist-timeout 3600 \
     --host 127.0.0.1 --port 31191 \
     --master-port 30191 --scheduler-port 56191 --strict-ports \
     --input-save-path "" --output-path "" \
     --vividvr-caption-bridge \
     --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
     --vividvr-caption-sidecar-timeout 1800 \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_$(date -u +%Y%m%dT%H%M%SZ).log'
```

等待服务就绪后检查：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31191/health
```

预期返回：

```json
{"status":"ok"}
```

### 4.4 可选：单卡主服务

单卡服务保留相同的 caption sidecar。将主服务改为 `CUDA_VISIBLE_DEVICES=0`、`--num-gpus 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1`，并使用 `--port 31190 --master-port 30190 --scheduler-port 56190`。其余模型、caption bridge、`fa` 与 `--enable-torch-compile` 参数不变。

## 5. 本地提交、查询与取消任务

FlowCut 是异步接口：提交成功只说明任务已被接受，最终结果应通过查询接口或 callback 确认。当前默认并发为 1，已有任务运行时新任务会返回 `code: 2`。

### 5.1 最小本地请求

`callbackUrl` 当前为必填字段。下面示例使用本地输入路径，便于同机测试；调用方应提供一个能接收 HTTP `POST` 的本地 callback 服务。

```bash
export BASE_URL=http://127.0.0.1:31191
export TASK_ID=vividvr-local-$(date -u +%Y%m%dT%H%M%SZ)
export INPUT_VIDEO=/absolute/path/to/input.mp4
export CALLBACK_URL=http://127.0.0.1:39090/vividvr/callback

curl --fail-with-body -X POST "$BASE_URL/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  -d "{
    \"taskId\": \"$TASK_ID\",
    \"callbackUrl\": \"$CALLBACK_URL\",
    \"timeout\": -1,
    \"video_input_path\": \"$INPUT_VIDEO\",
    \"numInferenceSteps\": 20,
    \"numTemporalProcessFrames\": 121,
    \"seed\": 42,
    \"upscale\": 1.0
  }"
```

`videoUrl` 与 `video_input_path` 二选一：

- `videoUrl`：服务端可访问的 URL。
- `video_input_path`：本机可访问的绝对路径，适用于本地调试。

成功接单的同步响应为：

```json
{"code":0,"message":"ok"}
```

### 5.2 查询进度和取消

```bash
# 任务详情
curl --fail "$BASE_URL/v1/videos/repairs/flowcut/$TASK_ID"

# 任务进度
curl --fail "$BASE_URL/v1/videos/repairs/flowcut/$TASK_ID/progress"

# 取消任务
curl --fail -X DELETE "$BASE_URL/v1/videos/repairs/flowcut/$TASK_ID"
```

进度通常会经历 `accepted`、`input_ready`、`caption_ready`、`denoising`、`uploading_result` 和 `succeeded`。取消任务后，对外状态按当前服务契约会写为 `failed`，原因是 `Request timed out.`。

### 5.3 输出与对象存储

本文件的服务命令使用 `--input-save-path "" --output-path ""`，因此服务会为每个请求创建临时 workdir。未指定请求级 `outputPath` 或对象存储时，结果会以本地临时路径返回；该路径不应作为长期留存方案。若需要可靠保留或共享结果，请在请求中传 `outputPath`，或传 `minioConfig`、`outputBucket` 与 `outputObjectKey`；上传成功后查询结果中的 `url` 会指向对象存储结果。

不要把真实 access key 或 secret key 写入命令历史、日志或文档。对象存储完整字段和 callback 数据格式见现有的《Vivid-VR 超分修复服务接口说明》。

## 6. 日志、查看与停止

| 产物 | 默认目录 |
| --- | --- |
| 服务与 CLI 日志 | `/home/zhiheng/sglang/Vivid_Acceptance/logs` |
| 离线输出视频 | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos` |
| 离线指标 JSON | `/home/zhiheng/sglang/Vivid_Acceptance/indicator` |

只读查看运行中的服务：

```bash
tmux attach -r -t vividvr_caption_sidecar
tmux attach -r -t vividvr_serve_dual
```

停止本地服务：

```bash
tmux kill-session -t vividvr_serve_dual
tmux kill-session -t vividvr_caption_sidecar
```

## 7. 常见检查项

1. 主环境和 caption 环境都存在，且没有混用 Python 可执行文件。
2. 三组 checkpoint 路径可读；主服务至少需要 CogVideoX 与 VividVR 权重，开启自动 caption 还需要 CogVLM2 权重。
3. 先确认 `31200/health`，再启动或检查 `31191/health`。
4. 服务请求不再传 `captionFilePath`；启用 bridge 后 caption 由 sidecar 自动生成。
5. `numTemporalProcessFrames` 必须符合 `(N - 1) % 8 == 0`。
6. 需要稳定留存结果时传请求级 `outputPath` 或配置对象存储；只用临时本地路径不适合作为长期输出方案。

## 8. 相关文档

- [正式服务启动命令](../run_command/vividvr_formal_service_start_commands.md)
- [默认离线与服务命令](../run_command/vividvr_default_run_and_serve_commands.md)
- [Vivid-VR 服务接口说明](../vividvr服务说明文档.md)
- [VividVR benchmark 说明](../run_vivid_benchmark.md)

性能数据、测试环境与正式/实验性配置的边界将在本目录的性能文档中单独维护。
