# Vivid-VR 当前默认运行命令

本文档记录当前 `Phase E` 长视频 `130f / 20 step` 口径下的默认命令。

当前默认配置：

- 单卡默认：`single_gpu_fa_compile`
- 双卡默认：`dual_gpu_fa_eager_compile`
- 本地 `offline / benchmark` 默认 `upscale=1.0`，与原版 `/home/zhiheng/Vivid-VR` 的 `up1` 语义对齐
- 双卡默认质量口径固定：
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
  - `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`
  - 请求侧 `--attention-backend fa`，双卡 `SP=2` 运行时有效 backend 记为 `fa_sp`

共同输入：

- 输入视频：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- prompt：`/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- reference：`/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- 固定参数：`seed=42`、`num_inference_steps=20`、`num_temporal_process_frames=121`

注意：

- 所有长时间推理都应放在 `tmux` 中启动。
- 如果要让 `serve` 自动生成 caption sidecar，先启动 `vividvr_caption_sidecar`，再启动主服务。
- caption sidecar 的代码、启动脚本和 HTTP 服务都在 `sglang` 仓库内；运行环境固定为 `/home/zhiheng/sglang/.venv-vividvr-caption`，可通过 `python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh` 创建；该脚本会把仓库 `python/` 路径写入 sidecar env 的 `.pth`，并在无 `PYTHONPATH` 的条件下自检导入。
- `/home/zhiheng/Vivid-VR` 当前只继续提供静态资源，例如 checkpoint、输入视频、`prompt.txt`、reference 和基线 caption 文件。
- 主推理和 `serve` 必须继续使用 `/home/zhiheng/sglang/.venv`；不要为了 caption 把主推理切回原版环境。
- 当前 caption bridge 的 sidecar 文本契约是：`caption.txt` 一行对应一个 temporal clip，行数、顺序和文本内容都必须与单卡基线逐字一致。
- 当前 `upscale` 已接入 `serve / FlowCut` 请求体，但它仍然表示**原版 Vivid-VR 的输入预缩放语义**，不是 `enable_upscaling / upscaling_scale` 那条后处理超分链。
- 单卡正式 benchmark 时同一时刻只能有一个单卡推理进程，避免并发导致耗时失真。
- 如果是 `serve` benchmark，必须先做一次 warmup，再记录第二次正式请求；warmup 不计入正式结果。
- 本文默认 `serve` 命令使用 `--host 0.0.0.0`，方便其他服务器直接请求当前机器上的 Vivid-VR 服务；如果只希望本机访问，可按“对外暴露端口”章节改回 `--host 127.0.0.1` 或指定网卡 IP。

## 1. 单卡直接运行命令

```bash
tmux new-session -d -s vividvr_single_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
    --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
    --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
    --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 \
    --output-dir /home/zhiheng/sglang/Vivid_Acceptance/result_videos \
    --report-dir /home/zhiheng/sglang/Vivid_Acceptance/indicator \
    --artifact-prefix phase_e_single_gpu_fa_compile_130f_20step \
    --phase-label E \
    --mode-label single_gpu_fa_compile \
    --num-temporal-process-frames 121 \
    --num-inference-steps 20 \
    --upscale 1.0 \
    --seed 42 \
    --num-gpus 1 \
    --tp-size 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --attention-backend fa \
    --enable-torch-compile \
    2>&1 | tee Vivid_Acceptance/logs/phase_e_single_gpu_fa_compile_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_single_default
```

## 2. 双卡直接运行命令

```bash
tmux new-session -d -s vividvr_dual_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30062 python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
    --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
    --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
    --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 \
    --output-dir /home/zhiheng/sglang/Vivid_Acceptance/result_videos \
    --report-dir /home/zhiheng/sglang/Vivid_Acceptance/indicator \
    --artifact-prefix phase_e_dual_gpu_fa_eager_compile_130f_20step \
    --phase-label E \
    --mode-label dual_gpu_fa_eager_compile \
    --num-temporal-process-frames 121 \
    --num-inference-steps 20 \
    --upscale 1.0 \
    --seed 42 \
    --num-gpus 2 \
    --tp-size 1 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --dist-timeout 3600 \
    --master-port 30062 \
    --attention-backend fa \
    --enable-torch-compile \
    2>&1 | tee Vivid_Acceptance/logs/phase_e_dual_gpu_fa_eager_compile_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_dual_default
```

如果要显式复现原版 `upscale` 语义，可只改这一个参数：

- `--upscale 0.0`：把输入短边缩放到 `1024`
- `--upscale 1.0`：保持输入分辨率不变
- `--upscale <正数且不等于 1.0>`：按倍率做推理前输入 resize

## 3. 当前推荐的 `serve` 启动顺序

当前正式推荐的服务形态是：

- caption sidecar：独立 HTTP 服务，运行在 `/home/zhiheng/sglang/.venv-vividvr-caption`
- 主服务：`sglang serve`，运行在 `/home/zhiheng/sglang/.venv`
- 当前默认正式配置：双卡 `dual_gpu_fa_eager_compile`
- 默认端口：
  - caption sidecar：`31200`
  - 单卡主服务：`31190`
  - 双卡主服务：`31191`

当前 bridge 路径的关键约束：

- 主服务必须显式打开 `--vividvr-caption-bridge`
- 主服务必须显式传 `--vividvr-caption-sidecar-url http://127.0.0.1:31200`
- sidecar 输出目录固定为 `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars`
- 自动 bridge 请求默认不传 `caption_file_path`
- Vivid-VR 服务请求统一走 `POST /v1/videos/repairs/flowcut`；共享 `POST /v1/videos/repairs` 在 Vivid server 下会拒绝并提示使用专用路由
- FlowCut 提交响应只返回 `{"code": 0|1|2, "message": "..."}`；`code=0` 只表示已接单，真正生成继续在后端执行，不能只看首个 HTTP 返回
- 成功 callback 的 `output` 只允许包含 `result_url` 和可选 `duration`，不能返回 `gen_video_url` 或 `file_path`

### 3.1 首次使用时创建 caption env

这一步通常只需要做一次：

```bash
cd /home/zhiheng/sglang
bash python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh
```

成功后，caption sidecar 独立环境应位于：

```bash
/home/zhiheng/sglang/.venv-vividvr-caption
```

### 3.2 启动 caption sidecar

当前正式验收使用的是 dual-worker sidecar，两个 worker 分别绑定 `cuda:0` 和 `cuda:1`：

```bash
tmux new-session -d -s vividvr_caption_sidecar \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31200 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看 sidecar：

```bash
tmux attach -r -t vividvr_caption_sidecar
```

检查 sidecar 健康：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31200/health
```

如果 sidecar 日志里出现：

```text
[VividVR Caption Sidecar] python_include=...
```

说明它已经找到可用的 Python dev headers，可以继续执行 caption。

### 3.3 启动双卡主服务

这是当前正式默认口径：

```bash
tmux new-session -d -s vividvr_serve_dual_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 2 \
    --tp-size 1 \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1 \
    --enable-torch-compile \
    --dist-timeout 3600 \
    --host 0.0.0.0 \
    --port 31191 \
    --master-port 30191 \
    --scheduler-port 56191 \
    --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    --vividvr-caption-bridge \
    --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
    --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
    --vividvr-caption-sidecar-timeout 1800 \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看主服务：

```bash
tmux attach -r -t vividvr_serve_dual_default
```

检查主服务健康：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31191/health
```

### 3.4 单卡主服务命令

如果只需要单卡 `serve`，使用下面这条命令。caption sidecar 仍按上一节先起：

```bash
tmux new-session -d -s vividvr_serve_single_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve \
    --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --model-id VividVR \
    --pipeline-class-name CogVideoXVividVRControlNetPipeline \
    --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
    --attention-backend fa \
    --num-gpus 1 \
    --tp-size 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --enable-torch-compile \
    --dist-timeout 3600 \
    --host 0.0.0.0 \
    --port 31190 \
    --master-port 30190 \
    --scheduler-port 56190 \
    --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    --vividvr-caption-bridge \
    --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
    --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars \
    --vividvr-caption-sidecar-timeout 1800 \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_single_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看单卡主服务：

```bash
tmux attach -r -t vividvr_serve_single_default
```

### 3.5 统一环境变量

下面请求命令默认以双卡主服务为例；如果你起的是单卡服务，把 `BASE_URL` 改成 `31190`。

```bash
export BASE_URL=http://127.0.0.1:31191
export OUTPUT_DIR=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark
export INDICATOR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/indicator
export LOG_DIR=/home/zhiheng/sglang/Vivid_Acceptance/logs
export CALLBACK_BASE_URL=http://<CALLBACK_SERVER_IP>:39090
mkdir -p "${OUTPUT_DIR}" "${INDICATOR_DIR}" "${LOG_DIR}"
```

默认 `2 clip` 输入：

```bash
export INPUT_VIDEO_130F=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
```

当前已正式验收过的 `4 clip` 输入：

```bash
export INPUT_VIDEO_301F=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_301f_h264.mp4
```

### 3.6 warmup 请求

正式 benchmark 前，先做一次 warmup。下面这条是当前推荐的自动 bridge warmup 方式，不显式传 `caption_file_path`：

```bash
export WARMUP_TASK_ID=vividvr-warmup-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${WARMUP_TASK_ID}.submit.log"
{
  "taskId": "${WARMUP_TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${WARMUP_TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 1,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_DIR}/${WARMUP_TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${WARMUP_TASK_ID}_perf.json"
}
JSON
```

### 3.7 正式 `2 clip` 请求

这是当前默认 `130f / 20 step` 正式请求：

```bash
export TASK_ID=vividvr-service-benchmark-130f-bridge-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json"
}
JSON
```

### 3.8 正式 `4 clip` 请求

如果要验证 `clip > 2` 的自动 bridge 路径，直接把输入换成当前已验收通过的 `301f` 样本：

```bash
export TASK_ID=vividvr-service-benchmark-301f-4clip-bridge-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_301F}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json"
}
JSON
```

当前这条 `4 clip` 路径已经完成端到端验收，sidecar 会自动生成 `4` 行 caption，并由主服务继续完成推理。

### 3.9 显式 caption replay 请求

如果你要绕过 bridge，复用已知 caption 文件，可以显式传入：

```bash
export TASK_ID=vividvr-manual-replay-$(date -u +%Y%m%dT%H%M%SZ)
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt

NO_PROXY=* curl -sS -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "caption_file_path": "${CAPTION_FILE}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json"
}
JSON
```

### 3.10 FlowCut 兼容请求

如果需要手写最小 FlowCut 兼容请求：

```bash
export TASK_ID=vividvr-flowcut-$(date -u +%Y%m%dT%H%M%SZ)
export FLOWCUT_CALLBACK_URL=http://<CALLBACK_SERVER_IP>:39090/tasks/${TASK_ID}/callback

NO_PROXY=* curl -sS -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${FLOWCUT_CALLBACK_URL}",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json"
}
JSON
```

推荐优先使用仓库内验收脚本提交并轮询，脚本会遵守 FlowCut 返回码语义：`code=2` 只重试提交，`code=1` 立即失败，只有 `code=0` 接单后才轮询进度；如果轮询阶段收到 `404`，按“服务可能已重启或该进程未接单”处理，不继续盲查旧任务。使用 `--callback-log` 时脚本会在本机临时拉起 callback receiver，并校验成功 callback 的 `output.result_url` 契约。

显式 caption replay 验收示例：

```bash
export TASK_ID=vividvr-flowcut-replay-$(date -u +%Y%m%dT%H%M%SZ)
export CALLBACK_LOG=${LOG_DIR}/${TASK_ID}_callback.jsonl

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-log "${CALLBACK_LOG}" \
  --input-video "${INPUT_VIDEO_130F}" \
  --caption-file "${CAPTION_FILE}" \
  --num-inference-steps 20 \
  --seed 42 \
  --num-temporal-process-frames 121 \
  --output-path "${OUTPUT_DIR}/${TASK_ID}.mp4" \
  --perf-dump-path "${INDICATOR_DIR}/${TASK_ID}_perf.json" \
  --poll-timeout-s 2400
```

自动 caption bridge 验收示例：

```bash
export TASK_ID=vividvr-flowcut-bridge-$(date -u +%Y%m%dT%H%M%SZ)
export CALLBACK_LOG=${LOG_DIR}/${TASK_ID}_callback.jsonl

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-log "${CALLBACK_LOG}" \
  --video-input-path "${INPUT_VIDEO_130F}" \
  --num-inference-steps 20 \
  --seed 42 \
  --num-temporal-process-frames 121 \
  --output-path "${OUTPUT_DIR}/${TASK_ID}.mp4" \
  --perf-dump-path "${INDICATOR_DIR}/${TASK_ID}_perf.json" \
  --submit-timeout-s 2400 \
  --poll-timeout-s 2400
```

### 3.11 查询进度与下载结果

请求提交后，先拿到的是排队结果，后续需要继续查进度：

```bash
curl --noproxy '*' -X GET "${BASE_URL}/v1/videos/${TASK_ID}/progress"
```

查询任务详情：

```bash
curl --noproxy '*' -X GET "${BASE_URL}/v1/videos/${TASK_ID}"
```

下载结果视频：

```bash
curl --noproxy '*' -X GET "${BASE_URL}/v1/videos/${TASK_ID}/content" \
  -o "${OUTPUT_DIR}/downloaded_${TASK_ID}.mp4"
```

bridge 场景下的 sidecar 文本默认写到：

```bash
/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/${TASK_ID}.txt
```

主日志和 sidecar 日志默认分别在：

```bash
/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_serve_*.log
/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_caption_sidecar_*.log
```

### 3.12 启动本地 S3/MinIO 模拟服务

如果要做 FlowCut `minioConfig` 上传验收，先起一个本地 `moto_server` 作为 S3/MinIO 模拟服务：

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566 2>&1 | tee Vivid_Acceptance/logs/vividvr_moto_s3_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看模拟 S3 服务：

```bash
tmux attach -r -t vividvr_moto_s3
```

准备环境变量并创建 bucket：

```bash
export MOTO_S3_ENDPOINT=127.0.0.1:4566
export MOTO_S3_BUCKET=flowcut
export MOTO_S3_ACCESS_KEY=test
export MOTO_S3_SECRET_KEY=test

PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
s3.create_bucket(Bucket="flowcut")
print([b["Name"] for b in s3.list_buckets()["Buckets"]])
PY
```

### 3.13 启动 FlowCut MinIO 单卡服务

当前本地 `moto_server` 验收口径走单卡 `fa eager`，不启用 caption bridge，直接复用已知 `caption_file_path`。这里如果要保持当前已验收基线，建议继续显式传 `upscale=1.0`。

```bash
tmux new-session -d -s vividvr_moto_minio_service \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && \
   export CUDA_VISIBLE_DEVICES=1 && \
   export PYTHONPATH=python && \
   export NO_PROXY=127.0.0.1,localhost && \
   export AWS_EC2_METADATA_DISABLED=true && \
   export SGLANG_FLOWCUT_PROGRESS_INTERVAL_SECONDS=5 && \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /home/zhiheng/sglang/.venv/bin/sglang serve \
     --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
     --model-id VividVR \
     --pipeline-class-name CogVideoXVividVRControlNetPipeline \
     --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR \
     --num-gpus 1 \
     --attention-backend fa \
     --host 127.0.0.1 \
     --port 31220 \
     --master-port 30220 \
     --scheduler-port 56220 \
     --strict-ports \
     --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
     --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
     2>&1 | tee Vivid_Acceptance/logs/vividvr_moto_minio_service_fa_eager_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看服务：

```bash
tmux attach -r -t vividvr_moto_minio_service
```

检查健康：

```bash
curl --noproxy '*' --silent --show-error --fail http://127.0.0.1:31220/health
```

### 3.14 FlowCut MinIO 模拟请求

先准备请求变量：

```bash
export MOTO_BASE_URL=http://127.0.0.1:31220
export MOTO_CALLBACK_BASE_URL=http://<CALLBACK_SERVER_IP>:39090
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
```

然后提交带 `minioConfig` 的 FlowCut 模拟请求：

```bash
export TASK_ID=vividvr-moto-minio-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* curl -sS -X POST "${MOTO_BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON | tee "${LOG_DIR}/${TASK_ID}.submit.log"
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${MOTO_CALLBACK_BASE_URL}/tasks/${TASK_ID}/callback",
  "video_input_path": "${INPUT_VIDEO_130F}",
  "caption_file_path": "${CAPTION_FILE}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "upscale": 1.0,
  "output_path": "${OUTPUT_DIR}/${TASK_ID}.mp4",
  "perf_dump_path": "${INDICATOR_DIR}/${TASK_ID}_perf.json",
  "minioConfig": {
    "endpoint": "${MOTO_S3_ENDPOINT}",
    "bucket_name": "${MOTO_S3_BUCKET}",
    "access_key": "${MOTO_S3_ACCESS_KEY}",
    "secret_key": "${MOTO_S3_SECRET_KEY}",
    "secure": false,
    "region": "us-east-1"
  }
}
JSON
```

如果你已经有 callback receiver，也可以把这条请求改成自动 bridge 版本；当前字段差异只在于不传 `caption_file_path`。

如果你用的是仓库内的 acceptance runner，也可以直接带 `--upscale` 提交：

```bash
export TASK_ID=vividvr-moto-minio-runner-$(date -u +%Y%m%dT%H%M%SZ)

NO_PROXY=* PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${MOTO_BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-log "${LOG_DIR}/${TASK_ID}.callback.jsonl" \
  --video-input-path "${INPUT_VIDEO_130F}" \
  --caption-file-path "${CAPTION_FILE}" \
  --output-path "${OUTPUT_DIR}/${TASK_ID}.mp4" \
  --perf-dump-path "${INDICATOR_DIR}/${TASK_ID}_perf.json" \
  --num-inference-steps 20 \
  --num-temporal-process-frames 121 \
  --upscale 1.0 \
  --seed 42
```

### 3.15 查询 FlowCut MinIO 结果

轮询任务进度：

```bash
curl --noproxy '*' -X GET "${MOTO_BASE_URL}/v1/videos/${TASK_ID}/progress"
```

检查模拟 S3 中是否已有上传对象：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
import os

task_id = os.environ["TASK_ID"]
s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)
head = s3.head_object(Bucket="flowcut", Key=f"outputs/{task_id}.mp4")
print({"content_length": head["ContentLength"]})
PY
```

成功 callback 返回的 `result_url` 形态应为：

```text
http://127.0.0.1:4566/flowcut/outputs/<TASK_ID>.mp4
```

## 4. 对外暴露端口

当前 `sglang.multimodal_gen` 的服务端支持通过 `--host` 控制 HTTP API 监听地址：

- `--host 127.0.0.1`：只监听本机 loopback，只有当前机器能访问
- `--host 0.0.0.0`：监听全部 IPv4 网卡，其他服务器也能访问；这是本文当前默认 `serve` 口径
- `--host <SERVER_LAN_IP>`：只监听指定网卡 IP

当前双卡和单卡默认命令已经使用 `0.0.0.0`。如果你只想让本机访问双卡服务，可把第 `3.3` 节命令里的：

```bash
--host 0.0.0.0 \
--port 31191 \
```

改为：

```bash
--host 127.0.0.1 \
--port 31191 \
```

如果你只想让本机访问单卡服务，可把第 `3.4` 节命令里的：

```bash
--host 0.0.0.0 \
--port 31190 \
```

改为：

```bash
--host 127.0.0.1 \
--port 31190 \
```

服务启动后，日志中应能看到类似：

```text
Uvicorn running on http://0.0.0.0:31191
```

在服务机器上查看可被其他机器访问的 IP：

```bash
hostname -I
```

假设服务机器 IP 是 `192.168.1.20`，双卡服务端口是 `31191`，则远程请求地址应写为：

```bash
export BASE_URL=http://192.168.1.20:31191
```

远程调用时还要注意：

- `video_input_path`、`caption_file_path`、`output_path`、`perf_dump_path` 都是服务机器上的路径
- 其他机器不能把自己的本地路径直接传给服务
- `callbackUrl` 必须是服务机器可访问的地址，远程调用时不要写 `127.0.0.1`
- 如果结果只写到 `output_path`，文件会落在服务机器本地；远程侧可以再调用 `/v1/videos/${TASK_ID}/content` 下载
