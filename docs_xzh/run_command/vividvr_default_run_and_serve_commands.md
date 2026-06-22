# Vivid-VR 当前默认运行命令

本文档记录当前 `Phase E` 长视频 `130f / 20 step` 口径下的默认命令。

当前默认配置：

- 单卡默认：`single_gpu_fa_compile`
- 双卡默认：`dual_gpu_fa_eager_compile`
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
- 单卡正式 benchmark 时同一时刻只能有一个单卡推理进程，避免并发导致耗时失真。
- 如果是 `serve` benchmark，必须先做一次 warmup，再记录第二次正式请求；warmup 不计入正式结果。
- 本文默认 `serve` 命令使用 `--host 127.0.0.1`，只允许本机访问；如果要让其他服务器请求当前机器上的 Vivid-VR 服务，需要按“对外暴露端口”章节改为 `--host 0.0.0.0` 或指定网卡 IP。

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

## 3. 单卡 `serve` 拉起命令

```bash
tmux new-session -d -s vividvr_serve_single_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/sglang serve \
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
    --host 127.0.0.1 \
    --port 31190 \
    --master-port 30190 \
    --scheduler-port 56190 \
    --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_single_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_serve_single_default
```

## 4. 双卡 `serve` 拉起命令

```bash
tmux new-session -d -s vividvr_serve_dual_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
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
    --host 127.0.0.1 \
    --port 31191 \
    --master-port 30191 \
    --scheduler-port 56191 \
    --strict-ports \
    --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark \
    --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

```bash
tmux new-session -d -s vividvr_serve_dual_default \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
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
    2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_default_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_serve_dual_default
```

## 5. 对外暴露端口

当前 `sglang.multimodal_gen` 的服务端支持通过 `--host` 控制 HTTP API 监听地址：

- `--host 127.0.0.1`：只监听本机 loopback，只有当前机器能用 `http://127.0.0.1:<port>` 访问。
- `--host 0.0.0.0`：监听全部 IPv4 网卡，局域网或其他服务器可以通过当前机器的真实 IP 访问。
- `--host <SERVER_LAN_IP>`：只监听指定网卡 IP，适合需要限制暴露范围的部署。

如果要让其他服务器调用单卡服务，把第 3 节命令中的：

```bash
--host 127.0.0.1 \
--port 31190 \
```

改为：

```bash
--host 0.0.0.0 \
--port 31190 \
```

如果要让其他服务器调用双卡服务，把第 4 节命令中的：

```bash
--host 127.0.0.1 \
--port 31191 \
```

改为：

```bash
--host 0.0.0.0 \
--port 31191 \
```

服务启动后，日志中应能看到类似：

```text
Uvicorn running on http://0.0.0.0:31191
```

在 Vivid-VR 服务所在机器上查看可被其他机器访问的 IP：

```bash
hostname -I
```

假设服务机器 IP 是 `192.168.1.20`，双卡服务端口是 `31191`，则其他电脑上的请求地址应写为：

```bash
export BASE_URL=http://192.168.1.20:31191
```

在其他电脑上先做健康检查：

```bash
curl --noproxy '*' --silent --show-error --fail "${BASE_URL}/health"
```

如果健康检查连不上，优先检查：

- 服务是否确实用 `--host 0.0.0.0` 或服务器网卡 IP 启动，而不是 `127.0.0.1`。
- 服务器防火墙、安全组、机房 ACL 是否放行对应 TCP 端口，例如 `31190` 或 `31191`。
- 如果服务跑在 Docker 或 Kubernetes 中，容器端口是否映射到宿主机，Kubernetes Service 是否暴露了对应端口。
- 其他电脑访问时不能使用 `127.0.0.1`；`127.0.0.1` 永远指向发起请求的那台电脑自己。

远程请求时还需要注意输入、输出和 callback 地址：

- `video_input_path`、`caption_file_path`、`reference_video_path`、`output_path`、`perf_dump_path` 是 Vivid-VR 服务所在机器上的路径，不是发起请求那台电脑上的路径。
- 如果输入视频在其他电脑上，不能直接把那台电脑的本地路径传给服务；应先放到 Vivid-VR 服务机器可读的位置，或使用服务可访问的 URL / 对象存储输入。
- `callbackUrl` 必须是 Vivid-VR 服务机器能访问到的地址；远程调用时不要写 `http://127.0.0.1:...`，除非 callback 服务也运行在 Vivid-VR 服务机器本机。
- 如果结果只保存在 `output_path`，文件会落在 Vivid-VR 服务机器本地；远程电脑可以通过 `/v1/videos/${TASK_ID}/content` 下载，或通过 `minioConfig` / 对象存储拿到可访问 URL。

## 6. `curl` 请求命令

单卡服务默认：

```bash
export BASE_URL=http://127.0.0.1:31190
export BASE_URL=http://10.119.16.10:31190
```

双卡服务默认：

```bash
export BASE_URL=http://127.0.0.1:31191
export BASE_URL=http://10.119.16.10:31191
```

如果在其他电脑上请求已对外暴露的 Vivid-VR 服务，把 `BASE_URL` 换成服务机器 IP：

```bash
export BASE_URL=http://<VIVIDVR_SERVER_IP>:31191
```

共同变量：

```bash
export INPUT_VIDEO=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
export CAPTION_FILE=/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
export REFERENCE_VIDEO=/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4
export OUTPUT_DIR=/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark
export INDICATOR_DIR=/home/zhiheng/sglang/Vivid_Acceptance/indicator
export TASK_ID=vividvr-manual-$(date -u +%Y%m%dT%H%M%SZ)
export OUTPUT_PATH=${OUTPUT_DIR}/${TASK_ID}.mp4
export PERF_DUMP_PATH=${INDICATOR_DIR}/${TASK_ID}_perf.json
```

检查服务健康：

```bash
curl --noproxy '*' --silent --show-error --fail "${BASE_URL}/health"
```

提交任务：

```bash
curl --noproxy '*' -X POST "${BASE_URL}/v1/videos/repairs" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "VividVR",
  "task_id": "${TASK_ID}",
  "video_input_path": "${INPUT_VIDEO}",
  "caption_file_path": "${CAPTION_FILE}",
  "reference_video_path": "${REFERENCE_VIDEO}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_PATH}",
  "perf_dump_path": "${PERF_DUMP_PATH}"
}
JSON
```

提交 FlowCut 兼容任务：

```bash
export FLOWCUT_CALLBACK_URL=http://<CALLBACK_SERVER_IP>:39090/tasks/${TASK_ID}/callback

curl --noproxy '*' -X POST "${BASE_URL}/v1/videos/repairs/flowcut" \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "taskId": "${TASK_ID}",
  "timeout": -1,
  "callbackUrl": "${FLOWCUT_CALLBACK_URL}",
  "video_input_path": "${INPUT_VIDEO}",
  "caption_file_path": "${CAPTION_FILE}",
  "reference_video_path": "${REFERENCE_VIDEO}",
  "num_inference_steps": 20,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "${OUTPUT_PATH}",
  "perf_dump_path": "${PERF_DUMP_PATH}"
}
JSON
```

推荐使用仓库内验收脚本提交并轮询，脚本会遵守 FlowCut 返回码语义：`code=2` 只重试提交，`code=1` 立即失败，只有 `code=0` 接单后才轮询进度；如果轮询阶段收到 `404`，按“服务可能已重启或该进程未接单”处理，不继续盲查旧任务。

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url "${BASE_URL}" \
  --task-id "${TASK_ID}" \
  --callback-url "${FLOWCUT_CALLBACK_URL}" \
  --input-video "${INPUT_VIDEO}" \
  --caption-file "${CAPTION_FILE}" \
  --reference-video "${REFERENCE_VIDEO}" \
  --num-inference-steps 20 \
  --seed 42 \
  --num-temporal-process-frames 121 \
  --output-path "${OUTPUT_PATH}" \
  --perf-dump-path "${PERF_DUMP_PATH}"
```

说明：

- `POST /v1/videos/repairs/flowcut` 是 FlowCut 专用兼容入口；普通 OpenAI 风格调用仍使用 `/v1/videos/repairs`。
- `timeout=-1` 表示 Vivid-VR 服务侧不对长推理设置超时；同步接单仍应快速返回。
- FlowCut callback 使用 `running`、`succeeded`、`failed` 状态，并通过 `progress` 上报中间进度。

查询进度：

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

如果要做正式 benchmark：

1. 先用一个单独的 `TASK_ID` 提交一次 warmup。
2. warmup 完成后，再换一个新的 `TASK_ID` 提交正式请求。
3. 正式统计只记录第二次请求，不把 warmup 计入最终结果。
