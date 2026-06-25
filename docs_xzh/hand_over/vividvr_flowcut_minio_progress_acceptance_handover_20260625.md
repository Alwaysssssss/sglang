# Vivid-VR FlowCut 进度回调与 MinIO 验收交接

日期：`2026-06-25 UTC`

## 背景

本交接承接 `vividvr_service_boundary_alignment_handover_20260624.md`。前一阶段已完成 Vivid-VR FlowCut 服务边界拆分、专用路由 `/v1/videos/repairs/flowcut`、`result_url` 契约、队列互斥、timeout 与本地文件生命周期的基础实现。

本轮继续完成两类收口：

- 进度回调从粗粒度 stage 进度改为 denoise 主阶段进度，并将 denoise 映射到整体进度的 `5% -> 95%`。
- 使用 `moto_server` 本地模拟 S3/MinIO，跑通真实 FlowCut 请求，验证结果上传、成功回调和本地生成结果删除。

## 当前实现状态

### FlowCut 服务边界

- Vivid-VR 只走专用入口：`POST /v1/videos/repairs/flowcut`。
- 通用入口 `POST /v1/videos/repairs` 在 Vivid-VR pipeline 下会拒绝，并提示使用专用 FlowCut 路由。
- 提交响应固定为 `{"code": 0|1|2, "message": str}`。
- 成功 callback 的 `output` 只包含 `result_url` 和可选 `duration`。
- 失败 callback 不暴露本地文件路径。
- 队列仍为单任务互斥；`code=2` 表示已有任务在运行。

### 进度语义

当前 FlowCut callback 进度含义如下：

- `1.0`：任务已接受，`reason=accepted`。
- `3.0`：输入已准备，`reason=input_ready`。
- `5.0`：caption 已准备，进入 denoise 起点，`reason=caption_ready`。
- `5.0 -> 95.0`：denoise runtime progress 映射区间，`reason=denoising`。
- `98.0`：结果上传阶段，`reason=uploading_result`。
- `100.0`：任务成功，`reason=succeeded`。

实现细节：

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py` 中定义 `FLOWCUT_DENOISE_START_PROGRESS=5.0` 和 `FLOWCUT_DENOISE_END_PROGRESS=95.0`。
- `flowcut_denoise_progress(runtime_progress)` 将 pipeline 内部 `0.0 -> 1.0` 映射为 FlowCut 外部 `5.0 -> 95.0`。
- `VividVRFlowCutProgressReporter.send_denoise_progress(...)` 只在新进度严格大于上一次 denoise 进度时发送 callback，不做相同值心跳。
- `python/sglang/multimodal_gen/runtime/vividvr/progress_file.py` 负责用原子替换方式写入/读取 runtime progress 文件。
- `VividVRDenoisingStage` 和 `VividVRMultiClipDenoisingStage` 会把当前 denoise step 进度写入 request-scoped progress 文件。
- `vividvr_flowcut_api.py` 中的 monitor task 轮询该 progress 文件，并通过 reporter 发送 denoise callback。

注意：验收脚本打印的 `{"event": "progress", "body": ...}` 是 runner 主动轮询 `/v1/videos/{task_id}/progress` 的日志，不是服务端 callback。轮询频率高于阶段变化时，会看到多个相同 `5.0`、`50.0` 或 `95.0`，这是正常现象。真实 callback 以 callback log 为准。

### 文件生命周期

启用 `minioConfig` 且上传成功时：

- 远端输入对象或用户原始输入文件不删除。
- 远端结果对象保留。
- 本地生成结果上传成功后删除。
- progress API 终态返回 `file_path=null`，`url=<result_url>`。

未启用 `minioConfig` 时：

- 本地结果作为 `result_url` 返回，需要保留。
- 因此不能粗暴删除整个 request workdir；后续如要做终态清理，需要按输入副本、caption 中间文件和输出结果分开处理。

## 关键文件

- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`：注册 Vivid-VR FlowCut 专用 router。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`：通用 repair 路径移除 Vivid FlowCut 语义。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py`：共享队列、Vivid 参数映射、caption 准备和 job metadata helper。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`：FlowCut 独立 router、dispatch、timeout、上传、callback 和 denoise monitor。
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py`：stage progress 与 denoise progress 映射。
- `python/sglang/multimodal_gen/runtime/vividvr/progress_file.py`：runtime progress 文件桥接。
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`：denoise stage 写入 runtime progress。
- `python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py`：FlowCut 服务验收工具，已收口 `result_url` 和 callback 预期。
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`：默认运行和 serve 命令已同步更新。

## 已完成验证

### 单测与静态检查

前一阶段已跑通：

```bash
/home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_progress.py \
  python/sglang/multimodal_gen/test/unit/test_video_job_runner.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py \
  python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_flowcut_service_acceptance_tool.py \
  -q
```

结果：`74 passed, 6 warnings in 11.09s`。

同时已跑：

```bash
/home/zhiheng/sglang/.venv/bin/python -m py_compile ...
git diff --check
```

结果：通过。

### 单卡 FlowCut 服务边界验收

已完成一次不带 MinIO 的真实单卡服务验收：

- `task_id`：`vividvr-flowcut-boundary-single-20260624T101647Z`
- 服务口径：`CUDA_VISIBLE_DEVICES=1`、`--num-gpus 1`、`--attention-backend fa`、`--enable-torch-compile`
- endpoint：`POST /v1/videos/repairs/flowcut`
- callback：最终 `status=succeeded`、`progress=100`，`output` 只包含 `result_url`
- 共享路由检查：Vivid-VR 服务上的 `POST /v1/videos/repairs` 返回 HTTP `400`，提示使用 `/v1/videos/repairs/flowcut`
- 结果视频：`Vivid_Acceptance/result_videos/service_boundary/vividvr-flowcut-boundary-single-20260624T101647Z.mp4`
- perf：`Vivid_Acceptance/indicator/vividvr-flowcut-boundary-single-20260624T101647Z_perf.json`

### Moto S3/MinIO 本地验收

已使用 `moto_server` 在本地模拟 S3/MinIO，完成真实 FlowCut 请求验收。

Moto 服务：

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566'
```

验收任务：

- `task_id`：`vividvr-moto-minio-121f-eager-20260624T155856Z`
- 服务口径：`CUDA_VISIBLE_DEVICES=1`、单卡、`--attention-backend fa`、不启用 torch compile
- 输入视频：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption sidecar：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference：`/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`
- `num_temporal_process_frames=121`
- `num_inference_steps=2`
- MinIO endpoint：`127.0.0.1:4566`
- bucket：`flowcut`
- result URL：`http://127.0.0.1:4566/flowcut/outputs/vividvr-moto-minio-121f-eager-20260624T155856Z.mp4`

验收结论：

- submit 返回 `{"code": 0, "message": "ok"}`。
- progress API 终态为 `status=completed`、`progress=100`、`file_path=null`、`url=<result_url>`。
- Moto S3 `head_object` 返回 `ContentLength=5661200`。
- 本地输出文件已删除：`local_file_exists=False`、`local_mp4_count=0`。
- 最终 callback 序列为 `1 -> 3 -> 5 -> 50 -> 95 -> 98 -> 100`，没有相同进度值重复 callback。

验收产物：

- `Vivid_Acceptance/logs/vividvr-moto-minio-121f-eager-20260624T155856Z.moto_minio_acceptance.log`
- `Vivid_Acceptance/logs/vividvr-moto-minio-121f-eager-20260624T155856Z_callback.jsonl`
- `Vivid_Acceptance/indicator/vividvr-moto-minio-121f-eager-20260624T155856Z_perf.json`

独立复核命令：

```bash
curl --noproxy '*' -s \
  http://127.0.0.1:31220/v1/videos/vividvr-moto-minio-121f-eager-20260624T155856Z/progress
```

返回：

```json
{"id":"vividvr-moto-minio-121f-eager-20260624T155856Z","status":"completed","progress":100,"file_path":null,"url":"http://127.0.0.1:4566/flowcut/outputs/vividvr-moto-minio-121f-eager-20260624T155856Z.mp4","error":null,"callback_status":null,"callback_error":null,"callback_attempts":null}
```

S3 与本地删除复核：

```bash
NO_PROXY=127.0.0.1,localhost AWS_EC2_METADATA_DISABLED=true \
  /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
from pathlib import Path

task = "vividvr-moto-minio-121f-eager-20260624T155856Z"
s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="ak",
    aws_secret_access_key="sk",
    region_name="us-east-1",
)
head = s3.head_object(Bucket="flowcut", Key=f"outputs/{task}.mp4")
local_file = Path(f"/home/zhiheng/sglang/inputs/uploads/{task}/outputs/{task}.mp4")
local_outputs = list(local_file.parent.glob("*.mp4")) if local_file.parent.exists() else []
print({
    "s3_content_length": head["ContentLength"],
    "local_file_exists": local_file.exists(),
    "local_mp4_count": len(local_outputs),
})
PY
```

返回：

```python
{'s3_content_length': 5661200, 'local_file_exists': False, 'local_mp4_count': 0}
```

## 验收中的失败与结论

### 121f + torch compile OOM

第一次 Moto MinIO 验收使用 `--enable-torch-compile`，121f / 2 step 在当前 GPU1 上 OOM：

- `task_id`：`vividvr-moto-minio-20260624T154757Z`
- 失败点：denoise 阶段
- 现象：任务终态 `status=failed`、`progress=50.0`、没有上传到 Moto S3
- 结论：这是生成阶段显存失败，不是 MinIO 上传或删除逻辑失败

### 81f caption mismatch

第二次尝试改成 `num_temporal_process_frames=81` 以降低显存，但 caption sidecar 与 temporal clip 数不匹配：

- `task_id`：`vividvr-moto-minio-81f-20260624T155549Z`
- 错误：`caption file does not contain enough entries for the requested temporal clips`
- 结论：不能随意改 temporal frames 做验收，否则会破坏 caption sidecar 与 clip 切分的一致性

### 最终通过口径

最终通过使用 `121f` 保持 caption sidecar 语义一致，并关闭 torch compile 降低显存压力：

- 单卡 GPU1
- `--attention-backend fa`
- 不启用 `--enable-torch-compile`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

这说明 MinIO 上传和本地删除功能已经通过验收，但在当前显存条件下，`single_gpu_fa_compile` 不是 Moto MinIO 验收的可靠运行口径。

## 复现建议

### 启动 Moto S3

```bash
tmux new-session -d -s vividvr_moto_s3 \
  'cd /home/zhiheng/sglang && /home/zhiheng/sglang/.venv/bin/moto_server -H 127.0.0.1 -p 4566'
```

创建 bucket：

```bash
NO_PROXY=127.0.0.1,localhost AWS_EC2_METADATA_DISABLED=true \
  /home/zhiheng/sglang/.venv/bin/python - <<'PY'
import boto3
s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:4566",
    aws_access_key_id="ak",
    aws_secret_access_key="sk",
    region_name="us-east-1",
)
s3.create_bucket(Bucket="flowcut")
print([b["Name"] for b in s3.list_buckets()["Buckets"]])
PY
```

### 启动单卡服务

正式验收命令需放在 tmux 中运行。当前推荐用 eager 口径复现 MinIO 验收：

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

### 提交 FlowCut MinIO 请求

本轮使用的临时验收 runner 位于：

```text
Vivid_Acceptance/logs/moto_minio_flowcut_acceptance_runner.py
```

该脚本在日志目录下，当前被 git ignore 忽略。后续如果希望固定成标准工具，建议迁移到：

```text
python/sglang/multimodal_gen/tools/run_flowcut_vividvr_moto_minio_acceptance.py
```

## 当前工作区状态

截至本交接编写时，工作区仍有未提交改动，主要属于本阶段实现：

- 服务入口与路由拆分：`http_server.py`、`video_api.py`、`vividvr_flowcut_api.py`、`video_repair_shared.py`
- progress 实现：`vividvr_flowcut_progress.py`、`progress_file.py`、`vividvr.py`
- caption manifest 与测试更新：`caption_manifest.py`、相关 unit tests
- acceptance 工具与命令文档：`run_flowcut_vividvr_service_acceptance.py`、`docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- 计划与交接文档：`.codex/plans/2026-06-24-vividvr-service-boundary-alignment-plan.md`、`docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md`

不要在未确认的情况下回退这些改动；它们共同构成本轮 FlowCut 服务边界、progress callback 与 MinIO 验收的实现。

## 后续建议

- 将 Moto MinIO 验收 runner 从 `Vivid_Acceptance/logs/` 提升为正式工具，避免关键验收逻辑只保存在 ignored 日志目录。
- 为 MinIO 上传成功后删除本地输出增加可复用的自动化测试或轻量集成测试；当前真实验收已经通过，但自动化入口还不够标准。
- 如果后续要求 `single_gpu_fa_compile` 也通过 MinIO 验收，需要单独处理 compile 显存峰值；当前 121f compile OOM 已记录，不应误判为存储逻辑问题。
- 不要用 81f 作为这个 caption sidecar 的验收替代口径，除非同步生成匹配 81f temporal clip 切分的 caption 文件。
- 如需进一步收口文件生命周期，优先实现“终态清理输入副本和中间文件，但保留无 MinIO 场景下的本地结果文件”的细粒度策略。
- 当前任务状态仍在进程内 `VIDEO_STORE`，服务重启后历史任务查询不可恢复；如果 FlowCut 需要跨重启查询，需要引入持久化状态。

