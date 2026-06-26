# Vivid-VR FlowCut 输入清理、输出格式与取消语义对齐交接（2026-06-26）

## 本轮目标

本轮继续保留 Vivid-VR 自己的独立入口：

- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py`

不把 FlowCut 并回共享 `video_api.py` 主线，只对齐请求端可感知的 3 项服务语义：

1. 输入缓存删除语义向 `share-tyx` 的 `video_edit` 对齐
2. 输出文件扩展名继承输入视频格式
3. `DELETE` 取消任务语义向 `origin/online_videoedit` 对齐

## 已完成的语义对齐

### 1. 输入缓存删除语义

- FlowCut 现在把“是否使用临时 request workdir”仅绑定到 `input_save_path`：
  - `input_save_path` 为空：使用临时 `request workdir`
  - `input_save_path` 非空：使用持久 `request workdir`
- `output_path` 不再决定输入缓存是否持久，只决定结果落点。
- 对齐后的删除语义是：
  - 删除的是服务为本次请求落地的本地缓存输入
  - 不是删除请求方原始本地 `video_input_path`
- 当结果已经外部化后，临时 request workdir 会在任务结束时被清理：
  - 输入视频副本
  - caption sidecar
  - request-local manifest / progress 文件

### 2. 输出格式继承输入视频格式

- FlowCut 输入视频 materialize 时会保留源扩展名，不再一律写成 `input.mp4`。
- FlowCut 结果文件名默认继承输入扩展名：
  - 输入 `.mov`，默认结果文件名也为 `.mov`
- 如果请求方显式给了 `output_path=/path/result.mp4`，但输入是 `.mov`，FlowCut 仍会把结果名收口为 `result.mov`。
- 如果请求方给了 `outputObjectKey` 但未带扩展名，服务会补成输入扩展名。
- `SamplingParams` 的输出文件扩展名校验已补齐视频格式白名单，避免 `.mov` 被二次改写成 `.mov.mp4`。

### 3. `DELETE` 取消任务语义

- FlowCut 新增独立取消入口：
  - `DELETE /v1/videos/repairs/flowcut/{video_id}`
- FlowCut 现在也有自己的读任务与读进度入口：
  - `GET /v1/videos/repairs/flowcut/{video_id}`
  - `GET /v1/videos/repairs/flowcut/{video_id}/progress`
- 语义对齐 `origin/online_videoedit`，取消后的对外状态不是 `cancelled`，而是：
  - `status = failed`
  - `reason = Request timed out.`
  - `error.message = Request timed out.`
- FlowCut 现在也有自己的取消骨架：
  - 任务注册表
  - cancel marker 文件
  - `request_cancel_path`
  - 任务级 `asyncio.Task.cancel()`
- 取消信号已传入 Vivid-VR runtime，并在长时间 denoise 循环里做协作式检查，避免 API 层删除后底层推理继续偷偷跑完。

### 4. FlowCut 专有响应模型收口

- 这轮把之前越界到共享层的补丁收回到了 FlowCut 自己的入口与协议中：
  - `video_api.py` 不再承载 FlowCut 专有 `reason/progress` 语义
  - 通用 `protocol.py` 的 `VideoResponse.reason` 已撤回
- FlowCut 现在改用自己的 response model：
  - `FlowCutVideoResponse`
  - `FlowCutProgressResponse`
- 这样后续请求方如果要读取 FlowCut 的 `reason`，必须走 FlowCut 自己的：
  - `GET /v1/videos/repairs/flowcut/{video_id}`
  - `GET /v1/videos/repairs/flowcut/{video_id}/progress`
  - `DELETE /v1/videos/repairs/flowcut/{video_id}`

## 本轮涉及的主要文件

- `python/sglang/multimodal_gen/runtime/request_timeout.py`
- `python/sglang/multimodal_gen/configs/sample/sampling_params.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/flowcut.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
- `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
- `python/sglang/multimodal_gen/test/unit/test_sampling_params.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_flowcut_storage.py`

## 本轮测试与验收

### 单测

本轮已覆盖并跑通：

- FlowCut 请求 alias 与 timeout 规则
- 临时 / 持久输入 workdir 语义
- 输入视频扩展名保留
- 默认 / 显式 `outputObjectKey` 与 `outputBucket`
- `.mov` 输出文件名保持不被追加 `.mp4`
- FlowCut `DELETE` 取消任务与失败回调
- FlowCut 自有 `GET /repairs/flowcut/{id}` 与 `GET /progress` 路由注册、返回模型与 `reason` 透出

### 真实验收

双卡 FlowCut `serve`、mock S3、callback receiver、caption sidecar 全部通过 `tmux` 拉起。

服务启动口径：

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve \
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
  --port 31193 \
  --master-port 30193 \
  --scheduler-port 56193 \
  --strict-ports \
  --output-path "" \
  --input-save-path "" \
  --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --vividvr-caption-bridge \
  --vividvr-caption-sidecar-url http://127.0.0.1:31200 \
  --vividvr-caption-work-dir "" \
  --vividvr-caption-sidecar-timeout 1800
```

验收时请注意：本机 localhost 请求需要显式带 `NO_PROXY=127.0.0.1,localhost`，否则客户端侧可能被代理链路误伤。

### 本轮最终验收结果

#### 成功链路

- task id：
  - `vividvr-align-success-20260626T062327Z`
- 请求结果：
  - `GET /v1/videos/repairs/flowcut/vividvr-align-success-20260626T062327Z`
    返回 `status=completed`
  - `GET /v1/videos/repairs/flowcut/vividvr-align-success-20260626T062327Z/progress`
    返回 `status=completed`、`url=http://127.0.0.1:4566/flowcut/service-semantics/vividvr-align-success-20260626T062327Z.mov`
- perf 文件：
  - `Vivid_Acceptance/indicator/input_format_cancel_alignment/vividvr-align-success-20260626T062327Z_perf.json`
- callback 日志：
  - `Vivid_Acceptance/logs/mock_callback_align_20260626T051656Z.jsonl`
  - 终态为 `{"status":"succeeded","reason":"succeeded","output":"{\"result_url\":\"http://127.0.0.1:4566/flowcut/service-semantics/vividvr-align-success-20260626T062327Z.mov\"}"}`
- S3 object：
  - bucket: `flowcut`
  - key: `service-semantics/vividvr-align-success-20260626T062327Z.mov`
  - `head_object` 已确认存在，`content_length=10899903`
- 清理结果：
  - 临时 request workdir `/tmp/sglang_vividvr_flowcut_5bp5swwr` 已清空
  - 输入副本与 caption sidecar 都已删除

#### 取消链路

- task id：
  - `vividvr-align-cancel-20260626T063429Z`
- 请求结果：
  - `DELETE /v1/videos/repairs/flowcut/vividvr-align-cancel-20260626T063429Z`
    返回 `status=failed`
  - `GET /v1/videos/repairs/flowcut/vividvr-align-cancel-20260626T063429Z`
    返回 `reason=Request timed out.`
  - `GET /v1/videos/repairs/flowcut/vividvr-align-cancel-20260626T063429Z/progress`
    返回 `reason=Request timed out.`
- perf 文件：
  - `Vivid_Acceptance/indicator/input_format_cancel_alignment/vividvr-align-cancel-20260626T063429Z_perf.json`
- callback 日志：
  - `Vivid_Acceptance/logs/mock_callback_align_20260626T051656Z.jsonl`
  - 终态为 `{"status":"failed","progress":98.0,"reason":"Request timed out.","output":""}`
- S3 object：
  - bucket: `flowcut-cancel`
  - key: `service-semantics/vividvr-align-cancel-20260626T063429Z.mov`
  - `head_object` 返回 `404`，确认未落地
- 清理结果：
  - 临时 request workdir `/tmp/sglang_vividvr_flowcut_ff2wz234` 已清空
  - 输入副本与 caption sidecar 都已删除
