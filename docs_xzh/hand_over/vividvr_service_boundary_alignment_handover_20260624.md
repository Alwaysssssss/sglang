# Vivid-VR 服务边界与 FlowCut 契约对齐交接

日期：`2026-06-24 UTC`

## Scope

本轮目标是把 Vivid-VR FlowCut 服务契约从共享 `video_api.py` 中拆出来，形成独立服务边界，同时保持外部入口不变：

- Vivid-VR 专用入口：`POST /v1/videos/repairs/flowcut`。
- 通用视频修复入口：`POST /v1/videos/repairs` 只保留 OpenAI 风格通用 video repair 语义。
- 当服务进程是 Vivid-VR pipeline 时，共享 `/v1/videos/repairs` 会直接拒绝，并提示使用 `/v1/videos/repairs/flowcut`。
- Vivid-VR FlowCut 成功 callback 的 `output` 只允许 `result_url` 和可选 `duration`，不再输出 `gen_video_url` 或 `file_path`。

## Changed Modules

- `vividvr_flowcut_protocol.py`：承载 `taskId`、`timeout`、`callbackUrl`、`minioConfig`、submit response 和 callback payload 的 Vivid 专属协议。
- `vividvr_flowcut_storage.py`：承载 request-scoped 输入 staging、输出路径、MinIO 上传和临时 workdir 清理。
- `vividvr_flowcut_progress.py`：承载 stage-based progress，避免用 elapsed time 伪造进度。
- `video_job_runner.py`：承载纯生成执行层，只调用底层 generation pipeline 并返回产物与指标。
- `vividvr_flowcut_api.py`：承载 Vivid 专用 router、校验、caption 准备、dispatch、timeout、callback 和上传流程。
- `video_repair_shared.py`：承载通用 repair 与 Vivid FlowCut 共用的队列、Vivid 参数映射、caption bridge 准备和 job metadata helper。
- `video_api.py`：移除 Vivid FlowCut 分支，通用 repair 路径只处理非 Vivid videoedit。
- `http_server.py`：显式注册 `vividvr_flowcut_api.router`。

## Contract

- 提交响应固定为 `{"code": 0|1|2, "message": str}`。
- `code=0` 表示接单成功并异步执行。
- `code=1` 表示业务错误，仍以 HTTP 200 JSON 返回给 FlowCut 客户端。
- `code=2` 表示队列已满，客户端只应重试提交，不应轮询未接单任务。
- `timeout=0` 会归一化为 `300` 秒。
- `timeout=-1` 表示不启用服务侧生成超时。
- `timeout<0` 除 `-1` 外由协议校验拒绝。
- 成功 callback 形态为 `{"status":"succeeded","progress":100,"reason":"","output":"{\"result_url\":\"...\",\"duration\":...}"}`。
- 失败 callback 不携带本地文件路径。

## Timeout Behavior

FlowCut 生成超时不会直接取消底层生成任务。服务会先把任务状态更新为 `failed` 并发送 failed callback，但会继续等待底层 generation task 结束后才释放单任务队列信号量。

这样处理的原因是 Vivid-VR 生成任务不能安全地被任意中断；如果超时后立即释放 semaphore，新的任务可能和仍在运行的旧任务并发抢占同一张卡。

## File Lifecycle

当前文件生命周期如下：

- 远端原视频对象：不删除。
- 远端结果视频对象：不删除。
- request workdir：接单前失败时会清理。
- 本地输入副本：保存在 request workdir 的 `inputs/` 下。
- 本地输出副本：如果启用 MinIO 且上传成功，会删除本地输出文件；如果没有启用 MinIO，则本地输出即 `result_url`，会保留。
- caption sidecar：如果请求显式传入 `caption_file_path`，不会删除用户提供的文件；如果通过 caption bridge 生成，当前位于输出目录下的 `caption_sidecars/`。

已知限制：

- 当结果以本地文件路径返回时，不能在任务终态直接删除整个 request workdir，否则会删除最终结果视频。因此“终态清理输入副本和 caption manifest，但保留本地结果”仍需要后续做更细粒度清理。
- 当前任务状态仍保存在进程内 `VIDEO_STORE`，服务重启后旧 `taskId` 查询仍可能返回 `404`。

## Verification

目标单测：

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

静态检查：

```bash
/home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_protocol.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_storage.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_progress.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_job_runner.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_repair_shared.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/vividvr_flowcut_api.py \
  python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py
git diff --check
```

本轮结果：

- 目标单测：`74 passed, 6 warnings in 11.09s`。
- `py_compile`：通过。
- `git diff --check`：通过。

单卡服务验收应只启动一个 Vivid-VR 服务进程，使用 `single_gpu_fa_compile` 口径，不启动双卡服务。推荐通过 acceptance 工具验证：

```bash
/home/zhiheng/sglang/.venv/bin/python \
  python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py \
  --base-url http://127.0.0.1:31190 \
  --task-id <task-id> \
  --callback-log Vivid_Acceptance/logs/<task-id>_callback.jsonl \
  --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
  --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
  --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 \
  --num-inference-steps 20 \
  --seed 42 \
  --num-temporal-process-frames 121 \
  --output-path Vivid_Acceptance/result_videos/service_boundary/<task-id>.mp4 \
  --perf-dump-path Vivid_Acceptance/indicator/<task-id>_perf.json
```

## Acceptance Notes

- 本轮验收优先使用显式 caption sidecar 文件，不启动 caption sidecar 服务。
- acceptance 工具现在会校验最终 callback `output` 必须包含 `result_url`，并拒绝 `gen_video_url` 和 `file_path`。
- 由于当前只有 1 张卡空闲，正式服务验收只应使用单卡服务，不运行双卡 `SP` 验收。
- 实际单卡服务验收使用 `CUDA_VISIBLE_DEVICES=1`、`--num-gpus 1`、`--attention-backend fa`、`--enable-torch-compile`，服务端口为 `127.0.0.1:31190`，未启动双卡服务或 caption sidecar。
- 实际验收 `task_id` 为 `vividvr-flowcut-boundary-single-20260624T101647Z`，submit 响应为 `{"code":0,"message":"ok"}`。
- callback 日志位于 `Vivid_Acceptance/logs/vividvr-flowcut-boundary-single-20260624T101647Z_callback.jsonl`，最终 payload 为 `status=succeeded`、`progress=100`，`output` 只包含 `result_url`。
- 结果视频已复制到标准验收目录 `Vivid_Acceptance/result_videos/service_boundary/vividvr-flowcut-boundary-single-20260624T101647Z.mp4`，`ffprobe` 显示 `960x720`、`130` 帧、`5.2s`。
- 性能指标位于 `Vivid_Acceptance/indicator/vividvr-flowcut-boundary-single-20260624T101647Z_perf.json`，`total_duration_ms=1027849.6591150761`。
- 共享路由检查：Vivid-VR 服务上的 `POST /v1/videos/repairs` 返回 HTTP `400`，detail 为 `Vivid-VR video repair must use /v1/videos/repairs/flowcut`。
- 验收结束后已停止 `vividvr_flowcut_single_service` tmux 服务，GPU 1 回到空闲状态。

注意：当前 FlowCut request-scoped storage 返回的本地 `result_url` 位于 `inputs/uploads/<task_id>/outputs/..._0.mp4`。本轮为了满足仓库验收目录约定，额外复制了一份到 `Vivid_Acceptance/result_videos/service_boundary/`；后续如需让请求内 `output_path` 直接决定最终本地目录，需要继续收口输出路径映射。
