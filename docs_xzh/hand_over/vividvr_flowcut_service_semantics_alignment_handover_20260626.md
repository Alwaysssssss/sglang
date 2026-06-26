# Vivid-VR FlowCut 服务语义对齐交接（2026-06-26）

## 本轮目标

本轮采用“保留 Vivid-VR 独立 `api/protocol/storage/progress`，只向 `share-tyx` 的 `video_edit` 对齐请求端可感知语义”的方案，不把 Vivid-VR 重新塞回共享 `video_api.py` 主线。

## 已完成的服务语义对齐

### 1. 请求契约与 failed submission

- `FlowCut` 请求在进入 `VividVRFlowCutRequest` 前，先做 alias 归一化：
  - `taskId -> task_id`
  - `callbackUrl -> callback_url`
  - `minioConfig.bucketName/accessKey/secretKey -> bucket_name/access_key/secret_key`
- 非法请求如果已经带 `taskId`，会在 `VIDEO_STORE` 中落 failed job，而不是只返回 HTTP 响应。
- `timeout` 规则收紧为：
  - `None / 0 -> 300`
  - `-1 -> 无限等待`
  - `< -1 -> 直接校验失败`

### 2. 对象存储契约

- `output_object_key` 与 `output_bucket` 已真正接入 Vivid-VR FlowCut 上传链路。
- 如果请求未显式传 `output_object_key`，服务端会生成默认 key：
  - `YYYY/MM/DD/HHMMSS_<request_id>.mp4`
- `upload_result()` 不再把 object key 写死为 `outputs/{request_id}.mp4`。

### 3. 生命周期与清理策略

- 保留 Vivid-VR 自己的 `request workdir` 结构：`inputs / outputs / manifests`。
- 当结果已经外化后，临时 request workdir 会在异步任务结束后清理：
  - 上传到 MinIO
  - 或者落到显式 `output_path`
- 仅本地临时输出、没有外化目标时，workdir 继续保留，避免返回给请求端的本地结果路径失效。

### 4. callback bookkeeping 与 timeout 语义

- FlowCut callback 现在会把结果记回 `VIDEO_STORE`：
  - `callback_status`
  - `callback_error`
  - `callback_attempts`
  - `callback_completed_at`
- timeout 失败语义与 `share-tyx` 对齐为固定文案：
  - `Request timed out.`
- timeout 时仍会等待后台 generation task 真正结束后再释放 semaphore，避免并发状态过早放开。

## 明确未做的事

- 没有把 Vivid-VR 改成复用 `video_edit` 的 `mask/reference image` 语义。
- 没有迁移到共享 `WanVideoEditSamplingParams`。
- 没有移除 Vivid-VR 自己的 `progress reporter / storage / callback payload`。
- 没有改动 caption bridge 的模型语义。

## 本轮单测覆盖

已新增或收紧的覆盖点包括：

- nested `minioConfig` alias 校验
- `timeout < -1` 拒绝
- invalid submission with `taskId` 落 failed job
- 默认 `output_object_key` 持久化
- 显式 `output_object_key / output_bucket` 上传
- 临时 workdir 清理/保留策略
- callback bookkeeping
- timeout 固定失败文案与 semaphore 释放顺序

## 后续建议

- 如果后续继续向 `share-tyx` 对齐，优先再看：
  - invalid submission 是否需要自动发失败 callback
  - callback retry 次数与错误细节是否需要更细粒度入库
  - 是否把“对象存储 key 生成规则”抽成更通用 helper
- 不建议把 Vivid-VR 的模型任务语义强行对齐到 `video_edit`。
