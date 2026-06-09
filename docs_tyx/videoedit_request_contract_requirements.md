# VideoEdit 请求协议改造需求文档

本文记录 FlowCut 对接 VideoEdit repair 接口的新增需求。目标是让调用方只提交业务必需信息，不再依赖服务端预先 `export` 环境变量或手动预处理参考帧。

## 1. 目标

- 支持请求中直接传入参考图。
- 如果请求提供参考图，服务端自动把参考图插入输入视频首帧。
- 如果请求提供参考图，服务端自动给 mask 视频首帧插入一张全 1 mask 图。
- 请求体携带任务、回调、超时、MinIO/S3、输入视频、输入 mask 等配置，消除对调用前 `export` 的依赖。
- 统一响应格式，调度层可按整数 `code` 判断任务是否提交成功、业务失败或并发受限。
- 默认参数面向生产调用收敛，调用方只需要传最基础参数即可。

## 2. 当前实现差距

当前 `/v1/videos/repairs` 使用 `VideoRepairRequest`，主要字段是 snake_case：

- `task_id`
- `video_input_path` / `video_url`
- `mask_input_path` / `mask_url`
- `callback_url`
- `output_storage`
- `output_path`
- `num_frames`
- `infer_len`
- `overlap`
- `drop_reference_frame`

当前不支持：

- `taskId`、`callbackUrl` 等 camelCase 字段。
- `timeout` 请求字段。
- `minioConfig` 请求字段。
- 单独的参考图输入字段。
- 请求级 MinIO/S3 endpoint、bucket、ak/sk、secure、region 配置。
- 固定 `{code:int, message:string}` 的提交响应格式；提交阶段失败时额外返回 `reason`。

当前云存储配置主要从服务端环境变量读取：

- `SGLANG_CLOUD_STORAGE_TYPE`
- `SGLANG_S3_BUCKET_NAME`
- `SGLANG_S3_ENDPOINT_URL`
- `SGLANG_S3_ACCESS_KEY_ID`
- `SGLANG_S3_SECRET_ACCESS_KEY`
- `SGLANG_S3_REGION_NAME`

本需求要求这些配置可以通过请求体传入。

## 3. 请求协议

### 3.1 最小请求体

调用方只需要提交任务 id、回调、MinIO 配置、输入视频、输入 mask、提示词，以及可选参考图。

```json
{
  "taskId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timeout": -1,
  "callbackUrl": "http://flowcut.example.com/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890/callback",
  "prompt": "remove the selected object and keep the background natural",
  "videoUrl": "http://minio.example.com:9000/flowcut/input/video.mp4",
  "maskUrl": "http://minio.example.com:9000/flowcut/input/mask.mp4",
  "referenceImageUrl": "http://minio.example.com:9000/flowcut/input/reference.png",
  "minioConfig": {
    "endpoint": "minio.example.com:9000",
    "bucket_name": "flowcut",
    "access_key": "admin",
    "secret_key": "******",
    "secure": false,
    "region": "us-east-1"
  }
}
```

`referenceImageUrl` 可选。没有参考图时，不插首帧参考图。

### 3.2 输入字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---:|---:|---:|---|
| `taskId` | string | 是 | 无 | 业务任务 id。服务端用作任务 id 和本地默认输出文件名基础。 |
| `timeout` | integer | 否 | `-1` | 本次任务允许的业务超时时间，单位秒；`-1` 表示不因任务超时取消。 |
| `callbackUrl` | string | 否 | `null` | 任务完成或失败后的回调地址；不传时服务端跳过回调。 |
| `prompt` | string | 是 | 无 | VideoEdit 提示词。 |
| `videoUrl` | string | 是 | 无 | 输入视频地址。 |
| `maskUrl` | string | 是 | 无 | 输入 mask 地址。 |
| `referenceImageUrl` | string | 否 | `null` | 参考图地址。存在时服务端插入为视频首帧。 |
| `minioConfig` | object | 是 | 无 | 本次请求使用的 MinIO/S3 配置。 |
| `outputObjectKey` | string | 否 | `YYYY/MM/DD/HHMMSS_{taskId}.mp4` | 输出对象 key。调用方传入时按传入值输出。 |

### 3.3 MinIO 配置字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---:|---:|---:|---|
| `endpoint` | string | 是 | 无 | MinIO/S3 endpoint，例如 `minio.example.com:9000`。 |
| `bucket_name` | string | 是 | 无 | bucket 名。 |
| `access_key` | string | 是 | 无 | access key。 |
| `secret_key` | string | 是 | 无 | secret key。 |
| `secure` | boolean | 否 | `false` | `true` 使用 HTTPS，`false` 使用 HTTP。 |
| `region` | string | 否 | `us-east-1` | S3 region。 |

## 4. 参考图首帧处理

### 4.1 有参考图

当请求包含 `referenceImageUrl` 时，服务端需要在进入 VideoEdit pipeline 前完成输入改写：

1. 下载或读取输入视频。
2. 下载或读取输入 mask 视频。
3. 下载或读取参考图。
4. 将参考图 resize 到输入视频帧尺寸。
5. 把参考图插入为视频第 0 帧。
6. 生成一张与视频尺寸一致的全 1 mask 图，插入为 mask 视频第 0 帧。
7. 后续原视频第 0 帧变为第 1 帧，原 mask 第 0 帧变为第 1 帧。
8. 设置或保持 `drop_reference_frame=true`，最终输出丢弃第 0 帧参考帧。

全 1 mask 图约定为单通道 `L` 图，像素值为 `255`。如果实现中内部 mask 语义存在反转，需要在 preprocess 层明确转换，不能让调用方感知内部细节。

### 4.2 无参考图

当请求不包含 `referenceImageUrl` 时：

- 不自动插入首帧。
- `drop_reference_frame` 的默认行为需要谨慎处理，避免误丢输入视频第 0 帧。
- 推荐默认 `drop_reference_frame=false`；只有服务端确实插入了参考图时，内部自动改为 `true`。

### 4.3 帧数语义

如果有参考图，`num_frames=-1` 表示使用全部业务视频帧，再额外插入 1 帧参考图。示例：

- 原视频 156 帧，原 mask 156 帧。
- 插入参考图后，模型实际输入 157 帧。
- 输出时丢弃参考帧，业务输出仍应为 156 帧。

## 5. 默认参数

调用方不传高级参数时，服务端使用以下默认值。

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `num_frames` | `-1` | 表示全帧。由服务端根据视频和 mask 实际帧数解析。 |
| `infer_len` | `81` | VideoEdit 当前窗口长度要求。 |
| `overlap` | `0` | 默认不做窗口重叠。 |
| `strength` | `1.0` | 默认完整编辑强度。 |
| `num_inference_steps` | `20` | 默认推理步数。 |
| `guidance_scale` | `5.0` | 默认 CFG。 |
| `seed` | `42` | 默认随机种子。 |
| `dtype` | `bf16` | 默认精度。 |
| `dynamic_cfg` | `true` | 默认开启动态 CFG。 |
| `dynamic_cfg_max_step` | `15` | 动态 CFG 最大步。 |
| `dynamic_cfg_min` | `1.0` | 动态 CFG 最小值。 |
| `bbox_padding` | `0` | 默认不额外扩展 bbox padding。 |
| `dilate_px` | `15` | 默认 mask 膨胀像素。 |
| `mask_scale` | `1.2` | 默认 mask 缩放比例。 |
| `feather_px` | `12` | 默认贴回羽化。 |
| `adain_boundary_dilate` | `15` | 默认边界处理膨胀值。 |
| `enable_paste_back` | `true` | 默认贴回原视频。 |
| `save_crop_only` | `false` | 默认不保存 crop sidecar。 |
| `drop_reference_frame` | 自动 | 有参考图时为 `true`，无参考图时为 `false`。 |
| `use_repaired_context` | `true` | 默认使用已修复上下文。 |
| `vary_seed_by_window` | `false` | 默认窗口之间不改变 seed。 |
| `enable_teacache` | `false` | 默认不开启 TeaCache。 |
| `outputObjectKey` | `YYYY/MM/DD/HHMMSS_{taskId}.mp4` | 默认输出对象 key，按年/月/日目录和时间命名。 |

如果调用方传入高级参数，服务端可覆盖默认值，但基础调用不需要填写。

## 6. 响应规范

提交接口需要统一返回 HTTP 200，并通过整数 `code` 表示业务结果。

### 6.1 接受任务成功

```http
HTTP/1.1 200 OK
```

```json
{
  "code": 0,
  "message": "ok"
}
```

含义：

- 服务端已接受任务。
- 任务进入队列或开始执行。
- `code` 必须是整数 `0`。

### 6.2 业务失败

```http
HTTP/1.1 200 OK
```

```json
{
  "code": 1,
  "message": "invalid mask image format",
  "reason": "invalid mask image format"
}
```

含义：

- 请求已到达服务端，但业务校验失败。
- 例如输入视频无法下载、mask 格式不合法、参考图格式不合法、MinIO 配置缺失等。
- `code` 必须是整数 `1`。
- `message` 和 `reason` 返回明确失败原因。

### 6.3 并发限制

```http
HTTP/1.1 200 OK
```

```json
{
  "code": 2,
  "message": "A task is running."
}
```

含义：

- 当前已有任务运行，服务端拒绝本次调度。
- 调度层收到此响应后，任务保持 pending 状态。
- 调度层约 5 秒后自动重试。
- `code` 必须是整数 `2`。

## 7. 回调规范

如果请求中提供了 `callbackUrl`，服务端会在任务运行中周期性上报进度，并在任务完成或失败后向 `callbackUrl` 发起最终 HTTP POST；未提供时不发送回调。

回调 payload 建议至少包含：

```json
{
  "status": "succeeded",
  "progress": 100,
  "reason": "",
  "output": "{\"result_url\":\"http://minio.example.com:9000/flowcut/a1b2c3d4-e5f6-7890-abcd-ef1234567890.mp4\",\"duration\":45}"
}
```

失败示例：

```json
{
  "status": "failed",
  "progress": 37,
  "reason": "invalid mask image format",
  "output": ""
}
```

进度上报示例：

```json
{
  "status": "running",
  "progress": 52,
  "reason": "",
  "output": ""
}
```

## 8. 实现注意事项

- 请求层需要支持 camelCase 字段，同时可以考虑短期兼容已有 snake_case 字段。
- `code` 必须返回 JSON number，不能返回字符串。
- 并发限制当前如果使用 HTTP 429，需要改为 HTTP 200 + `code=2`。
- 业务校验失败当前如果抛 HTTP 400，需要改为 HTTP 200 + `code=1`。
- MinIO/S3 客户端需要支持按请求创建或复用，不能只依赖进程级环境变量。
- 请求中携带 `secret_key`，日志必须脱敏，不能打印原文。
- 参考图插帧后，要同步修正视频帧数、mask 帧数、输出帧数和 metadata。
- 插入参考图时，输出文件不应包含参考图帧。
- 没有参考图时，不应因为默认 `drop_reference_frame=true` 丢掉原始第 0 帧。
- `timeout` 需要定义作用范围：任务总耗时超时、下载超时、推理超时、上传超时、回调超时可以分开实现；第一版至少要记录请求级 timeout，并用于任务总超时控制。默认 `-1` 表示不因任务超时取消。

## 9. 待确认项

- `videoUrl` 和 `maskUrl` 是否一定都是 MinIO/S3 HTTP 地址，还是也需要支持本地路径。
- `referenceImageUrl` 是否只支持图片 URL，还是也需要支持 base64 或本地路径。
- 全 1 mask 的业务语义是否固定为像素值 `255`，以及是否需要兼容内部 mask 反转逻辑。
- 输出对象 key 默认按 `YYYY/MM/DD/HHMMSS_{taskId}.mp4` 生成；调用方传 `outputObjectKey` 时使用调用方指定值。
- 是否需要在提交成功响应中返回 `taskId`。当前示例只要求 `code` 和 `message`。
