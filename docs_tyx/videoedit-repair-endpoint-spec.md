# FlowCut VideoEdit 修复服务接口说明

> 本文档对照当前发送到 **SGLang VideoEdit Serve** 的 `POST /v1/videos/repairs` 请求，说明请求字段、MinIO 输入输出、同步响应、进度查询、超时和回调行为。

### 调用方检查清单

- **准备输入资源**
  - 视频地址 `videoUrl` 可由服务端访问。
  - mask 地址 `maskUrl` 可由服务端访问。
  - 参考图地址 `referenceImageUrl` 可由服务端访问。
- **提交修复任务**
  - 请求方法为 `POST`，路径为 `/v1/videos/repairs`。
  - 请求头必须包含 `Content-Type: application/json`。
  - 请求体必须包含 `taskId`、`prompt`、`videoUrl`、`maskUrl`。
  - 当前任务需要参考图，因此携带 `referenceImageUrl`。
- **配置输出**
  - 携带 `minioConfig` 和 `outputObjectKey` 时，结果会上传到指定 bucket/object。
  - 当前输出对象为 `outputs/ubuntu-minio-videoedit-008.mp4`。
- **处理长任务**
  - `timeout: -1` 表示不因任务超时取消后台推理。
  - 提交后用 `GET /v1/videos/{taskId}/progress` 轮询状态。

## 一、概述

### 1.1 交互模型

```
调用方 / FlowCut ──POST /v1/videos/repairs──> SGLang VideoEdit Serve
     同步返回 code/message                    后台下载输入并执行修复

调用方 / FlowCut <──POST callbackUrl──────── SGLang VideoEdit Serve
     提供 callbackUrl 时回调                 输出上传到 MinIO 或保留本地文件

调用方 ──GET /v1/videos/{taskId}──────────> SGLang VideoEdit Serve
调用方 ──GET /v1/videos/{taskId}/progress──> SGLang VideoEdit Serve
调用方 ──GET /v1/videos/{taskId}/content───> SGLang VideoEdit Serve
调用方 ──DELETE /v1/videos/{taskId}────────> SGLang VideoEdit Serve
     查询详情、轮询进度、下载内容或删除任务记录
```

交互分为**三个阶段**：

| 阶段 | 方向 | 协议 | 说明 |
| --- | --- | --- | --- |
| 任务提交 | 调用方 → VideoEdit Serve | `POST /v1/videos/repairs` | 服务快速返回 `{"code":0,"message":"ok"}`，后台异步处理。 |
| 查询和管理 | 调用方 → VideoEdit Serve | `GET /v1/videos/{taskId}`<br>`GET /v1/videos/{taskId}/progress`<br>`GET /v1/videos/{taskId}/content`<br>`DELETE /v1/videos/{taskId}` | 查询任务详情、轮询进度、下载输出视频或删除任务记录。 |
| 结果回调 | VideoEdit Serve → 调用方 | `POST {callbackUrl}` | 提供 `callbackUrl` 时，任务完成或失败后服务主动通知调用方。 |

### 1.2 关键约束

- 接口是异步任务接口，提交成功只表示任务已被接受，不表示推理完成。
- 默认服务并发为 1。已有任务运行时，新请求会返回 `code: 2`。
- `callbackUrl` 为可选字段；未提供时服务不会发送回调，可通过进度接口查询任务状态。
- 输入 URL 必须能被 serve 所在机器访问，包含视频、mask 和参考图。
- 携带 `referenceImageUrl` 时，服务会把参考图作为参考首帧参与修复；未显式传 `drop_reference_frame` 时，默认会在输出中丢弃该参考首帧。

## 二、任务提交接口

### 2.1 接口定义

```
POST http://10.51.28.123:30000/v1/videos/repairs
Content-Type: application/json
```

### 2.2 当前请求示例

以下示例在原请求基础上加入了参考图 `referenceImageUrl`，并将 `timeout` 设置为 `-1`，表示后台任务不因超时被取消。

```
curl -v --fail-with-body -X POST "http://10.51.28.123:30000/v1/videos/repairs" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":"ubuntu-minio-videoedit-008",
    "timeout":-1,
    "callbackUrl":"https://example.com/videoedit/callback",
    "prompt":"A close-up of an orange flower with a yellow center, remaining in focus against a blurred green grass background throughout the video.",
    "videoUrl":"https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/video/15108907_3840_2160_50fps_short.mp4",
    "maskUrl":"https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/mask/mask.json",
    "referenceImageUrl":"https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/image/15108907_first_frame.png",
    "minioConfig":{
      "endpoint":"https://sites-hurricane-substantial-really.trycloudflare.com",
      "bucket_name":"flowcut",
      "access_key":"admin",
      "secret_key":"admin123456",
      "secure":true,
      "region":"us-east-1"
    },
    "outputObjectKey":"outputs/ubuntu-minio-videoedit-008.mp4",
    "num_frames":-1,
    "infer_len":81,
    "overlap":9,
    "bbox_expand_scale":0.3,
    "dilate_px":0,
    "mask_scale":1.0,
    "feather_px":0,
    "num_inference_steps":40,
    "guidance_scale":5.0,
    "seed":42,
    "dtype":"bf16",
    "enable_teacache":true,
    "teacache_thresh":0.3,
    "teacache_start_skipping":5,
    "teacache_end_skipping":1.0,
    "enable_paste_back":true
  }'
```

> **注意**
>
> 示例中的 `callbackUrl` 是占位地址。生产环境需要替换成调用方真实可访问的回调地址；也可以不传，此时任务完成后不会发送回调，但仍可通过进度接口查询任务状态。

### 2.3 请求体字段

| 字段 | 类型 | 必填 | 当前值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskId` | string | 是 | `ubuntu-minio-videoedit-008` | 任务唯一 ID。后续进度查询和回调中都会使用该 ID。 |
| `timeout` | int | 否 | `-1` | 后台任务超时秒数。`-1` 表示不设置任务超时；正数表示超过该秒数后标记为超时失败。 |
| `callbackUrl` | string | 否 | `https://example.com/videoedit/callback` | 任务完成或失败后，服务主动 POST 的回调地址；不传时跳过回调。 |
| `prompt` | string | 是 | 橙色花朵描述 | 视频修复提示词，用于约束生成内容和画面风格。 |
| `videoUrl` | string | 是 | `.../test/video/15108907_3840_2160_50fps_short.mp4` | 待修复输入视频地址。 |
| `maskUrl` | string | 是 | `.../test/mask/mask.json` | mask JSON 地址。服务端读取该 JSON 获取 mask 信息；内容需与输入视频帧范围匹配或可被预处理逻辑兼容。 |
| `referenceImageUrl` | string | 否 | `.../test/image/15108907_first_frame.png` | 参考图片地址。当前请求使用 MinIO 中的 `15108907_first_frame.png`。 |
| `minioConfig` | object | 是 | 见下表 | 用于下载输入资源和上传输出结果的存储配置。 |
| `outputObjectKey` | string | 否 | `outputs/ubuntu-minio-videoedit-008.mp4` | 输出视频在 bucket 中的对象路径。 |
| `num_frames` | int | 否 | `-1` | 处理帧数。`-1` 表示读取输入视频的全部可用帧，并按 mask 配置处理对应帧。 |
| `infer_len` | int | 否 | `81` | 单次窗口推理长度。 |
| `overlap` | int | 否 | `9（默认）` | 多窗口推理时相邻窗口重叠帧数。默认值为 `9`；`infer_len=81` 时窗口步长为 `72` 帧。增大重叠可增强窗口衔接，但会增加重复推理开销。 |
| `bbox_expand_scale` | float | 否 | `0.3（默认）` | 对所有 mask 合并得到的 bbox 进行比例外扩。`0.3` 表示左右各外扩原 bbox 宽度的 30%，上下各外扩原 bbox 高度的 30%；超出视频边界的部分会被裁剪。外扩可为修复区域提供更多上下文，但会增大裁剪区域和推理开销。 |
| `dilate_px` | int | 否 | `0（默认）` | 模型输入 mask 的像素膨胀量。`0` 表示不膨胀；正值使用大小为 `2 × dilate_px + 1` 的椭圆核向外扩张 mask，可覆盖目标边缘或减少残留，但过大会扩大实际生成区域。 |
| `mask_scale` | float | 否 | `1.0（默认）` | 在 `dilate_px` 膨胀之后，以 mask 前景中心为基准缩放 mask。`1.0` 表示不缩放，`>1.0` 扩大，`0~1.0` 收缩。该参数会影响模型编辑范围和最终回贴使用的 mask。 |
| `feather_px` | int | 否 | `0（默认）` | 生成区域回贴到原视频时的 mask 边缘羽化宽度。`0` 表示硬边界；正值通过高斯模糊形成渐变融合边缘。仅在 `enable_paste_back=true` 时影响最终合成，不改变模型输入 mask。 |
| `num_inference_steps` | int | 否 | `40` | 扩散采样步数。 |
| `guidance_scale` | float | 否 | `5.0` | 提示词引导强度。 |
| `seed` | int | 否 | `42` | 随机种子，用于结果复现。 |
| `dtype` | string | 否 | `bf16` | 推理精度。 |
| `enable_teacache` | bool | 否 | `true（默认）` | 是否启用 TeaCache。启用后会复用变化较小的扩散步骤计算结果，以降低推理耗时。 |
| `teacache_thresh` | float | 否 | `0.3（默认）` | TeaCache 累积相对 L1 距离阈值。阈值越大，通常缓存命中和加速机会越多，但画面质量变化风险也会增加；降低该值会更保守。 |
| `teacache_start_skipping` | int \| float | 否 | `5（默认）` | 允许 TeaCache 跳过计算的起始位置。整数表示先完整计算的步数；0 到 1 的浮点数表示总采样步数比例。默认先完整计算前 5 步。 |
| `teacache_end_skipping` | int \| float | 否 | `1.0（默认）` | 停止允许 TeaCache 跳过计算的位置。整数表示采样步索引，负整数表示从末尾倒数；0 到 1 的浮点数表示总采样步数比例。默认 `1.0` 表示允许区间延续到全部采样步骤的末端。 |
| `enable_paste_back` | bool | 否 | `true` | 启用修复区域回贴到原视频。 |

#### Mask 扩展与回贴说明

> **`dilate_px`、`mask_scale` 和 `feather_px` 的作用阶段不同**
>
> mask 预处理顺序为：先按阈值二值化，再应用 `dilate_px` 膨胀，最后应用 `mask_scale` 缩放。处理后的 mask 会用于模型编辑，也会在启用回贴时限定生成区域。`feather_px` 仅在最后回贴阶段平滑生成区域与原视频的边界。默认配置 `dilate_px=0`、`mask_scale=1.0`、`feather_px=0` 不额外扩大 mask，也不进行边缘羽化。参考帧的 mask 仍会被强制设为不可编辑。

#### BBox 外扩说明

> **`bbox_expand_scale` 控制模型可见的修复上下文**
>
> 服务先根据所有 mask 计算联合 bbox，再按该参数向四周扩展裁剪范围。默认值 `0.3` 在不触碰画面边界时，会使宽度和高度理论上分别扩大到原 bbox 的 `1.6` 倍。值过小可能缺少目标周边上下文；值过大会让模型处理更多无关区域，并增加显存和计算开销。设置为 `0` 可关闭这一步比例外扩，但小目标仍可能触发服务端的小区域自适应扩展。

#### TeaCache 说明

> **TeaCache 默认开启，用于减少重复的扩散计算**
>
> 默认配置为 `enable_teacache=true`、`teacache_thresh=0.3`、`teacache_start_skipping=5`、`teacache_end_skipping=1.0`。如更重视结果稳定性，可关闭 TeaCache，或适当降低 `teacache_thresh`；如更重视速度，可在验证画质后提高阈值。

#### 字段别名

接口支持当前请求中的驼峰字段名，并在服务端转换为内部 snake_case 字段。

| 请求字段 | 服务端字段 |
| --- | --- |
| `taskId` | `task_id` |
| `callbackUrl` | `callback_url` |
| `videoUrl` | `video_url` |
| `maskUrl` | `mask_url` |
| `referenceImageUrl` | `reference_image_url` |
| `bboxExpandScale` | `bbox_expand_scale` |
| `teacacheThresh` | `teacache_thresh` |
| `teacacheStartSkipping` | `teacache_start_skipping` |
| `teacacheEndSkipping` | `teacache_end_skipping` |
| `minioConfig` | `minio_config` |
| `outputObjectKey` | `output_object_key` |

### 2.4 MinIO 配置

| 字段 | 类型 | 当前值 | 说明 |
| --- | --- | --- | --- |
| `endpoint` | string | `https://sites-hurricane-substantial-really.trycloudflare.com` | MinIO/S3 访问入口。 |
| `bucket_name` | string | `flowcut` | 输入和输出对象所在 bucket。 |
| `access_key` | string | `admin` | 访问密钥。 |
| `secret_key` | string | `admin123456` | 访问密钥对应的 secret。 |
| `secure` | bool | `true` | 使用 HTTPS 访问。 |
| `region` | string | `us-east-1` | S3 region。 |

## 三、同步响应

### 3.1 响应规范

服务收到任务提交请求后，会同步返回一个 JSON 响应。`code` 表示提交阶段的处理结果，不代表最终视频修复结果。

#### 3.1.1 接受任务（成功）

**code: 0 — 成功接受**

```
// HTTP/1.1 200 OK
{
  "code": 0,
  "message": "ok"
}
```

- 表示任务已入队或已开始后台执行。
- 后续需要通过进度查询接口或回调确认最终结果。

#### 3.1.2 业务失败（永久失败）

**code: 1 — 业务失败**

```
// HTTP/1.1 200 OK
{
  "code": 1,
  "message": "videoUrl or video_input_path is required",
  "reason": "videoUrl or video_input_path is required"
}
```

- 表示请求参数错误、输入资源不可访问、MinIO 配置缺失等提交阶段错误。
- `message` 和 `reason` 会给出失败原因。

#### 3.1.3 并发限制（可重试拒绝）

**code: 2 — 并发限制**

```
// HTTP/1.1 200 OK
{
  "code": 2,
  "message": "A task is running."
}
```

- 表示当前 VideoEdit 服务已有任务运行，无法接受新任务。
- 调用方可以稍后重试。

#### 3.1.4 响应总览表

| 场景 | HTTP Status | code | 调用方行为 |
| --- | --- | --- | --- |
| 成功接受 | 200 | `0` | 开始轮询进度，等待回调或输出 URL。 |
| 提交阶段业务失败 | 200 | `1` | 读取 `message`，修正参数后重新提交。 |
| 服务忙 | 200 | `2` | 等待当前任务结束后重试。 |
| 非 2xx 或网络错误 | 其他 | - | 检查 serve 进程、端口、网络连通性。 |

## 四、查询、下载和删除接口

以下接口用于提交任务后的健康检查、任务详情查询、进度轮询、输出下载和任务记录删除。示例中的 `video_id` 使用当前请求的 `taskId`：`ubuntu-minio-videoedit-008`。

### 4.1 健康检查

```
GET http://10.51.28.123:30000/health
```

用于确认 VideoEdit Serve 进程和 HTTP 服务是否可访问。

```
{
  "status": "ok"
}
```

### 4.2 查询任务详情

```
GET http://10.51.28.123:30000/v1/videos/ubuntu-minio-videoedit-008
```

返回任务详情。该接口中的 `progress` 与 `/progress` 接口保持同步，都会读取当前窗口和采样 step 换算出的最新进度。

```
{
  "id": "ubuntu-minio-videoedit-008",
  "object": "video",
  "model": "videoedit",
  "status": "running",
  "progress": 52,
  "created_at": 1710000000,
  "url": null,
  "file_path": "/tmp/sglang_videoedit_output_xxx/ubuntu-minio-videoedit-008.mp4",
  "error": null,
  "reason": null
}
```

#### 主要字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 任务 ID，即提交时的 `taskId`。 |
| `status` | string | `queued`、`running`、`completed`、`failed` 或 `deleted`。 |
| `progress` | number | 当前百分比进度。运行中按窗口和采样 step 更新，完成后为 `100`。 |
| `url` | string/null | 上传到 MinIO/S3 后的输出视频地址。 |
| `file_path` | string/null | 服务端本地输出文件路径。上传到云存储并清理本地文件后可能为 `null`。 |
| `error` | object/null | 失败时包含错误信息。 |
| `reason` | string/null | 失败时包含可直接展示或写入任务表的失败原因。 |

### 4.3 查询进度

```
GET http://10.51.28.123:30000/v1/videos/ubuntu-minio-videoedit-008/progress
```

返回轻量级进度信息，适合调用方定时轮询。

```
{
  "id": "ubuntu-minio-videoedit-008",
  "status": "running",
  "progress": 52,
  "file_path": "/tmp/sglang_videoedit_output_vgfu4tlr/ubuntu-minio-videoedit-008.mp4",
  "url": null,
  "error": null,
  "reason": null,
  "callback_status": null,
  "callback_error": null,
  "callback_attempts": null
}
```

#### 状态字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `running` 表示处理中，`completed` 表示完成，`failed` 表示失败。 |
| `progress` | number | 当前百分比进度。VideoEdit 修复任务会按窗口和采样 step 换算当前进度。 |
| `file_path` | string | 服务端本地输出路径。 |
| `url` | string/null | 上传到 MinIO/S3 后的访问地址。运行中通常为 `null`。 |
| `error` | object/null | 失败时包含错误信息。 |
| `reason` | string/null | 失败时包含可直接展示或写入任务表的失败原因。 |
| `callback_status` | string/null | 回调发送状态。 |
| `callback_error` | string/null | 回调失败原因。 |
| `callback_attempts` | number/null | 回调尝试次数。 |

### 4.4 下载视频内容

```
GET http://10.51.28.123:30000/v1/videos/ubuntu-minio-videoedit-008/content
```

当任务完成且输出仍保留在服务端本地文件系统时，可以用该接口下载 MP4 文件。

```
curl -v --fail-with-body "http://10.51.28.123:30000/v1/videos/ubuntu-minio-videoedit-008/content" -o ubuntu-minio-videoedit-008.mp4
```

| 场景 | 响应 | 说明 |
| --- | --- | --- |
| 本地文件存在 | `200 video/mp4` | 直接返回视频文件。 |
| 任务仍在运行或文件不存在 | `404` | 返回 `Generation is still in-progress`。 |
| 输出已上传到云存储 | `400` | 返回提示信息，调用方应使用 `url` 字段中的云端地址下载。 |
| 任务不存在 | `404` | 返回 `Video not found`。 |

### 4.5 删除任务记录

```
DELETE http://10.51.28.123:30000/v1/videos/ubuntu-minio-videoedit-008
```

从服务端内存任务表中删除该任务记录，并返回一个 `status` 为 `deleted` 的任务响应。

```
{
  "id": "ubuntu-minio-videoedit-008",
  "object": "video",
  "model": "videoedit",
  "status": "deleted",
  "progress": 100
}
```

> **注意**
>
> 删除接口用于删除任务记录，不建议在任务运行中调用。该操作不等价于取消 GPU 推理，也不保证删除已经上传到 MinIO/S3 的输出对象。

### 4.6 输出位置

当前请求指定：

```
{
  "bucket_name": "flowcut",
  "outputObjectKey": "outputs/ubuntu-minio-videoedit-008.mp4"
}
```

因此最终输出对象路径为：

```
flowcut/outputs/ubuntu-minio-videoedit-008.mp4
```

如果上传成功，进度接口的 `url` 字段和回调体中的 `outputUrl` 应指向该输出视频。

> **本地内容下载**
>
> 如果任务没有上传到云存储，且 `file_path` 存在，可以通过 `GET /v1/videos/{taskId}/content` 下载本地文件。已上传到云存储时，应使用 `url` 返回的云端地址。

## 五、超时和回调

### 5.1 timeout 语义

| timeout 值 | 含义 | 后台任务行为 |
| --- | --- | --- |
| `-1` | 不设置任务超时 | 一直等待后台推理完成，不因超时取消。 |
| `> 0` | 超时秒数 | 超过指定秒数后，任务会标记为失败，错误信息为 `task timeout`。 |
| `0` | 非法值 | 提交阶段返回 `code: 1`。 |
| `< -1` | 非法值 | 提交阶段返回 `code: 1`。 |

> **说明**
>
> `timeout` 控制后台 VideoEdit 任务是否被服务端取消，不是 curl 命令本身的 HTTP 等待时间。任务提交接口仍会尽快同步返回。

### 5.2 回调请求

请求中提供 `callbackUrl` 时，服务会在任务运行中周期性上报进度，并在任务完成或失败后向该地址发送最终 POST 请求；未提供时不发送回调。

```
POST {callbackUrl}
Content-Type: application/json
```

**成功回调示例**

```
{
  "status": "succeeded",
  "progress": 100,
  "reason": "",
  "output": "{\"gen_video_url\":\"2026/06/09/060635_ubuntu-minio-videoedit-008.mp4\",\"duration\":45}"
}
```

**失败回调示例**

```
{
  "status": "failed",
  "progress": 37,
  "reason": "task timeout",
  "output": ""
}
```

**运行中进度回调示例**

```
{
  "status": "running",
  "progress": 52,
  "reason": "",
  "output": ""
}
```

## 六、自检

### 6.1 自检清单

| # | 检查项 | 说明 |
| --- | --- | --- |
| 1 | `POST /v1/videos/repairs` 可达 | 确认 `10.51.28.123:30000` 端口能访问。 |
| 2 | 输入视频 URL 可访问 | 服务端能下载 `videoUrl`。 |
| 3 | mask URL 可访问 | 服务端能下载 `maskUrl`。 |
| 4 | 参考图 URL 可访问 | 服务端能下载 `referenceImageUrl`。 |
| 5 | MinIO 凭据正确 | 可读取 bucket `flowcut` 并上传到 `outputs/`。 |
| 6 | `taskId` 唯一 | 重复 ID 会覆盖或混淆任务状态。 |
| 7 | `timeout` 符合预期 | 长任务建议使用 `-1`，避免后台推理被超时取消。 |
| 8 | 进度接口可查 | 提交后执行 `GET /v1/videos/{taskId}/progress`。 |
| 9 | 回调地址真实可达 | 生产环境不要使用 `https://example.com/...` 占位地址。 |
| 10 | 输出对象路径正确 | 完成后检查 `flowcut/outputs/ubuntu-minio-videoedit-008.mp4`。 |

### 6.2 对象路径对照

| 用途 | MinIO 对象路径 | HTTP URL |
| --- | --- | --- |
| 输入视频 | `flowcut/test/video/15108907_3840_2160_50fps_short.mp4` | `https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/video/15108907_3840_2160_50fps_short.mp4` |
| 输入 mask | `flowcut/test/mask/mask.json` | `https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/mask/mask.json` |
| 参考图 | `flowcut/test/image/15108907_first_frame.png` | `https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/test/image/15108907_first_frame.png` |
| 输出视频 | `flowcut/outputs/ubuntu-minio-videoedit-008.mp4` | `https://sites-hurricane-substantial-really.trycloudflare.com/flowcut/outputs/ubuntu-minio-videoedit-008.mp4` |
