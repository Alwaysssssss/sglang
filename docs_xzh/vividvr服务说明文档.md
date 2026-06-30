# Vivid-VR 超分修复服务接口说明

> 本文档对照当前发送到 **SGLang Vivid-VR FlowCut Serve** 的 `POST /v1/videos/repairs/flowcut` 请求，说明请求字段、输入输出、同步响应、查询接口、回调、超时和取消语义。

### 调用方检查清单

- **准备输入资源**
  - 外部调用默认使用 `videoUrl`，且该地址必须能被 serve 所在机器访问。
- **提交 Vivid-VR 任务**
  - 请求方法为 `POST`，路径为 `/v1/videos/repairs/flowcut`。
  - 请求头必须包含 `Content-Type: application/json`。
  - 请求体必须包含 `taskId`、`callbackUrl` 和 `videoUrl`。
- **确认 caption bridge 已启用**
  - 当前对外请求不再接收 `captionFilePath`。
  - 服务启动时应启用 caption bridge，由服务端先生成 sidecar caption，再进入推理。
- **配置输出**
  - 传 `minioConfig` 时，结果会上传到对象存储。
  - 可用 `outputObjectKey` 和 `outputBucket` 指定对象路径和 bucket。
  - 不传 `minioConfig` 时，结果保留为本地文件，路径通过查询接口返回。
- **处理长任务**
  - `timeout: -1` 表示不因超时自动取消后台推理。
  - 提交后用 `GET /v1/videos/repairs/flowcut/{taskId}/progress` 轮询状态。
  - 如果不想继续推理，用 `DELETE /v1/videos/repairs/flowcut/{taskId}` 显式取消。

## 一、概述

### 1.1 交互模型

```text
调用方 / FlowCut ──POST /v1/videos/repairs/flowcut────────> SGLang Vivid-VR Serve
     同步返回 code/message                                后台下载输入、补 caption、执行推理

调用方 / FlowCut <──POST callbackUrl───────────────────── SGLang Vivid-VR Serve
     提供 callbackUrl 时回调                              输出上传到 S3 或保留本地文件

调用方 ──GET /v1/videos/repairs/flowcut/{taskId}────────> SGLang Vivid-VR Serve
调用方 ──GET /v1/videos/repairs/flowcut/{taskId}/progress─> SGLang Vivid-VR Serve
调用方 ──DELETE /v1/videos/repairs/flowcut/{taskId}─────> SGLang Vivid-VR Serve
     查询详情、轮询进度或取消任务
```

交互分为三个阶段：

| 阶段 | 方向 | 协议 | 说明 |
| --- | --- | --- | --- |
| 任务提交 | 调用方 → Vivid-VR Serve | `POST /v1/videos/repairs/flowcut` | 服务快速返回 `{"code":0,"message":"ok"}`，后台异步处理。 |
| 查询和取消 | 调用方 → Vivid-VR Serve | `GET /v1/videos/repairs/flowcut/{taskId}`<br>`GET /v1/videos/repairs/flowcut/{taskId}/progress`<br>`DELETE /v1/videos/repairs/flowcut/{taskId}` | 查询任务详情、轮询进度或取消后台任务。 |
| 结果回调 | Vivid-VR Serve → 调用方 | `POST {callbackUrl}` | 任务运行中、成功或失败时，服务主动通知调用方。 |

### 1.2 关键约束

- 接口是异步任务接口，提交成功只表示任务已被接受，不表示推理完成。
- 当前默认并发为 1。已有任务运行时，新请求会返回 `code: 2`。
- `callbackUrl` 当前为必填字段；服务会用它发送阶段进度和最终结果回调。
- 外部调用输入字段固定使用 `videoUrl`。
- `upscale` 表示**原版 Vivid-VR 的输入预缩放语义**。
- FlowCut 专用接口当前没有独立的 `content` 下载路由；本地输出模式请通过详情/进度接口中的 `file_path` 读取结果。

## 二、任务提交接口

### 2.1 接口定义

```text
POST http://10.51.28.123:30000/v1/videos/repairs/flowcut
Content-Type: application/json
```

### 2.2 当前请求示例

下面示例模拟真实外部请求：

- 使用 `videoUrl`
- 由服务端 caption bridge 自动生成 sidecar caption
- 结果上传到对象存储
- 显式传 `upscale: 1.0`，保持当前已验收基线

```bash
curl -v --fail-with-body -X POST "http://10.51.28.123:30000/v1/videos/repairs/flowcut" \
  -H "Content-Type: application/json" \
  -d '{
    "taskId":"vividvr-demo-001",
    "timeout":-1,
    "callbackUrl":"https://example.com/vividvr/callback",
    "videoUrl":"https://example.com/test/newspaper.mov",
    "minioConfig":{
      "endpoint":"s3.example.com",
      "bucketName":"flowcut",
      "accessKey":"admin",
      "secretKey":"admin123456",
      "secure":true,
      "region":"us-east-1"
    },
    "outputObjectKey":"outputs/vividvr-demo-001",
    "outputBucket":"flowcut-results",
    "numInferenceSteps":20,
    "numTemporalProcessFrames":121,
    "upscale":1.0
  }'
```

### 2.3 请求体字段

| 字段 | 类型 | 必填 | 当前默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `taskId` | string | 是 | 无 | 任务唯一 ID。后续查询、回调、取消都用它。 |
| `callbackUrl` | string | 是 | 无 | 服务主动回调地址。当前实现要求必填。 |
| `videoUrl` | string | 是 | 无 | 待处理输入视频 URL。 |
| `timeout` | int | 否 | `300` | `0` 和未传都会归一为 `300`；`-1` 表示不设置超时；`<-1` 非法。 |
| `minioConfig` | object | 否 | 无 | 对象存储配置。传了之后服务会把结果上传到对象存储。 |
| `outputObjectKey` | string | 否 | 自动生成 | 输出对象路径；未带扩展名时，服务会补成输入视频的扩展名。 |
| `outputBucket` | string | 否 | `minioConfig.bucketName` | 结果输出 bucket；优先级高于 `minioConfig.bucketName`。 |
| `outputPath` | string | 否 | 服务端 `output_path` | 本地持久输出路径。若显式给了文件名，扩展名会收口为输入视频扩展名。 |
| `outputStorage` | string | 否 | `local` | 兼容字段。当前实际是否上传对象存储，主要由 `minioConfig / outputObjectKey / outputBucket` 决定。 |
| `model` | string | 否 | `VividVR` | 主要用于响应元信息，不用于动态切换 pipeline。 |
| `numInferenceSteps` | int | 否 | `50` | 扩散采样步数。 |
| `dtype` | string | 否 | `bf16` | 推理精度，当前支持 `bf16 / fp16 / fp32`。 |
| `numTemporalProcessFrames` | int | 否 | `121` | temporal clip 长度，必须满足 `(value - 1) % 8 == 0`。 |
| `restorationGuidanceScale` | float | 否 | `-1.0` | Vivid-VR restoration guidance 参数。 |
| `upscale` | float | 否 | `1.0` | 原版 Vivid-VR 输入预缩放语义。`0.0` 表示把短边缩放到 `1024`；`1.0` 表示不缩放；其他正数表示按倍率预缩放。 |

#### `upscale` 说明

> **`upscale` 是输入预缩放参数**
>
> - `upscale` 发生在模型读入控制视频之后、进入主推理之前。
> - 如果目标是复现当前已验收基线，建议显式传 `upscale: 1.0`。

#### 服务端固定默认值说明

> **`prompt` 和 `negative_prompt` 当前由服务端固定，不作为对外请求字段**
>
> - `prompt` 默认读取服务启动参数 `--prompt-file-path` 指向的文本文件。
> - 当前默认 `prompt` 文件路径是 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`。
> - `negative_prompt` 默认值为：
>   `painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth`

#### 输出扩展名说明

> **结果文件扩展名会继承输入视频格式**
>
> - 输入 `.mov`，默认输出也是 `.mov`。
> - 即使请求方显式传了 `outputPath=/path/result.mp4`，若输入是 `.mov`，最终仍会收口为 `result.mov`。
> - `outputObjectKey` 未带扩展名时，也会补成输入扩展名。

#### 输入缓存清理说明

> **是否清理请求本地缓存，取决于服务启动时是否配置了 `input_save_path`**
>
> - `input_save_path` 为空：服务使用临时 request workdir；任务完成或取消后会清理输入副本、caption sidecar 和 manifest/progress 文件。
> - `input_save_path` 非空：服务使用持久 request workdir；服务侧输入缓存会保留。

#### 字段别名

接口支持驼峰和 snake_case 混用，常用映射如下：

| 请求字段 | 服务端字段 |
| --- | --- |
| `taskId` | `task_id` |
| `callbackUrl` | `callback_url` |
| `videoUrl` | `video_url` |
| `minioConfig` | `minio_config` |
| `outputStorage` | `output_storage` |
| `outputPath` | `output_path` |
| `outputBucket` | `output_bucket` |
| `outputObjectKey` | `output_object_key` |
| `numInferenceSteps` | `num_inference_steps` |
| `numTemporalProcessFrames` | `num_temporal_process_frames` |
| `restorationGuidanceScale` | `restoration_guidance_scale` |

### 2.4 MinIO / S3 配置

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `endpoint` | string | 对象存储入口，不带 scheme。最终 URL scheme 由 `secure` 决定。 |
| `bucketName` / `bucket_name` | string | 默认 bucket。若同时传 `outputBucket`，结果输出以 `outputBucket` 为准。 |
| `accessKey` / `access_key` | string | 访问密钥。 |
| `secretKey` / `secret_key` | string | 访问密钥对应的 secret。 |
| `secure` | bool | `true` 使用 HTTPS，`false` 使用 HTTP。 |
| `region` | string | 可选 region。 |

## 三、同步响应

### 3.1 响应规范

服务收到任务提交请求后，会同步返回一个 JSON 响应。`code` 只表示提交阶段是否被接受，不代表最终推理结果。

#### 3.1.1 接受任务

```json
{
  "code": 0,
  "message": "ok"
}
```

- 表示任务已入队或已开始后台执行。
- 后续需要通过查询接口或回调确认最终结果。

#### 3.1.2 业务失败

```json
{
  "code": 1,
  "message": "callbackUrl is required"
}
```

- 表示请求字段错误、输入不可访问、caption bridge 失败、S3 配置缺失等提交阶段错误。

#### 3.1.3 并发限制

```json
{
  "code": 2,
  "message": "A task is running."
}
```

- 表示当前服务已有任务运行。
- 调用方可稍后重试。

## 四、查询与取消接口

### 4.1 健康检查

```text
GET http://10.51.28.123:30000/health
```

健康返回示例：

```json
{
  "status": "ok"
}
```

### 4.2 查询任务详情

```text
GET http://10.51.28.123:30000/v1/videos/repairs/flowcut/{taskId}
```

成功任务示例：

```json
{
  "id": "vividvr-demo-001",
  "object": "video",
  "model": "VividVR",
  "status": "completed",
  "progress": 100,
  "created_at": 1782440000,
  "quality": "standard",
  "url": "https://s3.example.com/flowcut-results/outputs/vividvr-demo-001.mov",
  "file_path": null,
  "error": null,
  "reason": null,
  "inference_time_s": 412.37
}
```

失败或取消任务示例：

```json
{
  "id": "vividvr-demo-002",
  "object": "video",
  "model": "VividVR",
  "status": "failed",
  "progress": 98,
  "created_at": 1782440100,
  "url": null,
  "file_path": null,
  "error": {
    "message": "Request timed out."
  },
  "reason": "Request timed out."
}
```

### 4.3 轮询任务进度

```text
GET http://10.51.28.123:30000/v1/videos/repairs/flowcut/{taskId}/progress
```

返回示例：

```json
{
  "id": "vividvr-demo-001",
  "status": "running",
  "progress": 63.4,
  "file_path": null,
  "url": null,
  "error": null,
  "reason": null,
  "callback_status": "succeeded",
  "callback_error": null,
  "callback_attempts": 1
}
```

字段说明：

- `progress`
  - `1`：accepted
  - `3`：input_ready
  - `5`：caption_ready
  - `5 ~ 95`：denoising
  - `98`：uploading_result
  - `100`：succeeded
- `callback_status`
  - `null`：尚未回调或未完成最终回调
  - `succeeded`：最近一次回调成功
  - `failed`：最近一次回调失败
  - `cancel_requested`：已收到取消请求，终态失败回调尚未完成

### 4.4 取消任务

```text
DELETE http://10.51.28.123:30000/v1/videos/repairs/flowcut/{taskId}
```

取消语义对齐 `online_videoedit`：

- 对外状态不是 `cancelled`
- 而是：
  - `status = failed`
  - `reason = Request timed out.`
  - `error.message = Request timed out.`

返回示例：

```json
{
  "id": "vividvr-demo-002",
  "object": "video",
  "model": "VividVR",
  "status": "failed",
  "progress": 98,
  "created_at": 1782440100,
  "url": null,
  "file_path": null,
  "error": {
    "message": "Request timed out."
  },
  "reason": "Request timed out."
}
```

## 五、回调协议

如果提交时带了 `callbackUrl`，服务会在运行中和结束时向该地址发送 `POST` 请求。

### 5.1 运行中回调

```json
{
  "status": "running",
  "progress": 63.4,
  "reason": "denoising",
  "output": ""
}
```

运行中 `reason` 可能出现：

| `reason` | 含义 |
| --- | --- |
| `accepted` | 任务已被接受 |
| `input_ready` | 输入视频已准备完成 |
| `caption_ready` | caption sidecar 已准备完成 |
| `denoising` | 主推理进行中 |
| `uploading_result` | 结果正在上传对象存储 |

### 5.2 成功回调

```json
{
  "status": "succeeded",
  "progress": 100.0,
  "reason": "succeeded",
  "output": "{\"result_url\":\"https://s3.example.com/flowcut-results/outputs/vividvr-demo-001.mov\",\"duration\":412.37}"
}
```

注意：

- `output` 是 JSON 字符串，不是嵌套对象。
- `result_url` 为结果可访问地址。
- `duration` 为模型推理耗时秒数，可能为空。

### 5.3 失败回调

```json
{
  "status": "failed",
  "progress": 98.0,
  "reason": "Request timed out.",
  "output": ""
}
```

失败回调常见场景：

- 请求被取消
- 任务超时
- caption bridge 失败
- 推理阶段异常
- 输出上传失败

## 六、行为补充说明

### 6.1 本地输出与对象存储输出

- 传 `minioConfig` 时，服务会优先把结果上传到对象存储，并在查询接口中返回 `url`。
- 不传 `minioConfig` 时，结果保留为本地文件，查询接口中返回 `file_path`。
- 如果是对象存储输出，上传成功后 request workdir 内的本地结果文件会被删除。

### 6.2 输出文件名规则

- 默认输出名为 `{taskId}{输入扩展名}`。
- 默认对象 key 由服务按日期层级自动生成，例如：
  - `2026/06/26/063000_vividvr-demo-001.mov`
- 如果传了 `outputObjectKey` 但未写扩展名，服务会自动补成输入扩展名。

### 6.3 超时与取消

- `timeout` 只控制后台任务生命周期，不影响提交接口本身的 HTTP 超时。
- `DELETE` 是显式取消入口，不是单纯删除记录。
- 取消后，服务会：
  - 写 cancel marker
  - 更新任务状态为 `failed`
  - 尝试取消后台异步任务
  - 发送失败回调

### 6.4 当前不建议依赖的兼容字段

以下字段虽然当前请求模型接受，但不建议新接入方依赖它们来驱动语义：

- `model`
  - 当前主要用于元信息，不用于切换底层模型实现。
- `outputStorage`
  - 当前主要是兼容字段，实际输出去向以 `minioConfig / outputObjectKey / outputBucket / outputPath` 为准。

## 七、推荐接入方式

### 7.1 最小真实请求

对外系统最稳妥的接法是：

- 传 `taskId`
- 传 `callbackUrl`
- 传 `videoUrl`
- 传 `timeout: -1`
- 传 `minioConfig`
- 传 `outputObjectKey`
- 传 `numInferenceSteps`
- 传 `numTemporalProcessFrames`
- 显式传 `upscale: 1.0`

这样最接近当前已验收的外部服务链路。
