# FlowCut VideoEdit 修复服务接口说明

> 本文档对照当前发送到 **SGLang VideoEdit Serve / 双后端网关** 的 `POST /v1/videos/repairs` 请求，说明每个请求字段的语义与默认值、MinIO 输入输出、同步响应、进度查询、超时和回调行为。

### 调用方检查清单

- **准备输入资源**

    - 视频地址 `videoUrl` 可由服务端访问。
    - mask 地址 `maskUrl` 可由服务端访问（支持视频、图片、`.npy/.npz`、COCO RLE `.json`）。
    - 参考图地址 `referenceImageUrl` 可由服务端访问；**当前服务强制要求参考图**。

- **提交修复任务**

    - 请求方法为 `POST`，路径为 `/v1/videos/repairs`。
    - 请求头必须包含 `Content-Type: application/json`。
    - 请求体必须包含 `taskId`、`prompt`、`videoUrl`、`maskUrl`、`referenceImageUrl`。
    - 字段名同时支持 `snake_case` 与 `camelCase`，二者等价。

- **配置输出**

    - 携带 `minioConfig` 时：输入按对象键下载，产物上传到该 bucket（本地文件上传后被清理）。
    - 不携带 `minioConfig` 且 `output_storage=local` 时：产物保留在服务端，用 `/content` 接口或 `file_path` 取回。

- **处理长任务**

    - `timeout: -1` 表示不因任务超时取消后台推理；`timeout: 6000` 表示 6000 秒（100 分钟）后取消。
    - 提交后用 `GET /v1/videos/{taskId}/progress` 轮询状态。
    - 任务运行中可用 `DELETE /v1/videos/{taskId}` 取消。

## 一、概述

### 1.1 交互模型

```
调用方 / FlowCut ──POST /v1/videos/repairs──> SGLang VideoEdit Serve
     同步返回 code/message                    后台下载输入并执行修复

调用方 / FlowCut <──POST callbackUrl──────── SGLang VideoEdit Serve
     提供 callbackUrl 时回调                 输出上传到 MinIO/COS 或保留本地文件

调用方 ──GET /v1/videos/{taskId}──────────> SGLang VideoEdit Serve
调用方 ──GET /v1/videos/{taskId}/progress──> SGLang VideoEdit Serve
调用方 ──GET /v1/videos/{taskId}/content───> SGLang VideoEdit Serve
调用方 ──DELETE /v1/videos/{taskId}────────> SGLang VideoEdit Serve
     查询详情、轮询进度、下载内容或取消任务
```

交互分为**三个阶段**：

| 阶段 | 方向 | 协议 | 说明 |
| --- | --- | --- | --- |
| 任务提交 | 调用方 → VideoEdit Serve | `POST /v1/videos/repairs` | 服务快速返回，后台异步处理。 |
| 查询和管理 | 调用方 → VideoEdit Serve | `GET /v1/videos/{taskId}`<br>`GET /v1/videos/{taskId}/progress`<br>`GET /v1/videos/{taskId}/content`<br>`DELETE /v1/videos/{taskId}` | 查询任务详情、轮询进度、下载输出视频或取消任务。 |
| 结果回调 | VideoEdit Serve → 调用方 | `POST {callbackUrl}` | 提供 `callbackUrl` 时，运行中周期上报进度，结束（完成/失败/超时）后上报最终结果。 |

### 1.2 关键约束

- 接口是异步任务接口，提交成功只表示任务已被接受，不表示推理完成。
- 同一时刻只允许 **1 个**任务（`VIDEOEDIT_QUEUE_CAPACITY=1`）。**当前部署不排队**：服务空闲则立即执行，否则返回 `code: 2`，调用方需稍后重试。
- 字段名兼容 `snake_case` 与 `camelCase`（如 `task_id` / `taskId`、`minio_config` / `minioConfig`）；同一字段两种写法同时出现时，**`camelCase` 覆盖 `snake_case`**，建议只写一种。
- `video_url`、`mask_url`、`reference_image_url` 与对应的 `*_input_path` 同时存在时，**以 URL 为准**（本地路径会被覆盖）。
- `video_url` 可以是 COS/MinIO 对象键（配合 `minio_config` 下载）、`s3://bucket/key`、与 `endpoint` 同域的 http(s) 链接，或服务端本地路径。
- 参考图作为**参考帧**参与生成：其内容被当作第 `ref_frame_idx` 帧的编辑结果，作为首个窗口的条件前缀帧，本身**不出现在输出帧序列中**（`drop_reference_frame` 已从请求中移除，服务内固定为 `False`）。
- `mask` 帧数必须与视频帧数一致，否则任务失败并返回 `VideoEdit video/mask length mismatch`。

## 二、任务提交接口

### 2.1 接口定义

```
POST http://10.51.28.123:30000/v1/videos/repairs
Content-Type: application/json
```

### 2.2 当前请求示例

以下为当前实际发送的请求（含逐字段注释）。**`#` 为文档注释，实际请求需去掉注释**。

```json
{
  "task_id": "objedit-teacache-off-09e3c2e9d0",   # 必填。原理：服务以它为键保存任务状态、生成取消标记文件、拼默认输出文件名；重复提交同 ID 会被网关拒（409）。
  "timeout": 6000,                                # 默认 -1（不超时）；可选 -1 或正整数秒；0 与 < -1 非法。原理：从拿到并发槽开始计时，覆盖下载+推理+保存+上传，到点后在"每步去噪/每窗口/保存前"的检查点抛错终止，结束原因 Request timed out.。
  "prompt": "鸟扇动翅膀",                          # 必填。原理：文本条件，每个窗口都会重新编码一次，描述"要生成什么动作/效果"而非"要保留什么"。
  "negative_prompt": null,                        # 默认 null → 服务内置默认负面词（中文画质类词表，启动时已预编码缓存）。原理：仅当 guidance_scale > 1（启用 CFG）时才真正参与编码。
  "model": "videoedit-normal",                    # 默认 "videoedit"。可选 videoedit-normal / normal / videoedit / ""（→ normal 后端，40 步完整采样）或 videoedit-dmd / dmd（→ DMD 蒸馏后端，服务强制 4 步、CFG=1.0、关 dynamic_cfg、负向置空）。原理：网关按该字段选后端并串行化，任务响应里的 variant 会回显实际路由。
  "video_input_path": null,                       # 默认 null。本地联调用的服务端视频路径；与 video_url 同时给出时被 URL 覆盖。
  "mask_input_path": null,                        # 默认 null。本地联调用的服务端 mask 路径；与 mask_url 同时给出时被 URL 覆盖。
  "video_url": "flowcut/test/2026/07/16/3aba7476ee0e275c85a32627bb7b1a6dd18a4ed8.mp4",  # 原理：先按 minio_config 的 endpoint+bucket 直读对象键（省公网流量），其次认 s3://bucket/key、同域 http(s) 链接，再退化到普通 HTTP 下载或本地路径。
  "mask_url": "flowcut/test/2026/07/17/feba817b185f2d7fad614a10c908feadb6f69b11.json",   # 可选格式：视频 / 图片 / .npy / .npz / COCO RLE .json（示例为 .json）。原理：逐帧 mask 取并集后二值化，再按 dilate_px、mask_scale 预处理；帧数必须与视频一致，否则报 VideoEdit video/mask length mismatch。
  "reference_image_path": null,                   # 默认 null。本地联调用的服务端参考图路径；与 reference_image_url 同时给出时被 URL 覆盖。
  "reference_image_url": "flowcut/test/2026/07/16/6d083c53f432e9d156660741dd0f3b8660b26a15.png",  # 必填。原理：作为第 ref_frame_idx 帧"编辑后应有的样子"（条件前缀帧）注入，用于锁定主体外观；它本身不进入输出帧序列。
  "callback_url": null,                           # 默认 null（不回调）。原理：任务进入运行后每 5 秒上报一次 running（仅进度变化时发），结束/失败/超时发终态；单次超时 10s，失败重试 3 次（间隔 1/2/4s），结果记录在 callback_status/callback_error/callback_attempts。
  "minio_config": {                               # 默认 null。可选子字段见下。原理：提供后输入按对象键直读，且"提供即触发上传"——即使 output_storage 仍为 local，产物也会传到这里并清理本地文件（file_path 变 null，url 给可访问地址）。
    "endpoint": "cos.ap-beijing.myqcloud.com",    # 必填。原理：可省略 scheme，由 secure 决定 http/https；同域链接会被识别为对象引用而非公网下载。
    "bucket_name": "vrs-mms-1258229344",          # 必填。原理：既作输入相对键的默认 bucket，也作输出的默认 bucket。
    "access_key": "AKID****************",         # 必填。访问密钥（本文档已脱敏，实际请求填真实值）。
    "secret_key": "****************",             # 必填。私钥（本文档已脱敏，实际请求填真实值）。
    "secure": true,                               # 默认 false。可选 true/false；true → https。
    "region": "ap-beijing",                       # 默认 us-east-1。
    "provider": "cos",                            # 默认 null。可选 cos 或其他；仅 "cos"（或 endpoint 为 *.myqcloud.com）会启用 prefix 语义。
    "prefix": "/flowcut"                          # 默认 null（不加前缀）。仅 COS 生效，作用在输出对象键上且幂等（键已带前缀则不重复添加）。
  },
  "output_storage": "local",                      # 默认 "local"。可选 local / s3。原理：s3 或显式给了 output_object_key 时，必须能拿到 minio_config（或服务端已开启云存储），否则提交阶段直接失败。
  "output_path": null,                            # 默认 null → 服务端配置的输出目录，仍为空则用临时目录（产物随任务清理）。原理：带视频后缀（.mp4/.mov/…）时视为完整文件名（后缀会替换成源视频后缀），否则视为目录。
  "output_bucket": null,                          # 默认 null → minio_config.bucket_name。
  "output_object_key": null,                      # 默认 null → "{YYYY/MM/DD}/{HHMMSS}_{task_id}{ext}"（本地时间）；COS 下再补 prefix。原理：给了就原样使用（自动去前导 /、去掉 .. 并替换后缀）。
  "num_frames": -1,                               # 默认 -1（= 与输入视频同长）。可选 -1 或正整数；0、< -1 非法。原理：正数时取 min(值, 视频帧数) 静默截断；输出帧数恒等于该值；mask 帧数必须与视频一致。
  "ref_frame_idx": 0,                             # 默认 0，可选 [0, num_frames)。原理：以该帧为起点、向帧数更多的一侧走 long pass，另一侧走反向 short pass；两段的首帧都由条件帧锚定，避免长视频单向漂移。
  "bridge_overlap": 5,                            # 默认 5。可选 1/5/9/…（满足 (b-1)%4==0）。原理：short pass 的条件前缀长度——取 long pass 首窗口已生成的前若干帧反向喂入，保证两侧接缝一致；实际值会按可用帧数裁剪并向下取整到 4k+1，ref_frame_idx=0 时该参数不起作用。
  "infer_len": 49,                                # 跟算法强相关，目前只能是 49。
  "overlap": 5,                                   # 跟算法强相关，目前只能是 5。
  "num_inference_steps": 40,                      # 默认 40。原理：去噪步数，每个窗口都跑满整条 schedule（不去噪整段视频，而是逐窗口各自采样）。
  "guidance_scale": 5.0,                          # 默认 5.0。原理：CFG 强度上限 max_cfg，靠"有条件 - 无条件"外推增强提示词跟随；<= 1.0 时直接关闭 CFG（只跑条件分支，更快、更贴原图但更不听提示词）。
  "seed": 42,                                     # 默认 42。原理：所有窗口共用同一 seed（当前版本不支持逐窗口变化），因此同输入+同参数可复现。
  "dtype": "bf16",                                # 默认 "bf16"。可选 bf16 / fp16 / fp32。原理：控制视频与条件张量的精度；模型权重精度由服务启动参数决定，不受此字段影响。
  "dynamic_cfg": true,                            # 默认 true。原理：扩散前期需要强引导（结构未定），后期需要保真（引导会过饱和），因此 CFG 随步数衰减。
  "dynamic_cfg_max_step": 15,                     # 默认 15。原理：衰减步数——第 i 步 CFG = min + (max-min) / pow(max-min, 1/max_step) ** i，即 15 步内从 guidance_scale 几何衰减到 dynamic_cfg_min。
  "dynamic_cfg_min": 1.0,                         # 默认 1.0。原理：衰减下限；=1.0 时第 15 步后 CFG 关闭，无条件前向被跳过（约省一半算力）。注意：与 guidance_scale 相等时公式除零，请避免。
  "bbox_padding": 0,                              # 默认 0（源视频像素，每边）。原理：在 mask 紧致框并集上再各边外扩固定像素，给模型更多原图上下文、并保证羽化有边界余量（需 >= feather_px+2）；只扩大裁剪范围，不扩大融合区；过大越界会直接失败。
  "bbox_expand_scale": 0.3,                       # 默认 0.3。原理：按紧致框自身宽高的比例每边外扩，内部倍率 2s+1（0.3 → 1.6×）；外扩后目标仍过小（面积比 < 0.2 且短边 < 480px）时自动把短边抬到 480。
  "dilate_px": 8,                                 # 默认 8（源视频像素）。原理：对二值 mask 做形态学膨胀（椭圆核 (2r+1)²），同时扩大"模型条件 mask"和"贴回融合区"（两者始终一致），用于覆盖目标位移与边缘未标注区域。
  "mask_scale": 1.0,                              # 默认 1.0。原理：膨胀后 mask 以自身质心为原点逐帧缩放（最近邻），1.0 为不变；用于整体放大/缩小编辑范围（注意运动目标每帧质心不同，会逐帧重定位）。
  "feather_px": 8,                                # 默认 8（裁剪图像素）。原理：贴回 alpha 用高斯模糊软化（核 2f+1、sigma=f/2），让生成区与原图平滑过渡；越大越能掩盖窗口接缝与边界错位，但会覆盖更多原图内容。
  "adain_boundary_dilate": 0,                     # 默认 0（裁剪像素）。原理：取 mask 外一圈原图作 AdaIN 颜色参考，把生成像素的通道均值/方差对齐过去，缓解生成区色偏；0 = 关闭（最容易出现色差的开关）。
  "crop_edge_feather": 0,                         # 默认 0（像素）。原理：裁剪框四条边做线性淡出（1→0），让生成内容在矩形边界前衰减，避免贴回后露出矩形硬边；0 = 硬贴回。
  "chunk_bbox_mode": "tight",                     # 默认 "tight"。可选 tight / fixed_size / global。原理：tight = 每个 chunk 用自身 mask 并集紧致框（16 对齐、不缩放，分辨率最高）；fixed_size = 各 chunk 中心不同但尺寸统一（所有窗口同分辨率，利于缓存/稳定）；global = 全片同一框（等同旧的非流式几何）。
  "stabilize_mask_union": false,                  # 默认 false。原理：窗口内所有帧取时间并集 mask，刚性目标形状固定、只保留位移，抑制逐帧抖动；与 stabilize_mask_shape 互斥（同时开会提交失败）。
  "stabilize_mask_shape": false,                  # 默认 false。原理：质心轨迹零相位滑动平均 + 窗口内平移对齐后投票重建轮廓，再按平滑质心摆放——保留真实形状（优于并集的矩形块）同时去抖；与 stabilize_mask_union 互斥。
  "stabilize_smooth_window": 5,                   # 默认 5。可选 >= 1（帧）。原理：参与投票/平滑的时间窗口，实际帧数 = 2*(w//2)+1；两个稳定开关都关闭时该参数被忽略。
  "preserve_audio": true,                         # 默认 true。原理：合成完成后直接把源视频的音轨复用过来（不重编码音频、不裁短），关闭则输出无声视频。
  "enable_paste_back": true,                      # 默认 true。原理：true = 逐窗口把裁剪区修复结果贴回原帧（输出与输入同尺寸）；false = 不做合成，直接输出拼接后的裁剪画布（更快，用于调试裁剪范围）。
  "save_crop_only": false,                        # 默认 false。原理：true 时额外落一个 "<output>_crop_only.<ext>" 裁剪版 sidecar 便于排查裁剪框是否合适，主输出不受影响。
  "use_clip": true,                               # 默认 true。原理：用 CLIP 视觉编码器编码窗口的条件帧（参考图/桥接帧/重叠帧），把图像 embedding 注入 DiT，锁定主体外观；关闭可省一次 ViT 前向，但一致性下降。
  "clip_preprocess": "diffuser",                  # 默认 "diffuser"。可选 diffuser / diffsynth。原理：diffuser = 用 checkpoint 自带的 HF image_processor（resize/crop/normalize）；diffsynth = 手工路径（/255 → CHW → 映射 [-1,1] → bicubic 到 224）。只影响 CLIP 的输入分布，须与训练时一致。
  "decode_mode": "stream",                        # 默认 "stream"。可选 stream / eager。原理：stream = 只解码当前窗口需要的帧区间（内存由窗口大小决定，长视频首选）；eager = 先把整段视频与 mask 读进内存（实现简单，长视频易 OOM）。
  "enable_teacache": false,                       # 请求模型默认 true（示例为做画质对比显式关闭）。原理：相邻去噪步的残差高度相似时复用缓存、跳过 DiT 前向；加速明显但误差会逐帧累积，长片尾部易糊化。
  "teacache_thresh": 0.3,                         # 默认 0.3。原理：累积相对 L1 误差阈值——低于阈值就跳过该步前向，越大越激进（更快、更易失真）。
  "teacache_start_skipping": 5,                   # 默认 5。可选整数（绝对步数）/ 浮点（占总步数比例）/ 负数（从尾部倒数）。原理：前 N 步必须真算，避免早期误差被后续放大；开启 CFG 时计数翻倍（按 DiT 调用次数算）。
  "teacache_end_skipping": 1.0,                   # 默认 1.0。原理：尾部强制真算的边界，1.0 表示不做尾部强制区（即最后一步也允许走缓存）；同样按 CFG 翻倍计数。
  # 常态关闭的，以下参数不用配置，走默认值即可
  "enable_frame_interpolation": false,            # 默认 false。原理：用 RIFE 在相邻帧之间插帧，提升流畅度/输出帧率；开启后帧数成倍增加，耗时与文件体积同步上涨。
  "frame_interpolation_exp": 1,                   # 默认 1。可选 1（输出帧率 ×2）/ 2（×4）。原理：插帧倍数指数。
  "frame_interpolation_scale": 1.0,               # 默认 1.0。原理：RIFE 推理分辨率尺度；高分辨率（如 1080p 以上）可降到 0.5 省显存，代价是光流精度下降。
  "frame_interpolation_model_path": null,         # 默认 null → elfgum/RIFE-4.22.lite（首次使用时自动下载）。
  "enable_upscaling": false,                      # 默认 false。原理：用 Real-ESRGAN 对输出超分；显存与耗时开销大，通常只在交付确需更高分辨率时开启。
  "upscaling_model_path": null,                   # 默认 null → ai-forever/Real-ESRGAN（RealESRGAN_x4.pth）。
  "upscaling_scale": 4,                           # 默认 4。原理：网络固定 x4，scale != 4 时在 x4 输出上再做 bicubic 缩放（不是换网络）。
  "output_quality": "default",                    # 默认 "default"。可选 default / maximum / high / medium / low → 50/100/90/55/35。注意：当前流式修复路径不读取该字段，输出沿用源视频的编码器与码率。
  "output_compression": null,                     # 默认 null（用画质档位的默认值）。同样不作用于流式修复路径。
  "perf_dump_path": null                          # 默认 null。原理：给定路径后写出性能报告 JSON（各阶段耗时、逐步去噪耗时、显存检查点、总时长），用于定位瓶颈；不影响生成结果。
}
```

> **注意**
> 示例中的 `callback_url` 为 `null`（不回调）。生产环境如需回调，请替换成调用方真实可访问的地址。
> 示例中的 `minio_config` 一旦提供，**即使 `output_storage` 仍是 `local`，产物也会上传到该 bucket**，并清理本地文件。
> 示例中的凭据已脱敏；实际请求请填真实 `access_key` / `secret_key`。

### 2.3 请求字段详解

#### 2.3.1 任务与输入输出

| 字段 | 默认 / 取值 | 说明 |
| --- | --- | --- |
| `task_id` | 必填 | 任务唯一 ID。用作查询/取消的键、回调标识、输入临时文件前缀与默认输出对象键的一部分。重复 ID 会被拒绝。 |
| `timeout` | `-1` | 任务超时秒数。`-1` 表示不超时；`> 0` 表示超时秒数；`0`、`< -1` 非法。计时覆盖**输入下载 + 全部窗口推理 + 保存 + 上传**，从拿到并发槽开始算。超时后任务以 `failed` 结束，`reason` 为 `Request timed out.`。 |
| `prompt` | 必填 | 正面提示词，描述期望的编辑效果（如"鸟扇动翅膀"）。每个窗口都会重新编码该文本。 |
| `negative_prompt` | `null` | 负面提示词。`null` 时使用服务内置默认负面词；仅在 `guidance_scale > 1.0` 时才实际参与编码。 |
| `model` | `"videoedit-normal"` | 生成模式，决定网关路由：`videoedit-normal` / `normal` / `videoedit` / 空 → normal 后端（完整步数）；`videoedit-dmd` / `dmd` → DMD 蒸馏后端（服务强制 `num_inference_steps=4`、`guidance_scale=1.0`、`dynamic_cfg=false`、`negative_prompt=null`）。其他取值会被拒绝。 |
| `video_input_path` | `null` | 服务端本地视频路径，用于本地联调。与 `video_url` 同时提供时**以 `video_url` 为准**。 |
| `video_url` | - | 待修复视频。支持：`minio_config` 所在 bucket 的对象键、`s3://bucket/key`、与 endpoint 同域的 http(s) 链接、可访问的 http(s) 链接（走 HTTP 下载）或服务端本地路径。 |
| `mask_input_path` | `null` | 服务端本地 mask 路径，规则同 `video_input_path`。 |
| `mask_url` | - | 编辑区域 mask。支持视频、图片、`.npy/.npz` 数组，以及 COCO RLE `.json`：`[{"frame":0,"size":[H,W],"counts":[{"mask":{"size":[H,W],"counts":...}}]}]`（顶层也可为 `{"frames":[...]}`）；压缩 RLE 需要 `pycocotools`。逐帧 mask 取并集后二值化。 |
| `reference_image_path` | `null` | 服务端本地参考图路径，规则同 `video_input_path`。 |
| `reference_image_url` | - | 参考图（**必填**）。其内容作为第 `ref_frame_idx` 帧的编辑结果，充当首个窗口的条件前缀帧，用于锁定主体外观。 |
| `callback_url` | `null` | 结果回调地址。提供后：运行中按固定间隔上报进度，结束/失败/超时上报最终结果；不提供则只靠进度接口。 |
| `minio_config` | `null` | 对象存储配置：`endpoint` / `bucket_name` / `access_key` / `secret_key` / `secure` / `region` / `provider` / `prefix`（子字段别名见 2.3.6）。**提供即触发产物上传**（即使 `output_storage=local`），上传成功后本地文件被清理。 |
| `output_storage` | `"local"` | `local` = 产物保留本地；`s3` = 上传云存储。`s3` 或显式 `output_object_key` 时若既无 `minio_config` 也未在服务端开启云存储，提交失败。 |
| `output_path` | `null` | 本地输出位置。带视频后缀（`.mp4/.mov/...`）时视为完整文件名（后缀会被替换为源视频后缀）；否则视为输出目录。`null` → 服务端配置目录，仍为空则用临时目录（产物随任务清理）。 |
| `output_bucket` | `null` | 输出 bucket，覆盖 `minio_config.bucket_name`。 |
| `output_object_key` | `null` | 输出对象键。给定则原样使用（后缀替换为输出后缀，自动去掉前导 `/`）；为 `null` 时用 `"{YYYY/MM/DD}/{HHMMSS}_{task_id}{ext}"`（本地时间）。COS 配置下会自动补 `prefix`。 |

#### 2.3.2 时间轴与窗口规划

修复按"参考帧切分 + 滑窗流式"执行：先按 `ref_frame_idx` 把时间轴切成两条 pass，再在每条 pass 内按 `infer_len`/`overlap` 滑窗。

| 字段 | 默认 / 取值 | 说明 |
| --- | --- | --- |
| `num_frames` | `-1` | 处理的总帧数。`-1` = 与输入视频同长；正数则取 `min(值, 视频帧数)`（静默截断）。mask 帧数与视频帧数必须一致。**输出帧数 == 解析后的 `num_frames`**。 |
| `ref_frame_idx` | `0` | 参考图对应的帧下标，即时间轴切分点：以它为首帧向较长一侧走 **long pass**，较短一侧走 **short pass**（方向相反，条件帧来自 long pass）。`0` 时只有 long pass，此时 `bridge_overlap` 不起作用。 |
| `infer_len` | `49` | 单窗口长度，单位为像素帧，必须满足 `(n-1)%4==0`（Wan VAE 时间压缩 4 倍，49 帧 = 13 个 latent 帧）。决定窗口跨度与 `stride = infer_len - overlap`。 |
| `overlap` | `5` | **同一 pass 内**相邻窗口的重叠帧数，取值 `[0, infer_len)`。除首个窗口外，每个窗口的前 `overlap` 帧以上一窗口的生成结果作为条件帧，其 mask 被清零，不参与贴回。 |
| `bridge_overlap` | `5` | **不是窗口内 overlap**，而是 short pass 的条件前缀长度：取 long pass 首窗口输出的前若干帧作为 short pass 的条件帧，用于跨侧一致性，必须满足 `(b-1)%4==0`。实际长度会被裁剪为可用帧数并按 4k+1 向下取整（最小 1），仅在 `ref_frame_idx > 0` 时生效。 |

#### 2.3.3 采样参数

| 字段 | 默认 / 取值 | 说明 |
| --- | --- | --- |
| `num_inference_steps` | `40` | 每个窗口的采样步数，所有窗口都跑满整个 schedule。 |
| `guidance_scale` | `5.0` | CFG 强度上限（`max_cfg`）。`<= 1.0` 时不启用 CFG（只跑条件分支）。 |
| `seed` | `42` | 随机种子，**所有窗口共用**（当前版本不支持逐窗口变化）。固定 seed + 固定输入可复现。 |
| `dtype` | `"bf16"` | 视频/条件张量精度，取值 `bf16` / `fp16` / `fp32`（DiT/VAE 权重精度由服务启动参数决定）。 |
| `dynamic_cfg` | `true` | 动态 CFG 总开关。开启时前 `dynamic_cfg_max_step` 步的 CFG 从 `guidance_scale` 几何衰减到 `dynamic_cfg_min`，之后只跑条件分支，约省一半算力。 |
| `dynamic_cfg_max_step` | `15` | 衰减步数：第 `i` 步（`i < max_step`）的 CFG = `min + (max-min) / (pow(max-min, 1/max_step) ** i)`。 |
| `dynamic_cfg_min` | `1.0` | 衰减下限。取 `1.0` 时第 `max_step` 步之后 CFG 关闭（`cfg = 1.0`，跳过无条件前向）。注意 `guidance_scale == dynamic_cfg_min` 且开启 `dynamic_cfg` 时会触发除零，请避免。 |

#### 2.3.4 裁剪框与 mask 处理

这些参数决定"取多大范围给模型看"以及"生成结果如何贴回"，单位为**源视频像素**（`feather_px`/`adain_boundary_dilate`/`crop_edge_feather` 为裁剪图像素；`tight`/`fixed_size` 模式下裁剪不再缩放，二者等价）。

| 字段 | 默认 / 取值 | 说明 |
| --- | --- | --- |
| `bbox_padding` | `0` | 在 mask 紧致框并集的基础上，每边额外外扩的**绝对像素**。只扩大裁剪范围（给模型更多原图上下文），不改变 mask 与贴回区域。过大导致越界时任务失败。 |
| `bbox_expand_scale` | `0.3` | 按紧致框自身宽高的比例每边外扩，内部倍率 `2*s+1`（0.3 → 1.6 倍）。外扩后若目标过小（面积比 < 0.2 且短边 < 480 px）会自动上采样短边。 |
| `dilate_px` | `8` | mask 二值化后的形态学膨胀半径（椭圆核 `(2r+1)²`）。同时扩大模型条件 mask 与贴回融合区，用于覆盖目标位移与边缘。 |
| `mask_scale` | `1.0` | 膨胀后 mask 以**自身质心**为原点逐帧缩放（最近邻），`1.0` 为不变。用于整体放大/缩小编辑区域。 |
| `feather_px` | `8` | 贴回 alpha 的高斯羽化半径（核 `2f+1`，`sigma=f/2`），用于软化 mask 边界。调大能掩盖窗口接缝与边界错位，但会更多覆盖原图；紧致框需距裁剪边 ≥ `feather_px + 2` px，否则告警。 |
| `adain_boundary_dilate` | `0` | AdaIN 颜色对齐的参考环宽度（裁剪像素）：用 mask 外一圈原图内容对生成像素做通道均值/方差对齐，缓解色偏。`0` = 关闭。 |
| `crop_edge_feather` | `0` | 裁剪框四条边的线性淡出宽度（像素）：让生成内容在裁剪矩形边界前衰减，避免出现矩形硬边。`0` = 硬贴回。 |
| `chunk_bbox_mode` | `"tight"` | 裁剪框策略：`tight` = 每个 chunk 用自身 mask 并集紧致框（16 对齐，不缩放）；`fixed_size` = 各 chunk 中心不同但尺寸统一（统一分辨率）；`global` = 全片同一框（等同旧的非流式几何）。 |
| `stabilize_mask_union` | `false` | mask 时序稳定（并集）：窗口内每帧都取时间并集 mask，刚性目标形状固定、只保留位移。与 `stabilize_mask_shape` **不可同时开启**。 |
| `stabilize_mask_shape` | `false` | mask 时序稳定（轮廓）：质心做零相位滑动平均，窗口内平移对齐后投票重建轮廓，再按平滑质心摆放，抑制抖动跳变。与 `stabilize_mask_union` 互斥。 |
| `stabilize_smooth_window` | `5` | 稳定化/投票的时间窗口（帧），必须 `>= 1`；实际参与帧数为 `2*(w//2)+1`。两个稳定开关都关闭时该参数被忽略。 |

#### 2.3.5 合成、解码与加速

| 字段 | 默认 / 取值 | 说明 |
| --- | --- | --- |
| `preserve_audio` | `true` | 输出保留源视频音轨（直接复用源音频流）。关闭则输出无声视频。 |
| `enable_paste_back` | `true` | `true` = 把每个窗口的修复结果贴回原视频帧，输出与输入同尺寸；`false` = 不做合成，直接输出裁剪画布（尺寸为最大裁剪框，可明显加速，用于调试）。 |
| `save_crop_only` | `false` | `true` 时**额外**写出 `<output>_crop_only.<ext>`（裁剪画布版本），主输出不受影响。 |
| `use_clip` | `true` | 用 CLIP 视觉编码器编码窗口的条件帧（参考图/桥接帧/重叠帧），把图像 embedding 一起喂给 DiT，提升主体一致性。关闭可省一次 ViT 前向。 |
| `clip_preprocess` | `"diffuser"` | CLIP 输入预处理：`diffuser` = 用 checkpoint 自带的 HF `image_processor`；`diffsynth` = 手工路径（uint8/255 → CHW → 映射到 [-1,1] → bicubic 到 224）。仅影响 CLIP 输入分布。 |
| `decode_mode` | `"stream"` | `stream` = 按窗口只解码所需帧区间，内存由窗口大小决定；`eager` = 先把整段视频与 mask 读入内存。长视频建议 `stream`。 |
| `enable_teacache` | `true`（示例关闭） | TeaCache 缓存加速：相邻步误差低于阈值时复用上一步残差、跳过 DiT 前向。**请求模型默认 `true`**，示例为做画质对比显式关闭。 |
| `teacache_thresh` | `0.3` | 累积相对 L1 阈值，越大越激进（越快、越易糊）。经验值 0.3 约 2.8× 加速，但画质损失会逐帧累积。 |
| `teacache_start_skipping` | `5` | 前 N 步强制真算（整数 = 绝对步数，浮点 = 步数比例，负数 = 从尾部倒数）。开启 CFG 时计数翻倍（按 DiT 调用次数计）。 |
| `teacache_end_skipping` | `1.0` | 尾部强制真算的边界，`1.0` 表示不做尾部强制区。计数同样按 CFG 翻倍。 |
| `enable_frame_interpolation` | `false` | 帧插值（RIFE），提升输出帧率/流畅度。开启后输出帧数成倍增加。 |
| `frame_interpolation_exp` | `1` | 插值倍数指数：`1` → 2 倍，`2` → 4 倍。 |
| `frame_interpolation_scale` | `1.0` | RIFE 推理尺度，高分辨率可降到 `0.5` 省显存。 |
| `frame_interpolation_model_path` | `null` | `null` 时使用默认模型 `elfgum/RIFE-4.22.lite`。 |
| `enable_upscaling` | `false` | 超分（Real-ESRGAN）。 |
| `upscaling_model_path` | `null` | `null` 时使用默认 `ai-forever/Real-ESRGAN`（`RealESRGAN_x4.pth`）。 |
| `upscaling_scale` | `4` | 目标放大倍数。网络固定 x4，`scale != 4` 时在 x4 输出上再做 bicubic 缩放。 |
| `output_quality` | `"default"` | 画质档位：`default`/`maximum`/`high`/`medium`/`low` → 50/100/90/55/35（未知取值按 `null` 处理）。**当前流式修复路径不读取该字段**：输出直接沿用源视频的编码参数（编码器、码率、像素格式），以保证与输入观感一致。 |
| `output_compression` | `null` | 与画质档位配套的压缩参数，同样不作用于流式修复路径；输出后缀默认沿用源视频后缀。 |
| `perf_dump_path` | `null` | 指定后写出性能报告 JSON（含阶段耗时、每步去噪耗时、显存检查点、总时长）。 |

#### 2.3.6 字段别名

`camelCase` 别名（与 `snake_case` 等价，网关与服务端都会先归一化）：

| camelCase | 规范字段 |
| --- | --- |
| `taskId` / `callbackUrl` / `videoUrl` / `maskUrl` | `task_id` / `callback_url` / `video_url` / `mask_url` |
| `referenceImagePath` / `referenceImageUrl` | `reference_image_path` / `reference_image_url` |
| `refFrameIdx` / `bridgeOverlap` / `decodeMode` | `ref_frame_idx` / `bridge_overlap` / `decode_mode` |
| `bboxExpandScale` / `chunkBboxMode` / `cropEdgeFeather` | `bbox_expand_scale` / `chunk_bbox_mode` / `crop_edge_feather` |
| `stabilizeMaskUnion` / `stabilizeMaskShape` / `stabilizeSmoothWindow` | `stabilize_mask_union` / `stabilize_mask_shape` / `stabilize_smooth_window` |
| `preserveAudio` / `useClip` / `clipPreprocess` | `preserve_audio` / `use_clip` / `clip_preprocess` |
| `teacacheThresh` / `teacacheStartSkipping` / `teacacheEndSkipping` | `teacache_thresh` / `teacache_start_skipping` / `teacache_end_skipping` |
| `minioConfig` / `outputObjectKey` | `minio_config` / `output_object_key` |

`minio_config` 子字段别名：`bucket`/`bucketName` → `bucket_name`，`rootUser`/`accessKey` → `access_key`，`rootPass`/`secretKey` → `secret_key`，`useSSL` → `secure`。

### 2.4 提交阶段的校验规则

以下情况会在提交阶段被拒绝（网关返回 HTTP 400，直连单服务返回 `code: 1`）：

| 校验 | 错误信息 |
| --- | --- |
| 缺少 `task_id` | `taskId is required` |
| `timeout` 为 `0` 或 `< -1` | `timeout must be positive or -1` |
| 视频、mask、参考图三项任一缺失 | `videoUrl or video_input_path is required` / `maskUrl or mask_input_path is required` / `referenceImageUrl or reference_image_path is required` |
| `num_frames` 为 `0` 或 `< -1` | `num_frames must be a positive integer or -1` |
| `ref_frame_idx < 0`，或 `>= num_frames` | `ref_frame_idx must be non-negative` / `ref_frame_idx must be smaller than num_frames` |
| `infer_len < 1` 或 `(infer_len-1)%4 != 0` | `infer_len must be >= 1 and satisfy (infer_len - 1) % 4 == 0` |
| `overlap` 不在 `[0, infer_len)` | `overlap must be in [0, infer_len)` |
| `bridge_overlap < 1` 或 `(bridge_overlap-1)%4 != 0` | `bridge_overlap must be >= 1 and satisfy (bridge_overlap - 1) % 4 == 0` |
| 同时开启 `stabilize_mask_union` 和 `stabilize_mask_shape` | `Choose at most one mask stabilization mode` |
| `output_storage=s3` 或给了 `output_object_key` 但没有 `minio_config` | `minioConfig is required for S3 output` |
| 已移除字段（`chunks`、`drop_reference_frame`、`init_latent_mode`、`strength` 等） | `removed VideoEdit request fields are not supported: ...` |

## 三、同步响应

### 3.1 响应规范

服务收到任务提交请求后，会同步返回一个 JSON 响应。`code` 表示提交阶段的处理结果，不代表最终视频修复结果。当前 `10.51.28.123:30000` 为**双后端网关**（normal + DMD），响应格式如下。

#### 3.1.1 接受任务（成功）

**code: 0 — 成功接受**

```
// HTTP/1.1 200 OK
{
  "code": 0,
  "message": "accepted",
  "task_id": "objedit-teacache-off-09e3c2e9d0",
  "status": "dispatching",
  "variant": "normal"
}
```

- `variant` 表示本次任务路由到的后端（`normal` / `dmd`）。
- 后续需要通过进度查询接口或回调确认最终结果。

#### 3.1.2 业务失败（永久失败）

**code: 1 / HTTP 400 — 参数或资源错误**

```
// HTTP/1.1 400 Bad Request（网关）
{
  "detail": "Invalid request body: ..."
}
```

```
// HTTP/1.1 200 OK（直连单后端服务）
{
  "code": 1,
  "message": "Invalid request body: maskUrl or mask_input_path is required",
  "reason": "Invalid request body: maskUrl or mask_input_path is required"
}
```

- 表示请求参数错误、输入资源不可访问、MinIO 配置缺失等提交阶段错误。
- 直连单服务时，带 `task_id` 的失败请求会写入一条 `failed` 任务记录（可被 `GET` 查到），并在提供 `callback_url` 时发送失败回调。

#### 3.1.3 并发限制（可重试拒绝）

**code: 2 — 并发限制**

```
// HTTP/1.1 200 OK
{
  "code": 2,
  "message": "A task is running."
}
```

- 表示当前服务已有任务运行（`VIDEOEDIT_QUEUE_CAPACITY=1`，当前部署不排队）。
- 调用方可以稍后重试。

#### 3.1.4 网关的其他 HTTP 错误

| 场景 | HTTP Status | 响应体 |
| --- | --- | --- |
| `task_id` 重复 | 409 | `{"detail": "Task already exists: <task_id>"}` |
| 目标后端不可用（健康检查失败） | 503 | `{"detail": "VideoEdit <variant> backend is unavailable"}` |
| 请求体不是合法 JSON | 400 | `{"detail": "Invalid JSON body"}` |

#### 3.1.5 响应总览表

| 场景 | HTTP Status | code | 调用方行为 |
| --- | --- | --- | --- |
| 成功接受 | 200 | `0` | 记录 `variant`，开始轮询进度，等待回调或输出 URL。 |
| 提交阶段业务失败 | 400（网关）/ 200（直连） | `1` | 读取 `detail`/`message`，修正参数后重新提交。 |
| 服务忙 | 200 | `2` | 等待当前任务结束后重试（建议指数退避）。 |
| 任务 ID 重复 | 409 | - | 换一个 `task_id`（或先查询该任务状态）。 |
| 后端不可用 | 503 | - | 稍后重试，必要时联系服务维护方。 |
| 其他非 2xx 或网络错误 | 其他 | - | 检查 serve 进程、端口、网络连通性。 |

## 四、查询、下载和取消接口

以下接口用于提交任务后的健康检查、任务详情查询、进度轮询、输出下载和任务取消。示例中的 `{taskId}` 使用当前请求的 `task_id`：`objedit-teacache-off-09e3c2e9d0`。

### 4.1 健康检查

```
GET http://10.51.28.123:30000/health
```

用于确认服务进程、后端和并发槽状态。

```
{
  "status": "ok",
  "backends": {"normal": true, "dmd": true},
  "queue": {"dispatching": 0, "running": 1, "cancelling": 0,
            "completed": 49, "failed": 3, "cancelled": 0, "queued": 0}
}
```

### 4.2 查询任务详情

```
GET http://10.51.28.123:30000/v1/videos/objedit-teacache-off-09e3c2e9d0
```

返回任务详情。该接口中的 `progress` 与 `/progress` 接口保持同步，都会读取当前窗口和采样 step 换算出的最新进度。

```
{
  "id": "objedit-teacache-off-09e3c2e9d0",
  "object": "video",
  "model": "videoedit-normal",
  "variant": "normal",
  "status": "running",
  "progress": 52,
  "created_at": 1791623925,
  "url": null,
  "file_path": ".../.local/videoedit-l40s.67-u2-5403/outputs/normal/objedit-teacache-off-09e3c2e9d0.mp4",
  "error": null,
  "reason": null
}
```

#### 主要字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | string | 任务 ID，即提交时的 `task_id`。 |
| `variant` | string | 实际执行的后端：`normal` 或 `dmd`。 |
| `status` | string | `dispatching`、`running`、`cancelling`、`completed`、`failed`、`cancelled`（`queued` 仅存在于历史记录）。 |
| `progress` | number | 当前百分比进度。运行中按窗口和采样 step 更新，完成后为 `100`。 |
| `url` | string/null | 上传到 MinIO/COS 后的输出地址。 |
| `file_path` | string/null | 服务端本地输出文件路径。上传到云存储并清理本地文件后为 `null`。 |
| `error` | object/null | 失败时包含错误信息。 |
| `reason` | string/null | 失败时包含可直接展示或写入任务表的失败原因。 |

### 4.3 查询进度

```
GET http://10.51.28.123:30000/v1/videos/objedit-teacache-off-09e3c2e9d0/progress
```

返回轻量级进度信息，适合调用方定时轮询。

```
{
  "id": "objedit-teacache-off-09e3c2e9d0",
  "variant": "normal",
  "status": "running",
  "progress": 52,
  "file_path": ".../.local/videoedit-l40s.67-u2-5403/outputs/normal/objedit-teacache-off-09e3c2e9d0.mp4",
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
| `status` | string | `running` 处理中，`cancelling` 取消中，`completed` 完成，`failed` 失败，`cancelled` 已取消。 |
| `progress` | number | 当前百分比进度，按窗口和采样 step 换算。 |
| `queue_position` | number | 排队位次；当前部署不排队，仅在历史 `queued` 记录上出现。 |
| `file_path` | string | 服务端本地输出路径。 |
| `url` | string/null | 上传到 MinIO/COS 后的访问地址。运行中通常为 `null`。 |
| `error` / `reason` | object/string/null | 失败时的错误对象与可直接展示的原因文本。 |
| `callback_status` | string/null | 回调发送状态（`succeeded` / `failed`）。 |
| `callback_error` | string/null | 回调失败原因。 |
| `callback_attempts` | number/null | 回调尝试次数。 |

### 4.4 下载视频内容

```
GET http://10.51.28.123:30000/v1/videos/objedit-teacache-off-09e3c2e9d0/content
```

当任务完成且输出仍保留在服务端本地文件系统时，可以用该接口下载视频文件。

```
curl -v --fail-with-body "http://10.51.28.123:30000/v1/videos/objedit-teacache-off-09e3c2e9d0/content" -o out.mp4
```

| 场景 | 响应 | 说明 |
| --- | --- | --- |
| 本地文件存在 | `200 video/mp4` | 直接返回视频文件。 |
| 任务仍在运行或文件不存在 | `404` | 返回 `Generation is still in-progress`。 |
| 输出已上传到云存储 | `400` | 返回提示信息，调用方应使用 `url` 字段中的云端地址下载。 |
| 任务不存在 | `404` | 返回 `Video not found`。 |

### 4.5 取消任务

```
DELETE http://10.51.28.123:30000/v1/videos/objedit-teacache-off-09e3c2e9d0
```

- 任务处于 `completed` / `failed` / `cancelled` 时，直接返回当前任务状态。
- 任务仍在运行时，服务会**请求取消**：状态先变为 `cancelling`，后端在下一个安全检查点（每步去噪 / 每个窗口 / 保存前）终止推理，随后状态落到终态。取消同超时一样会以 `Request timed out.` 作为结束原因。
- 取消是"请求"而非瞬时生效，调用方应继续用 `GET` 轮询直到终态。

```
{
  "id": "objedit-teacache-off-09e3c2e9d0",
  "object": "video",
  "model": "videoedit-normal",
  "variant": "normal",
  "status": "cancelling",
  "progress": 52
}
```

> **注意**
> `DELETE` 不再等价于"删除任务记录"，也不保证删除已经上传到 MinIO/COS 的输出对象；它用于中止仍在运行的任务。

### 4.6 队列状态（运维）

```
GET http://10.51.28.123:30000/admin/queue?limit=20
```

返回各状态任务计数与最近任务列表，仅供运维排查使用。

### 4.7 输出位置

当前请求提供 `minio_config`（COS），因此产物**会上传云存储并清理本地文件**：

```
bucket: vrs-mms-1258229344
object: flowcut/2026/10/10/093045_objedit-teacache-off-09e3c2e9d0.mp4
url:    https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/2026/10/10/093045_objedit-teacache-off-09e3c2e9d0.mp4
```

即：`output_object_key` 为空时用默认键 `"{YYYY/MM/DD}/{HHMMSS}_{task_id}.mp4"`（本地时间），再由 `minio_config.prefix`（`/flowcut`，仅 COS）补前缀。

如果需要"只留本地、不上传"，请**去掉 `minio_config`** 并保持 `output_storage: "local"`，再用 `GET /v1/videos/{taskId}/content` 下载；此时若 `output_path` 为空，产物位于服务端临时目录，任务记录清理后不可恢复。

## 五、超时和回调

### 5.1 timeout 语义

| timeout 值 | 含义 | 后台任务行为 |
| --- | --- | --- |
| `-1` | 不设置任务超时（默认） | 一直等待后台推理完成，不因超时取消。 |
| `> 0` | 超时秒数 | 从获取并发槽开始计时，覆盖下载 + 推理 + 保存 + 上传；超时后任务标记为失败，错误信息为 `Request timed out.`。 |
| `0` | 非法值 | 提交阶段被拒绝。 |
| `< -1` | 非法值 | 提交阶段被拒绝。 |

> **说明**
> `timeout` 控制后台 VideoEdit 任务是否被服务端取消，不是 curl 命令本身的 HTTP 等待时间。任务提交接口仍会尽快同步返回。
> 当前请求的 `timeout: 6000` 表示 100 分钟；正常短视频修复可在其内完成，超长视频或排队时建议传 `-1`。

### 5.2 回调请求

请求中提供 `callback_url` 时，服务会在任务运行中**默认每 5 秒**（有进度变化时）上报一次进度，并在任务完成、失败或超时后发送最终结果；未提供时不发送回调。

```
POST {callbackUrl}
Content-Type: application/json
```

回调失败会重试最多 3 次（间隔 1s / 2s / 4s，单次超时 10s），结果记录在 `callback_status` / `callback_error` / `callback_attempts`。

**成功回调示例**

```
{
  "status": "succeeded",
  "progress": 100,
  "reason": "",
  "output": "{\"gen_video_url\":\"flowcut/2026/10/10/093045_objedit-teacache-off-09e3c2e9d0.mp4\",\"duration\":45}"
}
```

- `output` 是字符串形式的 JSON；`gen_video_url` 为输出对象键（含 `prefix`）或本地路径，`duration` 为推理耗时（秒）。

**失败回调示例**

```
{
  "status": "failed",
  "progress": 37,
  "reason": "Request timed out.",
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
| 1 | `POST /v1/videos/repairs` 可达 | 确认 `10.51.28.123:30000` 端口能访问，且 `GET /health` 中目标后端为 `true`。 |
| 2 | 输入视频 URL 可访问 | 服务端能下载 `video_url`（对象键需与 `minio_config` 的 bucket/endpoint 匹配）。 |
| 3 | mask URL 可访问 | 服务端能下载 `mask_url`；`.json` 压缩 RLE 需要服务端安装 `pycocotools`。 |
| 4 | 参考图 URL 可访问 | `reference_image_url` 为必填项，缺失会在提交阶段被拒。 |
| 5 | mask 与视频帧数一致 | 否则任务失败：`VideoEdit video/mask length mismatch`。 |
| 6 | MinIO 凭据正确 | 凭据需同时具备读取输入对象与写入输出对象的权限。 |
| 7 | `task_id` 唯一 | 重复 ID 会返回 409。 |
| 8 | `timeout` 符合预期 | 长任务用 `-1` 或足够大的秒数，避免后台推理被取消。 |
| 9 | 进度接口可查 | 提交后执行 `GET /v1/videos/{taskId}/progress`，关注 `status` 与 `progress`。 |
| 10 | 回调地址真实可达 | 生产环境不要使用占位地址；未提供 `callback_url` 时靠轮询。 |
| 11 | 输出对象路径正确 | 完成后检查 `url` 或 `flowcut/<YYYY/MM/DD>/<HHMMSS>_<task_id>.mp4`。 |

### 6.2 对象路径对照

以当前请求的 COS 配置（bucket `vrs-mms-1258229344`，prefix `/flowcut`）为例：

| 用途 | 对象键 | 访问地址 |
| --- | --- | --- |
| 输入视频 | `flowcut/test/2026/07/16/3aba7476ee0e275c85a32627bb7b1a6dd18a4ed8.mp4` | `https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/test/2026/07/16/3aba7476ee0e275c85a32627bb7b1a6dd18a4ed8.mp4` |
| 输入 mask | `flowcut/test/2026/07/17/feba817b185f2d7fad614a10c908feadb6f69b11.json` | `https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/test/2026/07/17/feba817b185f2d7fad614a10c908feadb6f69b11.json` |
| 参考图 | `flowcut/test/2026/07/16/6d083c53f432e9d156660741dd0f3b8660b26a15.png` | `https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/test/2026/07/16/6d083c53f432e9d156660741dd0f3b8660b26a15.png` |
| 输出视频 | `flowcut/2026/10/10/093045_objedit-teacache-off-09e3c2e9d0.mp4` | `https://cos.ap-beijing.myqcloud.com/vrs-mms-1258229344/flowcut/2026/10/10/093045_objedit-teacache-off-09e3c2e9d0.mp4` |
