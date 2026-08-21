# VideoEdit 双模型服务快速使用说明（v2）

> 适用范围：`cos` 分支，基于 `HEAD 5e4e5e915` 及 2026-08-21 当前工作区的 VideoEdit 对齐改动。
>
> 本文由 [`videoedit_service_quickstart.md`](./videoedit_service_quickstart.md) 适配而来。命令默认在宿主机执行，容器名为 `videoedit_reset`，统一入口为 `http://127.0.0.1:30000`。

## 1. 当前版本必须先知道的变化

1. `strict_videoedit_math` 已在 VideoEdit 模型和所有 Transformer block 中固定为 `False`，不是 API 参数。当前 case0008、step_47500 的 crop/full golden 在该配置下均通过；客户端不要尝试发送这个字段。实现见 [`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)，验证证据见 [`performance-impact-review.md`](../docs_always/video-edit-compare/performance-impact-review.md#41-已关闭但保留对照严格-dit-数学路径)。
2. `drop_reference_frame` 已从请求协议删除，发送它或 `dropReferenceFrame` 会直接校验失败。当前算法语义固定为不删除参考帧。完整删除字段列表见 [`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L130)。
3. 请求接口默认 `num_inference_steps=40`、`decode_mode=stream`、`save_crop_only=false`、`enable_teacache=true`。数值 golden 请求应显式设置 `save_crop_only=true`、`enable_teacache=false`；默认值见 [`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L167)。
4. `videoedit-normal` 和 `videoedit-dmd` 共用两张 GPU，并由网关全局串行调度；它们不是两个可并发执行的 GPU 服务。路由和队列实现见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L34) 与 [`start_videoedit_container.sh`](../scripts/start_videoedit_container.sh#L5)。
5. DMD 请求会被网关固定改为 4 步、`guidance_scale=1.0`、关闭 dynamic CFG，并清空 negative prompt；请求体中的对应值不会生效。实现见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L73)。

当前双服务 [`start.sh`](../scripts/videoedit_dual_service/start.sh#L99) 没有显式传入 `--attention-backend`。在支持 FlashAttention 的 CUDA 环境中会自动优先选择 FA，而现有 strict-false golden 是用 `torch_sdpa` 采集的。因此，下文请求可复现 API 侧参数，但不能单独保证完整 golden 环境；需要复现 golden 时，还应给两个 backend 的 `sglang serve` 命令增加 `--attention-backend torch_sdpa` 并重启。自动选择逻辑见 [`selector.py`](../python/sglang/multimodal_gen/runtime/layers/attention/selector.py#L115) 和 [`cuda.py`](../python/sglang/multimodal_gen/runtime/platforms/cuda.py#L381)。

## 2. 服务拓扑与配置

默认拓扑如下：

| 组件 | 地址/端口 | 作用 |
| --- | --- | --- |
| Gateway | `0.0.0.0:30000` | 对外统一入口、持久队列、normal/DMD 路由 |
| normal backend | `127.0.0.1:31100` | normal checkpoint 推理 |
| DMD backend | `127.0.0.1:32100` | DMD checkpoint 推理 |

容器默认把宿主 GPU `2,3` 映射为容器内 `0,1`；两个 backend 都用容器内两张卡，并启用 2-way Ulysses、DiT 逐层 offload，以及 T5、CLIP、VAE CPU offload。只有 Gateway 的 `30000` 端口发布到宿主机。全局串行只限制任务执行，normal 和 DMD 的模型仍会同时驻留在这两张卡及主存中。启动参数来源见 [`start.sh`](../scripts/videoedit_dual_service/start.sh#L99) 和 [`config.env`](../scripts/videoedit_dual_service/config.env)。

当前 dual-service 脚本硬编码 `--num-gpus 2 --sp-degree 2 --ulysses-degree 2`，不能仅把 `CUDA_DEVICES` 改成单卡。需要单卡时必须另行调整服务拓扑和并行参数，不能直接套用本文的双模型 Gateway 启动方式。

启动前检查 [`config.env`](../scripts/videoedit_dual_service/config.env) 中至少以下配置：

| 字段 | 当前含义 |
| --- | --- |
| `BASE_MODEL` | 包含 `model_index.json` 的 VideoEdit 基础模型目录 |
| `NORMAL_TRANSFORMER` | normal Transformer checkpoint 目录 |
| `DMD_TRANSFORMER` | DMD Transformer checkpoint 目录 |
| `CUDA_DEVICES` | backend 进程可见的容器内 GPU，默认 `0,1` |
| `OUTPUT_DIR` | 未在请求中指定 `output_path` 时的服务输出根目录 |
| `INPUT_DIR` | URL 输入下载后的保存目录 |
| `QUEUE_DB` | Gateway 持久队列 SQLite 文件 |

Transformer 启动前会校验 `config.json`、`_class_name=WanVideoEditTransformer3DModel`、`in_channels=36`、`out_channels=16` 和 safetensors 权重。normal checkpoint 无效会导致启动失败；DMD checkpoint 无效时会降级为 normal-only。校验实现见 [`resource_probe.py`](../scripts/videoedit_dual_service/resource_probe.py#L246)。

## 3. 启动、检查与停止

### 3.1 重建容器

推荐显式重建，确保镜像、挂载、GPU 和环境变量口径一致：

```bash
RECREATE=1 bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
```

当前脚本在已有同名容器且未指定 `RESTART_EXISTING=1` 时也会删除并重建；`RECREATE=1` 用于明确表达这一意图。要更换宿主 GPU，可在重建时覆盖：

```bash
RECREATE=1 HOST_GPUS=0,1 \
  bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
```

默认镜像是 `sglang-mgtv:1.0`，容器是 `videoedit_reset`。变量及重建逻辑见 [`start_videoedit_container.sh`](../scripts/start_videoedit_container.sh#L12)。

### 3.2 只重启现有容器

```bash
RESTART_EXISTING=1 bash /root/VideoEdit/sglang/scripts/start_videoedit_container.sh
```

这会复用已有容器配置。代码和 `config.env` 位于绑定挂载中，进程重启后会重新加载；但镜像、端口映射、GPU 映射和 `docker run -e` 环境变量不会因此改变，这些情况应重建容器。

### 3.3 健康检查与状态

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/health \
  | python3 -m json.tool

docker exec videoedit_reset \
  bash scripts/videoedit_dual_service/status.sh
```

Gateway 健康状态含义：

- `ok`：normal 和 DMD 都可用；
- `degraded_normal_only`：normal 可用、DMD 不可用；
- `unavailable`：normal 不可用，此时不能接单。

`/health` 在上述三种状态下都会返回 HTTP 200，因此不能只看 `curl` 的退出码，必须检查响应中的 `status` 和 `backends`。

启动流程会先保证 normal 健康，再尝试启动 DMD；DMD 启动失败或空闲资源门禁失败时，会保留 normal 服务。实现见 [`start.sh`](../scripts/videoedit_dual_service/start.sh#L169) 和 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L108)。

### 3.4 停止

停止整个容器：

```bash
docker stop videoedit_reset
```

停止容器内服务进程：

```bash
docker exec videoedit_reset \
  bash scripts/videoedit_dual_service/stop.sh
```

容器主进程会监控 normal 和 Gateway；执行上述命令后，主进程会随之退出，容器也会停止。日常停服直接使用 `docker stop` 更清晰。

## 4. 本地 normal 请求：请求侧对齐口径

本地输入路径必须在容器内可读，输出目录也应位于持久挂载中。默认 `/root/VideoEdit` 会以相同路径挂载进容器。

```bash
curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "videoedit-normal-local-001",
    "model": "videoedit-normal",
    "timeout": -1,
    "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
    "video_input_path": "/root/VideoEdit/test/1080.mp4",
    "mask_input_path": "/root/VideoEdit/test/mask_1080_merged.mp4",
    "reference_image_path": "/root/VideoEdit/test/local.png",
    "output_storage": "local",
    "output_path": "/root/VideoEdit/test/output_normal_001.mp4",
    "num_frames": -1,
    "ref_frame_idx": 0,
    "bridge_overlap": 5,
    "infer_len": 49,
    "overlap": 5,
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "dynamic_cfg_min": 1.0,
    "seed": 42,
    "dtype": "bf16",
    "bbox_padding": 0,
    "bbox_expand_scale": 0.3,
    "dilate_px": 8,
    "mask_scale": 1.0,
    "feather_px": 8,
    "adain_boundary_dilate": 0,
    "enable_paste_back": true,
    "save_crop_only": false,
    "use_clip": true,
    "clip_preprocess": "diffuser",
    "decode_mode": "stream",
    "enable_teacache": false,
    "enable_frame_interpolation": false,
    "enable_upscaling": false
  }' | python3 -m json.tool
```

成功入队的 Gateway 响应类似：

```json
{
  "code": 0,
  "message": "queued",
  "task_id": "videoedit-normal-local-001",
  "status": "queued",
  "variant": "normal"
}
```

如果显式设置 `save_crop_only=true`，会在主输出外额外生成 `/root/VideoEdit/test/output_normal_001_crop_only.mp4`；这会增加一次 resize、视频编码、I/O 和磁盘占用。命名和写盘逻辑见 [`wan_videoedit_pipeline.py`](../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L592)。

服务还会写 `/root/VideoEdit/test/output_normal_001.videoedit.json`，记录 bbox、帧数和窗口物化信息。元数据写盘逻辑见 [`wan_videoedit_pipeline.py`](../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L528)。

`output_path` 可以是文件或目录：传视频文件名时使用其目录和基名，但最终扩展名优先跟随源视频；传目录时生成 `<task_id><源视频扩展名>`。未传时写入对应 backend 的 `OUTPUT_DIR/{normal,dmd}`。解析逻辑见 [`video_api.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py#L825)。每次重复执行示例前请更换 `task_id`，因为 Gateway 会拒绝数据库中已存在的 ID。

## 5. 请求参数口径

| 参数 | 当前行为 |
| --- | --- |
| `model` | `videoedit`、`videoedit-normal`、`normal` 路由 normal；`videoedit-dmd`、`dmd` 路由 DMD |
| `timeout` | `-1` 表示不限时；也可用正整数秒；`0` 或小于 `-1` 非法 |
| `num_frames` | `-1`/`null` 处理完整视频；正数取请求值与源帧数的较小值 |
| `ref_frame_idx` | 任意非负参考帧索引；显式 `num_frames>0` 时必须小于 `num_frames` |
| `infer_len` | 必须 `>=1` 且满足 `(infer_len - 1) % 4 == 0` |
| `overlap` | 必须满足 `0 <= overlap < infer_len` |
| `bridge_overlap` | 必须 `>=1` 且满足 `(bridge_overlap - 1) % 4 == 0` |
| `decode_mode` | 默认 `stream`，降低输入侧主存；`eager` 一次性解码全视频，但可避免 backward pass 缓存缺失时重复解码 |
| `enable_teacache` | API 默认 `true`；追求当前 golden 对齐时必须显式设为 `false` |
| `save_crop_only` | 默认 `false`；需要额外保存 crop sidecar 时显式设为 `true` |
| Attention backend | 不是 repair API 字段；由服务启动参数决定，当前双服务脚本未固定，完整 golden 需启动时指定 `torch_sdpa` |

视频和 mask 的总帧数必须相等，否则请求在预处理阶段失败；`num_frames=-1` 会解析为完整源帧数。实现见 [`preprocess.py`](../python/sglang/multimodal_gen/runtime/videoedit/preprocess.py#L97)。

normal 请求使用客户端给出的采样参数。DMD 请求则由 Gateway 覆盖为固定 4 步策略，因此不要用 DMD 结果验证 normal 的 40 步 golden。

## 6. 远程输入与 S3/MinIO 输出

远程输入使用 URL，输出上传到 S3/MinIO。独立部署时应随请求提供 `minio_config`：

```bash
curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:30000/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "videoedit-normal-remote-001",
    "model": "videoedit-normal",
    "timeout": -1,
    "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
    "video_url": "http://minio.example.com:9000/flowcut/input/1080.mp4",
    "mask_url": "http://minio.example.com:9000/flowcut/input/mask_1080.mp4",
    "reference_image_url": "http://minio.example.com:9000/flowcut/input/local.png",
    "minio_config": {
      "endpoint": "minio.example.com:9000",
      "bucket_name": "flowcut",
      "access_key": "your-access-key",
      "secret_key": "your-secret-key",
      "secure": false,
      "region": "us-east-1"
    },
    "output_storage": "s3",
    "output_bucket": "flowcut",
    "output_object_key": "test/output/remote_001.mp4",
    "num_frames": -1,
    "ref_frame_idx": 0,
    "bridge_overlap": 5,
    "infer_len": 49,
    "overlap": 5,
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    "dynamic_cfg": true,
    "dynamic_cfg_max_step": 15,
    "dynamic_cfg_min": 1.0,
    "seed": 42,
    "dtype": "bf16",
    "decode_mode": "stream",
    "save_crop_only": false,
    "enable_teacache": false,
    "enable_paste_back": true
  }' | python3 -m json.tool
```

注意：

- 未配置全局云存储时，`output_storage=s3` 或传入 `output_object_key` 都要求 `minio_config`；
- 未传 `output_object_key` 时，默认生成 `YYYY/MM/DD/HHMMSS_{task_id}.{源视频扩展名}`；MP4 输入对应 `.mp4`；
- `output_bucket` 未传时使用 `minio_config.bucket_name`；
- 当前上传流程只上传并清理主输出；如果生成了 `*_crop_only.mp4`，它和 `*.videoedit.json` sidecar 都仍保留在 backend 本地输出目录；
- 示例使用 snake_case。接口只兼容协议中明确列出的少量 camelCase 别名，不要假设所有字段都能自动转成驼峰。

存储校验和默认 object key 见 [`video_api.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py#L419) 与 [`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L159)。

## 7. 查询进度、队列和取消任务

设置任务 ID：

```bash
TASK_ID=videoedit-normal-local-001
```

查询完整任务：

```bash
curl --noproxy '*' -sS \
  "http://127.0.0.1:30000/v1/videos/${TASK_ID}" \
  | python3 -m json.tool
```

只查询进度：

```bash
curl --noproxy '*' -sS \
  "http://127.0.0.1:30000/v1/videos/${TASK_ID}/progress" \
  | python3 -m json.tool
```

查看 Gateway 队列：

```bash
curl --noproxy '*' -sS \
  'http://127.0.0.1:30000/admin/queue?limit=20' \
  | python3 -m json.tool
```

取消排队中或运行中的任务：

```bash
curl --noproxy '*' -sS \
  -X DELETE "http://127.0.0.1:30000/v1/videos/${TASK_ID}" \
  | python3 -m json.tool
```

队列是持久化且全局单并发的：一个 normal 或 DMD 任务运行时，其他任务保持 `queued`。重复提交同一个 `task_id` 返回 HTTP 409；目标 backend 不健康时返回 HTTP 503。接口定义见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L425)。

Gateway 不代理 backend 的 `/content` 下载接口。本地输出完成后读取响应中的 `file_path`；S3/MinIO 输出读取 `url` 或 `output_object_key`。

不要在 active 任务存在时重启服务。Gateway 队列保存在 SQLite 中，但 backend 任务状态保存在进程内存；重启后 Gateway 可能找不到原 backend 任务并暂停队列，以避免重复执行。操作前先检查 `/admin/queue`；遇到 stale active 记录时，应先备份 `QUEUE_DB` 再人工处置，不要直接删除生产队列文件。恢复保护见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L273)。

## 8. 日志与请求审计

查看容器聚合日志：

```bash
docker logs -f videoedit_reset
```

分别查看组件日志：

```bash
docker exec videoedit_reset \
  tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/normal.log

docker exec videoedit_reset \
  tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/dmd.log

docker exec videoedit_reset \
  tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/gateway.log
```

启动资源监控日志：

```bash
docker exec videoedit_reset \
  tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/normal-resource.log

docker exec videoedit_reset \
  tail -f /root/VideoEdit/tmp/sglang-videoedit-dual/logs/dmd-resource.log
```

资源与启动门禁记录位于：

```text
/root/VideoEdit/tmp/sglang-videoedit-dual/normal-startup.json
/root/VideoEdit/tmp/sglang-videoedit-dual/dmd-startup.json
/root/VideoEdit/tmp/sglang-videoedit-dual/dual-idle-gate.json
```

### 8.1 开启逐请求审计

当前 `ServerArgs` 默认关闭请求审计。虽然容器启动脚本创建并传入了 `VIDEOEDIT_REQUEST_LOG_DIR` 环境变量，但 [`start.sh`](../scripts/videoedit_dual_service/start.sh#L99) 尚未把该环境变量映射为 `sglang serve` 参数，因此只设置环境变量不会生成审计文件。

如需开启，在 `start_backend()` 的 `sglang serve` 命令中加入：

```text
--videoedit-request-log-dir "$VIDEOEDIT_REQUEST_LOG_DIR"
```

然后重启或重建容器。默认会脱敏 access key、secret key 等字段。生产环境不建议添加 `--videoedit-request-log-sensitive-values true`；审计开关定义见 [`server_args.py`](../python/sglang/multimodal_gen/runtime/server_args.py#L861)，脱敏实现见 [`request_audit.py`](../python/sglang/multimodal_gen/runtime/videoedit/request_audit.py#L43)。

查看审计文件：

```bash
docker exec videoedit_reset \
  ls -lt /root/VideoEdit/tmp/sglang-videoedit-request-logs
```

## 9. 常见问题

### 请求立即失败并提示 removed fields

删除 `drop_reference_frame`、`dropReferenceFrame`、`chunks`、`generator_device`、`strength` 等已移除字段。这些语义已经固定在服务端，不再允许请求覆盖。

### health 是 degraded_normal_only

normal 仍可用，但 `videoedit-dmd` 请求会返回 HTTP 503。检查 DMD checkpoint 校验结果、`dmd.log`、`dmd-resource.log` 和 `dual-idle-gate.json`。

### 请求一直 queued

Gateway 会串行调度 normal 和 DMD。先查询 `/admin/queue` 和当前 active 任务，再查看对应 backend 日志。不要同时直接调用内部 `31100/32100` 端口绕过 Gateway。

### 长视频主机内存过高

`stream` 和关闭 crop sidecar 已是默认值；若仍然过高，应检查请求是否显式覆盖为
`decode_mode=eager` 或 `save_crop_only=true`。注意任意参考帧导致 backward pass 时，
stream 缓存淘汰可能触发重复从头解码，需要结合视频长度实测时延。

### 只需要生产输出，不需要对齐 sidecar

设置：

```json
{
  "save_crop_only": false
}
```

这不会关闭主视频的 paste-back；主输出是否 paste-back 由 `enable_paste_back` 控制。

## 10. 主要实现依据

- 容器生命周期与挂载：[`start_videoedit_container.sh`](../scripts/start_videoedit_container.sh)
- 双 backend 启动和降级：[`videoedit_dual_service/start.sh`](../scripts/videoedit_dual_service/start.sh)
- Gateway 路由、串行队列和任务接口：[`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py)
- API 字段、默认值和删除字段：[`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)
- 请求校验、下载、输出和回调：[`video_api.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py)
- 当前 strict 配置：[`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)
