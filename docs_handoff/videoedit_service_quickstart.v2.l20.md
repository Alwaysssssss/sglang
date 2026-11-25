# VideoEdit 双模型服务快速使用说明（v2 / L20）

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

| 组件 | 地址/端口 |
| --- | --- |
| Gateway | 容器 `0.0.0.0:30000` → 宿主 `30000`，唯一对外入口 |
| normal backend | 容器内 `127.0.0.1:31100` |
| DMD backend | 容器内 `127.0.0.1:32100` |

两个 backend 共用 2 张 GPU，由 Gateway 串行执行任务；模型同时驻留在 GPU 和主存。

本指南使用 [`config.l20.env`](../scripts/videoedit_dual_service/config.l20.env)。启动前按本机情况检查其中的配置：

| 配置 | 本文设置 / 含义 |
| --- | --- |
| `CUDA_DEVICES` | `0,1`，容器内 GPU 编号 |
| `NUM_GPUS` / `SP_DEGREE` / `ULYSSES_DEGREE` / `RING_DEGREE` | `2 / 2 / 2 / 1` |
| `PROJECT_ROOT` | 容器内可见的源码仓库绝对路径 |
| `BASE_MODEL` / `NORMAL_TRANSFORMER` / `DMD_TRANSFORMER` | 基础模型及两个 checkpoint 的绝对路径 |
| `RUNTIME_DIR` / `LOG_DIR` / `PID_DIR` / `QUEUE_DB` | 运行数据、日志、PID 和持久队列 |
| `INPUT_DIR` / `OUTPUT_DIR` | 后端实际使用的输入、输出根目录 |
| `GATEWAY_PORT` | `30000`，与 Docker 的 `CONTAINER_PORT` 一致 |

脚本根据自身位置推导宿主仓库目录，并将仓库父目录同路径挂载进容器；配置中的模型、素材和运行目录须在挂载范围内。迁移机器时仍需修改配置文件中的绝对路径。容器脚本的 `PROJECT_ROOT` 表示挂载根，服务配置中的同名字段表示源码根。

**配置不会按机型自动选择。** 当前默认读取的 `config.env` 已不存在，需按 §3 显式选择配置。Docker 启动时，`DUAL_SERVICE_CONFIG_HOST` 用于检查宿主配置文件，`DUAL_SERVICE_CONFIG_CONTAINER` 被传入容器的 `VIDEOEDIT_DUAL_CONFIG`；容器入口以及 `start.sh`、`status.sh`、`stop.sh` 均通过 `source` 加载该配置。文件由仓库挂载提供，不会自动挂载任意外部配置。

直接调用 `start.sh` 时，外部 `CUDA_DEVICES` 和四个并行参数优先于文件中的值；通过容器入口启动时，请在配置文件中设置它们，外层脚本尚未透传这些并行参数。`HOST_GPUS` 是宿主 GPU 编号，与配置中的容器内编号分开设置。

## 3. 构建、启动、检查与停止

以下步骤在**宿主 Bash 终端、仓库根目录**执行，并沿用同一个终端的变量。

### 3.1 选择配置

```bash
export IMAGE_NAME=sglang-videoedit-src:l20
export CONTAINER_NAME=videoedit_reset
export HOST_PORT=30000 CONTAINER_PORT=30000
export HOST_GPUS=2,3  # 示例：替换为实际分配的 2 张宿主 GPU
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1
export DUAL_SERVICE_CONFIG_HOST="$PWD/scripts/videoedit_dual_service/config.l20.env"
export DUAL_SERVICE_CONFIG_CONTAINER=/sgl-workspace/sglang/scripts/videoedit_dual_service/config.l20.env
```

上述容器路径对应脚本默认的 `CONTAINER_REPO_DIR=/sgl-workspace/sglang`；自定义挂载目标时一并调整。确认 Docker 可用、GPU 分配正确，并检查配置中的模型和目录路径。

### 3.2 构建镜像

在 `sglang` 仓库根目录执行，Dockerfile 和构建上下文均使用 `.devcontainer`：

```bash
docker build -f .devcontainer/Dockerfile -t "$IMAGE_NAME" .devcontainer
```

L20 / L40S 共用 [`.devcontainer/Dockerfile`](../.devcontainer/Dockerfile)，分别沿用 §3.1 的镜像标签和机型配置。默认 `BASE_IMAGE` 固定为 `lmsysorg/sglang:nightly-dev-cu13-20260601-373cadc9` 及其 SHA256 digest；在基础镜像上补充 `ftfy==6.3.1`、`boto3==1.43.102`、`minio==7.2.20`、系统工具及开发工具。源码和服务配置在启动时由宿主挂载；此构建入口不执行本仓库 `python[diffusion]` 的完整安装。

若已将同一 digest 的基础镜像导入为本地标签 `sglang-base:20260601-local`，可通过 `BASE_IMAGE` 参数构建。以下命令适用于代理运行在宿主 `127.0.0.1:10808` 的情况；`--network=host` 使构建步骤能访问该代理：

```bash
docker build --pull=false --network=host \
  --build-arg BASE_IMAGE=sglang-base:20260601-local \
  --build-arg HTTP_PROXY=http://127.0.0.1:10808 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:10808 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -f .devcontainer/Dockerfile -t "$IMAGE_NAME" .devcontainer
```

本地标签须已存在且对应指定基础镜像；`--pull=false` 不能替代预先导入。上述代理参数供构建中的 `apt`、`pip`、`curl` 等使用，不会替 Docker daemon 配置拉取代理。已有兼容的最终镜像时可跳过构建，将 `IMAGE_NAME` 改为实际镜像名。

构建成功后同名标签指向新镜像，现有容器不会自动更新或重启。依赖说明、导入检查和本次 L40S 构建记录见 [L40S 指南附录 B](./videoedit_service_quickstart.v2.l40.md#附录-b-devcontainer-镜像说明与检查)；L20 仍需独立完成健康检查和推理验收。

`rebuild_image_create_videoedit_container.sh` 构建后启动的是单后端，不读取这两份配置；本文双模型部署使用上述构建命令和下面的启动脚本。

### 3.3 启动或重建容器

```bash
RECREATE=1 bash scripts/start_videoedit_container.sh
```

**该命令会删除并重建同名容器。** 已有服务时先按 §7 确认没有 active 任务。脚本未设置 `RECREATE=1` 时也可能重建，不能把它当作删除保护开关。

配置随仓库挂载进入容器，默认 `ENABLE_DMD=true`，启动顺序为 normal → DMD → Gateway。设置 `ENABLE_DMD=false` 时仅启动 normal 和 Gateway。启用 DMD 时，权重校验、资源检查或任一后端启动失败均会报错退出，不会自动降级；容器内 DMD 运行中退出也会停止整个服务栈。`SKIP_SECOND_SERVICE_RESOURCE_GATE` 只控制启动前的资源预判，不是 DMD 开关。

可在所选配置文件中设置开关，或在创建容器时显式覆盖（需要重建容器，不能在有 active 任务时执行）：

```bash
ENABLE_DMD=true RECREATE=1 bash scripts/start_videoedit_container.sh   # 开启（默认）
# 或：
ENABLE_DMD=false RECREATE=1 bash scripts/start_videoedit_container.sh  # 手动关闭
```

直接启动脚本同样支持 `ENABLE_DMD=false VIDEOEDIT_DUAL_CONFIG=... bash scripts/videoedit_dual_service/start.sh`。Docker 启动时显式传入的开关会保存在容器环境中，切换该值需要重建容器。

### 3.4 检查状态

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:$HOST_PORT/health" | python3 -m json.tool
docker exec "$CONTAINER_NAME" bash scripts/videoedit_dual_service/status.sh
docker logs --tail 100 "$CONTAINER_NAME"
```

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:$HOST_PORT/health" | python3 -m json.tool
docker exec videoedit_l40s bash scripts/videoedit_dual_service/status.sh
docker logs --tail 100 videoedit_l40s
```

`ok` 表示两个后端健康；`normal_only` 表示用户显式关闭 DMD 且 normal 健康；`unavailable` 表示某个已启用的后端不健康。三种状态均返回 HTTP 200，须检查响应内容。健康通过后按 §4–5 分别提交 normal 和 DMD 请求验收。

### 3.5 重启和停止

按需执行：

```bash
# 重新读取已选配置文件和挂载的源码。
docker restart "$CONTAINER_NAME"

# 停止整个容器。
docker stop "$CONTAINER_NAME"

# 启动已停止的容器。
docker start "$CONTAINER_NAME"
```

修改已选配置文件的内容后，重启即可读取；**切换配置文件、镜像、GPU、端口或挂载时，需要重跑 §3.1 和 §3.3 重建容器**。`docker restart` 不会更新这些容器创建参数。修改 GPU 数时，同时调整 `HOST_GPUS`、`CONTAINER_CUDA_VISIBLE_DEVICES` 和配置中的 GPU / 并行参数。

### 3.6 可选：直接管理服务进程

已有 Python / CUDA 环境、无需 Docker 时，使用 `VIDEOEDIT_DUAL_CONFIG`。以下命令为启动、状态和停止，按需执行：

```bash
export VIDEOEDIT_DUAL_CONFIG="$PWD/scripts/videoedit_dual_service/config.l20.env"
bash scripts/videoedit_dual_service/start.sh
bash scripts/videoedit_dual_service/status.sh
bash scripts/videoedit_dual_service/stop.sh
```

直接在宿主运行时，`CUDA_DEVICES` 应填写该环境实际可见的 GPU 编号。在已创建的容器中，`VIDEOEDIT_DUAL_CONFIG` 已设置，`docker exec` 无需重复指定。日常停止容器使用 `docker stop`；仅停止内部进程可能触发容器自动重启。

## 4. 本地 normal 请求：请求侧对齐口径

本地输入路径必须在容器内可读，输出目录也应位于持久挂载中。以下示例沿用 L20 的 `/root/VideoEdit` 路径；迁移机器或修改容器名、端口时，请同步替换请求和日志示例。

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
  "message": "accepted",
  "task_id": "videoedit-normal-local-001",
  "status": "dispatching",
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

normal 请求使用客户端给出的采样参数。normal 完成后，使用新的 `task_id`，将 `model` 改为 `videoedit-dmd` 再提交；分别检查任务完成、输出文件及视频解码。DMD 请求由 Gateway 覆盖为固定 4 步策略，因此不要用 DMD 结果验证 normal 的 40 步 golden。

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

取消当前任务：

```bash
curl --noproxy '*' -sS \
  -X DELETE "http://127.0.0.1:30000/v1/videos/${TASK_ID}" \
  | python3 -m json.tool
```

任务记录持久化，normal 和 DMD 共享一个执行名额，不支持排队。任务处于 `dispatching`、`running` 或 `cancelling` 时，新请求返回 HTTP 200、`code: 2`、`message: "A task is running."`，不保存新任务；空闲时返回 `code: 0`、`status: "dispatching"`。升级时，旧数据库里的 `queued` 任务会被标记为 `cancelled`，需要空闲后使用新 `task_id` 重新提交。重复提交同一个 `task_id` 返回 HTTP 409；目标 backend 不健康时返回 HTTP 503。接口定义见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L425)。

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
  bash -lc 'ls -lt "$VIDEOEDIT_REQUEST_LOG_DIR"'
```

## 9. 常见问题

### 请求立即失败并提示 removed fields

删除 `drop_reference_frame`、`dropReferenceFrame`、`chunks`、`generator_device`、`strength` 等已移除字段。这些语义已经固定在服务端，不再允许请求覆盖。

### health 是 normal_only 或 unavailable

`normal_only` 表示配置了 `ENABLE_DMD=false`，DMD 请求返回 HTTP 503。默认启用 DMD 时，DMD 故障不会自动切为 normal-only；检查权重校验结果、`dmd.log`、`dmd-resource.log`、`second-service-gate.json` 和 `dual-idle-gate.json`。容器脚本会在已启用的 DMD 进程退出后停止整个服务栈，Docker 根据 restart 策略处理后续重启。

### 请求返回 code: 2

Gateway 忙碌时拒绝新任务，不会自动排队。先查询 `/admin/queue` 和当前 active 任务，再查看对应 backend 日志。不要同时直接调用内部 `31100/32100` 端口绕过 Gateway。

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
- 双 backend 启动与开关：[`videoedit_dual_service/start.sh`](../scripts/videoedit_dual_service/start.sh)
- Gateway 路由、串行队列和任务接口：[`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py)
- API 字段、默认值和删除字段：[`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)
- 请求校验、下载、输出和回调：[`video_api.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py)
- 当前 strict 配置：[`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)
