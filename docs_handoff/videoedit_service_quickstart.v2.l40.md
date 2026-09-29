# VideoEdit 双模型服务快速使用说明（v2 / L40S）

> 章节和操作流程与 [L20 快速说明](./videoedit_service_quickstart.v2.l20.md) 保持一致。命令在宿主 Bash 终端执行；L40S 使用独立配置、容器 `videoedit_l40s`，默认入口为 `http://127.0.0.1:5402`。
>
> 本文按当前工作区整理配置；没有重新启动服务或执行推理。2026-09-21 的双卡部署与测试记录保留在附录 A，不能作为当前四卡配置的验收结果。

## 1. 当前版本必须先知道的变化

1. `strict_videoedit_math` 已在 VideoEdit 模型和所有 Transformer block 中固定为 `False`，不是 API 参数。当前 case0008、step_47500 的 crop/full golden 在该配置下均通过；客户端不要尝试发送这个字段。实现见 [`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)，验证证据见 [`performance-impact-review.md`](../docs_always/video-edit-compare/performance-impact-review.md#41-已关闭但保留对照严格-dit-数学路径)。
2. `drop_reference_frame` 已从请求协议删除，发送它或 `dropReferenceFrame` 会直接校验失败。当前算法语义固定为不删除参考帧。完整删除字段列表见 [`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L130)。
3. 请求接口默认 `num_inference_steps=40`、`decode_mode=stream`、`save_crop_only=false`、`enable_teacache=true`。数值 golden 请求应显式设置 `save_crop_only=true`、`enable_teacache=false`；默认值见 [`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py#L167)。
4. `videoedit-normal` 和 `videoedit-dmd` 共用配置指定的 GPU，并由网关全局串行调度；它们不是两个可并发执行的 GPU 服务。路由和队列实现见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L34) 与 [`start_videoedit_container.sh`](../scripts/start_videoedit_container.sh#L5)。
5. DMD 请求会被网关固定改为 4 步、`guidance_scale=1.0`、关闭 dynamic CFG，并清空 negative prompt；请求体中的对应值不会生效。实现见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L73)。

当前双服务 [`start.sh`](../scripts/videoedit_dual_service/start.sh#L99) 没有显式传入 `--attention-backend`。在支持 FlashAttention 的 CUDA 环境中会自动优先选择 FA，而现有 strict-false golden 是用 `torch_sdpa` 采集的。因此，下文请求可复现 API 侧参数，但不能单独保证完整 golden 环境；需要复现 golden 时，还应给两个 backend 的 `sglang serve` 命令增加 `--attention-backend torch_sdpa` 并重启。自动选择逻辑见 [`selector.py`](../python/sglang/multimodal_gen/runtime/layers/attention/selector.py#L115) 和 [`cuda.py`](../python/sglang/multimodal_gen/runtime/platforms/cuda.py#L381)。

## 2. 服务拓扑与配置

| 组件 | 地址/端口 |
| --- | --- |
| Gateway | 容器 `0.0.0.0:30000` → 宿主 `5402`，唯一对外入口 |
| normal backend | 容器内 `127.0.0.1:31100` |
| DMD backend | 容器内 `127.0.0.1:32100` |

两个 backend 共用 4 张 GPU，由 Gateway 串行执行任务；模型同时驻留在 GPU 和主存。

本指南使用 [`videoedit-l40s.0123.env`](../scripts/videoedit_dual_service/videoedit-l40s.0123.env)。启动前按本机情况检查其中的配置：

| 配置 | 本文设置 / 含义 |
| --- | --- |
| `CUDA_DEVICES` | `0,1,2,3`，容器内 GPU 编号 |
| `NUM_GPUS` / `SP_DEGREE` / `ULYSSES_DEGREE` / `RING_DEGREE` | `4 / 4 / 4 / 1` |
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
export IMAGE_NAME=sglang-videoedit-dev-v2:l40s
export CONTAINER_NAME=videoedit_l40s-v1
export HOST_PORT=5403 CONTAINER_PORT=30000
export HOST_GPUS=0,1,2,3 
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3
export DUAL_SERVICE_CONFIG_HOST="$PWD/scripts/videoedit_dual_service/config.l40s.0123.env"
export DUAL_SERVICE_CONFIG_CONTAINER=/sgl-workspace/sglang/scripts/videoedit_dual_service/config.l40s.0123.env
```

```bash
export IMAGE_NAME=sglang-videoedit-dev-v2:l40s
export CONTAINER_NAME=videoedit_l40s-v1
export HOST_PORT=5402 CONTAINER_PORT=30000
export HOST_GPUS=4,5,6,7 
export CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3
export DUAL_SERVICE_CONFIG_HOST="$PWD/scripts/videoedit_dual_service/config.l40s.4567.env"
export DUAL_SERVICE_CONFIG_CONTAINER=/sgl-workspace/sglang/scripts/videoedit_dual_service/config.l40s.4567.env
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

配置随仓库挂载进入容器，启动顺序为 normal → DMD → Gateway。每个后端的启动监测超时默认 `900` 秒；normal 失败会启动失败，DMD checkpoint 或资源检查失败可能降级为 normal-only。

### 3.4 检查状态

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:$HOST_PORT/health" | python3 -m json.tool
docker exec "$CONTAINER_NAME" bash scripts/videoedit_dual_service/status.sh
docker logs --tail 100 "$CONTAINER_NAME"
```

`ok` 表示两个后端健康；`degraded_normal_only` 表示仅 normal 健康；`unavailable` 表示 normal 不健康。三种状态均返回 HTTP 200，须检查响应内容。健康通过后按 §4–5 分别提交 normal 和 DMD 请求验收。

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
export VIDEOEDIT_DUAL_CONFIG="$PWD/scripts/videoedit_dual_service/config.l40s.0123.env"
bash scripts/videoedit_dual_service/start.sh
bash scripts/videoedit_dual_service/status.sh
bash scripts/videoedit_dual_service/stop.sh

export VIDEOEDIT_DUAL_CONFIG="$PWD/scripts/videoedit_dual_service/config.l40s.4567.env"
bash scripts/videoedit_dual_service/start.sh
bash scripts/videoedit_dual_service/status.sh
bash scripts/videoedit_dual_service/stop.sh
```

直接在宿主运行时，`CUDA_DEVICES` 应填写该环境实际可见的 GPU 编号。在已创建的容器中，`VIDEOEDIT_DUAL_CONFIG` 已设置，`docker exec` 无需重复指定。日常停止容器使用 `docker stop`；仅停止内部进程可能触发容器自动重启。

## 4. 本地 normal 请求：请求侧对齐口径

本地输入路径必须在容器内可读，输出目录也应位于持久挂载中。下面素材路径沿用本机 `/home/zhouhao6/VideoEdit/test/`，迁移时替换为实际路径；修改 §3.1 的端口或容器名时，也须同步替换后续示例。首次请求前确认视频、mask 和参考图存在且可解码。

```bash
curl --noproxy '*' -sS \
  -X POST http://127.0.0.1:5402/v1/videos/repairs \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "videoedit-normal-local-001",
    "model": "videoedit-normal",
    "timeout": -1,
    "prompt": "一个男人站在舞台中央演讲，背后有两排巨大的立体文字。",
    "video_input_path": "/home/zhouhao6/VideoEdit/test/1080.mp4",
    "mask_input_path": "/home/zhouhao6/VideoEdit/test/mask_1080_merged.mp4",
    "reference_image_path": "/home/zhouhao6/VideoEdit/test/local.png",
    "output_storage": "local",
    "output_path": "/home/zhouhao6/VideoEdit/test/output_normal_001.mp4",
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

如果显式设置 `save_crop_only=true`，会在主输出外额外生成 `/home/zhouhao6/VideoEdit/test/output_normal_001_crop_only.mp4`；这会增加一次 resize、视频编码、I/O 和磁盘占用。命名和写盘逻辑见 [`wan_videoedit_pipeline.py`](../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L592)。

服务还会写 `/home/zhouhao6/VideoEdit/test/output_normal_001.videoedit.json`，记录 bbox、帧数和窗口物化信息。元数据写盘逻辑见 [`wan_videoedit_pipeline.py`](../python/sglang/multimodal_gen/runtime/pipelines/wan_videoedit_pipeline.py#L528)。

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
  -X POST http://127.0.0.1:5402/v1/videos/repairs \
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
  "http://127.0.0.1:5402/v1/videos/${TASK_ID}" \
  | python3 -m json.tool
```

只查询进度：

```bash
curl --noproxy '*' -sS \
  "http://127.0.0.1:5402/v1/videos/${TASK_ID}/progress" \
  | python3 -m json.tool
```

查看 Gateway 队列：

```bash
curl --noproxy '*' -sS \
  'http://127.0.0.1:5402/admin/queue?limit=20' \
  | python3 -m json.tool
```

取消当前任务：

```bash
curl --noproxy '*' -sS \
  -X DELETE "http://127.0.0.1:5402/v1/videos/${TASK_ID}" \
  | python3 -m json.tool
```

任务记录持久化，normal 和 DMD 共享一个执行名额，不支持排队。任务处于 `dispatching`、`running` 或 `cancelling` 时，新请求返回 HTTP 200、`code: 2`、`message: "A task is running."`，不保存新任务；空闲时返回 `code: 0`、`status: "dispatching"`。升级时，旧数据库里的 `queued` 任务会被标记为 `cancelled`，需要空闲后使用新 `task_id` 重新提交。重复提交同一个 `task_id` 返回 HTTP 409；目标 backend 不健康时返回 HTTP 503。接口定义见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L425)。

Gateway 不代理 backend 的 `/content` 下载接口。本地输出完成后读取响应中的 `file_path`；S3/MinIO 输出读取 `url` 或 `output_object_key`。

不要在 active 任务存在时重启服务。Gateway 队列保存在 SQLite 中，但 backend 任务状态保存在进程内存；重启后 Gateway 可能找不到原 backend 任务并暂停队列，以避免重复执行。操作前先检查 `/admin/queue`；遇到 stale active 记录时，应先备份 `QUEUE_DB` 再人工处置，不要直接删除生产队列文件。恢复保护见 [`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py#L273)。

## 8. 日志与请求审计

查看容器聚合日志：

```bash
docker logs -f videoedit_l40s
```

分别查看组件日志：

```bash
docker exec videoedit_l40s \
  tail -f /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/normal.log

docker exec videoedit_l40s \
  tail -f /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/dmd.log

docker exec videoedit_l40s \
  tail -f /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/gateway.log
```

启动资源监控日志：

```bash
docker exec videoedit_l40s \
  tail -f /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/normal-resource.log

docker exec videoedit_l40s \
  tail -f /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/dmd-resource.log
```

资源与启动门禁记录位于：

```text
/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/normal-startup.json
/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/dmd-startup.json
/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/dual-idle-gate.json
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
docker exec videoedit_l40s \
  bash -lc 'ls -lt "$VIDEOEDIT_REQUEST_LOG_DIR"'
```

## 9. 常见问题

### 请求立即失败并提示 removed fields

删除 `drop_reference_frame`、`dropReferenceFrame`、`chunks`、`generator_device`、`strength` 等已移除字段。这些语义已经固定在服务端，不再允许请求覆盖。

### health 是 degraded_normal_only

normal 仍可用，但 `videoedit-dmd` 请求会返回 HTTP 503。检查 DMD checkpoint 校验结果、`dmd.log`、`dmd-resource.log` 和 `dual-idle-gate.json`。

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
- 双 backend 启动和降级：[`videoedit_dual_service/start.sh`](../scripts/videoedit_dual_service/start.sh)
- Gateway 路由、串行队列和任务接口：[`dual_service_gateway.py`](../python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py)
- API 字段、默认值和删除字段：[`protocol.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py)
- 请求校验、下载、输出和回调：[`video_api.py`](../python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py)
- 当前 strict 配置：[`wan_videoedit.py`](../python/sglang/multimodal_gen/runtime/models/dits/wan_videoedit.py#L84)

## 附录 A. 历史部署与测试记录（2026-09-21 UTC）

- 当时容器：`videoedit_l40s`；旧容器 `videoedit_reset` 保留，未删除。以下状态均为当日记录。
- 镜像：`sglang-videoedit-src:l40s`，ID `sha256:da597cfcbcfa9628bafd63ab9963353dfe2d325a0d0f1ec1ffd951ceefbdb8b9`。镜像由用户提前构建，本次完成启动与验收。
- GPU：宿主 4、5（容器内 0、1）；测试结束每卡占用 11134 MiB、空闲 34326 MiB。
- 入口：`http://127.0.0.1:5402`；宿主端口实际映射 `0.0.0.0:5402 → 30000`，同时有 IPv6 映射。
- 最终健康：`status=ok`，`normal=true`、`dmd=true`；队列 `completed=2`、`failed=0`、`queued=0`、`running=0`。
- 本机启动参数保存在 [start-container.sh](../.local/videoedit-l40s/start-container.sh)，配置为 [videoedit-l40s.0123.env](../scripts/videoedit_dual_service/videoedit-l40s.0123.env)  [videoedit-l40s.4567.env](../scripts/videoedit_dual_service/videoedit-l40s.4567.env)。启动脚本有同名容器保护；已有容器日常使用 `docker start/restart videoedit_l40s`。

| 模型 | 测试任务 ID | 接口报告推理耗时 | 结果 |
| --- | --- | --- | --- |
| normal | `l40s-smoke-normal-91181993171d` | 205.88 秒 | completed，H.264 / 1920×1080 / 9 帧 |
| DMD | `l40s-smoke-dmd-f13174a7f0ea` | 120.42 秒 | completed，H.264 / 1920×1080 / 9 帧 |

测试使用已有 105 帧视频及 mask 的前 9 帧，`infer_len=9`、`overlap=1`、`bridge_overlap=1`、4 步，开启 CLIP 和 paste-back、关闭 TeaCache。normal 的 CFG 为 5，DMD 按网关规则覆盖参数。两份输出均通过 ffprobe 帧数检查及 ffmpeg 全量解码检查。本次证明服务链路可用，不代表完整 105 帧、normal 40 步、并发队列或画质验收已完成。

产物及请求、状态记录位于 [.local/videoedit-l40s/test/](../.local/videoedit-l40s/test/)；汇总见 [results.json](../.local/videoedit-l40s/test/results.json)。复测命令（会提交新的两次 GPU 推理）：

```bash
docker exec videoedit_l40s python3 -u \
  /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/test/service_smoke.py
```

**冷启动记录：**第一次 normal 加载超过 900 秒，容器自动重启一次，没有 OOM 记录。加载进程曾处于文件页等待；顺序预读 normal 的 30.54 GiB 权重耗时 9.28 秒，随后第二次启动完成。DMD 某个约 4.65 GiB 分片顺序读耗时 142.27 秒，其余多数为 1–2 秒。文件读取耗时不均是已观测现象，尚未证实底层根因；未修改宿主驱动、模型文件或算法代码，也未增加启动超时。后续冷启动若再次超时，检查加载日志和存储 I/O，不要将此记录解读为问题已永久修复。

`.local/` 包含缓存、日志、队列和测试输出，应作为本机运行数据保留，不应整体提交到 Git。

### A.1 105 帧任务停在 99% 的修复与完整复测

任务 `videoedit-normal-l40s-3aae5381c0244f139d98398a52f4be55` 的最后一次去噪于
13:37:01 完成、VAE 解码于 13:38:09 完成、贴回后的元数据于 13:38:23 写出。
两个 rank 的调用栈随后均停留在 `_pil_frames_to_video_tensor()` 的 `np.stack()`。
进度只按去噪步数计算，120/120 步对应 99%，并不表示剩余耗时为 1%。

修复分两层：

- 文件输出且关闭插帧、超分时，只由 rank 0 将 PIL 帧交给视频编码器，直接返回
  `OutputBatch.output_file_paths`，不再整段构造 float32 视频张量。其余调用保留张量路径。
- 本机 `videoedit-l40s.0123.env` 设置 `VIDEOEDIT_DISABLE_THP=true`。启动脚本通过
  `disable_thp_exec.py` 调用 Linux `PR_SET_THP_DISABLE`，仅影响新启动的后端及其
  Python worker、FFmpeg 子进程，不修改宿主 `/sys` 设置。通用配置示例默认关闭此选项。

依据：仅绕过张量转换时，1080p 编码实验仍在 FFmpeg 中观察到
`try_to_migrate_one` 页迁移停顿；新进程禁用 THP 后，同样 105 帧、1920×1080、25 fps
合成视频通过修复后的输出路径约 2.17 秒写出，并通过 ffprobe 帧数、尺寸和帧率检查。
这是输出阶段的隔离验证，使用重复的合成帧，不代表实际编辑画面的编码耗时，
也不代表完整 40 步任务已重新验收。

2026-09-21 已按用户授权重启 `videoedit_l40s`，两个后端健康，4 个 scheduler 均实测
`THP_enabled: 0`。随后使用用户原始 105 帧、49 帧窗口、40 步、seed 42、dynamic CFG、
stream decode 和 paste-back 参数完成一次完整 normal 请求：

- 任务：`videoedit-normal-l40s-fixed-561be2b7a11a48b0a575cebd501fcc93`。
- 14:59:52 提交，15:39:05 的定时查询确认 `completed / 100`，观察耗时约 39 分 13 秒。
- 最后一轮去噪于 15:37:30 完成，解码于 15:38:39 完成，贴回等收尾于 15:38:54
  完成，15:38:57 视频写出。去噪结束至写出约 87 秒，其中最后一次解码约 69 秒、
  贴回等收尾约 15 秒、编码约 3 秒，未复现之前的长期停滞。
- ffprobe 确认 H.264、1920×1080、25 fps、105 帧；FFmpeg 全量解码通过。
- 抽查第 0、52、104 帧可见两排大字，但字形和颜色与参考图明显不同；本次确认
  输出链路稳定性，不能将其等同于文字准确性或完整画质验收。

请求、状态采样、日志、结果及输入/输出抽帧对照保存在
[full-retest](../.local/videoedit-l40s/full-retest/)；
[result.json](../.local/videoedit-l40s/full-retest/result.json) 记录检查详情。
推理期间每 3 分钟轮询一次，此次轮询直接从 92% 到完成，因此 99% 后耗时采用日志计算，
不声称轮询精确测得了 99% 的停留时间。未来重启仍会中断在途任务，不能从内存结果断点续跑。


## 附录 B. Devcontainer 镜像说明与检查

L20 / L40S 均使用 [`.devcontainer/Dockerfile`](../.devcontainer/Dockerfile)，构建命令见 §3.2。

| 项目 | 来源 / 设置 |
| --- | --- |
| 基础镜像 | `lmsysorg/sglang:nightly-dev-cu13-20260601-373cadc9`，默认固定 digest，支持 `BASE_IMAGE` 覆盖 |
| 基础镜像 digest | `sha256:0068fe3bf3f78f42d3d18314e8b0b6d7a721a4968b37a7d689257a6ea3915d36` |
| Python 依赖 | 继承基础镜像，额外安装 `ftfy==6.3.1`、`boto3==1.43.102`、`minio==7.2.20` |
| 安装环境 | 切换用户前通过基础镜像的 `python3 -m pip` 安装；不创建 `/opt/venv` |
| 系统工具 | sudo、ffmpeg / ffprobe、curl、flock（util-linux）、procps、zsh |
| 开发工具 | 在 `devuser` 下安装 uv、Rust |
| 镜像默认用户 | `devuser`；服务启动脚本默认使用 `--user root` |
| 构建参数 | `HOST_UID` / `HOST_GID` 默认均为 `1003`；`BASE_IMAGE` 默认如上 |
| 源码及配置 | 此 Dockerfile 不复制宿主源码及机型配置；启动时由宿主绑定挂载 |

2026-09-28 使用本地基础镜像 `sglang-base:20260601-local` 构建成功，最终镜像为 `sglang-videoedit-dev-v2:l40s`，ID 为 `sha256:5a738194cb3b16f3dcc7ab3760500ac660ac9743f93076398082068c6f9d2fe6`。实测依赖为 PyTorch `2.11.0+cu130`、`sglang-kernel 0.4.3`；挂载当前源码后，`fp8_blockwise_scaled_mm` 导入和 pipeline 模块查找均通过。检查中出现 `torchao` Tensor 对象导入警告，未进行模型加载或推理验收；现有服务容器未重启。该记录不代表 L20 已验收，也不代表镜像使用了 `sglang-kernel 0.4.2.post2`。

构建完成后，先挂载当前源码检查 CUDA 导入路径，无需加载模型：

```bash
docker run --rm --user root --gpus "\"device=${HOST_GPUS}\"" \
  -v "$PWD:/sgl-workspace/sglang:ro" \
  -e PYTHONPATH=/sgl-workspace/sglang/python \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 "$IMAGE_NAME" -B -c '
import importlib.util
import torch
import sgl_kernel
import sglang
print("source:", sglang.__file__)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("sgl_kernel:", sgl_kernel.__version__)
from sgl_kernel import fp8_blockwise_scaled_mm
spec = importlib.util.find_spec("sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline")
assert spec is not None
print(spec)
'
```

检查必须通过后再重建服务容器。`fp8_blockwise_scaled_grouped_mm` 是另一接口，不能根据 ImportError 的名称提示直接替换。也不要只在 CUDA 13 的 dev 镜像内降级 kernel：PyTorch、CUDA 扩展和其余依赖需要一起匹配。服务健康和双模型推理仍须按 §3.4、§4–5 验收。

需要保存环境记录时：

```bash
mkdir -p .local/videoedit-l40s/build
docker image inspect "$IMAGE_NAME" --format '{{.Id}}' \
  > .local/videoedit-l40s/build/image-id.txt
docker run --rm --user root "$IMAGE_NAME" python3 -m pip freeze \
  > .local/videoedit-l40s/build/pip-freeze.txt
```

附录 A 是旧镜像的历史测试记录，不能作为此构建入口的验收结果。构建镜像不会更新现有容器；使用新镜像需按 §3.3 重建。
