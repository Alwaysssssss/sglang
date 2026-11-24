# VSR 服务交接

## 当前实例

本机使用原生 SGLang VSR HTTP API，通过 Docker 容器 `vsr-l40s` 运行。

| 项目 | 当前配置 |
| --- | --- |
| 镜像 | `sglang-vsr:l40s` |
| 镜像 ID | `sha256:df3a9c71ff2130cccf71fb4f1e93c277f34834ece97a6ed04d91864e8dc34542` |
| GPU | 物理 GPU0，NVIDIA L40S；容器内为 `cuda:0` |
| HTTP 地址 | `http://127.0.0.1:30176` |
| 环境 | Python 3.12、Torch 2.9.1+cu129、Triton 3.5.1 |
| 模型精度 | BF16 |
| 并发 | 单恢复任务；忙时返回业务码 `2` |

服务只映射到宿主回环地址；远程机器可通过 SSH 转发访问：

```bash
ssh -N -L 30176:127.0.0.1:30176 用户名@服务器地址
```

## 模型、输入与输出

| 宿主路径 | 容器路径 | 用途 |
| --- | --- | --- |
| `/home/zhouhao6/vsr/model/best/checkpoint-1300` | `/models/checkpoint` | Transformer EMA 与 VAE decoder，只读 |
| `/home/zhouhao6/vsr/model/Wan2.2-TI2V-5B-Diffusers` | `/models/wan` | Wan VAE，只读 |
| `/home/zhouhao6/vsr/sglang/output_results/vsr/docker-input` | `/input` | 输入视频，只读 |
| `/home/zhouhao6/vsr/sglang/output_results/vsr/docker-output` | `/output` | 输出，读写 |
| `/home/zhouhao6/vsr/sglang/output_results/vsr/docker-cache` | `/cache` | 编译缓存，读写 |

请求中的本地路径填写**容器路径**。将视频放入宿主输入目录后，用 `/input/文件名.mp4` 提交。结果写入宿主输出目录的 `vsr/<taskId>.mp4`。

## 启动与维护

已有容器直接启动；启动前确认 GPU0 可用：

```bash
nvidia-smi
docker start vsr-l40s
docker logs --since 5m --tail 100 -f vsr-l40s
```

在另一个终端检查：

```bash
docker inspect vsr-l40s --format '{{.State.Status}} {{.State.Health.Status}}'
curl --noproxy '*' -fsS http://127.0.0.1:30176/health
```

模型加载期间端口可能暂时不可用。健康检查成功仅表示服务就绪，首次推理仍可能编译；更换视频形状后也可能重新编译。

```bash
# 停止前确认没有待完成任务
docker stop vsr-l40s
# 需要重启时
docker restart vsr-l40s
```

任务状态保存在内存中，重启后无法查询旧任务；已生成的视频和编译缓存保留。当前容器未配置开机自动重启，宿主重启后需要执行 `docker start vsr-l40s`。

## 提交、查询与下载

以下示例使用现有的 128×128、8 帧测试视频，输出 256×256。每次提交使用新的任务 ID：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:30176/v1/videos/restorations \
  -H 'Content-Type: application/json' \
  -d '{"taskId":"vsr-demo-001","video_input_path":"/input/smoke-generated.mp4","target_resolution":"256x256"}'

curl --noproxy '*' -fsS http://127.0.0.1:30176/v1/videos/vsr-demo-001

# 等待 status=completed 后下载
curl --noproxy '*' -f http://127.0.0.1:30176/v1/videos/vsr-demo-001/content \
  -o result.mp4
```

提交成功返回 `code=0`；`code=1` 为校验或提交失败，`code=2` 为服务忙碌。应检查 JSON 业务码，不能只看 HTTP 状态码。状态依次为 `queued`、`running`、`completed` 或 `failed`；提交后每隔几秒查询，失败时检查返回的错误信息及容器日志。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `taskId` | 可选；字母、数字、下划线、连字符，最长 128 字符，不能重复 |
| `video_input_path` / `videoUrl` | 本地容器路径或远程视频地址，二选一 |
| `target_resolution` | **高×宽**；横向 1920×1080 视频填 `1080x1920` |
| `long_edge` | 最长边尺寸；固定分辨率与最长边通常选择其一 |
| `preserve_audio` | 默认 `true`，保留源音轨 |
| `callbackUrl` | 可选终态回调地址，须从容器内可访问 |
| `timeout` | 默认 `-1`，正数为处理期限秒数 |

`GET /v1/videos/<id>/progress` 可查询状态和回调结果；`DELETE /v1/videos/<id>` 可请求取消，需等待后端停止。

## 重建实例与多卡配置

现有实例使用镜像内源码，宿主代码修改不会自动进入容器。需要更新源码时重新构建镜像并重建实例。具体步骤见 [L40S Docker 部署](../docs_always/add_new_mode/add_vsr/docker/videoedit_l40s.md)。

如果容器已被删除，使用现有镜像创建单卡实例：

```bash
cd /home/zhouhao6/vsr/sglang/docs_always/add_new_mode/add_vsr/docker
set -a
source env.videoedit.example
set +a
export VSR_NUM_GPUS=1 VSR_GPUS=0
bash run.sh --dry-run
bash run.sh
```

脚本不会覆盖同名容器。多卡可设置 `VSR_NUM_GPUS=2 VSR_GPUS=2,3`，选卡前检查占用，并为同时运行的实例分别设置容器名、端口和输出目录。每卡持有完整模型，tile 并行；框架仍使用一个调度器。多卡及高分辨率性能需要另行验证。

完整接口参数见 [VSR API](../docs_always/add_new_mode/add_vsr/server_api.md)，音频规则见 [音频保留](../docs_always/add_new_mode/add_vsr/audio.md)。

## 本次启动验证（2026-10-09）

- 已启动既有容器 `vsr-l40s`，使用 GPU0；Docker 健康状态为 `healthy`，`GET /health` 返回 `{"status":"ok"}`。
- 任务 `handoff-smoke-20261009-001` 提交返回 `code=0`，最终状态 `completed`；128×128、8 帧输入恢复为 256×256、8 帧输出。
- 服务端推理耗时约 48.09 秒。这是重启后首次请求，不作为稳态性能指标。
- 结果下载成功，下载副本与服务输出 SHA256 一致。验证记录与下载副本保存在 `output_results/vsr/service_check_20261009/`，服务输出为 `output_results/vsr/docker-output/vsr/handoff-smoke-20261009-001.mp4`。
- 本次验证为单卡小视频的启动、提交、查询、推理和下载；未执行完整 API 验收、高分辨率性能或参考视频精度对齐。服务保留运行。
