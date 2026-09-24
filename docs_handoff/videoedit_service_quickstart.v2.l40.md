# VideoEdit 双模型容器部署指南（本机 L40S / 5402）

> 更新日期：2026-09-21。仓库：`/home/zhouhao6/VideoEdit/sglang`，分支：`cos_l40s`。
>
> 服务已于 2026-09-21 在本机启动，并通过 normal / DMD 短视频冒烟测试。算法与请求参数规则参见 [v2 文档](./videoedit_service_quickstart.v2.md)。

## 0. 本次部署与测试结果（2026-09-21 UTC）

- 容器：`videoedit_l40s`，保持运行；旧容器 `videoedit_reset` 保留，未删除。
- 镜像：`sglang-videoedit-src:l40s`，ID `sha256:da597cfcbcfa9628bafd63ab9963353dfe2d325a0d0f1ec1ffd951ceefbdb8b9`。镜像由用户提前构建，本次完成启动与验收。
- GPU：宿主 4、5（容器内 0、1）；测试结束每卡占用 11134 MiB、空闲 34326 MiB。
- 入口：`http://127.0.0.1:5402`；宿主端口实际映射 `0.0.0.0:5402 → 30000`，同时有 IPv6 映射。
- 最终健康：`status=ok`，`normal=true`、`dmd=true`；队列 `completed=2`、`failed=0`、`queued=0`、`running=0`。
- 本机启动参数保存在 [start-container.sh](../.local/videoedit-l40s/start-container.sh)，配置为 [config.l40s.env](../scripts/videoedit_dual_service/config.l40s.env)。启动脚本有同名容器保护；已有容器日常使用 `docker start/restart videoedit_l40s`。

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

### 0.1 105 帧任务停在 99% 的修复与完整复测

任务 `videoedit-normal-l40s-3aae5381c0244f139d98398a52f4be55` 的最后一次去噪于
13:37:01 完成、VAE 解码于 13:38:09 完成、贴回后的元数据于 13:38:23 写出。
两个 rank 的调用栈随后均停留在 `_pil_frames_to_video_tensor()` 的 `np.stack()`。
进度只按去噪步数计算，120/120 步对应 99%，并不表示剩余耗时为 1%。

修复分两层：

- 文件输出且关闭插帧、超分时，只由 rank 0 将 PIL 帧交给视频编码器，直接返回
  `OutputBatch.output_file_paths`，不再整段构造 float32 视频张量。其余调用保留张量路径。
- 本机 `config.l40s.env` 设置 `VIDEOEDIT_DISABLE_THP=true`。启动脚本通过
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

## 1. 本机核对结果

| 项目 | 本次确认结果 |
| --- | --- |
| 双模型入口 | `scripts/start_videoedit_container.sh` 存在，默认镜像为 `sglang-mgtv:1.0` |
| 服务配置 | `scripts/videoedit_dual_service/config.env` 存在，路径仍为原机器 `/root/VideoEdit` |
| 本机专用文件 | 已新增 `docker/videoedit-l40s.Dockerfile` 及专用构建上下文过滤文件；`config.l40s.env` 已按 §4 生成 |
| 模型 | 基础模型及两个 transformer 已成功加载并完成推理 |
| 本地素材 | `/home/zhouhao6/VideoEdit/test/` 下已有 `1080.mp4`、`mask_1080_merged.mp4`、`local.png`；视频与 mask 均已确认 1920×1080、25 fps、105 帧 |
| 运行状态 | 经宿主访问权限验证：容器运行、GPU 正常、5402 双后端健康，见 §0 |

不沿用旧版中未经本次核实的“已上线”“镜像冒烟全绿”“GPU 4/5 空闲”等结论。下文命令均在具备 Docker 权限的**宿主 Bash 终端**执行。

## 2. 部署前检查

```bash
cd /home/zhouhao6/VideoEdit/sglang
id
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
docker image ls
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
free -h
df -h .
ss -ltn '( sport = :5402 )'
```

确认 Docker 可访问、宿主驱动正常、5402 可用，并确认分配给服务的两张 GPU。本文使用 `4,5` 作为示例，不表示这些卡目前空闲或已获分配。

若 Docker 报权限不足，检查登录用户的组权限。仅在用户已经获 docker 组权限、当前会话尚未刷新时，可重新登录或运行 `sg docker` 进入具有该组身份的 shell。不能据此宣称所有会话都无需提权。

### 2.1 拓扑与路径

| 项目 | 配置 |
| --- | --- |
| 容器名 | `videoedit_l40s` |
| Gateway | 容器 `0.0.0.0:30000` → 宿主 `5402`，唯一对外入口 |
| normal 后端 | 容器内 `127.0.0.1:31100` |
| DMD 后端 | 容器内 `127.0.0.1:32100` |
| GPU | 宿主示例 `4,5`，容器内 `0,1`；两个后端共用两张卡 |
| 本机运行数据 | 仓库内 `.local/videoedit-l40s/`，隔离历史队列 |

Gateway 串行调度请求。两后端均使用 `--num-gpus 2 --sp-degree 2 --ulysses-degree 2 --ring-degree 1`，开启 DiT 逐层 offload 和 T5 / CLIP / VAE CPU offload。不能只改 GPU 列表就切换到单卡运行。

启动脚本将 `/home/zhouhao6/VideoEdit` 同路径挂载进容器，并额外挂载仓库到 `/sgl-workspace/sglang`。请求可以直接使用挂载范围内的宿主绝对路径。内部通信端口不映射到宿主。

注意：启动脚本的 `PROJECT_ROOT` 是 `/home/zhouhao6/VideoEdit`（挂载根），服务配置文件中的 `PROJECT_ROOT` 是 `/home/zhouhao6/VideoEdit/sglang`（源码根）。

## 3. 准备镜像

```bash
# 后续命令沿用本终端中的变量；镜像名称按实际情况修改。
export IMAGE_NAME=sglang-mgtv:1.0
export HOST_GPUS=4,5
export CONTAINER_NAME=videoedit_l40s
docker image inspect "$IMAGE_NAME" --format '{{.Id}}'
```

若镜像不存在，先从原部署机器导出并导入，或从实际持有镜像的内部仓库获取。导出 / 导入示例：

```bash
# 在原部署机器的仓库目录执行，然后将归档传到本机仓库目录。
docker save -o sglang-mgtv-1.0.tar sglang-mgtv:1.0

# 在本机仓库目录执行。
docker load -i sglang-mgtv-1.0.tar
docker image inspect "$IMAGE_NAME" --format '{{.Id}}'
```

当前 `python/pyproject.toml` 声明 `torch==2.9.1`、`sglang-kernel==0.4.1`、`flashinfer_python==0.6.7.post2` 和 `flashinfer_cubin==0.6.7.post2`。镜像名称不能证明兼容性，必须执行 §4 的检查。

### 3.1 基于当前源码构建（无需原服务镜像）

使用本次新增的 [videoedit-l40s.Dockerfile](../docker/videoedit-l40s.Dockerfile)。它从 CUDA 开发镜像安装本地 `python[diffusion]`，包含当前工作区未提交的 Python 改动。这里的“源码构建”指 SGLang 使用当前源码，PyTorch 和 sglang-kernel 等依赖仍优先使用发行 wheel，不是将全部依赖从 C++ 源码编译。

构建配置：

| 项目 | 设置 |
| --- | --- |
| 基础镜像 | `nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04`，可用 `BASE_IMAGE` 覆盖 |
| Python | Ubuntu 24.04 的 Python 3.12，虚拟环境 `/opt/venv` |
| PyTorch | 先从 cu129 索引安装 `torch==2.9.1`、`torchaudio==2.9.1` |
| 应用依赖 | 安装当前 `python/pyproject.toml` 的 `diffusion` extra，不手工删减依赖 |
| CUDA 扩展 | `TORCH_CUDA_ARCH_LIST=8.9`，`MAX_JOBS=4`；面向 L40S |
| 媒体和进程工具 | ffmpeg / ffprobe、curl、flock、procps |
| CLI | 提供 `/usr/local/bin/sglang`，兼容当前服务配置 |

在仓库根目录执行：

```bash
cd /home/zhouhao6/VideoEdit/sglang
export IMAGE_NAME=sglang-videoedit-src:l40s
export HOST_GPUS=4,5  # 先核实实际分配
export CONTAINER_NAME=videoedit_l40s
mkdir -p .local/videoedit-l40s/build
# 记录工作区状态：revision 标签只记录 HEAD，不包含未提交改动。
git rev-parse HEAD > .local/videoedit-l40s/build/source-revision.txt
git status --short > .local/videoedit-l40s/build/source-status.txt
(
  set -euo pipefail
  docker build --progress=plain \
    -f docker/videoedit-l40s.Dockerfile \
    --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" \
    -t "$IMAGE_NAME" . \
    2>&1 | tee .local/videoedit-l40s/build/build.log
)
```

专用 `.dockerignore` 只发送 Python 源码和服务脚本，不发送模型、运行数据库和测试视频。本机 `config.l40s.env` 也不烘焙进镜像，启动时由宿主挂载。构建不需要把 GPU 传入 Docker；扩展仍可能进行较长时间的 CUDA 编译。

如果需要使用可访问的镜像仓库或包索引，可在上述 `docker build` 增加：

```text
--build-arg BASE_IMAGE=<可访问仓库中的同版本CUDA镜像>
--build-arg PIP_INDEX_URL=<可访问的Python包索引/simple>
--build-arg TORCH_INDEX_URL=<提供torch-2.9.1-cu129的索引>
```

这些是占位参数，不要原样执行。本指南没有验证任何代理站点的可达性。Docker 拉取基础镜像的代理由 Docker daemon 配置；构建步骤内的下载代理可通过 Docker 的 `HTTP_PROXY` / `HTTPS_PROXY` 构建参数传入。使用宿主回环代理时还需按实际构建器配置网络，不能假设容器的 `127.0.0.1` 就是宿主。

构建结束先检查**镜像自身**，本步骤不挂载宿主源码，防止挂载掩盖镜像打包问题：

```bash
docker run --rm "$IMAGE_NAME" bash -lc '
  set -euo pipefail
  python3 -m pip check
  python3 -c "import sglang; print(sglang.__file__)"
  test -x /usr/local/bin/sglang
  /usr/local/bin/sglang serve --help
  ffmpeg -version
  ffprobe -version
'
docker image inspect "$IMAGE_NAME" --format '{{.Id}}'
docker run --rm "$IMAGE_NAME" cat /opt/videoedit-build/pip-freeze.txt \
  > .local/videoedit-l40s/build/pip-freeze.txt
```

然后依次执行 **§4 配置与 GPU / 模型预检 → §5 启动 → §6 健康检查 → §7 双模型推理验收**，保持 `IMAGE_NAME=sglang-videoedit-src:l40s`。

如果替换已有容器，先按 §8 清空任务并停止、删除旧容器，再执行 §5；单纯 `docker restart` 不会切换到新镜像。构建新镜像本身不会修改正在运行的容器。

### 3.2 源码构建的边界与排错

- 镜像由用户构建；本次已验证当前镜像的源码导入、GPU 可见性及双模型推理，见 §0。新构建产物仍须重新预检和验收。
- 当前依赖包含 `st_attn`、`vsa` 等扩展。Dockerfile 先安装 torch，再用 `--no-build-isolation` 安装应用依赖，使扩展能找到 torch；这不保证所有扩展在 L40S 上都有可用内核。若失败，保留首次构建错误并检查对应包，不要用 `--no-deps` 跳过后认定成功。
- `pyproject.toml` 中部分依赖没有固定版本，因此相同源码在不同日期构建可能解析出不同依赖。`pip-freeze.txt` 记录实际环境，但不是带哈希的完整锁文件；稳定部署后应保存镜像 digest / 归档及依赖记录。
- 当前启动脚本会将宿主仓库挂载到镜像源码路径上。线上实际运行的是宿主源码；若修改依赖声明，需要重建镜像，只有 Python 代码变化时通常可重启生效。
- `.devcontainer/Dockerfile` 使用浮动的 `lmsysorg/sglang:dev`；`scripts/rebuild_image_create_videoedit_container.sh` 启动单后端，均不作为本文双模型重建入口。

## 4. 本机配置与镜像预检

生成本机配置副本，保留通用 `config.env`。运行文件全部放在仓库内；若本机配置已存在，下面的命令保留它，请人工检查路径：

```bash
cd /home/zhouhao6/VideoEdit/sglang
python3 - <<'PY'
from pathlib import Path
repo = Path.cwd()
source = repo / 'scripts/videoedit_dual_service/config.env'
target = source.with_name('config.l40s.env')
if target.exists():
    print(f'保留已有配置，请检查：{target}')
else:
    config = source.read_text().replace('/root/VideoEdit', '/home/zhouhao6/VideoEdit')
    config = config.replace('/home/zhouhao6/VideoEdit/tmp/sglang-videoedit-dual', str(repo / '.local/videoedit-l40s/dual'))
    config = config.replace('/home/zhouhao6/VideoEdit/tmp/sglang-videoedit-outputs', str(repo / '.local/videoedit-l40s/outputs'))
    target.write_text(config)
    print(f'已生成：{target}')
PY
mkdir -p .local/videoedit-l40s/{dual,inputs,outputs,request-logs,cache}
cat scripts/videoedit_dual_service/config.l40s.env
```

模型路径应为：

```text
BASE_MODEL=/home/zhouhao6/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
NORMAL_TRANSFORMER=/home/zhouhao6/VideoEdit/model/DifusserEdit/pretrain_models/VideoEdit-diffusers-model/transformer
DMD_TRANSFORMER=/home/zhouhao6/VideoEdit/model/DifusserEdit/merged_dit_lightx2v_lora_scale_1p0
```

使用选定镜像检查源码导入、GPU、媒体工具和模型，不启动推理服务：

```bash
docker run --rm -i --gpus "\"device=${HOST_GPUS}\"" --user root \
  -v /home/zhouhao6/VideoEdit:/home/zhouhao6/VideoEdit \
  -w /home/zhouhao6/VideoEdit/sglang \
  -e PYTHONPATH=/home/zhouhao6/VideoEdit/sglang/python \
  "$IMAGE_NAME" bash -s <<'CHECK'
set -euo pipefail
source scripts/videoedit_dual_service/config.l40s.env
nvidia-smi
command -v "$PYTHON_BIN"
test -x "$SGLANG_BIN"
command -v curl
command -v flock
command -v ffmpeg
command -v ffprobe
ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264|libx265'
"$PYTHON_BIN" - <<'PY'
import torch
import sglang
from importlib.metadata import version
from sgl_kernel import fused_add_rmsnorm, rmsnorm
import sglang.multimodal_gen.runtime.pipelines.wan_videoedit_pipeline
import sglang.multimodal_gen.runtime.videoedit.dual_service_gateway
print('sglang:', sglang.__file__)
for name in ('torch', 'sglang-kernel', 'flashinfer-python', 'flashinfer-cubin'):
    print(name, version(name))
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 2
PY
"$SGLANG_BIN" serve --help
"$PYTHON_BIN" scripts/videoedit_dual_service/resource_probe.py validate-transformer "$NORMAL_TRANSFORMER"
"$PYTHON_BIN" scripts/videoedit_dual_service/resource_probe.py validate-transformer "$DMD_TRANSFORMER"
test -r "$BASE_MODEL/model_index.json"
CHECK
```

确认帮助包含 `--dit-layerwise-offload`、`--num-gpus`、`--sp-degree`、`--ulysses-degree`、`--ring-degree`。若 CLI 不在 `/usr/local/bin/sglang`，将本机配置的 `SGLANG_BIN` 改为镜像中的实际绝对路径，`PYTHON_BIN` 也须对应同一 Python 环境。

GPU、依赖或权重检查失败时先修复再启动。预检通过仅表示具备基本启动条件，不代表推理通过。

## 5. 启动容器

**脚本默认会删除同名容器，即使没有设置 `RECREATE=1`。** 首次启动命令增加了同名容器检查；已有容器时先按 §8 查看队列，再决定重启或重建。

```bash
cd /home/zhouhao6/VideoEdit/sglang
(
  set -euo pipefail
  : "${IMAGE_NAME:?先完成镜像准备}"
  : "${HOST_GPUS:?先确认两张GPU的分配}"
  : "${CONTAINER_NAME:?先设置容器名}"
  docker image inspect "$IMAGE_NAME" >/dev/null
  test -r scripts/videoedit_dual_service/config.l40s.env
  existing="$(docker ps -aq -f "name=^/${CONTAINER_NAME}$")"
  if [ -n "$existing" ]; then
    echo '同名容器已存在，请先按第8节检查队列并决定重启或重建。' >&2
    exit 1
  fi
  env \
    IMAGE_NAME="$IMAGE_NAME" CONTAINER_NAME="$CONTAINER_NAME" \
    PROJECT_ROOT=/home/zhouhao6/VideoEdit \
    HOST_REPO_DIR=/home/zhouhao6/VideoEdit/sglang \
    WORKDIR_IN_CONTAINER=/home/zhouhao6/VideoEdit/sglang \
    DUAL_SERVICE_DIR_HOST=/home/zhouhao6/VideoEdit/sglang/scripts/videoedit_dual_service \
    DUAL_SERVICE_CONFIG_HOST=/home/zhouhao6/VideoEdit/sglang/scripts/videoedit_dual_service/config.l40s.env \
    DUAL_SERVICE_CONFIG_CONTAINER=/home/zhouhao6/VideoEdit/sglang/scripts/videoedit_dual_service/config.l40s.env \
    HOST_GPUS="$HOST_GPUS" CONTAINER_CUDA_VISIBLE_DEVICES=0,1 \
    HOST_PORT=5402 CONTAINER_PORT=30000 \
    INPUT_SAVE_DIR=/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/inputs \
    VIDEOEDIT_OUTPUT_DIR=/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/outputs \
    VIDEOEDIT_REQUEST_LOG_DIR=/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/request-logs \
    VIDEOEDIT_REQUEST_LOG_SENSITIVE_VALUES=false \
    CACHE_DIR=/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/cache \
    bash scripts/start_videoedit_container.sh
)
```

`FLASHINFER_WORKSPACE_BASE` 和 `XDG_CACHE_HOME` 默认由 `CACHE_DIR` 派生。容器以 root 运行，生成文件可能属于 root。

启动顺序是 normal → DMD → Gateway。`STARTUP_TIMEOUT=900` 是每个后端的启动监测超时，不是整个服务的总超时；等待期间查看日志，不要反复重建。

normal 失败会导致启动失败；DMD 失败时降级为 normal-only。配置默认跳过第二服务的预测门禁，但保留双服务空闲门禁：GPU 余量 4 GiB、宿主和 cgroup 余量各 40 GiB。门禁通过不保证所有分辨率的推理都不会 OOM。

## 6. 健康验收与日志

以下命令按默认容器名编写，若自定义了名称，请一并替换。

```bash
curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:5402/health | python3 -m json.tool
docker exec videoedit_l40s bash scripts/videoedit_dual_service/status.sh
docker exec videoedit_l40s nvidia-smi
docker logs --tail 200 videoedit_l40s
```

| `status` | 含义 |
| --- | --- |
| `ok` | normal 和 DMD 均健康，下一步执行推理验收 |
| `degraded_normal_only` | 仅 normal 健康，尚未完成双模型部署 |
| `unavailable` | normal 不健康，不可验收 |

三种状态均返回 HTTP 200，不能只看 curl 退出码，须检查 `status` 和 `backends`。

```bash
docker exec videoedit_l40s tail -n 100 /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/normal.log
docker exec videoedit_l40s tail -n 100 /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/dmd.log
docker exec videoedit_l40s tail -n 100 /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual/logs/gateway.log
```

脚本将 5402 发布到宿主所有接口。另从访问方机器请求 `http://<宿主IP>:5402/health` 验证网络；本地成功、远端失败时检查监听、路由和防火墙。

## 7. 本地推理验收

先检查视频与 mask 可解码、帧数一致，并确认参考图有效：

```bash
for name in 1080.mp4 mask_1080_merged.mp4; do
  docker exec videoedit_l40s ffprobe -v error -select_streams v:0 \
    -count_frames -show_entries stream=width,height,r_frame_rate,nb_read_frames \
    -of json "/home/zhouhao6/VideoEdit/test/$name"
done
```

下面根据 v2 的 normal 示例提交本机素材；自动生成唯一任务 ID，输出写入仓库运行目录。`num_frames=-1` 处理完整视频，执行前应了解素材长度和资源需求。

```bash
TASK_ID="videoedit-normal-l40s-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export TASK_ID
python3 - <<'PY' | curl --noproxy '*' -fsS \
  -X POST http://127.0.0.1:5402/v1/videos/repairs \
  -H 'Content-Type: application/json' --data-binary @-
import json
import os
print(json.dumps({
    'task_id': os.environ['TASK_ID'],
    'model': 'videoedit-normal',
    'timeout': -1,
    'prompt': '两行字幕固定在人物后方。',
    'video_input_path': '/home/zhouhao6/VideoEdit/test/0008/video.mp4',
    'mask_input_path': '/home/zhouhao6/VideoEdit/test/0008/mask.json',
    'reference_image_path': '/home/zhouhao6/VideoEdit/test/0008/reference.png',
    'output_storage': 'local',
    'output_path': '/home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/outputs/' + os.environ['TASK_ID'] + '.mp4',
    'num_frames': -1, 'ref_frame_idx': 0,
    'num_inference_steps': 40, 
    'seed': 1785978278, 
    'bbox_expand_scale': 0.3,
    'dilate_px': 8, 'mask_scale': 1.0, 'feather_px': 8
}))
PY
```

返回 `queued` 仅代表入队。在同一个终端查询：

```bash
curl --noproxy '*' -fsS "http://127.0.0.1:5402/v1/videos/${TASK_ID}" | python3 -m json.tool
curl --noproxy '*' -fsS "http://127.0.0.1:5402/v1/videos/${TASK_ID}/progress" | python3 -m json.tool
curl --noproxy '*' -fsS 'http://127.0.0.1:5402/admin/queue?limit=20' | python3 -m json.tool

# 仅在需要取消该任务时执行。
curl --noproxy '*' -fsS -X DELETE "http://127.0.0.1:5402/v1/videos/${TASK_ID}" | python3 -m json.tool
```

normal 完成后，将请求中的 `model` 改为 `videoedit-dmd`，重新生成任务 ID 后再提交；DMD 参数覆盖规则按 v2 文档执行。验收要求两种模型均完成任务，返回的输出文件存在，并通过 `ffprobe` 及播放检查。

远程输入、`output_storage=s3` 和 `minio_config` 使用 v2 文档示例，将入口端口替换为 5402，并填写实际存储配置。不要在文档中保存密钥。

## 8. 停止、重启和重建

先查看 `/admin/queue`，确认没有执行中或待处理任务。Gateway 的 SQLite 队列持久化，后端任务状态在内存中；执行中重启可能导致队列暂停。不要删除数据库来绕过问题，也不要复用历史队列作为首次部署配置。

```bash
# 停止整个容器。
docker stop videoedit_l40s

# 启动已停止的容器。
docker start videoedit_l40s

# 重启并重新读取挂载的源码和服务配置。
docker restart videoedit_l40s
```

修改挂载的 Python 代码或服务配置通常只需重启；修改镜像、端口、GPU、挂载或容器环境变量需要重建。重建前完成预检、记录旧镜像 ID 和部署参数，确认队列已清空，再明确执行 `docker rm -f videoedit_l40s`，随后重跑 §5。运行数据保存在宿主目录中，不随容器删除。

脚本使用 `--restart unless-stopped`。正常停止使用 `docker stop`；仅杀容器内进程可能触发自动重启。

## 9. 常见问题与验证边界

| 现象 | 检查与处理 |
| --- | --- |
| 镜像不存在 | 按 §3 准备镜像，不能只改名称假定镜像可用 |
| 导入失败 / CLI 不识别参数 | 按 §4 检查 Python、CLI 路径与依赖版本 |
| GPU 容器启动失败 | 检查宿主驱动和 NVIDIA runtime；若 CDI 引用了不存在的驱动库，由宿主管理者检查并刷新 CDI。当前脚本不支持 `GPU_MODE=legacy` |
| DMD 降级 | 查看 `dmd.log`、`dmd-resource.log`、`dual-idle-gate.json`，检查权重与资源门禁 |
| Gateway 长时间不可达 | 查看后端加载日志、启动超时与容器是否反复重启 |
| 请求审计目录为空 | 当前 `start.sh` 未传入 `--videoedit-request-log-dir`，仅设置环境变量不会开启审计；开启需另行修改启动参数，并保持敏感值记录关闭 |
| 与历史 golden 不一致 | 当前脚本未固定 attention backend；依赖、后端和算法参数需另行核对，健康检查不证明数值等价 |

本次已完成源码与路径核对、GPU 检查、宿主 5402 健康验证及双模型短视频端到端测试；完整视频、正式步数和画质验收仍需另行执行，范围见 §0。
