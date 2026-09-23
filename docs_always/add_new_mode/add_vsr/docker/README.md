# VSR 原生 API Docker 部署

本目录提供当前 VSR 服务的镜像构建、单卡/双卡启动和 HTTP 验收脚本。**当前机器不能使用 Docker，因此尚未验证镜像构建、容器启动或容器内性能。** 已验证的是原宿主环境的算法/API；本次仅在宿主机做脚本语法、命令生成、版本和无 GPU 导入检查。

## 环境与文件

| 项目 | 配置 |
| --- | --- |
| 平台 | Linux x86_64 / Python 3.11.12 |
| PyTorch | 2.10.0+cu126，torchvision 0.25.0+cu126，Triton 3.6.0 |
| 模型库 | diffusers 0.37.0 / transformers 5.3.0 |
| 视频 | decord 0.6.0 / imageio 2.36.0 / imageio-ffmpeg 0.6.0 |
| Kernel | 本地针对 torch 2.10 编译的 sglang-kernel 0.4.1；FA3/FlashMLA 关闭 |
| API | `POST /v1/videos/restorations`，异步任务、查询、下载、回调 |
| 默认设备 | 单卡物理 GPU7；双卡物理 GPU6、GPU7 |

`Dockerfile` 使用 Python Debian 基础镜像，CUDA 12.6 用户态库由官方 PyTorch cu126 wheel 安装，不依赖容器内另装完整 CUDA toolkit。保留 C/C++ 编译器供 torch.compile 使用；NVIDIA 驱动由宿主机提供。目标运行机需要能运行 torch 2.10/cu126 的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。当前 kernel 在原有 H20 环境验证，其他 GPU 型号需要重新验收。

| 文件 | 用途 |
| --- | --- |
| `Dockerfile` / `Dockerfile.dockerignore` | 镜像与精简构建上下文，不携带权重、宿主虚拟环境或历史输出 |
| `requirements.txt` / `constraints.txt` | 服务依赖及核心版本约束 |
| `build.sh` | 校验 kernel SHA256、准备 wheel、构建镜像 |
| `run.sh` / `env.example` | 配置挂载、端口、单卡/双卡及共享内存 |
| `entrypoint.sh` / `check_runtime.py` | 检查模型路径、环境和可见 GPU，然后启动原生服务 |
| `smoke.sh` / `warmup.py` | 全分辨率 warmup 后执行已有 HTTP 验收 |
| `package_context.py` | 无需 Docker，将当前源码和 kernel 打成可转移构建包 |

复用 [`serve_vsr.py`](../scripts/serve_vsr.py) 中的 BF16、cuDNN benchmark、channels-last-3d、encoder/decoder 编译、decoder implicit padding 和 GPU 后处理。双卡叠加这些单卡优化；`num_gpus=1` 表示一个 SGLang 调度器，`tile_devices=[cuda:0,cuda:1]` 创建两份模型按 tile 并行。不要改成框架 `num_gpus=2`。

## 在当前机器准备可转移构建包

在仓库根目录运行，不调用 Docker：

```bash
python docs_always/add_new_mode/add_vsr/docker/package_context.py
```

输出 `output_results/vsr/vsr-docker-context.tar.gz`。包含当前工作区源码，包括尚未提交的 VSR API 改动，以及约 53 MB 的 kernel wheel 和 provenance。**权重、输入视频另行复制或通过共享文件系统挂载。** 脚本拒绝覆盖已有构建包，需要更新时通过 `--output /another/path.tar.gz` 指定新文件。

将包复制到 Docker 构建机器，解压到空目录：

```bash
mkdir -p vsr-build
tar -xzf vsr-docker-context.tar.gz -C vsr-build
cd vsr-build
bash docs_always/add_new_mode/add_vsr/docker/build.sh
```

也可在完整源码目录直接运行 `build.sh`，默认读取 `output_results/vsr/kernel210-wheel/`。通过 `VSR_KERNEL_WHEEL` 可改变 wheel 所在位置，但 SHA256 必须与已验证文件一致。不要替换成 PyPI 通用 kernel wheel。

构建需要联网拉取基础镜像和 Python 依赖。核心算法版本固定；其余依赖允许指定范围内解析，**不是完整锁定的离线环境**。原环境存在 requests 开发版、compressed-tensors 预发布版等拼接依赖，这里不复制它们，而由 pip 求解服务依赖；其 API 兼容性需要首次 Docker 构建与验收确认。构建会运行 `pip check`、核心版本断言以及 VSR 服务/模型导入检查，失败会终止。成功后的完整版本写在镜像 `/opt/vsr/installed-requirements.txt`，建议保留镜像 digest 和该文件，后续直接分发同一镜像。

源码通过 `PYTHONPATH` 使用，不执行 `pip install .[diffusion]`，避免仓库通用依赖中的 torch 2.9.1 覆盖本环境。该镜像只面向 VSR，不保证其他 SGLang 模型能力。系统安装 `ffmpeg/ffprobe` 以支持 VSR 默认保留源音轨，构建时检查命令可用性；音频开关及限制见 [音频保留](../audio.md)。

## 启动单卡

在完整仓库或解压后的构建目录：

```bash
cd docs_always/add_new_mode/add_vsr/docker
cp env.example .env
# 编辑 .env 中的绝对路径，随后加载：
set -a
source .env
set +a
bash run.sh --dry-run
bash run.sh
docker logs -f "$VSR_CONTAINER"
```

权重目录结构：

```text
VSR_CHECKPOINT_HOST/
  transformer_ema/          # 完整配置、权重及索引文件
  vae_decoder_ema.pt
VSR_WAN_HOST/
  vae/                     # 完整 Wan VAE 配置及权重
VSR_INPUT_HOST/
  input.mp4
```

模型和输入只读挂载；结果写到宿主 `VSR_OUTPUT_HOST/vsr/`，Inductor/Triton 缓存写到 `VSR_CACHE_HOST`。若权重使用跨目录软链接，应把真实目标一起放入挂载范围，或准备已展开链接的完整权重目录。大分辨率推理需要充足 CPU 内存和显存；`--shm-size=16g` 是容器共享内存上限，不代表进程总内存限额。

默认仅暴露 `127.0.0.1:30176`，通过内部反向代理接入；需要远程直连时显式设 `VSR_BIND_IP=0.0.0.0`。当前 API 没有内置鉴权，部署的访问控制由入口代理承担。容器内监听 `0.0.0.0:30176`，仅映射 HTTP 端口，内部调度端口无需开放。

## 切换双卡

先停止并移除旧容器，保留输出与缓存目录：

```bash
docker stop "$VSR_CONTAINER"
docker rm "$VSR_CONTAINER"
export VSR_MODE=dual VSR_GPUS=6,7
bash run.sh --dry-run
bash run.sh
```

Docker 限定暴露的两张卡，服务使用容器内 `cuda:0`、`cuda:1`。不要再设置 `CUDA_VISIBLE_DEVICES=6,7`。双卡每张卡持有完整模型，单请求 tile 并行；服务仍一次接收一个恢复任务，忙时返回业务码 2。

## API 与验收

健康检查通过仅表示 HTTP 服务可响应，**不表示编译和 warmup 已完成**。

```bash
curl -fsS http://127.0.0.1:30176/health
curl -fsS http://127.0.0.1:30176/v1/videos/restorations \
  -H 'Content-Type: application/json' \
  -d '{"taskId":"docker-demo-001","video_input_path":"/input/input.mp4","target_resolution":"3840x2160"}'
curl -fsS http://127.0.0.1:30176/v1/videos/docker-demo-001
# 待 status=completed：
curl -f http://127.0.0.1:30176/v1/videos/docker-demo-001/content -o result.mp4
```

`target_resolution` 是 **高×宽**。本地路径填写容器路径，不能填宿主路径。`videoUrl`、`callbackUrl` 必须从容器内可访问，容器中的 `127.0.0.1` 指容器自身。完整接口字段、业务码和限制见 [`server_api.md`](../server_api.md)。MinIO/S3 依赖已列入，但外部存储上传尚未验收。

服务空闲时执行：

```bash
export VSR_TEST_INPUT=/input/input.mp4
bash smoke.sh
```

验收会先对同一输入执行一次完整 `3840x2160` warmup，排除模型首推和首次形状编译；随后运行已有验收脚本的三次小分辨率预热和两次完整分辨率请求。双卡预热覆盖两卡。只比较预热后的两次请求；更换输入形状、tile 参数或 GPU 后需要重新 warmup。首次部署时总验收耗时包含编译，不能拿脚本总用时计算推理加速比。

测试在容器内启动临时视频下载/回调服务，覆盖本地输入、URL 输入、回调、输出下载、重复任务、忙时拒绝、超时及恢复；报告写到宿主输出目录 `acceptance-时间戳/`。`results.json` 的 `response.inference_time_s` 是服务端耗时，`request_wall_s` 包含每 2 秒轮询引入的误差。两次输出的 SHA256 一致性不等于与原算法的精度对齐；更换镜像环境后仍需用原参考视频验证 SSIM≥0.985、MSE≤36、MAE≤6。

任务状态保存在内存中，重启后不恢复任务记录；已写出的结果文件保留。停止使用 `docker stop`，入口会清理调度器及双卡子进程。编译缓存持久化能减少后续编译开销，但重启服务仍需 warmup。

## 本次验证范围

- Bash 语法、Python 语法、单卡/双卡命令 dry-run、重复 GPU 参数拒绝。
- 已验证 kernel wheel 的 SHA256。
- 宿主现有 torch 2.10/cu126 环境中的无 GPU 服务导入与版本检查。
- 未运行 Docker，也未在新依赖集合中构建/启动镜像；未进行本次容器 GPU 推理、性能或精度测试。
