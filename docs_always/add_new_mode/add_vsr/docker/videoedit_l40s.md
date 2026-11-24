# 基于 sglang-videoedit-src:l40s 的 VSR 容器全流程

本流程复用本机镜像的 Python 3.12、Torch 2.9.1+cu129、torchvision 0.24.1、Triton 3.5.1 和 sglang-kernel 0.4.1，补齐 decord 0.6.0、imageio-ffmpeg 0.6.0 和 accelerate 1.12.0，复制当前仓库 VSR 源码。无需旧方案的 Torch 2.10 kernel wheel，也不安装仓库的整套 diffusion 依赖。基础镜像已有 ffmpeg/ffprobe。

## 1. 准备

在 Docker 宿主机的 Bash 中操作。需要 NVIDIA 驱动、可用的 NVIDIA Container Toolkit、模型权重，以及下载新增 Python wheel 的网络。基础镜像的 CUDA 驱动要求会由容器运行时检查。

```bash
cd /home/zhouhao6/vsr/sglang
docker image inspect sglang-videoedit-src:l40s
nvidia-smi
cd docs_always/add_new_mode/add_vsr/docker
cp env.videoedit.example .env
# 编辑 .env：模型、输入、输出、缓存路径必须是宿主机绝对路径。
set -a
source .env
set +a
mkdir -p "$VSR_INPUT_HOST" "$VSR_OUTPUT_HOST" "$VSR_CACHE_HOST"
# 将待处理视频复制到 "$VSR_INPUT_HOST/input.mp4"
```

权重结构：

```text
VSR_CHECKPOINT_HOST/
  transformer_ema/       # 配置和完整权重
  vae_decoder_ema.pt
VSR_WAN_HOST/
  vae/                  # 配置和完整权重
```

跨目录符号链接的目标必须在挂载范围内。模型和输入只读挂载；输出、编译缓存持久化到宿主路径。已有同名容器不会被脚本自动删除。

## 2. 构建镜像

```bash
bash build_videoedit.sh --dry-run
bash build_videoedit.sh
```

默认生成 `sglang-vsr:l40s`。可通过 `VSR_BASE_IMAGE` 和 `VSR_IMAGE` 覆盖基础与目标标签。构建会检查核心版本、VSR 服务导入和 ffmpeg，并记录 `/opt/vsr/installed-requirements.txt`。继承基础镜像的依赖组合，不宣称整个环境通过 `pip check`；旧的 `build.sh` / `Dockerfile` 仍用于 Torch 2.10 路线。

## 3. 选择 GPU 数量并启动

只指定数量时，从宿主 GPU0 开始选择 N 张卡；不会自动寻找空闲卡，运行前用 `nvidia-smi` 确认。

```bash
# 单卡 GPU0
export VSR_NUM_GPUS=1
unset VSR_GPUS VSR_MODE
bash run.sh --dry-run
bash run.sh
```

双卡或四卡示例（每次启动一个新容器；切换已有容器见下一节）：

```bash
# 任意两张卡：物理 GPU2、GPU3
export VSR_NUM_GPUS=2 VSR_GPUS=2,3
bash run.sh

# 四张卡：物理 GPU0、GPU1、GPU2、GPU3
export VSR_NUM_GPUS=4
unset VSR_GPUS
bash run.sh
```

参数规则：

- `VSR_NUM_GPUS` 指定数量；`VSR_GPUS` 可选，逗号分隔物理卡号，数量必须一致且不能重复。
- 未设置数量时，可从 `VSR_GPUS` 推断；兼容旧的 `VSR_MODE=single/dual`。显式数量优先。
- Docker 限定可见 GPU；服务使用容器内 `cuda:0 ... cuda:N-1`，不要把宿主卡号再次写入 `CUDA_VISIBLE_DEVICES`。
- 一个 SGLang 调度器，`num_gpus=1` 保持不变；N≥2 时自动生成 `--tile-devices`，每张卡驻留完整模型。增加卡数不降低每卡模型内存，实际收益取决于 tile 数量和通信成本。
- 启动检查可见卡数、每卡 SDPA 和 kernel 导入。N 卡配置支持不等于所有卡数均已完成性能、精度验收。

默认访问 `http://127.0.0.1:30176`。远程直连可设置 `VSR_BIND_IP=0.0.0.0`；端口用 `VSR_HOST_PORT` 修改。API 无内置鉴权，请按部署环境配置入口访问控制。

## 4. 检查、提交任务、下载

```bash
docker logs --tail 100 -f "$VSR_CONTAINER"
# 另一个终端，或 Ctrl-C 退出日志跟踪后：
curl --noproxy '*' -fsS "http://127.0.0.1:$VSR_HOST_PORT/health"
docker exec "$VSR_CONTAINER" nvidia-smi
curl --noproxy '*' -fsS "http://127.0.0.1:$VSR_HOST_PORT/v1/videos/restorations" \
  -H 'Content-Type: application/json' \
  -d '{"taskId":"vsr-demo-001","video_input_path":"/input/input.mp4","target_resolution":"1080x1920"}'
curl --noproxy '*' -fsS "http://127.0.0.1:$VSR_HOST_PORT/v1/videos/vsr-demo-001"
# status=completed 后：
curl --noproxy '*' -f "http://127.0.0.1:$VSR_HOST_PORT/v1/videos/vsr-demo-001/content" -o result.mp4
```

分辨率是 **高×宽**。输入路径是容器内路径。HTTP 健康不等于模型首推/编译已完成；首次任务需要等待。任务提交后每隔数秒查询状态，不要重复提交同一 taskId。结果位于宿主输出目录的 `vsr/` 下；任务记录在内存中，重启不保留。

需要完整 API 验收时，在服务空闲且已准备真实输入视频后：

```bash
export VSR_TEST_INPUT=/input/input.mp4
bash smoke.sh
```

该脚本包含完整分辨率预热及多项 API 请求，会花费明显长于单次推理的时间。多于两卡时，较小预热形状不保证覆盖所有 worker；应以实际目标分辨率的预热为准。性能比较使用预热后请求，并另行做参考视频精度对齐。接口详见 [server_api.md](../server_api.md)。

## 5. 切换卡数、停止与分发

确认旧容器没有待完成任务后，停止并删除容器，再按第 3 节设置卡数启动（输出和缓存保留）：

```bash
docker stop "$VSR_CONTAINER"
docker rm "$VSR_CONTAINER"
export VSR_NUM_GPUS=2 VSR_GPUS=2,3
bash run.sh
```

需要同时保留多个实例，分别设置 `VSR_CONTAINER`、`VSR_HOST_PORT`、GPU 卡号和输出目录，避免端口与资源冲突。

```bash
docker image inspect "$VSR_IMAGE" --format '{{.Id}}'
docker exec "$VSR_CONTAINER" cat /opt/vsr/installed-requirements.txt
docker save "$VSR_IMAGE" | gzip > sglang-vsr-l40s.tar.gz
# 目标机器：gunzip -c sglang-vsr-l40s.tar.gz | docker load
```

权重和输入不在镜像内，需要另行传输。目标机加载镜像后仍需配置挂载和 GPU；更新源码后重新构建并重建容器。

## 本机验证记录（2026-09-30）

- 实际构建成功：`sglang-vsr:l40s`，镜像 ID `sha256:df3a9c71ff2130cccf71fb4f1e93c277f34834ece97a6ed04d91864e8dc34542`。
- 容器 `vsr-l40s` 使用 GPU0，端口 `127.0.0.1:30176`，Docker 健康状态为 `healthy`，保留运行。
- 自动生成 128×128、8 帧测试视频；任务 `l40s-build-smoke-001` 输出 256×256、8 帧，提交、查询、推理、下载均通过。输出 `/output/vsr/l40s-build-smoke-001.mp4`，下载副本为宿主 `output_results/vsr/docker-output/smoke-download.mp4`。
- 首次请求服务端耗时 130.43 秒，包含编译，不能用作预热后性能指标。
- 在物理 GPU1、GPU2 上完成双卡可见性、每卡 SDPA 和 kernel 导入检查；未完成多卡视频推理、完整 `smoke.sh`、高分辨率或参考视频精度验收。
- Bash 语法、diff 空白检查、4 组合法 GPU 配置和 5 组非法 GPU 配置检查通过。
- 首次模型启动发现缺少 accelerate，已补入镜像并重建，最终启动和推理通过。

## 当前双卡实例（2026-10-09）

当前容器 `vsr-l40s` 采用物理 GPU6、GPU7，容器内部为 `cuda:0`、`cuda:1`，自动使用 `--tile-devices cuda:0 cuda:1`。宿主端口映射为 `127.0.0.1:5409 → 容器 30176`；API 地址为 `http://127.0.0.1:5409`。镜像、模型、输入、输出和缓存路径沿用原实例。

双卡配置保存为 [env.videoedit.gpu67.example](env.videoedit.gpu67.example)。创建新容器时使用：

```bash
cd /home/zhouhao6/vsr/sglang/docs_always/add_new_mode/add_vsr/docker
set -a
source env.videoedit.gpu67.example
set +a
bash run.sh --dry-run
bash run.sh
```

现有容器直接启动或查询：

```bash
docker start vsr-l40s
curl --noproxy '*' -fsS http://127.0.0.1:5409/health
docker logs --tail 100 -f vsr-l40s
```

提交示例：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:5409/v1/videos/restorations \
  -H 'Content-Type: application/json' \
  -d '{"taskId":"vsr-gpu67-demo-001","video_input_path":"/input/smoke-generated.mp4","target_resolution":"320x1216"}'
```

原单卡容器保留为已停止的 `vsr-l40s-single-backup-20261009`；端口调整前的双卡容器保留为已停止的 `vsr-l40s-dual-30176-backup-20261009`。两者仅供回退，不要与当前实例同时启动。任务记录在内存中，重建后旧任务无法查询，持久化输出仍保留。

宿主仍仅绑定回环地址；远程访问使用 SSH 转发：

```bash
ssh -N -L 5409:127.0.0.1:5409 用户名@服务器地址
```

本次启动验证通过：Docker 状态 `running healthy`，`GET /health` 返回 `{"status":"ok"}`；任务 `gpu67-port5409-smoke-20261009-001` 完成，输入 128×128、8 帧，输出高 320×宽 1216、8 帧。日志显示 2 个空间 tile，两张指定 GPU 均加载模型。服务端耗时约 110.78 秒，包含首次编译，不作为稳态性能指标。结果下载成功，与服务输出 SHA256 一致。验证记录保存在 `output_results/vsr/gpu67-port5409-check/`；本次未执行高分辨率性能或参考视频精度验收。服务保留运行。
