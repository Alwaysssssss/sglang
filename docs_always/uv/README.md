# SGLang uv 开发环境安装指南（LLM + Diffusion）

> 目标：在当前 SGLang 仓库中，用 `uv` 建立一个同时支持 **LLM / VLM serving 开发** 和 **SGLang Diffusion 开发** 的 Python 环境。
>
> 本文以仓库当前配置为准：`python/pyproject.toml` + `python/uv.lock`。

---

## 1. 适用场景

这份文档适合以下开发工作：

- 修改 `python/sglang/` 下的 SGLang Runtime、OpenAI API、调度、模型执行、sampling、cache、分布式等 LLM/VLM 逻辑。
- 修改或调试 `docs/diffusion/`、`python/sglang/srt/` 中 diffusion 相关入口、pipeline、模型加载、缓存、offload、attention backend 等逻辑。
- 需要同一个 editable 开发环境同时具备：
  - SGLang 基础依赖；
  - `dev/test` 依赖；
  - `diffusion` extra 依赖；
  - 当前锁文件固定的 Torch / Transformers / Diffusers / FlashInfer / sglang-kernel 版本。

如果只做文档或纯静态代码阅读，可以不安装完整 GPU 环境；如果要运行 LLM 或 diffusion 服务，建议在 NVIDIA GPU + CUDA 12.x 环境中安装。

---

## 2. 前置条件

### 2.1 系统依赖

推荐环境：

- Linux x86_64。
- NVIDIA GPU，驱动可支持 CUDA 12.x 运行时。
- `git`、`curl`、`gcc/g++`、`ninja`、`cmake` 等基础构建工具。
- 足够的磁盘空间：Torch、CUDA wheel、diffusion/video 依赖与模型缓存会占用较多空间。

Ubuntu/Debian 可先安装基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake ninja-build pkg-config
```

如需处理视频、图片或 diffusion 调试，建议也安装：

```bash
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
```

### 2.2 安装 uv

官方安装方式见 Astral uv 文档：<https://docs.astral.sh/uv/>。

Linux/macOS：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重新加载 shell，或手动加入 PATH：

```bash
source ~/.bashrc 2>/dev/null || true
uv --version
```

如果不能使用安装脚本，也可以临时用 pip 安装：

```bash
python3 -m pip install --user uv
python3 -m uv --version
```

---

## 3. 仓库路径与核心配置

从仓库根目录开始：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
```

当前 Python 包在 `python/` 目录：

```text
python/
├── pyproject.toml      # SGLang Python 包、依赖、extras、uv index/source 配置
├── uv.lock             # 当前锁定依赖版本
└── sglang/             # Python 源码
```

关键 extras：

| extra | 用途 | 说明 |
| --- | --- | --- |
| 默认依赖 | LLM/VLM serving 基础环境 | 包含 `torch==2.9.1`、`transformers==5.3.0`、`flashinfer_python==0.6.7.post2`、`sglang-kernel==0.4.1` 等当前锁定依赖。 |
| `dev` | 开发/测试环境 | 当前定义为 `sglang[test]`，包含 `pytest`、`accelerate`、`lm-eval`、`peft` 等测试开发依赖。 |
| `diffusion` | SGLang Diffusion 环境 | 包含 `diffusers==0.37.0`、`moviepy`、`opencv-python-headless`、`cache-dit`、`vsa`、`st_attn` 等 diffusion/video 依赖。 |
| `all` | diffusion + tracing | 当前包含 `sglang[diffusion]` 和 `sglang[tracing]`，但不包含 `dev/test`。 |

推荐开发安装使用 `dev + diffusion`，而不是只用 `all`。

---

## 4. 推荐安装：同一环境开发 LLM + Diffusion

> 推荐优先使用仓库根目录脚本 `./install_uv_env.sh` 完成安装；本节同时保留等价的手工 `uv` 命令，方便排障和自定义。

### 4.1 进入 Python 项目目录

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
```

### 4.2 选择 Python 版本

`pyproject.toml` 要求 `>=3.10`。推荐使用 Python 3.11 或 3.12，优先选与团队/CI 一致的版本。

用 uv 安装并固定一个项目 Python：

```bash
uv python install 3.11
uv python pin 3.11
```

这会在 `python/` 下生成或更新 `.python-version`。如果不希望提交该文件，请按团队习惯处理。

### 4.3 同步依赖

推荐把 uv 虚拟环境建到 `/home`，避免在仓库所在盘生成大型 `.venv`。`UV_PROJECT_ENVIRONMENT` 可以指定当前项目环境路径：

```bash
mkdir -p /home/$USER/uv-envs
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
```

如果当前 shell 的 `$USER` 不符合预期，先显式设置用户名或直接传 `--env-dir`：

```bash
export USER=$(id -un)
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
```

等价的根目录脚本命令：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
./install_uv_env.sh --env-dir "$UV_PROJECT_ENVIRONMENT"
```

然后同步依赖：

```bash
uv sync --locked --extra dev --extra diffusion
```

含义：

- `--locked`：只使用当前 `uv.lock`，不重新解析或改写锁文件。
- `--extra dev`：安装开发/测试依赖。
- `--extra diffusion`：安装 SGLang Diffusion 依赖。
- `UV_PROJECT_ENVIRONMENT`：把虚拟环境放到指定目录，而不是默认的 `python/.venv`。
- 默认项目会以 editable 方式安装，因此修改 `python/sglang/` 源码后无需重新 `pip install -e .`。

如果你明确需要 tracing：

```bash
uv sync --locked --extra dev --extra diffusion --extra tracing
```

或使用脚本：

```bash
./install_uv_env.sh --with-tracing
```

如需强制重建环境：

```bash
rm -rf "$UV_PROJECT_ENVIRONMENT"
uv sync --locked --extra dev --extra diffusion
```

为了避免误删，脚本重建环境需要显式确认：

```bash
CONFIRM_DELETE_ENV=1 ./install_uv_env.sh --recreate
```

### 4.4 激活环境

```bash
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
python -V
python -c "import sglang, torch; print('sglang', sglang.__file__); print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

也可以不激活，直接使用：

```bash
uv run python -c "import sglang, torch; print(torch.__version__)"
```

---

## 5. 验证 LLM 开发环境

### 5.1 基础 import 验证

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
uv run python - <<'PY'
import torch
import transformers
import sglang
import sglang.srt
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('sglang:', sglang.__file__)
print('srt import: ok')
PY
```

### 5.2 启动一个 LLM 服务

示例：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
uv run python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

另一个终端请求：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32
  }'
```

如果机器没有 GPU，可只做 import 与单元测试；完整 serving 需要按模型大小准备显存。

---

## 6. 验证 Diffusion 开发环境

### 6.1 基础 import 验证

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
uv run python - <<'PY'
import diffusers
import imageio
import cv2
import moviepy
import sglang
print('diffusers:', diffusers.__version__)
print('imageio:', imageio.__version__)
print('cv2:', cv2.__version__)
print('sglang:', sglang.__file__)
print('diffusion env import: ok')
PY
```

### 6.2 启动 diffusion 服务

具体模型参数以 `docs/diffusion/api/cli.md` 和对应模型文档为准。基础形态如下：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
uv run python -m sglang.launch_server \
  --model-path <diffusion-model-or-local-path> \
  --model-type diffusion \
  --host 0.0.0.0 \
  --port 30000
```

常见调试入口：

- `docs/diffusion/api/cli.md`：diffusion CLI、backend、overlay repo、diffusers fallback、Cache-DiT 参数。
- `docs/diffusion/performance/`：diffusion 性能与并行相关文档。
- `docs_always/multimodal_gen/`：本仓库中关于多模态生成架构、pipeline、runtime、loader、优化的长期笔记。

---

## 7. 本地开发 sgl-kernel（可选）

默认 Python 环境会安装 `pyproject.toml` 锁定的 `sglang-kernel==0.4.1` wheel，适合大多数 SGLang Python 开发。

如果你正在修改仓库内 `sgl-kernel/`，需要改用本地 editable/install 版本。通常流程是先安装上面的 Python 环境，再进入 kernel 子项目构建：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/sgl-kernel
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pip install -e . --no-build-isolation
```

验证：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
uv run python - <<'PY'
import sgl_kernel
print(sgl_kernel.__file__)
PY
```

注意：

- 本地 kernel 编译依赖 CUDA 编译工具链、GPU 架构、PyTorch/CUDA ABI，失败时先确认 `nvcc --version`、`torch.version.cuda` 与驱动兼容性。
- 如果只是开发 Python 层 LLM/diffusion 逻辑，不建议优先切本地 kernel，避免引入额外编译变量。

---

## 8. 常用开发命令

所有命令默认从 `python/` 目录运行：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
```

### 8.1 运行指定测试

```bash
uv run pytest path/to/test_file.py -q
```

示例：

```bash
uv run pytest sglang/test/test_utils.py -q
```

仓库级测试也可从根目录指定虚拟环境 Python：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
"$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest test/srt -q
```

### 8.2 临时安装调试包

优先不要直接改锁文件。临时调试可使用：

```bash
uv pip install <package>
```

如果确定要纳入项目依赖，需要修改 `python/pyproject.toml` 后重新锁定：

```bash
uv lock
uv sync --extra dev --extra diffusion
```

提交前确认 `python/pyproject.toml` 与 `python/uv.lock` 是否都应该进入变更。

### 8.3 查看依赖来源

```bash
uv tree | less
uv pip show torch diffusers transformers sglang sglang-kernel flashinfer-python
```

### 8.4 清理环境和缓存

只重建当前项目环境：

```bash
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
rm -rf "$UV_PROJECT_ENVIRONMENT"
uv sync --locked --extra dev --extra diffusion
```

清 uv 全局缓存需谨慎：

```bash
uv cache clean
```

---

## 9. 常见问题

### 9.1 `uv sync --locked` 提示 lockfile 不匹配

说明 `python/pyproject.toml` 与 `python/uv.lock` 不一致。开发者应先判断是否需要更新依赖：

- 只是想复现当前环境：不要改依赖，恢复相关文件后再执行 `uv sync --locked ...`。
- 确实新增/升级依赖：在 `python/` 目录运行 `uv lock`，再 `uv sync --extra dev --extra diffusion`，并提交 `pyproject.toml`/`uv.lock` 的合理变更。

### 9.2 Torch/CUDA 安装不符合预期

当前 `pyproject.toml` 配置了 `torch-cu129` index，并对部分平台设置了 `tool.uv.sources`。如果安装后 CUDA 不可用，先检查：

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
nvidia-smi
```

如果 `nvidia-smi` 正常但 PyTorch CUDA 不可用，优先核对当前平台、Python 版本、lockfile 解析结果和 uv index 配置。

### 9.3 下载 NVIDIA CUDA wheel 失败

现象示例：

```text
Failed to download `nvidia-cuda-nvrtc-cu12==12.8.93`
Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).

Failed to download `nvidia-cufft-cu12==11.3.3.83`
Failed to write to the distribution cache
error decoding response body
peer closed connection without sending TLS close_notify
```

根因：`nvidia-cuda-nvrtc-cu12`、`nvidia-cufft-cu12` 等是 `torch==2.9.1` 依赖的大型 CUDA wheel。默认 `UV_HTTP_TIMEOUT=30` 秒，在当前网络或 PyPI/CDN 抖动时，wheel 下载或解压阶段容易超过 30 秒，被 uv 主动中断；即使把超时调大，远端也可能在大文件传输中提前关闭 TLS 连接，导致 `peer closed connection without sending TLS close_notify` 或半截 wheel 写入 uv cache。这不是 `sglang` 代码错误，也不是 `/home/$USER/uv-envs/sglang-llm-diffusion` 路径错误。

解决：使用安装脚本的较长超时和自动重试。脚本失败后会清理 NVIDIA CUDA wheel 的 uv 缓存条目，再重新执行 `uv sync`：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
./install_uv_env.sh --http-timeout 3000 --sync-retries 5
```

等价手工命令：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
export UV_HTTP_TIMEOUT=3000
uv cache clean nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12
uv sync --locked --extra dev --extra diffusion
```

如果上一次失败留下了不完整环境，先安全重建：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
CONFIRM_DELETE_ENV=1 ./install_uv_env.sh --recreate --http-timeout 3000 --sync-retries 5
```

### 9.4 `flash-attn`、`st_attn`、`vsa` 或 kernel wheel 安装失败

这些包和 CUDA/PyTorch/平台强相关。排查顺序：

1. 使用推荐 Python 版本重新创建 `/home/$USER/uv-envs/sglang-llm-diffusion`。
2. 确认 `uv sync --locked --extra dev --extra diffusion` 没有改写 lockfile。
3. 确认系统有编译工具、`ninja`、CUDA 编译/运行库。
4. 在 ARM/aarch64 平台上注意 `pyproject.toml` 中部分 diffusion 包带有平台 marker，行为会和 x86_64 不同。

### 9.5 Diffusion import 正常但运行模型失败

常见原因不是环境安装本身，而是：

- 模型路径不是 SGLang 支持的 diffusion repo 或 diffusers/componentized repo。
- 模型需要 `--backend diffusers`、overlay materialize 或 custom pipeline。
- 显存不足，需要降低分辨率、帧数、batch，或启用 offload/cache 相关配置。
- 本地缺少模型文件、权限或 Hugging Face token。

优先查看 `docs/diffusion/api/cli.md` 中的模型加载、backend、overlay 和 Cache-DiT 说明。

---

## 10. 推荐工作流

### 10.0 使用根目录脚本

当前仓库根目录提供了两个便捷脚本：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
./install_uv_env.sh
./start_sglang_uv.sh --model-path Qwen/Qwen2.5-0.5B-Instruct
```

启动 diffusion：

```bash
./start_sglang_uv.sh --model-type diffusion --model-path <diffusion-model-or-local-path>
```

脚本默认把环境建到 `/home/$USER/uv-envs/sglang-llm-diffusion`，可用 `--env-dir` 或 `UV_PROJECT_ENVIRONMENT` 覆盖。

安装脚本常用参数：

| 命令 | 作用 |
| --- | --- |
| `./install_uv_env.sh` | 安装 Python 3.11、固定 `.python-version`、同步 `dev + diffusion`。 |
| `./install_uv_env.sh --python 3.12` | 使用 Python 3.12 创建环境。 |
| `./install_uv_env.sh --env-dir /home/$USER/uv-envs/sglang-llm-diffusion` | 显式指定默认环境目录。 |
| `./install_uv_env.sh --http-timeout 3000 --sync-retries 5` | 大型 CUDA wheel 下载慢或 TLS 中断时，提高超时并自动重试。 |
| `./install_uv_env.sh --with-tracing` | 额外安装 `tracing` extra。 |
| `CONFIRM_DELETE_ENV=1 ./install_uv_env.sh --recreate` | 删除并重建目标环境。 |

启动脚本常用参数：

| 命令 | 作用 |
| --- | --- |
| `./start_sglang_uv.sh --model-path Qwen/Qwen2.5-0.5B-Instruct` | 启动 LLM/VLM 服务。 |
| `./start_sglang_uv.sh --model-type diffusion --model-path <model>` | 启动 diffusion 服务。 |
| `./start_sglang_uv.sh --model-path <model> --host 127.0.0.1 --port 30001` | 指定监听地址和端口。 |
| `./start_sglang_uv.sh --model-path <model> -- --tp 1 --log-level info` | `--` 后面的参数原样传给 `sglang.launch_server`。 |

### 10.1 首次安装

推荐使用脚本：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
./install_uv_env.sh
```

等价手工命令：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
uv python install 3.11
uv python pin 3.11
uv sync --locked --extra dev --extra diffusion
uv run python -c "import sglang, torch, diffusers; print('ok')"
```

### 10.2 每天开发前同步

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
git pull
./install_uv_env.sh
```

### 10.3 修改依赖时

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
# edit pyproject.toml
uv lock
uv sync --extra dev --extra diffusion
uv tree | less
```

### 10.4 提交前检查

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/python
export UV_PROJECT_ENVIRONMENT=/home/$USER/uv-envs/sglang-llm-diffusion
uv run python -c "import sglang, torch, diffusers; print('import ok')"
uv run pytest <changed-related-tests> -q
```

---

## 11. 参考文件

- `python/pyproject.toml`：项目依赖、extras、uv index/source 配置。
- `python/uv.lock`：当前锁定依赖版本。
- `docs/diffusion/api/cli.md`：SGLang Diffusion CLI 与 backend 说明。
- `docs/diffusion/performance/`：SGLang Diffusion 性能相关文档。
- `docs/developer_guide/`：SGLang 开发与 benchmark 文档。
- uv 官方文档：<https://docs.astral.sh/uv/>。
