# VSR 环境文档

本文记录两端（基线 `vsr` 与 SGLang）实际可运行的环境、模型权重、输入数据和命令行。
版本一律以实际导入的解释器为准，不采信系统 Python、另一环境的 `pip list` 或旧文档结论。

最近核实：2026-09-20。历史环境信息保留如下；本次恢复的解释器见 [本次验收记录](acceptance_20260920.md#环境恢复)。
旧 `/home/root/uv-envs/sglang-llm-diffusion/bin/python` 当前不存在，改用 `$SGLANG_REPO/output_results/vsr/migration_env/bin/python`。

## 1. 两套环境

### 1.1 基线环境 `swiftvr`（conda）

```bash
conda activate swiftvr
# 等价于直接调用
VSR_PYTHON=/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python
```

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu126 |
| Decord | 0.6.0 |
| Diffusers | 0.36.0 |
| imageio | 2.37.2 |
| imageio-ffmpeg | 0.6.0 |
| OpenCV | 5.0.0 |
| NumPy | 1.26.0 |
| Transformers | 5.2.0 |
| Safetensors | 0.7.0 |

### 1.2 SGLang 环境（uv）

```bash
export VE_SGLANG_PYTHON="$SGLANG_REPO/output_results/vsr/migration_env/bin/python"
# 2026-09-18 原路径 /home/root/uv-envs/sglang-llm-diffusion/bin/python 已不存在。
```

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.12 |
| PyTorch | 2.9.1+cu128 |
| Diffusers | 0.37.0 |
| sglang | `0.0.0.dev11368`（editable，指向本仓库 `python/sglang`） |
| imageio | 2.36.0 |
| imageio-ffmpeg | 0.6.0（2026-09-18 由 0.5.1 升级，见 §3） |
| OpenCV | 4.10.0 |
| NumPy | 2.4.6 |
| Transformers | 5.3.0 |
| Decord | 0.6.0（2026-09-18 安装） |

两点操作注意：

- 该环境是 uv 环境，**没有 `pip` 模块**。安装依赖走
  `uv pip install --python "$VE_SGLANG_PYTHON" <pkg>`。
- `sglang` 是 **editable 安装，直接指向本仓库的 `python/sglang`**，仓库内改动即刻生效，
  无需重装。`sglang.multimodal_gen` 可正常导入。
- `decord` 已于 2026-09-18 安装（`uv pip install --python "$VE_SGLANG_PYTHON" decord`，
  解析到 `decord-0.6.0-py3-none-manylinux2010_x86_64.whl`，Python 版本无关）。
  基线的解码全部依赖 decord，第一阶段沿用 decord，不做解码器替换（`requirements.md` §5）。

> 注：`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang` 是另一个已失效的 venv
> （`bin/python` 指向不存在的 `/root/miniconda3/envs/qwen3_vl/bin/python3.11`），**不要使用**。

### 1.3 GPU

`8 × NVIDIA A100-SXM4-80GB`（`nvidia-smi` 实测）。基线示例命令用 `CUDA_VISIBLE_DEVICES=3` 指定单卡。

### 1.4 两端差异（不只是 torch）

| | 基线 `swiftvr` | SGLang |
| --- | --- | --- |
| torch | `2.10.0+cu126` | `2.9.1+cu128` |
| **diffusers** | **`0.36.0`** | **`0.37.0`** |
| OpenCV | `5.0.0` | `4.10.0` |
| NumPy | `1.26.0` | `2.4.6` |
| 捆绑 ffmpeg | `v7.0.2` | `v7.0.2`（已对齐，见 §3） |

**diffusers 版本差异（`0.36.0` / `0.37.0`）已做静态核对，结论是「在 VSR 的用法下应等价」，但仍需实测确认。**
第一阶段两端都使用 diffusers 的 `AutoencoderKLWan` 与 `WanTransformer3DModel`，因此该版本差异会进入
`requirements.md` §4.1.2 的 `ε_torch`——若不成立，`ε_torch` 就不再是纯 torch 差异，容差将失去意义。

逐文件核对两端 `site-packages/diffusers/`：

| 文件 | 差异 |
| --- | --- |
| `models/autoencoders/autoencoder_kl_wan.py` | 仅类型注解现代化（`Optional[X]` → `X \| None` 等），无数值路径改动 |
| `models/transformers/transformer_wan.py` | 除注解外有三处实质重构，逐条判定见下 |

`transformer_wan.py` 的三处实质差异：

1. **`parallel_config` 传递方式**（[PR #12909](https://github.com/huggingface/diffusers/pull/12909)）。
   `0.36.0` 两处调用都传 `self._parallel_config`；`0.37.0` 改为 `encoder_hidden_states` 非空时传 `None`。
   但 `_parallel_config` 在两版中都是**类属性 `None`**（`transformer_wan.py:70`），VSR 单卡且无分布式上下文，
   两个分支取值都是 `None`，**行为一致**。若将来启用分布式注意力，此处会分叉。
2. **`is_cross_attention` 的引入。** `0.37.0` 新增该参数，并把
   `if self.cross_attention_dim_head is None` 改为 `if not self.is_cross_attention`；
   未显式传入时 `is_cross_attention = cross_attention_dim_head is not None`，**默认语义与原式等价**。
3. **LoRA API 改名**（`USE_PEFT_BACKEND` / `scale_lora_layers` → `apply_lora_scale`）。
   VSR 不使用 LoRA、不传 `lora_scale`，**无影响**。

另有 `fused_projections` 分支依赖 `fuse_projections()` 被显式调用，VSR 不调用，同样无影响。

**结论**：静态核对下这三处差异在 VSR 的调用方式（单卡、无 LoRA、无 fused projections、
`cross_attention_dim_head` 已设置）中都不改变数值。

**并且无法把 diffusers 从变量中消掉**：`sglang` 的 `diffusion` extra 硬性要求
`diffusers==0.37.0`（`importlib.metadata.requires('sglang')` 实测），因此不能把 SGLang 环境
固定到基线的 `0.36.0`。`requirements.md` §4.1.2 的 `ε_torch` 因此按 torch + diffusers 两项差异的
合计实测值采纳，M0 结果为 `ssim_min 0.988793 / mse_max 2.7160 / mae_max 1.1283`，53/53 帧通过门限。

**捆绑的 ffmpeg 原为 `v4.2.2`（基线 `v7.0.2`），已于 2026-09-18 对齐，见 §3。**

OpenCV 与 NumPy 的版本差异暂不单独处理：

- OpenCV 不产生跨端不对称——`compare.py` 是独立工具，在同一个环境下解码两端的产物。
- NumPy 在数据路径上（`to_uint8_hwc` 输出的 uint8 数组经它交给 imageio）。基线代码的几何运算
  全部在 torch 上做，numpy 只承担搬运；但 `1.26.0` 与 `2.4.6` 跨了大版本（NEP 50 等语义变更），
  M0 在 SGLang 环境下跑基线时会一并暴露实际问题，届时再处理。

## 2. 资产路径

```bash
export VSR_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/vsr
export SGLANG_REPO=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
export VSR_PYTHON=/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python
export VSR_WAN_ROOT=/mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers
export VSR_CHECKPOINT=/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300
export VSR_INPUT=$VSR_REPO/input/input.mp4
export VSR_OUTPUT_ROOT=$SGLANG_REPO/output_results/vsr
```

### 2.1 模型权重

检查权重配对：

```bash
test -s "$VSR_WAN_ROOT/vae/config.json"
test -s "$VSR_CHECKPOINT/transformer_ema/config.json"
test -s "$VSR_CHECKPOINT/vae_decoder_ema.pt"
```

| 路径 | 内容 | 实测 |
| --- | --- | --- |
| `$VSR_WAN_ROOT` | 完整 diffusers 目录：`model_index.json`、`vae/`、`transformer/`、`text_encoder/`、`tokenizer/`、`scheduler/` | 存在 |
| `$VSR_WAN_ROOT/vae` | 基础 VAE（encoder 权重 + `latents_mean`/`latents_std`） | `config.json` 1701 B |
| `$VSR_CHECKPOINT/transformer_ema/` | 微调 EMA DiT | `config.json` 592 B |
| `$VSR_CHECKPOINT/vae_decoder_ema.pt` | 微调 EMA VAE decoder，**裸 state_dict** | `1 110 143 159 B`（约 `1.03 GiB`） |

`$VSR_CHECKPOINT` 下同时存在 `transformer/` 与 `vae_decoder.pt`（非 EMA 回退），以及训练产物
`ema_state.pt`、`trainer_state.json`、`discriminator_engine/`、`generator_engine/` 等。
加载规则见 `infer/models/stage3.py:from_pretrained`：**优先 EMA，EMA 缺失时回退非 EMA**。

两个必须注意的点：

- `$VSR_WAN_ROOT` 里还有一个普通 DiT（`transformer/`），**不是**微调权重。DiT 一律来自
  `$VSR_CHECKPOINT`。
- EMA DiT 必须与 `vae_decoder_ema.pt` 配对使用，不能与 `vae_decoder.pt` 混搭。

`$VSR_WAN_ROOT` 中的 `text_encoder/` 与 `tokenizer/` **本管线不使用**——文本条件是全零张量，
不是文本编码器的输出（见 `requirements.md` §7-3）。

### 2.2 测试视频

```bash
export VSR_TEST_MEDIA=/mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/zy_test
```

| 路径 | 宽×高 | T | fps | pix_fmt |
| --- | --- | --- | --- | --- |
| `$VSR_REPO/input/input.mp4` | 1080×1918 | 53 | 25 | `yuv422p10le` |
| `$VSR_TEST_MEDIA/val_input_480x832.mp4` | 832×480 | 54 | 25 | `yuv422p10le` |
| `$VSR_TEST_MEDIA/val_input_512x512.mp4` | 512×512 | 54 | 25 | `yuv422p10le` |
| `$VSR_TEST_MEDIA/val_input_768x1280.mp4` | 1280×768 | 54 | 25 | `yuv422p10le` |

文件名里的尺寸是**高×宽**（`480x832` = `H480 × W832`，即横屏 832×480），与
`--target_resolution` 的 `H×W` 语义一致。

`$VSR_REPO/input/input.mp4` 与 `zhouyang/SwiftVR/zy_test/dog2_1080P.mp4` 是**同一个文件**
（md5 `12c49181d5559847e71ac877d4b19d4c`），后者是 `vsr/stream.log` 那次实际运行的输入。

探测方式：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,nb_frames,pix_fmt \
  -of json "$VSR_INPUT"
```

素材与参数的完整覆盖矩阵见 `requirements.md` §3.1 / §3.2。注意**现有素材全部只有 2 个时间窗口**，
三窗口路径需要循环拼接长片。

## 3. ffmpeg 与编码

### 3.1 现状：两端已对齐

`imageio.get_writer(..., codec="libx264")` 默认使用 `imageio_ffmpeg` **自带**的二进制。

| 来源 | 路径 | 版本 |
| --- | --- | --- |
| 系统 | `/usr/bin/ffmpeg` | `8.0.1`（未被 imageio 使用） |
| 基线捆绑 | `…/swiftvr/…/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2` | `7.0.2` |
| SGLang 捆绑 | `…/sglang-llm-diffusion/…/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2` | `7.0.2` |

**2026-09-18 修正**：升级前 SGLang 环境的 `imageio-ffmpeg` 是 `0.5.1`，自带 `ffmpeg-linux64-v4.2.2`
（ffmpeg `4.2.2`），与基线的 `v7.0.2` 相差三个大版本，libx264 编码结果必然不同。已执行：

```bash
uv pip install --python "$VE_SGLANG_PYTHON" "imageio-ffmpeg==0.6.0"
```

修正依据：两端 `imageio_ffmpeg` 的 Python 代码**逐字节相同**（`_io.py`、`_utils.py`、`_parsing.py`、
`__init__.py` 均 0 行差异），唯一差异在 `_definitions.py` 的 `FNAME_PER_PLATFORM`——
`0.5.1` 指向 `ffmpeg-linux64-v4.2.2`，`0.6.0` 指向 `ffmpeg-linux-x86_64-v7.0.2`。
因此升级只换二进制，不换命令行构造逻辑。

验证结果：两端 `imageio_ffmpeg` 包现已**逐字节一致**，二进制 md5 同为
`d5c698e0e98e5cf7de03fe14bd1cda6a`。

```bash
$VE_SGLANG_PYTHON -c "import imageio_ffmpeg; print(imageio_ffmpeg.__version__, imageio_ffmpeg.get_ffmpeg_exe())"
# 0.6.0 …/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
```

### 3.2 对验收的影响

编码器已不再是跨端变量，`requirements.md` §4.2.3 的「编码器确定性检查」按原样执行即可：
若该检查通过，则「mp4 差异 ⇒ 帧差异」成立，§4.2.2 的 mp4 门限可解释。

仍需在 `requirements.md` §3 的可复现记录中记下 ffmpeg 版本，因为它是输出的一部分。

若日后再出现两端编码器不一致的情况，两种处理方式：**(a)** 统一指定 `IMAGEIO_FFMPEG_EXE`
指向同一个二进制；**(b)** 保留差异并把编码器差异当作待测误差 `ε_encoder` 纳入容差——
后者会让 mp4 门限更难解释，仅在无法统一时才用。

## 4. 命令行

### 4.1 基线：流式（唯一验收基线）

```bash
CUDA_VISIBLE_DEVICES=3 "$VSR_PYTHON" -B "$VSR_REPO/run_inference_stream.py" \
  --input "$VSR_INPUT" \
  --output "$VSR_OUTPUT_ROOT/reference_stream.mp4" \
  --checkpoint_dir "$VSR_CHECKPOINT" \
  --wan_root "$VSR_WAN_ROOT" \
  --target_resolution 3840x2160 \
  --tile_h 320 --tile_w 640 --tile_t 33 \
  --color_ref global --read_queue 2
```

`3840x2160` 是**高×宽**（`H×W`），不是常见的 `W×H`。输入是竖屏 1080×1918，输出 3840×2160
即 H=3840、W=2160，同样是竖屏。

`color_ref` 的另外两档：`chunk`（纯流式，每个 chunk 对齐自身输入）、`none`（关闭颜色校正）。
`requirements.md` §3.2 的矩阵要求三档各跑一次。

`global` 会先做一次颜色统计预扫描：53 帧输入在默认 `color_ref_samples=64` 下会采样全部 53 帧，
但不会把完整视频留在内存里。

`run_inference.py`（整段视频路径）**不是验收基线**，不要混用两个入口的默认参数。

### 4.2 基线：环境自检

```bash
"$VSR_PYTHON" -B -c '
import sys, torch, decord, diffusers, imageio
print(sys.executable)
print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(diffusers.__version__, decord.__version__, imageio.__version__)
'

"$VE_SGLANG_PYTHON" -c '
import sys, torch, diffusers
print(sys.executable)
print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(diffusers.__version__)
import sglang.multimodal_gen
print("multimodal_gen OK")
'
```

自检须确认 GPU 可用、decord 能探测输入、imageio 能创建 x264 writer，且输入、权重可读、输出目录可写。

### 4.3 SGLang 侧（第一阶段，已实现）

形态见 `requirements.md` §5.1：独立的 `runtime/vsr/cli.py`，保留基线的权重路径语义与全部输出相关参数。

```bash
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.vsr.cli restore \
  --via-pipeline \
  --checkpoint_dir "$VSR_CHECKPOINT" \
  --wan_root "$VSR_WAN_ROOT" \
  --input "$VSR_INPUT" \
  --output "$VSR_OUTPUT_ROOT/sglang_stream.mp4" \
  --target_resolution 3840x2160 \
  --tile_h 320 --tile_w 640 --tile_t 33 \
  --color_ref global --read_queue 2
```

`--via-pipeline` 通过 SGLang 原生调度器和 `WanVSRPipeline` 运行；省略该参数则直接运行相同流式核心。
实际推理命令前加 `CUDA_VISIBLE_DEVICES=7`，避免使用其他物理 GPU。

## 5. 资源与目录约束

- 输入、外部 VSR 仓库和模型权重**只读**。
- 日志、输出、调试 tile 和临时文件写入 `$VSR_OUTPUT_ROOT`（即
  `$SGLANG_REPO/output_results/vsr`）。
- `read_queue` 和 `write_queue` 是流式内存的主要调节项；内存不足时先降低队列，不改变 tile、
  overlap、精度或颜色语义。
- `global` 颜色统计使用恒定大小的 float64 累加器，但会增加一次读取。
- 仍需为当前窗口、空间融合 accumulator、时间累积与权重、编码队列和解码器预留主存。
  4K 下单 chunk（`[1,3,33,3840,2176]` fp32）即约 `3.3 GB`；`stream.py` 自估 53 帧峰值约 `22.6 GiB`。
- 输出写入失败、GPU 异常或取消时必须清理 writer 和本任务临时文件。

## 6. 本次核实的命令

```bash
# 两套环境的版本
$VSR_PYTHON -c "import torch, diffusers, decord, imageio, cv2, numpy; ..."
$VE_SGLANG_PYTHON -c "import torch, diffusers, sglang.multimodal_gen; ..."

# 权重存在性与体积
test -s "$VSR_WAN_ROOT/vae/config.json"
test -s "$VSR_CHECKPOINT/transformer_ema/config.json"
ls -la "$VSR_CHECKPOINT/vae_decoder_ema.pt"

# 输入与素材
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames,pix_fmt -of json "$VSR_INPUT"
md5sum "$VSR_REPO/input/input.mp4" /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/zy_test/dog2_1080P.mp4

# GPU
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ffmpeg
which ffmpeg && ffmpeg -version | head -1
$VSR_PYTHON -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
$VE_SGLANG_PYTHON -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```
