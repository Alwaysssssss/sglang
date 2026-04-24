# Wan2.1 完整运行指南

本文档介绍如何使用 `sglang.multimodal_gen` 运行 [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) 系列模型。覆盖环境准备、模型选择、命令行/服务两种运行方式、参数调优、显存优化、Cache-DiT 加速、LoRA 以及常见问题排查。

- 代码相关入口：`python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py`
- Pipeline 配置：`python/sglang/multimodal_gen/configs/pipeline_configs/wan.py`
- 采样参数：`python/sglang/multimodal_gen/configs/sample/wan.py`
- 模型注册：`python/sglang/multimodal_gen/registry.py`（`_register_configs` 中的 Wan 系列条目）

## 1. 模型简介与支持的变体

Wan2.1 是万相团队开源的视频生成模型族，支持 T2V / I2V，SGLang 针对它做了原生 Pipeline（`WanPipeline`）、LoRA、UniPC 调度器、TeaCache、USP 序列并行、CFG 并行、Cache-DiT 等整套优化。

SGLang 原生已注册的 Wan2.1 权重（`registry.py`）：

| HuggingFace 模型 ID | 任务 | 默认分辨率 | 采样参数类 | Pipeline 配置类 |
| --- | --- | --- | --- | --- |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | T2V | 480×832, 81 帧, 16fps | `WanT2V_1_3B_SamplingParams` | `WanT2V480PConfig` |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | T2V | 720×1280, 81 帧, 16fps | `WanT2V_14B_SamplingParams` | `WanT2V720PConfig` |
| `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | I2V | 480×832, 81 帧, 16fps | `WanI2V_14B_480P_SamplingParam` | `WanI2V480PConfig` |
| `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | I2V | 720×1280, 81 帧, 16fps | `WanI2V_14B_720P_SamplingParam` | `WanI2V720PConfig` |
| `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers` | T2V（DMD 加速） | 480×832 | `WanT2V_1_3B_SamplingParams` | `TurboWanT2V480PConfig` |
| `IPostYellow/TurboWan2.1-T2V-14B-Diffusers` / `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers` | T2V（DMD 加速） | 720×1280 | `WanT2V_14B_SamplingParams` | `TurboWanT2V480PConfig` |
| `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | T2V（FastVideo） | 480×832 | `FastWanT2V480PConfig` | `FastWan2_1_T2V_480P_Config` |
| `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` | I2V（Fun） | 480×832 | `Wan2_1_Fun_1_3B_InP_SamplingParams` | `WanI2V480PConfig` |

> 只要模型名里含 `wanpipeline` / `wanimagetovideo`，或传入的是本地 HuggingFace 缓存目录（形如 `.../models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/<hash>`），`registry.py` 中的检测器也能自动识别为 Wan 系列。

## 2. 环境准备

### 2.1 硬件要求（参考）

| 变体 | 单卡最低显存（无 offload） | 建议配置 |
| --- | --- | --- |
| T2V-1.3B | 12 GiB | 单卡 RTX 4090 / A10 即可，480p 5 秒 |
| T2V-14B / I2V-14B 480P | ~48 GiB | 单卡 H20/H100/MI300X；或 4×24G 打开 `--text-encoder-cpu-offload --dit-layerwise-offload` |
| T2V-14B / I2V-14B 720P | ~80 GiB | 单卡 H100/H200/MI300X；或 4–8 卡 USP |

长度/分辨率越大显存越吃，必要时叠加下文的「显存优化」部分。

### 2.2 安装

SGLang 的扩散能力打包在 `diffusion` extras 中，可以直接基于仓库源码安装：

```bash
cd /path/to/sglang/python
uv venv --python 3.12 --seed /opt/venv
source /opt/venv/bin/activate
uv pip install --prerelease=allow ".[diffusion]"
```

也可以直接使用仓库自带的 Docker 镜像 `sglang/docker/diffusion_sm.Dockerfile`：

```bash
cd /path/to/sglang
docker build -f docker/diffusion_sm.Dockerfile -t sglang-diffusion:latest .
docker run --gpus all --rm -it -p 30000:30000 sglang-diffusion:latest
```

### 2.3 模型权重

- 在线：不指定本地路径时，`DiffGenerator` 会自动从 HuggingFace Hub 拉取权重。
- 离线：设置 `HF_HOME` / `HUGGINGFACE_HUB_CACHE` 指向本地缓存路径，或者直接给 `--model-path` 传本地目录。
- 国内加速：可预先 `export HF_ENDPOINT=https://hf-mirror.com` 再运行。

## 3. 两条核心运行链路

Wan2.1 的所有用法都走两条链路之一：

- **离线生成**：`sglang generate`（或 Python `DiffGenerator.from_pretrained`），一次跑完就退出。
- **在线服务**：`sglang serve` 启动 FastAPI + Scheduler + Worker，通过 HTTP / OpenAI API 持续接请求。

底层都会经过：`registry.get_model_info → ServerArgs → DiffGenerator → Scheduler → GPUWorker → WanPipeline → Stage`（细节见 [03_runtime_execution.md](./03_runtime_execution.md)、[04_pipeline_and_stage.md](./04_pipeline_and_stage.md)）。

## 4. 快速开始：单卡离线生成

### 4.1 CLI 一行生成视频

T2V-1.3B 是最轻量的变体，单卡起步最简单：

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A curious raccoon walking through an autumn forest, cinematic lighting" \
  --save-output \
  --output-path outputs \
  --output-file-name "raccoon_forest.mp4"
```

跑完会在 `outputs/raccoon_forest.mp4` 看到 5 秒的 480p 视频。默认采样参数（`WanT2V_1_3B_SamplingParams`）：

- `height=480, width=832, num_frames=81, fps=16`
- `num_inference_steps=50, guidance_scale=3.0, seed=42`

CLI 允许覆盖这些默认值，例如：

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A neon-lit Tokyo street at night, rainy reflections" \
  --height 480 --width 832 \
  --num-frames 81 --fps 16 \
  --num-inference-steps 40 \
  --guidance-scale 4.5 \
  --seed 2024 \
  --save-output --output-path outputs
```

### 4.2 使用配置文件

`sglang generate` 支持 `--config` 读取 JSON/YAML。仓库里已经有示例 `python/sglang/multimodal_gen/test/test_files/launch_wan.json`：

```json
{
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "text_encoder_cpu_offload": true,
    "pin_cpu_memory": true,
    "save_output": true,
    "width": 720,
    "height": 720,
    "output_path": "outputs",
    "output_file_name": "Wan2.1-T2V-1.3B-Diffusers, single gpu"
}
```

运行：

```bash
sglang generate --config python/sglang/multimodal_gen/test/test_files/launch_wan.json
```

配置文件中的 key 同时支持 `ServerArgs` 和 `SamplingParams` 的字段（见 `runtime/server_args.py` 与 `configs/sample/sampling_params.py`）。

### 4.3 Python API

如果需要嵌入到自己的流水线，可以直接用 `DiffGenerator`：

```python
from sglang.multimodal_gen import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    local_mode=True,
    text_encoder_cpu_offload=True,
    pin_cpu_memory=True,
)

results = generator.generate(
    sampling_params_kwargs={
        "prompt": "A dragon flying through stormy clouds at sunset",
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "num_inference_steps": 40,
        "seed": 7,
        "save_output": True,
        "output_path": "outputs",
        "output_file_name": "dragon_clouds",
    }
)

for r in results if isinstance(results, list) else [results]:
    print(r.output_path, r.metrics.get("total_duration_ms"), "ms")
```

`DiffGenerator.from_pretrained` 接受所有 `ServerArgs` 字段（如 `num_gpus`, `tp_size`, `sp_size`, `attention_backend`, `dit_cpu_offload`）。`generate` 的 `sampling_params_kwargs` 接受所有 `SamplingParams` 字段（如 `prompt`, `negative_prompt`, `height`, `width`, `num_frames`, `fps`, `seed`, `num_inference_steps`, `guidance_scale`, `enable_teacache`, `enable_frame_interpolation` 等）。

## 5. 多卡并行（14B 模型必读）

14B T2V / I2V 单卡 480p 可以跑，720p 一般需要 CPU offload 或者多卡 USP 序列并行 + CFG 并行：

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --enable-cfg-parallel \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --prompt "A majestic eagle soaring above snowy mountains" \
  --save-output --output-path outputs \
  --output-file-name "eagle_mountains.mp4"
```

关键参数含义：

- `--num-gpus N`：总 GPU 数量，自动分配给 TP/SP/CFG。
- `--tp-size`：张量并行尺寸。大量 CPU offload 场景建议保持 1。
- `--sp-size` / `--ulysses-degree` / `--ring-degree`：USP 序列并行。Ulysses 通信少但需要 head 可整除，Ring 适合长序列。两者乘积 = `sp_size`。
- `--enable-cfg-parallel`：把 CFG 的 positive / negative 两路拆到不同 GPU 上，等价把并行度再翻倍。
- `--master-port / --nccl-port`：分布式通讯端口，单机多实例时避免冲突。

一个常见的 4 卡最佳实践组合：`--num-gpus 4 --ulysses-degree 2 --enable-cfg-parallel`，相当于 sp=2、cfg=2，总并行度 4。

## 6. 启动在线服务

### 6.1 启动

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --host 0.0.0.0 --port 30000 \
  --num-gpus 4 --ulysses-degree 2 --enable-cfg-parallel \
  --text-encoder-cpu-offload \
  --pin-cpu-memory
```

默认 `host=127.0.0.1, port=30000`。启动完成后可访问：

- `GET /health`：存活探针
- `GET /server_info` / `GET /model_info`：返回加载的模型与 ServerArgs 摘要
- `GET /stats`：运行时统计
- `POST /v1/videos`：生成视频（OpenAI 风格，见 6.2）
- `POST /v1/images/generations`：生成图像（Wan2.1 不常用）

### 6.2 生成视频（OpenAI 风格）

T2V-14B：

```bash
curl -N http://127.0.0.1:30000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "prompt": "A cute baby sea otter floating on its back, holding a small pebble",
        "seconds": 5,
        "fps": 16,
        "size": "1280x720",
        "num_inference_steps": 40,
        "guidance_scale": 5.0,
        "seed": 42
      }' \
  -o response.json
```

返回的 `response.json` 里含 `id`，可以继续：

```bash
VIDEO_ID=$(jq -r '.id' response.json)
curl -OJ http://127.0.0.1:30000/v1/videos/${VIDEO_ID}/content
```

I2V 则需要上传参考图（`multipart/form-data`）：

```bash
curl http://127.0.0.1:30000/v1/videos \
  -F "model=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" \
  -F "prompt=Camera slowly zooms in while the subject smiles" \
  -F "seconds=5" -F "fps=16" -F "size=832x480" \
  -F "input_reference=@./ref.jpg;type=image/jpeg" \
  -o i2v.json
```

其余常用字段：`negative_prompt`, `num_inference_steps`, `guidance_scale`, `guidance_scale_2`（Wan2.2），`enable_teacache`, `enable_frame_interpolation`, `frame_interpolation_exp`, `enable_upscaling` 等，字段定义见 `runtime/entrypoints/openai/video_api.py` 与 `configs/sample/sampling_params.py`。

### 6.3 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="not-needed")

resp = client.videos.generate(
    model="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    prompt="A cyberpunk city at night, flying cars passing by",
    seconds=5,
    fps=16,
    size="1280x720",
    extra_body={
        "num_inference_steps": 40,
        "guidance_scale": 5.0,
        "seed": 2024,
    },
)
print(resp.id)
```

## 7. 显存优化

14B/720P 经常吃爆显存，`ServerArgs` 里有一整套 offload 选项：

| 参数 | 作用 |
| --- | --- |
| `--text-encoder-cpu-offload` | T5 文本编码器常驻 CPU，只在需要时搬上 GPU。几乎零损失，14B 推荐默认开启。 |
| `--dit-cpu-offload` | 整个 DiT 放 CPU，按需搬入；速度下降较明显，适合极端显存紧张场景。 |
| `--dit-layerwise-offload` | 按层 offload DiT，精度无损，速度折损有限。14B 单卡强烈推荐。 |
| `--dit-layerwise-offload-prefetch` | 预取层数，可给浮点比例（0–1）或绝对层数；越大越接近非 offload。 |
| `--vae-cpu-offload` | 解码阶段前把 VAE 放 CPU，解码时上 GPU。 |
| `--image-encoder-cpu-offload` | I2V 的 CLIP vision 编码器 offload。 |
| `--pin-cpu-memory` | 固定住 CPU 缓冲区以加速 H2D 拷贝；偶尔能缓解 `CUDA error: invalid argument`。 |

SGLang 还会基于显存自动调整 `dit_layerwise_offload`：H200（≥130 GiB）默认会被自动关掉（`WAN_LAYERWISE_OFFLOAD_AUTO_DISABLE_MEM_GB`），以保持最快速度。

组合示例（单卡 24G 跑 14B-480P）：

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --text-encoder-cpu-offload \
  --dit-layerwise-offload true \
  --dit-layerwise-offload-prefetch 0.25 \
  --vae-cpu-offload \
  --pin-cpu-memory
```

## 8. Cache-DiT 加速

Wan 系列已经接入了 [Cache-DiT](https://github.com/vipshop/cache-dit)，实测 14B T2V 能从 ~1958s 降到 ~557s（仓库 benchmark，MI300X 单卡）。基础用法：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers
```

进阶组合（按 `envs.py` 中环境变量）：

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang serve --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers
```

> 注意：Cache-DiT 与 `--dit-layerwise-offload` 不能同时开启，`ServerArgs` 会校验并报错。

## 9. TeaCache / 帧插值 / 超分

Wan2.1 原生支持 TeaCache，14B 的 `WanI2V_14B_*` 和 `WanT2V_14B_SamplingParams` 已经内置了合法系数。按需在采样参数里打开：

- `--enable-teacache`：跳步加速，精度有轻微损失。
- `--enable-frame-interpolation --frame-interpolation-exp 1`：RIFE 插帧（1=2×，2=4×），默认用 `elfgum/RIFE-4.22.lite`。
- `--enable-upscaling --upscaling-scale 4`：Real-ESRGAN 超分，默认 `ai-forever/Real-ESRGAN`。

这三个可以叠加，CLI 用法：

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --prompt "An astronaut riding a horse on Mars" \
  --enable-teacache \
  --enable-frame-interpolation --frame-interpolation-exp 1 \
  --enable-upscaling --upscaling-scale 2 \
  --save-output --output-path outputs
```

服务端等价于 `POST /v1/videos` 里的 `enable_teacache`、`enable_frame_interpolation` 等字段。

## 10. LoRA

Wan2.1 的 LoRA 走 `LoRAPipeline`（`WanPipeline` 继承自它）。

### 10.1 启动时直接挂载

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --lora-path NIVEDAN/wan2.1-lora
```

### 10.2 运行时动态挂载

服务启动后可通过 `DiffGenerator` 或 HTTP 管理接口 `set_lora` / `merge_lora_weights` / `list_loras` / `unmerge_lora_weights`（见 `runtime/entrypoints/utils.py` 中的 `SetLoraReq` 等）。

已验证的组合：

| 基座 | LoRA |
| --- | --- |
| `Wan-AI/Wan2.1-T2V-14B` | `NIVEDAN/wan2.1-lora` |
| `Wan-AI/Wan2.1-I2V-14B-720P` | `valiantcat/Wan2.1-Fight-LoRA` |

## 11. Turbo / FastWan 加速版

如果对质量要求不是 SOTA、但想要几倍速度，可以直接换权重：

```bash
sglang generate \
  --model-path IPostYellow/TurboWan2.1-T2V-14B-Diffusers \
  --prompt "An explosion of colorful paint in slow motion" \
  --save-output --output-path outputs
```

这些权重在 `registry.py` 中绑定了 `TurboWanT2V480PConfig`，默认 `flow_shift=8.0`、`dmd_denoising_steps=[988, 932, 852, 608]`，通常 4 步即可完成去噪，速度提升显著。FastWan 同理 (`FastVideo/FastWan2.1-T2V-1.3B-Diffusers`)。

## 12. 关键 ServerArgs / SamplingParams 速查

### 12.1 ServerArgs（`runtime/server_args.py`）

```
--model-path                # HF repo id 或本地路径
--model-id                  # 强制走哪个注册条目（调试用）
--backend {auto,sglang,diffusers}
--num-gpus / --tp-size / --sp-size / --ulysses-degree / --ring-degree
--enable-cfg-parallel
--attention-backend {fa,sage_attn,torch_sdpa,...}
--text-encoder-cpu-offload  --dit-cpu-offload  --dit-layerwise-offload
--dit-layerwise-offload-prefetch
--vae-cpu-offload  --image-encoder-cpu-offload
--pin-cpu-memory
--host / --port / --nccl-port / --master-port / --scheduler-port
--lora-path
--webui / --webui-port
--input-save-path / --output-path
```

### 12.2 SamplingParams（`configs/sample/sampling_params.py` + `configs/sample/wan.py`）

```
--prompt / --negative-prompt / --prompt-path
--height / --width / --num-frames / --fps
--num-inference-steps / --guidance-scale / --guidance-scale-2
--seed / --generator-device
--enable-teacache
--enable-frame-interpolation --frame-interpolation-exp --frame-interpolation-scale
--enable-upscaling --upscaling-scale --upscaling-model-path
--save-output / --output-path / --output-file-name / --output-file-path
```

## 13. 常见问题

1. **`CUDA out of memory` 跑 14B 720P**：优先 `--dit-layerwise-offload true --text-encoder-cpu-offload --vae-cpu-offload --pin-cpu-memory`；仍不够再叠 `--dit-cpu-offload` 或多卡 USP。
2. **`CUDA error: invalid argument` 相关 offload 报错**：打开 `--pin-cpu-memory`。
3. **提示 “Pipeline class 'X' is not a registered EntryClass”**：说明用的权重没有 SGLang 原生实现；把 `--backend diffusers` 加上，会回退到 diffusers 原版 pipeline，全部 Wan 参数仍然兼容。
4. **模型路径识别不到**：可以显式加 `--model-id Wan2.1-T2V-1.3B-Diffusers` 强制命中 `_MODEL_HF_PATH_TO_NAME` 中的注册。
5. **USP 报错 “head dim 不能整除”**：改用 `--ring-degree` 或降低 `--ulysses-degree`；最极端场景用 `--attention-backend torch_sdpa`。
6. **视频保存为空 / 帧数不对**：Wan I2V 要求 `num_frames % vae_scale_factor_temporal == 1`，`WanI2VCommonConfig.adjust_num_frames` 会自动向下取整；若你硬传 num_frames 又报错，多为 81 / 65 / 121 这类合法值。
7. **HuggingFace 下载慢**：`export HF_ENDPOINT=https://hf-mirror.com` 或先 `huggingface-cli download Wan-AI/...` 到本地。

## 14. 一步步 Checklist（面向第一次跑通）

1. 准备 GPU + CUDA 12.x 环境；`pip install -e '.[diffusion]'` 或起 `diffusion_sm.Dockerfile`。
2. `export HF_HOME=/data/hf`，提前 `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers`。
3. 跑一次离线 1.3B：`sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "hello world" --save-output --output-path outputs`，确认 `outputs/*.mp4` 能放。
4. 切 14B + CPU offload：`sglang generate --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers --text-encoder-cpu-offload --dit-layerwise-offload true ...`。
5. 升级到多卡：`--num-gpus 4 --ulysses-degree 2 --enable-cfg-parallel`。
6. 切到服务模式：`sglang serve ...`，用 `curl /v1/videos` 或 OpenAI SDK 发起请求。
7. 有稳定基线后再开 Cache-DiT、TeaCache、LoRA、RIFE、Real-ESRGAN 等加速/增强选项。

## 参考

- 架构与运行时细节：[01_architecture_overview.md](./01_architecture_overview.md) → [07_disaggregation_and_optimization.md](./07_disaggregation_and_optimization.md)
- Pipeline 实现：`python/sglang/multimodal_gen/runtime/pipelines/wan_pipeline.py`、`wan_i2v_pipeline.py`、`wan_dmd_pipeline.py`
- 官方模型与论文：[Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)、[Wan-AI HuggingFace](https://huggingface.co/Wan-AI)
