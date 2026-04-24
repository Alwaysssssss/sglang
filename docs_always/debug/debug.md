# SGLang Wan2.1 调试命令汇总

本文只收集 `sglang generate` 下的 Wan2.1 离线调试命令，覆盖当前仓库注册表里所有 Wan2.1 `T2V / I2V` case。

## 0. 进入环境

```bash
docker exec -it sglang-dev zsh
source /opt/venv/bin/activate
```

说明：

- 下文统一写成本地 HuggingFace cache snapshot 路径；如果你直接走 Hub，也可以把 `--model-path` 替换成对应的 HF ID。
- `I2V / Fun` 在 CLI 中用的是 `--image-path`；`input_reference` 是服务端 OpenAI API 字段，不是 CLI 参数。
- 参考图路径请替换成你自己的实际文件，例如 `/path/to/ref.jpg`。

当前仓库已注册的 Wan2.1 变体如下：

| 类型 | 模型 |
| --- | --- |
| 标准 T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| 标准 T2V | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| 标准 I2V | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
| 标准 I2V | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` |
| Turbo T2V | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers` |
| Turbo T2V | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers` |
| Turbo T2V | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers` |
| FastWan T2V | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` |
| Fun I2V | `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` |

## 1. 通过 HF 下载模型

### 1.1 下载到默认 HuggingFace cache

如果你希望继续使用本文里的 snapshot/cache 路径写法，可以直接下载到 HF 默认缓存目录：

```bash
export HF_HOME=/root/.cache/huggingface
# 如果网络慢，可以打开镜像
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
huggingface-cli download IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers
huggingface-cli download IPostYellow/TurboWan2.1-T2V-14B-Diffusers
huggingface-cli download IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers
huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers
huggingface-cli download weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

下载后可用下面的方式查看实际 snapshot 目录：

```bash
ls -d /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/*
ls -d /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*
ls -d /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/*
ls -d /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-720P-Diffusers/snapshots/*
```

### 1.2 下载到固定本地目录

如果你不想手动找 snapshot hash，推荐直接下载到固定目录：

```bash
mkdir -p /data/models

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir /data/models/Wan2.1-T2V-1.3B-Diffusers

huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --local-dir /data/models/Wan2.1-T2V-14B-Diffusers

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --local-dir /data/models/Wan2.1-I2V-14B-480P-Diffusers

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
    --local-dir /data/models/Wan2.1-I2V-14B-720P-Diffusers

huggingface-cli download IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers \
    --local-dir /data/models/TurboWan2.1-T2V-1.3B-Diffusers

huggingface-cli download IPostYellow/TurboWan2.1-T2V-14B-Diffusers \
    --local-dir /data/models/TurboWan2.1-T2V-14B-Diffusers

huggingface-cli download IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers \
    --local-dir /data/models/TurboWan2.1-T2V-14B-720P-Diffusers

huggingface-cli download FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --local-dir /data/models/FastWan2.1-T2V-1.3B-Diffusers

huggingface-cli download weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers \
    --local-dir /data/models/Wan2.1-Fun-1.3B-InP-Diffusers
```

下载到固定目录后，`--model-path` 可以直接写：

```bash
--model-path /data/models/Wan2.1-T2V-1.3B-Diffusers
```

## 2. 已跑通过的命令

### 2.1 Wan2.1 T2V 1.3B 单卡

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A curious raccoon walking through an autumn forest, cinematic lighting" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 4.5 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_sp1.mp4"
```

[04-23 03:36:29] Running pipeline stages: ['InputValidationStage', 'TextEncodingStage', 'LatentPreparationStage', 'TimestepPreparationStage', 'DenoisingStage', 'DecodingStage']
[04-23 03:36:29] [InputValidationStage] started...
[04-23 03:36:29] [InputValidationStage] finished in 0.0000 seconds
[04-23 03:36:29] [TextEncodingStage] started...
[04-23 03:36:32] [TextEncodingStage] finished in 2.3317 seconds
[04-23 03:36:32] [LatentPreparationStage] started...
[04-23 03:36:32] [LatentPreparationStage] finished in 0.0010 seconds
[04-23 03:36:32] [TimestepPreparationStage] started...
[04-23 03:36:32] [TimestepPreparationStage] finished in 0.0002 seconds
[04-23 03:36:32] [DenoisingStage] started...
100%|██████████████████████████████████████████████████████████████████████| 40/40 [03:31<00:00,  5.28s/it]
[04-23 03:40:03] [DenoisingStage] average time per step: 5.2824 seconds
[04-23 03:40:03] [DenoisingStage] finished in 211.2998 seconds
[04-23 03:40:03] [DecodingStage] started...
[04-23 03:40:17] [DecodingStage] finished in 14.0261 seconds
[04-23 03:40:19] Output saved to outputs/wan21_sp1.mp4
[04-23 03:40:19] Pixel data generated successfully in 229.70 seconds
[04-23 03:40:19] Completed batch processing. Generated 1 outputs in 229.70 seconds

### 2.2 Wan2.1 T2V 1.3B 双卡 USP

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b \
    --num-gpus 2 \
    --ulysses-degree 2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A curious raccoon walking through an autumn forest, cinematic lighting" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 4.5 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_sp2.mp4"
```

[04-23 03:24:47] Running pipeline stages: ['InputValidationStage', 'TextEncodingStage', 'LatentPreparationStage', 'TimestepPreparationStage', 'DenoisingStage', 'DecodingStage']
[04-23 03:24:47] [InputValidationStage] started...
[04-23 03:24:47] [InputValidationStage] finished in 0.0001 seconds
[04-23 03:24:47] [TextEncodingStage] started...
[04-23 03:24:50] [TextEncodingStage] finished in 3.4621 seconds
[04-23 03:24:50] [LatentPreparationStage] started...
[04-23 03:24:50] [LatentPreparationStage] finished in 0.0013 seconds
[04-23 03:24:50] [TimestepPreparationStage] started...
[04-23 03:24:50] [TimestepPreparationStage] finished in 0.0004 seconds
[04-23 03:24:50] [DenoisingStage] started...
100%|███████████████████████████████████████████████████| 40/40 [02:01<00:00,  3.03s/it]
[04-23 03:26:51] [DenoisingStage] average time per step: 3.0259 seconds
[04-23 03:26:51] [DenoisingStage] finished in 121.0393 seconds
[04-23 03:26:51] [DecodingStage] started...
[04-23 03:26:59] [DecodingStage] finished in 7.9133 seconds
[04-23 03:27:00] Output saved to outputs/wan21_sp2.mp4
[04-23 03:27:01] Pixel data generated successfully in 133.86 seconds
[04-23 03:27:01] Completed batch processing. Generated 1 outputs in 133.86 seconds

## 3. 标准 T2V

### 3.1 Wan2.1 T2V 14B

推荐用 4 卡起步：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/<snapshot-hash> \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --enable-cfg-parallel \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A majestic eagle soaring above snowy mountains, cinematic aerial shot" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 5.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_t2v_14b.mp4"
```

## 4. 标准 I2V

### 4.1 Wan2.1 I2V 14B 480P

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/<snapshot-hash> \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --enable-cfg-parallel \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --image-path /path/to/ref.jpg \
    --prompt "The character slowly turns around and smiles, natural motion, cinematic lighting" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 5.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_i2v_480p.mp4"
```

### 4.2 Wan2.1 I2V 14B 720P

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-720P-Diffusers/snapshots/<snapshot-hash> \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --enable-cfg-parallel \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --image-path /path/to/ref.jpg \
    --prompt "The camera pushes in while the subject blinks and hair moves with the wind" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 40 \
    --guidance-scale 5.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_i2v_720p.mp4"
```

## 5. Turbo / Fast / Fun 变体

### 5.1 TurboWan2.1 T2V 1.3B

Turbo 通常走 4 步去噪：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--IPostYellow--TurboWan2.1-T2V-1.3B-Diffusers/snapshots/<snapshot-hash> \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A sports car drifts through a rainy neon city street" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 4 \
    --guidance-scale 3.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "turbo_wan21_t2v_1p3b.mp4"
```

### 5.2 TurboWan2.1 T2V 14B

`TurboWan2.1-T2V-14B-Diffusers` 和 `TurboWan2.1-T2V-14B-720P-Diffusers` 二选一即可：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--IPostYellow--TurboWan2.1-T2V-14B-Diffusers/snapshots/<snapshot-hash> \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A giant whale swims through clouds above a sunset ocean" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 4 \
    --guidance-scale 5.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "turbo_wan21_t2v_14b.mp4"
```

如果你下载的是 720P 别名目录，也可以直接用：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--IPostYellow--TurboWan2.1-T2V-14B-720P-Diffusers/snapshots/<snapshot-hash> \
    --num-gpus 4 \
    --ulysses-degree 2 \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A giant whale swims through clouds above a sunset ocean" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 4 \
    --guidance-scale 5.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "turbo_wan21_t2v_14b_720p.mp4"
```

### 5.3 FastWan2.1 T2V 1.3B

FastWan 默认就是 3 步、`448x832`、`61` 帧：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--FastVideo--FastWan2.1-T2V-1.3B-Diffusers/snapshots/<snapshot-hash> \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --prompt "A robot barista making latte art in a cozy cafe" \
    --height 448 \
    --width 832 \
    --num-frames 61 \
    --fps 16 \
    --num-inference-steps 3 \
    --guidance-scale 3.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "fastwan21_t2v_1p3b.mp4"
```

### 5.4 Wan2.1 Fun 1.3B InP

这个变体在 SGLang 里走 `I2V` 管线，所以同样使用 `--image-path`：

```bash
sglang generate \
    --model-path /root/.cache/huggingface/hub/models--weizhou03--Wan2.1-Fun-1.3B-InP-Diffusers/snapshots/<snapshot-hash> \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --image-path /path/to/ref.jpg \
    --prompt "Keep the main subject consistent while adding subtle motion and richer scene details" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --fps 16 \
    --num-inference-steps 50 \
    --guidance-scale 6.0 \
    --seed 2024 \
    --save-output \
    --output-path outputs \
    --output-file-name "wan21_fun_1p3b_inp.mp4"
```

## 6. 直接使用 HF ID 的等价写法

如果不想写本地 snapshot 目录，`--model-path` 可以直接替换成下面这些 HF ID：

```text
Wan-AI/Wan2.1-T2V-1.3B-Diffusers
Wan-AI/Wan2.1-T2V-14B-Diffusers
Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers
IPostYellow/TurboWan2.1-T2V-14B-Diffusers
IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers
FastVideo/FastWan2.1-T2V-1.3B-Diffusers
weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```
