# Vivid-VR Benchmark 命令

本文档记录当前 `sglang` 集成版 `Vivid-VR` 的标准验收命令，以及用于和原版 `Vivid-VR` 做公平对比的推理命令。后续如果推理参数、脚本路径或环境路径发生变化，需要同步更新本文件和仓库根目录 `AGENTS.md`。

## 1. 当前 `sglang` 标准验收命令

当前默认使用 `/home/zhiheng/sglang/.venv`，并通过 `Phase C` 单次验收脚本产出标准指标 JSON 和候选视频。

### 1.1 在 `tmux` 中启动

```bash
tmux new-session -d -s vividvr_phase_c \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py 2>&1 | tee Vivid_Acceptance/logs/phase_c_single_$(date -u +%Y%m%dT%H%M%SZ).log'
```

### 1.2 查看进度

```bash
tmux attach -t vividvr_phase_c
```

### 1.3 标准产物位置

- 指标 JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator`
- 候选视频：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos`
- 当前格式基准：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json`

### 1.4 当前标准字段补充要求

在现有基准 JSON 的基础上，后续验收结果必须额外包含：

- `total_runtime_seconds`
- `model_inference_runtime_seconds`

其中 `model_inference_runtime_seconds` 默认统计 `pipeline.forward(...)` 对应的纯模型推理时段。

## 2. 与原版 `Vivid-VR` 公平对比的推理命令

下面这条命令用于重新运行原版 `Vivid-VR`，尽量和当前 `sglang` 集成版保持一致的输入、seed 和主要推理超参。

需要注意：

- 当前 `sglang` `Phase C` 验收语义固定读取 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`，不走 live `CogVLM2`。
- 原版 `Vivid-VR` 默认会实时 caption，因此它更适合做“原版行为复现 / 侧向性能与画面对比”，而不是直接替代当前 `Phase C` 的 acceptance gold。
- 当前 `Phase C` 的正式 gold 仍然是现有 reference 视频：`/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`。

### 2.1 原版 `Vivid-VR` 默认对比命令

```bash
tmux new-session -d -s vividvr_ori_benchmark \
  'cd /home/zhiheng/Vivid-VR && mkdir -p /home/zhiheng/sglang/Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && /home/zhiheng/sglang/.venv/bin/python VRDiT/inference.py \
    --ckpt_dir=/home/zhiheng/Vivid-VR/ckpts \
    --cogvideox_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --cogvlm2_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption \
    --input_dir=/home/zhiheng/Vivid-VR/input/720p \
    --output_dir=/home/zhiheng/Vivid-VR/result/720p_benchmark_cogvlm2 \
    --num_temporal_process_frames=121 \
    --num_inference_steps=50 \
    --guidance_scale=6 \
    --restoration_guidance_scale=-1.0 \
    --upscale=0 \
    --seed=42 \
    2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_benchmark_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -t vividvr_ori_benchmark
```

### 2.2 可选：原版 `Vivid-VR` + 本地 `SGLang` caption 服务

如果需要把原版 `Vivid-VR` 的 caption 端替换成本地 `SGLang` OpenAI-compatible 服务，用于缩小 caption backend 差异，可以先启动本地服务，再运行：

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://huggingface.co \
/home/zhiheng/sglang/.venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-3B-Instruct \
  --host 127.0.0.1 \
  --port 31000 \
  --log-level warning \
  --disable-cuda-graph \
  --skip-server-warmup
```

```bash
tmux new-session -d -s vividvr_ori_sglang_caption \
  'cd /home/zhiheng/Vivid-VR && mkdir -p /home/zhiheng/sglang/Vivid_Acceptance/logs && CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True /home/zhiheng/sglang/.venv/bin/python VRDiT/inference.py \
    --ckpt_dir=/home/zhiheng/Vivid-VR/ckpts \
    --cogvideox_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --input_dir=/home/zhiheng/Vivid-VR/input/720p \
    --output_dir=/home/zhiheng/Vivid-VR/result/720p_benchmark_sglang_caption \
    --num_temporal_process_frames=121 \
    --num_inference_steps=50 \
    --guidance_scale=6 \
    --restoration_guidance_scale=-1.0 \
    --upscale=0 \
    --seed=42 \
    --caption_backend=sglang \
    --caption_sglang_base_url=http://127.0.0.1:31000/v1 \
    --caption_sglang_model=Qwen/Qwen2.5-VL-3B-Instruct \
    2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_sglang_caption_$(date -u +%Y%m%dT%H%M%SZ).log'
```

## 3. 对比时必须保持一致的关键参数

- 输入目录：`/home/zhiheng/Vivid-VR/input/720p`
- seed：`42`
- `num_temporal_process_frames=121`
- `num_inference_steps=50`
- `guidance_scale=6`
- `restoration_guidance_scale=-1.0`
- `upscale=0`
- 默认不启用 `textfix`
- 默认不加 `--save_images`

如果后续这些参数发生变化，必须同步修改：

- `/home/zhiheng/sglang/AGENTS.md`
- `/home/zhiheng/sglang/docs_xzh/run_vivid_benchmark.md`
