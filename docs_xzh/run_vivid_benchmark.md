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
tmux attach -r -t vividvr_phase_c
```

默认建议使用只读 attach。这样即使本地终端或 shell 集成误发 `Ctrl-C`，也不会把 `tmux` 里的推理进程一并中断。

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
- 原版 `Vivid-VR` 默认会实时 caption，因此做公平对比时，不能直接拿“原版 live caption”去和 `sglang` 的固定 `prompt.txt` 结果比较。
- 原版 benchmark 必须使用 `/home/zhiheng/Vivid-VR/.venv/bin/python`，不要用 `/home/zhiheng/sglang/.venv/bin/python` 代跑原版；否则当前机器上的高版本 `transformers` 会让 `CogVLM2` 产出错误 caption。
- 当前机器上，如果原版 `Vivid-VR` 在启动时因为 Python 头文件路径报错，需要先额外导出：`CPATH=/home/zhiheng/tmp_py310_headers/extracted/libpython3.10-dev/usr/include/python3.10:/home/zhiheng/tmp_py310_headers/extracted/libpython3.10-dev/usr/include`。
- 正确做法是：先跑原版 `Vivid-VR`，从日志里提取它实际生成的 raw caption，保存到 `/home/zhiheng/Vivid-VR/input/captions/<video_stem>.txt`，再让 `sglang` 版本用这个 caption sidecar 重跑。
- 当前 `Phase C` 的正式 gold 仍然是现有 reference 视频：`/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`。

### 2.1 原版 `Vivid-VR` 默认对比命令

```bash
tmux new-session -d -s vividvr_ori_benchmark \
  'cd /home/zhiheng/Vivid-VR && mkdir -p /home/zhiheng/sglang/Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=/home/zhiheng/Vivid-VR/src && /home/zhiheng/Vivid-VR/.venv/bin/python VRDiT/inference.py \
    --ckpt_dir=/home/zhiheng/Vivid-VR/ckpts \
    --cogvideox_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --cogvlm2_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption \
    --input_dir=/home/zhiheng/Vivid-VR/input/720p \
    --output_dir=/home/zhiheng/Vivid-VR/result/720p_benchmark_cogvlm2 \
    --num_temporal_process_frames=121 \
    --num_inference_steps=50 \
    --guidance_scale=6 \
    --restoration_guidance_scale=-1.0 \
    --upscale=1 \
    --seed=42 \
    2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_benchmark_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_ori_benchmark
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
  'cd /home/zhiheng/Vivid-VR && mkdir -p /home/zhiheng/sglang/Vivid_Acceptance/logs && export PYTHONPATH=/home/zhiheng/Vivid-VR/src && CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True /home/zhiheng/Vivid-VR/.venv/bin/python VRDiT/inference.py \
    --ckpt_dir=/home/zhiheng/Vivid-VR/ckpts \
    --cogvideox_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --input_dir=/home/zhiheng/Vivid-VR/input/720p \
    --output_dir=/home/zhiheng/Vivid-VR/result/720p_benchmark_sglang_caption \
    --num_temporal_process_frames=121 \
    --num_inference_steps=50 \
    --guidance_scale=6 \
    --restoration_guidance_scale=-1.0 \
    --upscale=1 \
    --seed=42 \
    --caption_backend=sglang \
    --caption_sglang_base_url=http://127.0.0.1:31000/v1 \
    --caption_sglang_model=Qwen/Qwen2.5-VL-3B-Instruct \
    2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_sglang_caption_$(date -u +%Y%m%dT%H%M%SZ).log'
```

### 2.3 Phase D 长视频公平对比命令

当前 `Phase D` 的正式长视频 benchmark 不再使用旧的 `x3 duplicate` 输入，而是固定使用下面这条 `130f` 输入视频：

```text
/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4
```

当前日常 benchmark 默认使用 `num_inference_steps=20`，便于后续做性能迭代；`num_inference_steps=50` 只保留给最终回归和阶段验收。

需要注意：

- 当前这条 `130f` 基准对应 2 个 temporal clip，不再是旧的 `x3` / 3 clip 流程。
- `sglang` benchmark 和后续 `Phase E` 默认统一使用通用脚本：`python/sglang/multimodal_gen/tools/run_vividvr_inference.py`
- 这份脚本是单视频通用入口，直接在启动命令里修改 `--input-video`、`--caption-file`、`--reference-video`、`--num-inference-steps` 等参数即可，不再依赖 `Phase D` 专项 helper 或 preset。

原版 `Vivid-VR` long-video reference：

```bash
tmux new-session -d -s vividvr_ori_phase_d_20step \
  'cd /home/zhiheng/Vivid-VR && mkdir -p /home/zhiheng/sglang/Vivid_Acceptance/logs && export PYTHONUNBUFFERED=1 && export PYTHONPATH=/home/zhiheng/Vivid-VR/src && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/Vivid-VR/.venv/bin/python VRDiT/inference.py \
    --ckpt_dir=/home/zhiheng/Vivid-VR/ckpts \
    --cogvideox_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B \
    --cogvlm2_ckpt_path=/home/zhiheng/Vivid-VR/ckpts/cogvlm2-llama3-caption \
    --input_dir=/home/zhiheng/Vivid-VR/input/720p_long \
    --output_dir=/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step \
    --num_temporal_process_frames=121 \
    --num_inference_steps=20 \
    --guidance_scale=6 \
    --restoration_guidance_scale=-1.0 \
    --upscale=1 \
    --seed=42 \
    2>&1 | tee /home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_ori_phase_d_$(date -u +%Y%m%dT%H%M%SZ).log'
```

原版跑完后，需要把日志里每次 `Generating 1 video with prompt: ...` 对应的原始 caption 依次提取出来，保存到：

```text
/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt
```

要求：

- 一行一个 caption，顺序必须与原版 clip/tile 生成顺序一致。
- 保存原始 caption 文本，不要把正向 prompt suffix 手工拼进去。
- 当前这条 `130f` 基准的 caption 文件名必须与输入视频 stem 一致。
- 当前这条 `test_video_long_960x720_130f.txt` 应包含 2 行 raw caption，对应 2 个 temporal clip。

`sglang` Phase D 验收：

```bash
tmux new-session -d -s vividvr_phase_d_20step \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && CUDA_VISIBLE_DEVICES=0 /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
    --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
    --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
    --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 \
    --output-dir /home/zhiheng/sglang/Vivid_Acceptance/result_videos \
    --report-dir /home/zhiheng/sglang/Vivid_Acceptance/indicator \
    --artifact-prefix phase_d_130f_20step \
    --phase-label D \
    --mode-label temporal_windowed_reference_alignment \
    --num-temporal-process-frames 121 \
    --num-inference-steps 20 \
    --guidance-scale 6 \
    --restoration-guidance-scale -1.0 \
    --seed 42 \
    2>&1 | tee Vivid_Acceptance/logs/phase_d_130f_20step_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_phase_d_20step
```

如果要做最终 `50 step` 回归，把上面两条 `Phase D` 命令里的 `20` 同步改成 `50`，并同时更新：

- 原版命令里的输出目录改成 `720p_long_up1_result_vivid_ori_50step`
- `sglang` 命令里的 `--caption-file` 改成 `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f_50step.txt`
- `sglang` 命令里的 `--reference-video` 改成 `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_50step/videos/test_video_long_960x720_130f.mp4`
- `sglang` 命令里的 `--artifact-prefix` 改成 `phase_d_130f_50step`
- `sglang` 命令里的 `--num-inference-steps` 改成 `50`

### 2.4 Phase E4.1 native SP 多卡 formal 命令

下面三条命令用于当前 `Phase E4.1` 的双卡 `SP` 正式 benchmark。三条命令共用同一套输入、caption、prompt、reference、`20 step` 和 `FA + compile` 口径，只改变 `native fast path / v1 / v2` 的 connector 语义入口。

共同约束：

- 必须使用真实多进程入口：`torchrun --nproc_per_node=2`
- 当前默认并行口径固定为：
  - `num_gpus=2`
  - `tp_size=1`
  - `sp_degree=2`
  - `ulysses_degree=2`
  - `ring_degree=1`
- `v1 / v2` 通过环境变量 `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE` 选择语义
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE` 默认值固定为 `1`，也就是 SP 默认不做 control pooling；只有显式设置 `=2` 等值时才启用池化压缩
- `native` fast path 不额外设置该环境变量，使用当前默认 native `SP` 快路径

`native` fast path：

```bash
tmux new-session -d -s vividvr_e41_native_sp_fast \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30062 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --num-inference-steps 20 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30062 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 --artifact-prefix phase_e41_native_sp_only_130f_20step_compile 2>&1 | tee Vivid_Acceptance/logs/phase_e41_native_sp_formal_$(date -u +%Y%m%dT%H%M%SZ).log'
```

`v1`：

```bash
tmux new-session -d -s vividvr_e41_v1_recheck \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=deferred_global && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30063 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --num-inference-steps 20 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30063 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 --artifact-prefix phase_e41_native_sp_quality_opt_v1_130f_20step_compile 2>&1 | tee Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v1_recheck_$(date -u +%Y%m%dT%H%M%SZ).log'
```

`v2`：

```bash
tmux new-session -d -s vividvr_e41_v2_recheck \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && /home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30064 python/sglang/multimodal_gen/tools/run_vividvr_inference.py --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 --num-inference-steps 20 --seed 42 --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --dist-timeout 3600 --master-port 30064 --attention-backend fa --enable-torch-compile --warmup --warmup-steps 1 --artifact-prefix phase_e41_native_sp_quality_opt_v2_130f_20step_compile 2>&1 | tee Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_recheck_$(date -u +%Y%m%dT%H%M%SZ).log'
```

查看进度：

```bash
tmux attach -r -t vividvr_e41_native_sp_fast
tmux attach -r -t vividvr_e41_v1_recheck
tmux attach -r -t vividvr_e41_v2_recheck
```

对应语义标签：

- `native` fast path：
  - 当前默认 native `SP` 快路径，不恢复 full global control context
- `v1`：
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=deferred_global`
  - runtime snapshot 中应表现为 `connector_context_mode = sp_exact_local_attention`
- `v2`：
  - `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
  - runtime snapshot 中应表现为 `connector_context_mode = sp_exact_global_control_attention`

## 3. 对比时必须保持一致的关键参数

- 输入视频：`/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- seed：`42`
- `num_temporal_process_frames=121`
- `num_inference_steps=20`
- `guidance_scale=6`
- `restoration_guidance_scale=-1.0`
- `upscale=1`
- 默认不启用 `textfix`
- 默认不加 `--save_images`

补充说明：

- `20 step` 是当前日常 benchmark 默认档位。
- `50 step` 仍然是最终回归和阶段验收档位。

如果后续这些参数发生变化，必须同步修改：

- `/home/zhiheng/sglang/AGENTS.md`
- `/home/zhiheng/sglang/docs_xzh/run_vivid_benchmark.md`
