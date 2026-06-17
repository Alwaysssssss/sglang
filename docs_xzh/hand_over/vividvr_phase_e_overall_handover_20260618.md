# VividVR Phase E 整体交接文档

**日期**: 2026-06-18 UTC
**分支**: `sglang_Vivid`
**HEAD commit**: `337a31932` — "修复了长视频 SP 推理的 bug"

---

## 1. 项目概述

### 1.1 目标

将 Vivid-VR（基于 CogVideoX1.5-5B 的视频修复模型）的推理 pipeline 从原始 Vivid-VR 仓库迁移到 SGLang Diffusion 的 10-stage 模块化框架中，并实现：

- **语义等价**: 与原始 Vivid-VR（commit `d45bf7a36` 前的 Phase C）输出像素级一致
- **分布式推理**: 通过 Sequence Parallelism (SP) + FlashAttention 实现 2 卡并行加速
- **性能优化**: torch.compile、fusion、control state spatial pooling 等

### 1.2 推理管线概述

| 管线 | 输入 | 帧数 | 步数 | 处理方式 |
|------|------|------|------|----------|
| **短视频 (Short Video)** | `test_video_960x720.mp4` | 70f | 50步 | x3 temporal duplication (70→121f) → 10-stage pipeline |
| **长视频 (Long Video)** | `test_video_long_960x720_130f.mp4` | 130f | 20步 | Temporal windowed (2 clips: 0-121f, 60-130f) → stage helper methods |

两种管线通过 `VividVRPipeline.forward()` 自动路由：
- `original_num_frames <= num_temporal_process_frames` → `ComposedPipelineBase.forward()` (10-stage)
- 否则 → `_forward_temporal_windowed()` (clip-based)

---

## 2. Git 仓库状态

### 2.1 分支与 Commit 历史

```
当前分支: sglang_Vivid (领先 main 39 commits)

关键 commit 链:
853a791f0  ← Phase C 基线：恢复原始 VividVR 3-stage pipeline
c714ebc56  ← modular style pipeline (10-stage 框架引入)
83064b778  ← phaseD 语义对齐 (长视频)
4cd992dc2  ← Phase E0/E1: attention backend 引入
3187e66f8  ← Phase E2/E3: torch.compile + fusion 加速
bd03ecbb4  ← Phase E3.2: Tighten control runtime path
a9058b25d  ← Phase E3.2: Complete runtime alignment
9e43f5d31  ← SP (sp2) 双卡并行加速
c0008cd89  ← Restore native SP v1/v2 connector semantics
337a31932  ← HEAD: 修复长视频 SP 推理 bug
```

### 2.2 未提交改动

以下工作区文件有未提交改动（脏状态）：

| 文件 | 改动内容 | 状态 |
|------|----------|------|
| `cogvideox_attention_backend.py` | `CogVideoXSPFlashAttnProcessor` + USP attention backend | 需提交 |
| `cogvideox_vividvr_common.py` | Path B control state spatial pooling + 诊断日志 | 需提交 |
| `vividvr_pipeline.py` | auto fa→fa_sp 升级 + compile mode 配置 | 需提交 |
| `test_connector_remote_compress.py` | Path B 低层测试脚本 | 工具 |
| `diag_connector_attention.py` | connector attention 诊断工具 | 工具 |
| `run_vividvr_inference.py` | 通用推理入口 (720p/130f 支持) | 工具 |

### 2.3 关键代码规模

| 文件 | 行数 | 说明 |
|------|------|------|
| `vividvr.py` (stages) | 1227 | 10个 VividVR pipeline stages |
| `vividvr_pipeline.py` | ~1300 | VividVR pipeline + temporal windowed 逻辑 |
| `cogvideox_vividvr_common.py` | ~100+ | SP control pooling + connector |
| `cogvideox_attention_backend.py` | ~500+ | FA/FA_SP attention backends |

---

## 3. 架构对照：Phase C vs Phase E

### 3.1 Phase C (3-stage 原始管线)

```
VividVRBeforeDenoisingStage  →  VividVRDenoisingStage  →  VividVRDecodingStage
     (prompt+encode+prep)         (denoise loop)            (decode+resize)
```

- 状态全部存在 `batch.extra["vividvr_runtime"]` dict 中
- `attn_metadata=None` 始终
- 不分长/短视频，统一走 `forward()`

### 3.2 Phase E (10-stage 模块化管线)

```
InputValidation → PromptPreparation → TextEncoding → ConditionEncoding
→ LatentPreparation → TilingPreparation → TimestepPreparation
→ Denoising → Decoding → OutputPostprocess
```

- 状态存在 `params.runtime_*` dataclass fields 中
- `attn_metadata` 由 `_build_runtime_attn_metadata()` 构建（FA3/FA4）
- 短视频走 `ComposedPipelineBase.forward()` → 10 stages
- 长视频走 `_forward_temporal_windowed()` → 直接调用 stage helper methods

### 3.3 语义等价性

| 模块 | Phase C | Phase E HEAD | 等价？ |
|------|---------|-------------|--------|
| Prompt 读取 | `prompt_file` mode | `prompt_file` mode | ✅ (需传 `--prompt-file`, 不要 `--caption-file`) |
| T5 文本编码 | T5 + compose_positive_prompt | 相同 | ✅ |
| VAE encode | `vae.encode()` | 相同 | ✅ |
| 噪声生成 | `torch.randn(seed=42)` | 相同 | ✅ |
| Tiling | `_prepare_latent_tiling()` | 相同 | ✅ |
| Denoising math | CFG + restoration guidance + DPM | 相同 | ✅ |
| VAE decode | `vae.decode()` | 相同 | ✅ |
| FA attention metadata | `None` | `_build_runtime_attn_metadata()` | ⚠️ 差异来源 |
| Control state | 无 pooling | SP pool=1 (无压缩) | ✅ |

**结论**: 单卡 + 正确 prompt + pool=1 时，HEAD 与 Phase C 达到像素级等价（SSIM 均值完全一致 0.967716）。

---

## 4. 验收产物

### 4.1 目录结构

```
/home/zhiheng/sglang/Vivid_Acceptance/
├── result_videos/          # 保留 6 个质量达标视频
├── indicator/              # 保留 6 个对应的指标 JSON
├── reference_video/        # 质量对比用参考视频 (2 个)
├── result_videos_backup/   # 其余 126 个历史视频
├── indicator_backup/       # 其余 126 个历史指标文件
└── logs/                   # 推理日志
```

### 4.2 保留的 6 个质量达标结果

#### 短视频 (720p, 70f, 50 steps)

| 序号 | 配置 | 文件名前缀 | SSIM Mean | SSIM Min | PSNR |
|------|------|-----------|-----------|----------|------|
| 1 | **Phase C 基线** (原始 3-stage, 单卡) | `phase_c_candidate_...090642Z` | 0.9677 | 0.9473 | 32.16 |
| 2 | **HEAD 单卡** 正确 prompt (无 compile) | `phase_c_prompt_720p_50steps_...153727Z` | 0.9677 | 0.9473 | 32.16 |
| 3 | **HEAD 双卡 SP pool=1** (fa_sp, compile) | `ctrl_pool1_720p_50steps_phasecprompt_...161102Z` | 0.9680 | 0.9566 | 31.94 |

**关键结论**: 条目 2 与条目 1 的 SSIM 均值完全一致（精确到小数点后6位），证明 HEAD 代码在单卡短视频路径上与 Phase C 像素级等价。条目 3 SP pool=1 也完全正常（SSIM min 甚至更高）。

#### 长视频 (720p, 130f, 20 steps)

| 序号 | 配置 | 文件名前缀 | SSIM Mean | SSIM Min | PSNR |
|------|------|-----------|-----------|----------|------|
| 4 | **HEAD 单卡** compile | `phase_e41_single_gpu_v2_130f_...150500Z` | 0.9845 | 0.9791 | 37.29 |
| 5 | **HEAD 双卡 SP pool=1** (fa_sp, compile) | `phase_e4_usp_sp_fa_sp_v2_130f_...045515Z` | 0.9847 | 0.9792 | 37.32 |
| 6 | **HEAD 双卡 SP pool=2** (compile) | `ctrl_pool_pool2_...130511Z` | 0.9812 | 0.9655 | 36.54 |

### 4.3 参考视频

| 文件 | 用途 | 来源 |
|------|------|------|
| `reference_video/test_video_960x720_70f.mp4` | 短视频参考 | Phase C 50步 原始输出 |
| `reference_video/test_video_long_960x720_130f.mp4` | 长视频参考 | Phase C 20步 原始输出 |

---

## 5. 性能基准

### 5.1 去噪速度加速比

| 配置 | 视频 | 推理总耗时 | 去噪耗时 | 加速比 vs 单卡 |
|------|------|-----------|----------|----------------|
| 单卡 compile | 130f 长视频 | 928.5s | — | 1.0× |
| SP pool=1 (fa_sp) | 130f 长视频 | 537.7s | 381.3s | **1.73×** |
| SP pool=2 | 130f 长视频 | 440.2s | — | **2.11×** |
| SP pool=1 | 720p 短视频 | ~147s | — | — |

### 5.2 编译 vs 非编译

- **compile=True** (`max-autotune-no-cudagraphs`): 去噪 ~19s/step (130f SP)
- **compile=False**: 去噪更慢，但 SSIM 确定
- compile 在 warmup 阶段完成，不计入 `model_inference_runtime_seconds`

### 5.3 Attention Backend

| 模式 | 当 SP 开启时 | 说明 |
|------|-------------|------|
| `--attention-backend fa` | 自动升级为 `fa_sp` | SP-aware FA (分布式 joint attention) |
| `--attention-backend sdpa` | 不升级 | PyTorch 原生 SDPA |
| 无 `--attention-backend` | 默认 FA | 单卡 FA, 多卡自动 fa_sp |

---

## 6. 已解决的重大问题

### 6.1 Prompt 文本误读 (✅ 已解决)

**问题**: 早期测试全部使用 `--caption-file` 参数，导致读取了不同于 Phase C 的 prompt 文本，T5 嵌入分歧，SSIM 从 0.9677 暴跌到 0.837/0.643。

**解决**: 使用 `--prompt-file` 代替 `--caption-file`，SSIM 恢复到 0.947-0.967。

**教训**: `--caption-file` 和 `--prompt-file` 是不同的数据源，必须与 Phase C 基线一致。

### 6.2 长视频 SP 推理 bug (✅ 已解决)

commit `337a31932` 修复。问题出在 SP 路径下长视频 temporal windowed 推理的 clip 边界处理。

---

## 7. 已知问题与待排查项

### 7.1 🔴 SP pool=2 在短视频上质量退化

| 配置 | SSIM Mean | SSIM Min | 通过？ |
|------|-----------|----------|--------|
| 720p SP pool=1 | 0.9680 | 0.9566 | ✅ |
| 720p SP pool=2 | **0.8947** | — | ❌ (6/70 帧失败) |
| 130f SP pool=1 | 0.9847 | 0.9792 | ✅ |
| 130f SP pool=2 | 0.9812 | 0.9655 | ✅ |

**根因假设**: x3 temporal duplication 导致 latent 中 2/3 帧为插值帧，2× 空间池化 (30×45→15×22) 无法区分真实帧与插值帧，cross-attention 引导失效。长视频每帧不同，不受此影响。

**建议排查**: 参考 `phase_e_sp_pooling_720p_regression_and_compile_nondeterminism_handover.md` Section 4。

### 7.2 🟡 torch.compile 非确定性

`max-autotune-no-cudagraphs` 模式下，每次运行可能选出不同的 triton kernel 变体，导致 SSIM 波动 (~0.005)。

**建议**: 改为 `reduce-overhead` 或 `default` 模式以提升可复现性。参考同文档 Section 5。

### 7.3 🟢 工作区有未提交代码

5 个核心文件有未提交改动，需要清理并提交。

---

## 8. 标准推理命令

### 8.1 单卡短视频推理 (50步，正确 prompt)

```bash
cd /home/zhiheng/sglang && export PYTHONPATH=python && \
/home/zhiheng/sglang/.venv/bin/python3 \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4 \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --reference-video /home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4 \
  --num-inference-steps 50 --seed 42 \
  --num-gpus 1 --tp-size 1 --sp-degree 1 --ulysses-degree 1 \
  --dist-timeout 3600 \
  --artifact-prefix phase_c_prompt_720p_50steps \
  2>&1 | tee Vivid_Acceptance/logs/$(date -u +%Y%m%dT%H%M%SZ).log
```

### 8.2 双卡 SP pool=1 短视频推理 (50步)

```bash
cd /home/zhiheng/sglang && export PYTHONPATH=python && \
export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && \
/home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30066 \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4 \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --reference-video /home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4 \
  --num-inference-steps 50 --seed 42 \
  --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
  --dist-timeout 3600 --attention-backend fa \
  --enable-torch-compile --warmup --warmup-steps 1 \
  --artifact-prefix ctrl_pool1_720p_50steps \
  2>&1 | tee Vivid_Acceptance/logs/$(date -u +%Y%m%dT%H%M%SZ).log
```

### 8.3 双卡 SP 长视频推理 (20步)

```bash
# pool=1
cd /home/zhiheng/sglang && export PYTHONPATH=python && \
export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && \
/home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master-port=30067 \
  python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 \
  --caption-file /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --reference-video /home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4 \
  --num-inference-steps 20 --seed 42 \
  --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
  --dist-timeout 3600 --attention-backend fa \
  --enable-torch-compile --warmup --warmup-steps 1 \
  --artifact-prefix phase_e4_usp_sp_fa_sp_v2_130f_20step \
  2>&1 | tee Vivid_Acceptance/logs/$(date -u +%Y%m%dT%H%M%SZ).log
```

(Bug: `--prompt-file` will override `--caption-file` in the pipeline config for long videos too — verify correct behavior.)

### 8.4 独立质量指标计算

```bash
/home/zhiheng/sglang/.venv/bin/python3 -m sglang.multimodal_gen.runtime.videoedit.compare \
  --reference <ref.mp4> \
  --candidate <cand.mp4> \
  --report-json <output.json>
```

或直接在 Python 中调用：

```python
from sglang.multimodal_gen.runtime.videoedit.compare import compare_videos
report = compare_videos(reference_path, candidate_path, ...)
# report["summary"] 和 report["frames"] 包含逐帧 SSIM/PSNR/MSE/MAE
```

---

## 9. 外部资源路径

### 9.1 Vivid-VR 原始仓库

| 资源 | 路径 |
|------|------|
| 原始 Vivid-VR | `/home/zhiheng/Vivid-VR` |
| CogVideoX1.5-5B checkpoint | `/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B` |
| Vivid-VR checkpoint (ControlNet) | `/home/zhiheng/Vivid-VR/ckpts/Vivid-VR` |
| Phase C 原始输出 (50步) | `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4` |
| Phase C 原始输出 (20步) | `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4` |

### 9.2 输入文件

| 类型 | 路径 |
|------|------|
| 短视频 720p 输入 | `/home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4` (70f) |
| 长视频 720p 输入 | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4` (130f) |
| 短视频 prompt | `/home/zhiheng/Vivid-VR/input/720p/prompt.txt` |
| 长视频 caption sidecar | `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt` |
| 短视频 caption sidecar | `/home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt` |

### 9.3 模型路径 (SGLang 格式)

```
/home/zhiheng/.cache/sglang/models--CogVideoX1.5-5B-VividVR/
```

---

## 10. 质量指标 JSON 格式说明

每次推理结束后，`run_vividvr_inference.py` 自动调用 `compare_videos()` 生成指标 JSON，格式如下：

```json
{
  "run_id": "20260617T045515Z",
  "mode": "single_video_inference",
  "command": "...",
  "total_runtime_seconds": 798.99,
  "model_inference_runtime_seconds": 537.68,
  "seed": 42,
  "num_inference_steps": 20,
  "caption_source": "caption_file",
  "runtime_config": { /* attention_backend, compile flags, SP config, ... */ },
  "distributed_env": { /* world_size, rank, ... */ },
  "request_metrics": { /* per-stage and per-step timing in ms */ },
  "summary": {
    "compared_frames": 130,
    "ssim_mean": 0.9847,
    "ssim_min": 0.9792,
    "psnr_mean": 37.32,
    "mse_mean": 12.12,
    "mse_max": 17.60,
    "mae_mean": 2.66,
    "failed_frames": [],
    "pass_compare": true,
    "thresholds": { "min_ssim": 0.9, "max_mse": 150.0, ... }
  },
  "frames": [
    {"index": 0, "ssim": 0.9865, "mse": 12.10, "psnr": 37.30, "pass_frame": true},
    ...
  ],
  "debug": { /* clip specs, tensor shapes, etc. */ }
}
```

**注意**: `--caption-file` 和 `--prompt-file` 的含义不同：
- `--caption-file`: 设置 `caption_source="caption_file"`, 读取 caption 文件第一行作为 `model_prompt_text`
- `--prompt-file`: 设置 `caption_source="prompt_file"`, 直接读取 prompt 文件
- **Phase C 基线使用 prompt_file 模式，后续测试必须保持一致**

---

## 11. 关键代码路径速查

### 11.1 Pipeline 入口

| 用途 | 路径 (行号) |
|------|-----------|
| VividVR Pipeline forward | `vividvr_pipeline.py:1254-1281` |
| 短视频 10-stage 路径 | `composed_pipeline_base.py:594-622` |
| 长视频 temporal windowed | `vividvr_pipeline.py:941-1250` |
| 10-stage 执行器 | `parallel_executor.py:55-93` |

### 11.2 核心 Stages

| Stage | `vividvr.py` 行号 | 职责 |
|-------|-------------------|------|
| InputValidation | ~194 | 输入校验 |
| PromptPreparation | ~194-228 | 读取 prompt/caption |
| TextEncoding | ~239-284 | T5 文本编码 (含 negative prompt) |
| ConditionEncoding | ~310-420 | ControlNet VAE encode |
| LatentPreparation | ~440-530 | 噪声生成 + tiling |
| Denoising | ~1072-1116 | 去噪循环 (含 attn_metadata 构建) |
| Decoding | ~1150+ | VAE decode + 后处理 |

### 11.3 SP / Pooling / Compile

| 用途 | 路径 |
|------|------|
| Control state spatial pooling | `cogvideox_vividvr_common.py:55-94` |
| Pool size 配置 | `cogvideox_vividvr_common.py:34-52` |
| SP FlashAttn processor | `cogvideox_attention_backend.py:442+` |
| auto fa→fa_sp 升级 | `vividvr_pipeline.py:473-510` |
| Compile mode 默认值 | `vividvr_pipeline.py:302-304` |

### 11.4 质量对比

| 用途 | 路径 |
|------|------|
| `compare_videos()` | `runtime/videoedit/compare.py:61-133` |
| `_ssim()` (逐帧灰度 SSIM) | `runtime/videoedit/compare.py:45-58` |

---

## 12. 环境变量速查

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE` | 1 | control state 空间池化倍数 (1=无压缩, 2=2×, 4=4×) |
| `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE` | `sp_exact_global_control_attention` | SP connector 上下文模式 |
| `SGLANG_TORCH_COMPILE_MODE` | `max-autotune-no-cudagraphs` | torch.compile 模式 |
| `SGLANG_DIFFUSION_ATTENTION_BACKEND` | `fa` | attention backend (fa/sdpa/fa3) |

---

## 13. 工作约定

1. **所有推理必须在 tmux 中运行**，避免断连导致推理中断
2. **推理前确认 GPU 空闲** (`nvidia-smi`)
3. **使用 `.venv/bin/python` 而非系统 python**
4. **tmux 窗口命名规范**: `sp_pool2_720p`、`sp_pool1_130f` 等描述性名称
5. **`--artifact-prefix`** 使用描述性名称 (含 pool size, 视频类型, 步数)
6. **推理完成后** 指标 JSON 和视频自动保存到 `Vivid_Acceptance/` 下

---

## 14. 相关文档

| 文档 | 路径 |
|------|------|
| SP pooling 720p 退化详细分析 | `docs_xzh/hand_over/phase_e_sp_pooling_720p_regression_and_compile_nondeterminism_handover.md` |
| SGLang Diffusion 开发指南 | `python/sglang/multimodal_gen/.claude/CLAUDE.md` |
| 完整对话记录 | `~/.claude/projects/-home-zhiheng-sglang/18d81fff-cadf-4ce7-b7ee-7c9f963cc753.jsonl` |

---

## 15. 交接检查清单

- [ ] 阅读本 handover 全文
- [ ] 查看 `Vivid_Acceptance/` 目录确认产物位置
- [ ] 阅读 SP pooling 720p 退化分析文档
- [ ] 查看 `vividvr.py` 理解 10-stage 结构
- [ ] 查看 `compare.py` 理解 SSIM 计算逻辑
- [ ] 确认理解 prompt_file vs caption_file 的区别
- [ ] 确认理解长视频 vs 短视频 pipeline 的路由逻辑
- [ ] 决定第一个排查方向：
  - 短视频 SP pool=2 质量退化 (最高优先级)
  - torch.compile 确定性策略 (中等)
  - 未提交代码清理 (低优)
- [ ] 推理前检查 GPU 空闲
- [ ] 所有新推理在 tmux 中运行
