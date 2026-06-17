# VividVR Phase E: SP Control Pooling 720p 质量回归 & Compile 不确定性交接

更新时间：`2026-06-17 UTC`

## 1. 文档目的

本 handover 面向下一位接手 `Phase E` SP Path B (control state spatial pooling) 以及 compile 确定性问题的 Codex，完整呈现：

1. 公平计时口径的确立过程及最终约定
2. SP pool=2 在不同视频 pipeline 上的质量表现（130f 正常 vs 720p 严重退化）
3. `max-autotune-no-cudagraphs` compile 模式导致的 SSIM 非确定性
4. 当前仍未解决的根因问题及建议的下一步排查方向

## 2. 代码与环境锚点

| 项 | 值 |
|----|-----|
| **分支** | `sglang_Vivid` |
| **HEAD commit** | `c0008cd89` — "Restore vividvr native SP v1/v2 connector semantics" |
| **前续关键 commit** | `9e43f5d31` — "sp2 双卡并行加速" |
| | `3187e66f8` — "Enable VividVR phase E2/E3 acceleration acceptance"（首次引入 torch.compile + max-autotune）|
| **Python** | 3.10（`/home/zhiheng/sglang/.venv/bin/python`） |
| **torch** | 2.9.1 |
| **GPU** | 2× NVIDIA A100-SXM4-80GB |
| **脏文件** | `cogvideox_attention_backend.py`（SP-aware FA processor） |
| | `cogvideox_vividvr_common.py`（Path B control pooling + 诊断日志） |
| | `vividvr_pipeline.py`（auto fa→fa_sp upgrade） |
| | `test_connector_remote_compress.py`（Path B 低层测试脚本） |
| | `run_vividvr_inference.py`（通用推理入口，已修改支持 720p） |
| | `diag_connector_attention.py`（connector attention 诊断工具，新增） |

## 3. 公平计时口径（已确立）

### 3.1 最终约定

```python
# Warmup: compile 在此时发生，不计入推理耗时
warmup_request = request.copy_as_warmup(warmup_steps=1)
pipeline.forward(warmup_request, server_args)
dist.barrier()

# 计时开始：compile 已完成，测量端到端推理
model_inference_start = time.perf_counter()
result = pipeline.forward(request, server_args)
torch.cuda.synchronize()
model_inference_runtime = round(time.perf_counter() - model_inference_start, 6)
```

### 3.2 关键澄清

- `ServerArgs.warmup=True` **不做任何实际 warmup**，只在 `server_args.py:393` 打一行日志
- 实际 warmup 必须通过 `request.copy_as_warmup()` → 单独的 `pipeline.forward()` 实现
- compile 发生在 warmup forward 期间，不计入 `model_inference_runtime_seconds`
- 指标 JSON 中必须包含 `model_inference_runtime_seconds`（不是 `denoise_time_seconds`）

### 3.3 当前基准结果（公平口径）

| 配置 | 视频 | inference_time | SSIM | pass |
|------|------|---------------|------|------|
| 单卡 compile | 130f 长视频 | 928.5s | 0.9845 | ✅ |
| SP (fa_sp, 无pool) | 130f 长视频 | 537.7s | 0.9847 | ✅ |
| SP pool=2 (Path B) | 130f 长视频 | 440.2s | 0.9655 | ✅ |
| **SP pool=2 (Path B)** | **720p 短视频** | **147.4s** | **0.6428** | **❌** |
| Phase C 单卡 baseline | 720p 短视频 | N/A | 0.9677 | ✅ |

## 4. 核心问题一：720p 短视频 SP pool=2 质量崩溃

### 4.1 现象

同一套 SP pool=2 配置（`SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=2`），在 130f 长视频上表现正常（SSIM=0.9655，close to single GPU 0.9791），但在 720p 短视频（`test_video_960x720.mp4`, 70f）上 SSIM 从单卡的 0.9677 暴跌到 0.614-0.643。

### 4.2 已验证的两条独立路径

| 测试脚本 | 720p SSIM | 结果文件 |
|----------|-----------|----------|
| `test_connector_remote_compress.py`（Path B 低层测试） | 0.6192 | `ctrl_pool_pool2_20260617T133134Z.json` |
| `run_vividvr_inference.py`（通用推理入口） | 0.6428 | `ctrl_pool2_720p_metrics_seed42_20260617T133902Z.json` |

两条路径独立运行，结果高度一致（SSIM ~0.62-0.64），排除了脚本差异因素。

### 4.3 关键差异：短视频 vs 长视频 pipeline

| 维度 | 720p 短视频 (70f) | 130f 长视频 |
|------|-------------------|-------------|
| 帧数 | 70 | 130 |
| 时序处理 | x3 duplication (70→73→121f) | clip split (2 temporal clips) |
| SP tokens global | 27000 | 27000 |
| SP tokens local | 13500 | 13500 |
| control pool | 13500→3300→6600 global | 13500→3300→6600 global |
| 压缩比 | 2.0× | 2.0× |
| 单卡 SSIM | 0.9677 | 0.9845 |
| pool=2 SSIM | **0.6428** | **0.9655** |

两者的 SP token 量完全一致（global=27000, local=13500, pooled global=6600），但质量差异巨大。这说明问题不在于 token 数量，而在于数据本身的特性。

### 4.4 怀疑方向

**假设 1（最可能）：x3 帧复制 + 空间池化 = 控制信号丢失**

720p pipeline 做了 x3 temporal duplication（70f→121f），这意味着 latent space 中有 2/3 的时间帧是插值出来的。控制状态是从原始 70 帧计算的，pool_size=2 将每帧的空间分辨率从 30×45 压缩到 15×22（~4×）。对于包含大量插值帧的场景，pooled control 缺乏足够的分辨率来区分真实帧和插值帧，导致 cross-attention 引导失效。

**假设 2：不同的 caption 结构**

720p 使用 `test_video_960x720_x3.txt`（3 个 caption entries），130f 使用其自身的 caption sidecar。caption 数量/内容可能影响 connector 行为。

**假设 3：control latent 的实际帧数不同**

虽然 debug 显示 control_context 都是 20 frames，但 720p pipeline 的 73 padded frames 经 VAE temporal compression 得到的实际 latent 帧数可能与 130f 不同（130f 经 VAE 压缩后也是 ~32+ frames），导致 per-frame token 分布有差异。

### 4.5 下一步排查建议（按优先级）

1. **跑 720p pool=1（即不压缩）验证 SP 本身不影响 720p 质量**：设置 `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`，用 `run_vividvr_inference.py` 跑一次 720p SP 推理。如果 SSIM 恢复正常（~0.97），则确认问题是 pool_size=2 特有的，不是 SP 本身的问题。

2. **如果 pool=1 正常，尝试 pool_size=2 但用不同的池化策略**：当前用 `F.adaptive_avg_pool2d`，可以尝试 `F.adaptive_max_pool2d` 或混合策略。

3. **如果 pool=1 也异常**：问题可能在 SP shard/gather 路径与 x3 duplication 的交互。需要深入排查 VividVRPipeline 中短视频 x3 路径的 SP shard 逻辑。

4. **检查 720p pipeline 的 temporal 处理细节**：确认 x3 duplication 后 latent 的实际帧数、control state 的帧数、以及 SP shard 策略是否正确对齐。

## 5. 核心问题二：`max-autotune-no-cudagraphs` compile 模式导致 SSIM 非确定性

### 5.1 现象

同样的配置（pool=2, compile, warmup, seed=42），三次独立运行 SSIM 分别为 0.9711、0.9679、0.9655。

### 5.2 根因

```python
# vividvr_pipeline.py:303
mode = os.environ.get("SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
```

`max-autotune-no-cudagraphs` 模式每次运行会 benchmark 多个 triton kernel 变体，选择当前最快的。由于 GPU 频率/温度微小波动，不同运行可能选出不同的"最优" kernel。不同 kernel 的浮点累加顺序不同 → 像素输出略微不同 → SSIM 波动。

### 5.3 该参数引入历史

首次引入于 commit `3187e66f8`（2026-06-06，"Enable VividVR phase E2/E3 acceleration acceptance"），此后一直作为默认值保留。

### 5.4 已确认的事实

- seed=42 的一致性完全可靠：`torch.Generator` 每次 forward 都重新创建并 manual_seed(42)
- 130f 单卡 compile 的 SSIM 极高（0.9845），compile 本身不降低质量
- 130f SP (无 pool) compile 的 SSIM（0.9847）与单卡一致
- 上述三次数值波动（0.9655-0.9711）都在 pool=2 路径上观察到

### 5.5 决策建议

需要决定是否将 compile mode 改为确定性模式：

| mode | 确定性 | 性能 | 建议场景 |
|------|--------|------|----------|
| `max-autotune-no-cudagraphs`（当前） | ❌ 不确定 | 最快 | 开发/探索阶段 |
| `max-autotune`（带 cudagraphs） | ❌ 不确定 | 快+更多内存 | 不推荐 |
| `reduce-overhead` | ✅ 确定 | 较快 | 推荐作为正式默认 |
| `default` | ✅ 确定 | 基准 | 最保守 |

如要切换，修改 `vividvr_pipeline.py:303` 的默认值，或通过 `SGLANG_TORCH_COMPILE_MODE=reduce-overhead` 环境变量。

## 6. 当前工作区未提交改动说明

以下文件有未提交改动，下一任接手时需要注意：

| 文件 | 改动性质 | 是否需提交 |
|------|----------|-----------|
| `cogvideox_attention_backend.py` | 新增 `CogVideoXSPFlashAttnProcessor` + USP attention | ✅ 核心功能 |
| `cogvideox_vividvr_common.py` | Path B control state spatial pooling (`_pool_control_state_2d`) + 诊断打印 | ✅ 核心功能 |
| `vividvr_pipeline.py` | 自动 fa→fa_sp 升级逻辑 | ✅ 核心功能 |
| `test_connector_remote_compress.py` | Path B 低层测试脚本（当前指向 720p 路径） | 测试工具，酌情提交 |
| `diag_connector_attention.py` | connector attention 诊断工具 | 诊断工具，酌情提交 |

## 7. 关键文件路径速查

### 7.1 代码

| 用途 | 路径 |
|------|------|
| SP control pooling 实现 | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:55-94` |
| Pool size 配置 | `cogvideox_vividvr_common.py:34-52` |
| SP attention processor | `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py:442+` |
| Compile mode 默认值 | `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:302-304` |
| Auto fa→fa_sp 升级 | `vividvr_pipeline.py:473-510` |
| 公平计时 warmup | `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py` (`copy_as_warmup()`) |

### 7.2 验收产物

| 用途 | 路径 |
|------|------|
| 指标目录 | `/home/zhiheng/sglang/Vivid_Acceptance/indicator/` |
| 结果视频目录 | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/` |
| 日志目录 | `/home/zhiheng/sglang/Vivid_Acceptance/logs/` |

### 7.3 关键指标文件

| 描述 | 文件 |
|------|------|
| Phase C 720p 单卡 baseline（SSIM=0.9677） | `indicator/phase_c_metrics_seed42_20260604T090642Z.json` |
| 130f pool=2 SP 正常结果（SSIM=0.9655） | `indicator/ctrl_pool_pool2_20260617T130511Z.json` |
| 720p pool=2 坏结果 - test_connector 脚本（SSIM=0.619） | `indicator/ctrl_pool_pool2_20260617T133134Z.json` |
| 720p pool=2 坏结果 - run_vividvr 脚本（SSIM=0.614） | `indicator/ctrl_pool2_720p_metrics_seed42_20260617T133902Z.json` |
| 130f 单卡 compile baseline（SSIM=0.9845） | `indicator/phase_e41_single_gpu_v2_130f_20step_compile_metrics_seed42_20260616T150500Z.json` |
| 130f SP fa_sp v2（SSIM=0.9847） | `indicator/phase_e4_usp_sp_fa_sp_v2_130f_20step_compile_metrics_seed42_20260617T045515Z.json` |

### 7.4 外部资源

| 用途 | 路径 |
|------|------|
| Vivid-VR 原版仓库 | `/home/zhiheng/Vivid-VR` |
| CogVideoX checkpoint | `/home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B` |
| Vivid-VR checkpoint | `/home/zhiheng/Vivid-VR/ckpts/Vivid-VR` |
| 720p 输入视频 | `/home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4` |
| 720p prompt | `/home/zhiheng/Vivid-VR/input/720p/prompt.txt` |
| 720p caption sidecar | `/home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt` |
| 720p 原版 reference 视频 | `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4` |

## 8. 标准推理命令（更新版）

### 8.1 720p SP pool=2 推理（当前已知失败的配置）

```bash
cd /home/zhiheng/sglang && export PYTHONPATH=python && \
export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && \
export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=2 && \
/home/zhiheng/sglang/.venv/bin/torchrun --nproc_per_node=2 --master_port=30603 \
python/sglang/multimodal_gen/tools/run_vividvr_inference.py \
  --input-video /home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4 \
  --caption-file /home/zhiheng/Vivid-VR/input/captions/test_video_960x720_x3.txt \
  --prompt-file /home/zhiheng/Vivid-VR/input/720p/prompt.txt \
  --reference-video /home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4 \
  --num-inference-steps 20 --seed 42 \
  --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 \
  --dist-timeout 3600 --attention-backend fa \
  --enable-torch-compile --warmup --warmup-steps 1 \
  --artifact-prefix ctrl_pool2_720p \
  2>&1 | tee Vivid_Acceptance/logs/ctrl_pool2_720p_$(date -u +%Y%m%dT%H%M%SZ).log
```

### 8.2 720p SP pool=1 验证（建议下一步第一步跑）

```bash
# 同上，但设置 pool_size=1（无压缩）
export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1
# ...其余参数相同
```

### 8.3 确定性 compile 验证

```bash
export SGLANG_TORCH_COMPILE_MODE=reduce-overhead
# ...其余参数相同
```

## 9. 未完成任务清单

1. **🔴 最紧急：排查 720p pool=2 质量崩溃根因**
   - 先跑 pool=1 确认 SP 本身是否影响 720p
   - 如 pool=1 正常，排查 pool_size=2 的池化策略是否适合 x3 duplication pipeline
   - 如 pool=1 也异常，排查 SP shard/gather 与短视频 x3 路径的交互

2. **🟡 中等：决定 compile 确定性策略**
   - 是否将 `max-autotune-no-cudagraphs` 改为 `reduce-overhead` 或 `default`
   - 权衡性能损耗 vs 可复现性
   - 如果保留 max-autotune，需要在验收流程中记录该不确定性

3. **🟢 低优：提交已完成的 SP 功能代码**
   - `CogVideoXSPFlashAttnProcessor`（fa_sp）
   - Path B control pooling
   - Auto fa→fa_sp 升级逻辑
   - 在提交前确保工作区干净且通过基本验证

4. **🟢 低优：720p 短视频 pool=4 验证（依赖问题 1 先解决）**

5. **🟢 低优：更新 benchmark 文档**
   - 将 Path B 130f pool=2 的结果（SSIM=0.9655, 440.2s）写入 `docs_xzh/run_vivid_benchmark.md`

## 10. 交接检查清单

- [ ] 阅读本 handover 全文
- [ ] 确认理解公平计时口径（Section 3）
- [ ] 确认理解 720p vs 130f pipeline 差异（Section 4.3）
- [ ] 查看关键指标 JSON 确认数字
- [ ] 查看 `cogvideox_vividvr_common.py:55-94` 理解 pooling 实现
- [ ] 决定第一个排查方向（建议先跑 pool=1 验证）
- [ ] 跑 inference 前确认 GPU 空闲（`nvidia-smi`），避免资源争抢
- [ ] 所有推理必须在 tmux 中运行
