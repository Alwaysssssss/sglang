# VividVR Phase E E3 手工融合优先级与 QK norm + RoPE 评估交接

更新时间：`2026-06-08 UTC`

## 1. 这份交接文档覆盖什么

本文档总结本轮对话里围绕 `Phase E / E3` 的几个关键结论，供后续 Codex 直接接手：

- 澄清当前 `E3` 正式验收通过的到底是哪条算子融合路径。
- 说明 `QK norm + RoPE` 为什么当前不能升格为正式 `E3` 配置。
- 说明 `QK norm + RoPE` 实际带来的加速量级，以及为什么端到端收益不大。
- 说明当前结果对“`torch.compile` vs 手工融合”的真实启示。
- 给出后续应优先推进的“真正值得做的手工融合”方向，而不是继续在低收益路径上消耗时间。


## 2. 当前最重要的结论

### 2.1 当前正式通过的 E3 不是 QK norm + RoPE

当前 `E3` 正式验收通过的主路径是：

- `modulation / residual fusion`

不是：

- `QKV fusion`
- `QK norm + RoPE`

后两者都只能视为“已接线或已实验过的候选路径”，不能写成当前正式 `E3` 结果。

### 2.2 当前单卡最好正式结果仍然是 E2

截至本次对话结束，当前最好正式单卡结果仍然是：

- `E2 = FA + torch.compile`
- `model_inference_runtime_seconds = 923.9699`

这比当前已通过的 `E3` 单项融合更快，也比“当前已测到的全开组合”更快。

### 2.3 QK norm + RoPE 目前不能作为正式可用方案

`QK norm + RoPE` 这条链路已经端到端跑通，但正式 A/B 中：

- 有小幅提速
- 质量 compare 明显失败

所以当前结论只能写成：

- “可运行、可测得局部和小幅端到端收益”

不能写成：

- “正式通过的 E3 加速方案”

### 2.4 后续应该继续做真正的手工融合

本轮对话的最终判断是：

- 当前这版 `E3` 小链路融合，整体不如 `torch.compile` 有效。
- 但这不等于“手工融合没有价值”。
- 后续应该继续做的是 `compile` 不容易自动合成、且位于真正热点上的深层 fused kernel。


## 3. 当前正式结果与配置口径

原版 `20 step` 长视频基线：

- `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f_report.json`
- `model_inference_runtime_seconds = 1047.001905`

### 3.1 E1 正式通过结果

配置：

- `attention_backend = fa`
- `dit_cpu_offload = false`
- `text_encoder_cpu_offload = false`
- `vae_cpu_offload = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e1_130f_20step_fa_metrics_seed42_20260606T045306Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 1016.256615`

### 3.2 E2 正式通过结果

配置：

- `attention_backend = fa`
- `enable_torch_compile = true`
- `dit_cpu_offload = false`
- `text_encoder_cpu_offload = false`
- `vae_cpu_offload = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e2_130f_20step_compile_metrics_seed42_20260606T084506Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 923.9699`
- 相对原版 `20 step`：
  - `1047.001905 / 923.9699 = 1.133156x`

说明：

- 这是当前最好正式单卡结果。

### 3.3 E3 正式通过结果

配置：

- `attention_backend = fa`
- `enable_cogvideox_modulation_fusion = true`
- `cogvideox_modulation_fusion_targets = transformer,controlnet`
- `enable_torch_compile = false`
- `dit_cpu_offload = false`
- `text_encoder_cpu_offload = false`
- `vae_cpu_offload = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e3_130f_20step_modulation_fusion_both_metrics_seed42_20260606T082346Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 1007.328337`
- 相对 `E1` 单项收益：
  - `1016.256615 / 1007.328337 = 1.008863x`
- 相对原版 `20 step`：
  - `1047.001905 / 1007.328337 = 1.039385x`

说明：

- 当前正式通过的 `E3` 是 `modulation / residual` 融合。
- 它能带来收益，但明显弱于 `E2 = torch.compile`。

### 3.4 当前已测到的单卡全开组合

配置：

- `attention_backend = fa`
- `enable_torch_compile = true`
- `enable_cogvideox_qkv_fusion = true`
- `enable_cogvideox_modulation_fusion = true`
- `dit_cpu_offload = false`
- `text_encoder_cpu_offload = false`
- `vae_cpu_offload = false`

formal report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e123_130f_20step_fa_compile_qkv_modfusion_metrics_seed42_20260606T091719Z.json`

结果：

- `pass_compare = true`
- `model_inference_runtime_seconds = 928.20107`
- `total_runtime_seconds = 1229.96182`
- 相对原版：
  - `1047.001905 / 928.20107 = 1.127990x`

说明：

- 组合全开虽然过了质量验收，但仍然略慢于 `E2 = 923.9699`。
- 不要把“所有加速都开”误写成“当前最佳单卡默认配置”。


## 4. 当前 E3 真实做了哪些算子融合

### 4.1 正式通过的主路径：modulation / residual fusion

核心实现文件：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_operator_fusion.py`

主要融合内容：

- `LayerNorm + scale/shift`
  - `LayerNormScaleShift`
- `residual add + gate + LayerNorm + scale/shift`
  - `ScaleResidualLayerNormScaleShift`
- `residual + gate * ff_output`
  - `MulAdd`

这条路径的特点是：

- 主要减少反复触发的小 kernel launch
- 不直接改变 attention 主核
- 因此端到端收益偏温和

### 4.2 已实现但不是正式通过主路径：QKV fusion

核心实现文件：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

当前接入方式本质上是：

- `to_q / to_k / to_v -> to_qkv`
- 然后再 `split` 回 `q / k / v`

需要特别记住：

- 当前这条路径不是更深的 `sglang packed-QKV attention kernel`
- 只是 diffusers 层的 fused linear
- 因此理论收益本来就有限

### 4.3 新实验过但未验收通过：QK norm + image RoPE

同样接入在：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`

这条路径尝试把：

- `Q/K LayerNorm`
- `image token RoPE`

接入 `sglang` / `flashinfer` 底层能力，但当前未通过正式 compare。


## 5. QK norm + RoPE 的当前状态

### 5.1 它已经端到端跑通，但未通过正式验收

本轮对话完成了 `QK norm + RoPE` 的正式 A/B：

baseline report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e3_130f_20step_qk_norm_rope_baseline_dit_offload_metrics_seed42_20260606T154018Z.json`

fusion report：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e3_130f_20step_qk_norm_rope_fusion_dit_offload_metrics_seed42_20260606T155846Z.json`

fusion log：

- `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e3_130f_20step_qk_norm_rope_fusion_dit_offload_20260606T155836Z.log`

结果：

- baseline：`pass_compare = true`
- fusion：`pass_compare = false`

fusion 的失败指标：

- `ssim_min = 0.607216`
- `mse_max = 6038.493652`
- `mae_max = 62.649200`
- `failed_frame_ratio = 1.000000`

结论：

- 这不是“边缘漂移”
- 而是明显语义回归

### 5.2 它带来的加速有多大

在 formal offload A/B 中：

- `model_inference_runtime_seconds`：
  - `1040.474058 -> 1008.275354`
  - `1.031934x`
- `total_runtime_seconds`：
  - `1088.132330 -> 1063.044362`
  - `1.023600x`

解释：

- 端到端确实有加速
- 但量级不大，模型段大约 `3.2%`

### 5.3 为什么 microbench 看起来更好

之前 synthetic attention hot-path microbench 结果是：

- `baseline`: `prepare 2.90 ms`, `processor 4.41 ms`
- `qkrope_only`: `prepare 1.82 ms`, `processor 3.40 ms`

即：

- attention processor 局部大约 `1.30x`

但端到端只剩约 `1.03x`，原因主要是：

- 这条融合只优化了 attention 前后的局部预处理，不是整个 denoise 主核。
- attention 主核本身已经是 `FA`。
- formal A/B 这次使用了 `offload`，host-device 搬运稀释了局部收益。
- `LayerNorm` 那侧 Triton kernel 这次没有完整吃上，测到的不是最理想速度。


## 6. 为什么 QK norm + RoPE 没通过验收

当前最合理的判断是：

- 问题更像是“实际 CUDA fast path 语义没有严格对齐”
- 而不是简单的 CLI 接线问题
- 也不是“它没有真正生效”

支持这个判断的证据有三条。

### 6.1 日志显示 fast path 确实被打开了

fusion log 明确出现：

- `Enabled CogVideoX QK-norm/RoPE fusion on VividVR transformer; attention_modules=42, effective_impl=sglang_layernorm+rope_accel.`

这说明：

- 开关不是假的
- runtime 侧确实走到了这条路径

### 6.2 现有单测主要证明 fallback 路径，不足以证明 CUDA fast path

关键单测文件：

- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

本轮对话里的判断是：

- 现有测试更多证明 Python/fallback 路径数值对齐
- 不能等价推出“正式验收时使用的 CUDA fast path 一定数值对齐”

所以当前缺的不是“再多写几个普通单测”，而是：

- 针对真正 CUDA fast path 的 parity 验证

### 6.3 LayerNorm Triton launcher 编译环境有问题

fusion log 中还有：

- `fatal error: Python.h: No such file or directory`

这说明：

- `LayerNorm` 对应的 Triton launcher 编译环境不完整
- 当前实跑路径很可能是“RoPE 生效，LayerNorm 部分 fallback”

这件事本身不一定解释 compare 失败，但至少说明：

- 当前测到的速度不是理论上限
- 当前路径状态不够干净，不能直接拿来做正式默认方案

### 6.4 当前最可疑的语义风险点

后续若继续排查，优先怀疑：

- `flashinfer` image-token RoPE fast path 的实际语义对齐

比起优先怀疑：

- 纯 fallback 的 LayerNorm 路径

原因是：

- 当前真正新的数值路径主要在 image RoPE CUDA fast path
- compare 失败是 catastrophic 级别
- 现有测试并没有充分覆盖这条真实 CUDA 路径


## 7. 这次为什么出现 offload，是否应理解为新的正式配置

不要这样理解。

### 7.1 之前正式通过的 E1 / E2 / E3 都没有使用 offload

这一点必须明确：

- `E1` formal：`offload = false`
- `E2` formal：`offload = false`
- `E3` formal：`offload = false`

这几项正式结果都建立在“非 offload”配置上。

### 7.2 QK norm + RoPE 这次使用 offload，只是当时的环境性 workaround

本轮排查时，非 offload 的 `QK norm + RoPE` smoke 先遇到了 OOM。

关键日志：

- `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e3_smoke_130f_2step_qk_norm_rope_20260606T152707Z.log`

日志里明确写到：

- GPU 总显存 `79.32 GiB`
- 只剩 `1.89 GiB` 可用
- 其他进程已经占用 `48.04 GiB` 和 `29.37 GiB`

因此：

- 这次切到 `dit/text/vae offload`
- 主要是为了在共享 GPU 环境下把实验跑完

它不应被重新解释成：

- “QK norm + RoPE 天然就必须用 offload”
- “Phase E 的默认正式配置已经变成 offload”

### 7.3 当前 GPU 环境提示

用户在本轮对话中提供的 `nvidia-smi` 快照时间是：

- `2026-06-08 03:10:41 UTC`

快照显示：

- `GPU 0`：`61213 MiB / 81920 MiB`，`100% util`
- `GPU 1`：基本空闲

这只能说明：

- 当前机器是共享状态
- 后续如果还要做正式 rerun，优先考虑独占空闲 GPU，必要时优先试 `GPU 1`

不要把这条环境快照写成长期稳定事实。


## 8. 这轮对话对“torch.compile vs 手工融合”的真实结论

### 8.1 当前这版 E3，不如直接用 torch.compile

数据上非常明确：

- `E1 (FA)`: `1016.256615`
- `E2 (FA + compile)`: `923.9699`
- `E3 (FA + modulation fusion)`: `1007.328337`

因此当前可以明确写：

- `torch.compile` 的单项收益明显强于当前已通过的 `E3` 小链路融合

### 8.2 这不等于“手工融合天然不如 compile”

更准确的理解是：

- 当前正式通过的 `E3` 打到的热点不够深
- `torch.compile` 能覆盖更长的图，端到端收益更大
- 当前 `QKV fusion` 也不是真正深层 packed-QKV kernel
- 因此当前这版 `E3` 看起来不如 `compile`

但如果后面做的是：

- `compile` 不容易自动合成的专用 fused kernel
- 且位于 attention 主热点或高频高成本路径

那么手工融合仍然很可能有价值。

### 8.3 当前“所有加速全开”并没有赢过 E2

组合全开结果：

- `928.20107`

仍然慢于：

- `E2 = 923.9699`

这再次说明：

- 当前手工融合与 `compile` 之间已经存在收益重叠
- “全开”不等于“最佳”


## 9. 后续应优先做什么真正的手工融合

本轮对话的最终建议是：

- 后续继续做手工融合
- 但不要继续把主要时间投入在当前这类低收益、浅层、易与 `compile` 重叠的路径上

### 9.1 第一优先级：补真正适配 CogVideoX 的 LayerNorm+bias + RoPE fused kernel

目标：

- 不是现在这种“LayerNorm 尝试走 kernel，失败就 fallback；RoPE 单独走 flashinfer”
- 而是给 `CogVideoX` 真实使用的 `nn.LayerNorm` 语义补齐一体化 fused kernel

原因：

- 当前 `QK norm + RoPE` 局部收益已经证明这个方向不是完全没价值
- 但现有实现仍然是半拼接态，不够干净

### 9.2 第二优先级：补 diffusion 路径可直接使用的 packed-QKV attention 接口

目标：

- 在 `sgl_kernel.flash_attn` / `sglang` wrapper 层提供 diffusion 路径可直接吃的 packed-QKV 接口

原因：

- 当前 `QKV fusion` 只是 `to_qkv -> split -> flash_attn`
- 不是真正意义上的深层 attention 融合
- 这也是它收益不大的根本原因

### 9.3 低优先级：继续打磨当前这版小链路融合

例如：

- 再多堆几条 pointwise 小链路融合

当前不建议把这类工作放在最前面，因为：

- 端到端 ROI 不高
- 与 `compile` 容易重叠
- 很可能继续出现“实现复杂，但正式收益很小”的情况


## 10. 下一个 Codex 的建议起手顺序

1. 先保护当前已验收基线：
   - `Phase D`
   - `E1`
   - `E2`
   - 已通过的 `E3 modulation fusion`

2. 不要把 `QK norm + RoPE` 当前实现升格为正式 `E3` 默认方案。

3. 如果继续做 `QK norm + RoPE` 排查，先处理运行环境：
   - 优先使用独占 GPU
   - 必要时优先试空闲 GPU
   - 使用 `CC=/home/zhiheng/sglang/scripts/gcc_python310_headers_wrapper.sh`

4. 在继续做 formal benchmark 前，先补真正的 CUDA parity 验证：
   - fused vs unfused 的 `q/k` 数值对齐
   - image token RoPE 前后对齐
   - 真实 CUDA fast path，而不是只测 fallback

5. 如果继续追求手工融合的真实收益，优先开新工作线：
   - `LayerNorm+bias + RoPE` fused kernel
   - packed-QKV attention 接口

6. 继续遵守单项 A/B 纪律：
   - 每次只引入一个新的主要加速变量
   - 不要把组合实验的结果反向当成单项收益结论

7. 所有重型推理与验收仍然必须放在 `tmux` 中运行。


## 11. 当前工作区提醒

截至写这份文档时，工作区不是干净状态。

已可见的相关改动包括：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
- `python/sglang/multimodal_gen/runtime/models/schedulers/cogvideox_dpm_vividvr.py`
- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`
- `python/sglang/multimodal_gen/runtime/utils/common.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_inference.py`

以及多份未提交的验收产物与 handover 文档。

默认要求：

- 不要对这些文件做破坏性清理
- 不要因为它们是 untracked 或 modified 就直接回滚

