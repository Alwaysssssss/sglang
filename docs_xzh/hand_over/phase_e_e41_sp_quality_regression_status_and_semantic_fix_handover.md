# VividVR Phase E4.1 SP 质量回退现状与语义修复交接

更新时间：`2026-06-15 UTC`

## 1. 这份文档解决什么问题

本 handover 面向下一位继续处理 `Phase E4.1` 的 Codex，目标是把当前项目状态、已经做过的验证、当前明确的坏现象，以及下一步应如何修正 `SP` 并行引入的视频质量问题，一次性讲清楚。

当前主判断是：

- `Phase C` 与 `Phase D` 基线仍然有效，不能回归。
- `Phase E4.1` 的双卡 `SP` 路径已经证明可以真实提速。
- 但当前 `native SP` / `v2` 相关路径存在质量回退，回退后的结果会稳定落到 `native-like` 的坏簇。
- 本轮更倾向于把问题归因为“推理语义变化”，而不是底层环境变化。

本轮没有修改代码，只整理现状并形成交接。

## 2. 当前项目实现状态

### 2.1 已冻结的阶段基线

以下阶段基线仍然视为已验收、必须继续保护：

- `Phase C`
  - 单 clip 主链语义已对齐并验收。
- `Phase D`
  - 长视频 `clip split / timestep orchestration / latent merge / trim / stitch` 已验收。
- `Phase E3.2`
  - 当前单卡质量控制基线仍是：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`
  - `model_inference_runtime_seconds = 935.243947`
  - `ssim_mean = 0.9844564849526698`
  - `ssim_min = 0.9799698863913181`

### 2.2 当前 `E4.1` 已完成的实现层面

当前仓库已经具备并跑通过的能力包括：

- 双卡 `SP` runtime 接线
- `native SP` 的 shard / gather 基础路径
- `v1 / v2` connector 语义恢复代码
- 长视频 `130f / 20 step` formal benchmark 工具链
- 标准化产物输出：
  - `Vivid_Acceptance/logs`
  - `Vivid_Acceptance/indicator`
  - `Vivid_Acceptance/result_videos`

### 2.3 当前代码锚点

当前分支与代码锚点：

- branch: `sglang_Vivid`
- `HEAD = c0008cd89c1fb56f348b2c29820703da8b6a511b`
- commit message:
  - `Restore vividvr native SP v1/v2 connector semantics`

当前工作区脏状态只有文档：

- `M docs_xzh/add_strategy/README.md`
- `?? docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`
- `?? docs_xzh/hand_over/phase_e_e41_native_sp_v2_quality_control_and_next_speedup_handover.md`

没有发现当前运行时代码的未提交改动。

## 3. 当前几条关键结果线

### 3.1 单卡质量基线

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e32_runtime_e2_align_130f_20step_compile_metrics_seed42_20260609T025514Z.json`
- 结论：
  - `model_inference_runtime_seconds = 935.243947`
  - `ssim_mean = 0.9844564849526698`
  - `ssim_min = 0.9799698863913181`

### 3.2 runtime-only 双卡对照

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_sp_only_130f_20step_compile_metrics_seed42_20260611T041018Z.json`
- 结论：
  - 速度几乎不变
  - 质量接近单卡

这条对照线的重要意义是：

- 双卡 runtime / distributed 环境本身不是质量问题根因。
- 问题更像是模型内部 `SP` 语义改变，而不是“只要一上双卡就坏”。

### 3.3 最快 native SP fast path

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_only_130f_20step_compile_metrics_seed42_20260611T052918Z.json`
- 结论：
  - `model_inference_runtime_seconds = 396.745880`
  - `ssim_mean = 0.9627860811380421`
  - `ssim_min = 0.9152052581958419`

这条线说明：

- 双卡 `native SP` 的速度收益是真实成立的。
- 但这条最激进 fast path 不能作为质量候选。

### 3.4 `quality opt_v1`

- recheck 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v1_130f_20step_compile_metrics_seed42_20260611T133530Z.json`
- 结论：
  - `model_inference_runtime_seconds = 470.026965`
  - `ssim_mean = 0.9785842231502916`
  - `ssim_min = 0.9529226959647417`

这条线说明：

- `v1` 比 fast path 质量明显更好。
- 但 `v1` 仍弱于单卡 / 冻结 `v2`，不是最终质量目标。

### 3.5 冻结的好 `v2`

当前应视为质量真值的正式 `v2` 基线：

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_metrics_seed42_20260611T134903Z.json`
- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_recheck_20260611T134851Z.log`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_quality_opt_v2_130f_20step_compile_seed42_20260611T134903Z.mp4`

冻结结论：

- `model_inference_runtime_seconds = 539.324976`
- `total_runtime_seconds = 805.179149`
- `ssim_mean = 0.9846050631221304`
- `ssim_min = 0.9778964153159052`
- `attention_backend_resolved = fa`
- `attn_metadata_builder = FlashAttentionMetadataBuilder`
- `control_context_shape_local = [2, 13500, 3072]`
- `control_context_shape_global = [2, 27000, 3072]`

当前所有后续提速，都必须以这条结果为质量对照真值。

## 4. 当前已经确认的坏现象

### 4.1 纯双卡 `sp2`、无 FA、无 compile

- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_pure_sp2_sdpa_no_compile_130f_20step_metrics_seed42_20260615T054934Z.json`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_quality_opt_v2_pure_sp2_sdpa_no_compile_130f_20step_seed42_20260615T054934Z.mp4`

结果：

- `model_inference_runtime_seconds = 522.485439`
- `ssim_mean = 0.9650776068823614`
- `ssim_min = 0.912228775788142`

这个结果说明：

- 它虽然跑通了 `sp2`，也进入了 `v2` connector 语义标签
- 但视频结果并没有对齐冻结 `v2`
- 它实际落回了之前 `native-like` 的失败簇

### 4.2 当前 `HEAD` 上重跑 `FA + compile + eager_global`

为了排除“只是因为关掉 FA/compile 才坏”，已经在当前 `HEAD c0008cd89` 上重跑了名义上与冻结好结果一致的组合：

- 日志：
  - `/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_e41_native_sp_quality_opt_v2_fa_compile_recheck_20260615T065232Z.log`
- 指标：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_e41_native_sp_quality_opt_v2_fa_compile_recheck_130f_20step_metrics_seed42_20260615T065242Z.json`
- 视频：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_e41_native_sp_quality_opt_v2_fa_compile_recheck_130f_20step_seed42_20260615T065242Z.mp4`

结果：

- `model_inference_runtime_seconds = 451.88314`
- `total_runtime_seconds = 696.711053`
- `ssim_mean = 0.9645150142503939`
- `ssim_min = 0.9114373253071709`
- `attention_backend_resolved = fa`
- `attn_metadata_builder = FlashAttentionMetadataBuilder`
- `control_context_shape_local = [2, 13500, 3072]`
- `control_context_shape_global = [2, 27000, 3072]`

这说明：

- 当前坏结果不是单纯“因为没用 FA / compile”
- 即便名义配置已经回到 `v2 + FA + compile + eager_global`
- 结果仍然落到 `0.964 / 0.911` 附近的坏簇

## 5. 当前最重要的判断

### 5.1 不是“没有进入 v2 标签”

从日志和指标看，当前坏结果都已经进入了 `v2` 相关的显式语义标签：

- `connector_context_mode = sp_exact_global_control_attention`
- `control_context_shape_local = [2, 13500, 3072]`
- `control_context_shape_global = [2, 27000, 3072]`

所以问题不能再简单表述为：

- “命令没进 `v2`”
- “只是 attention backend 选错了”

### 5.2 当前更像是“同名 v2，实际语义已经变了”

在用户给定前提“底层环境没有变化”下，当前更合理的工作假设是：

- 问题主要来自 `SP` 并行相关推理语义变化
- 而不是 CUDA / torch / flash-attn / checkpoint 漂移

换句话说，当前最需要追的不是“环境变了没有”，而是：

- 当前代码路径里，哪一步虽然仍然产出了 `global_control` 形状
- 但它的数值语义、排列语义、local/global 对位关系，已经不再等价于冻结 `v2`

### 5.3 速度异常变快本身也是信号

冻结好 `v2`：

- `model_inference_runtime_seconds = 539.324976`

当前 `HEAD` 上 `FA + compile` 重跑坏结果：

- `model_inference_runtime_seconds = 451.88314`

如果配置名相同、环境不变，但纯推理时间少了约 `87.4s`，更应怀疑：

- 有一段本来属于 `v2` 真语义的成本被省掉了
- 而省掉的恰好就是保证质量的那部分计算或数据恢复语义

因此，“更快但更差”本身就是语义退化的重要旁证。

## 6. 下一步任务应如何定义

当前下一步任务不应再定义成泛化的“继续提速”，而应收缩为：

- 修正 `SP` 并行造成的视频结果质量问题
- 先恢复与冻结 `v2` 真值一致的推理语义
- 再讨论进一步提速

更具体地说，下一轮主线应该是：

1. 先确认冻结 `v2` 的关键张量级合同
2. 找出当前 `HEAD` 上哪一段 `SP` 路径虽然保留了接口标签，但已经改变了语义
3. 恢复该段语义，重新得到与冻结 `v2` 对齐的结果
4. 只有在质量重新对齐之后，才继续做保语义优化

## 7. 下一轮建议重点排查的代码位置

以下位置应视为下一轮优先排查对象：

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`
  - `Connector.forward()`
  - `unpack_vividvr_connector_context()`
  - `build_vividvr_connector_control_states()`
  - `restore_vividvr_connector_global_control_state()`
  - `restore_vividvr_connector_global_control_states()`
  - `run_vividvr_connector_attention()`
  - `run_vividvr_connector_sequence_parallel_attention()`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - `connector_context_mode` 相关分支
  - `runtime_config` / `debug` 记录位置
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_attention_backend.py`
  - `fa` / `native` backend 归一化与 processor 安装逻辑

重点不是先做大改，而是回答以下问题：

- 当前 `local_control` 与 `global_control` 的构造顺序是否仍与冻结 `v2` 一致
- gather 前后的 `dtype / contiguous / reshape / unbind` 顺序是否改变了数值语义
- `Connector` 中 `attention` 使用的 `global_control` 与 `c_mlp` 使用的 `local_control` 是否仍保持正确对位
- 是否有某条 `SP` fast path 在名义上保留了 `global shape`，但实际上绕过了原本完整的 `v2` 语义成本

## 8. 下一轮工作纪律

下一轮应遵守以下原则：

- 不要再把“跑通 + pass_compare=true”当作 `v2` 验收通过
- 任何质量判断都必须对照冻结 `v2` 视频或其严格指标
- 任何策略如果再次落到 `ssim_mean ≈ 0.964 / ssim_min ≈ 0.911` 这一坏簇，可快速判为失败
- 在没有恢复 `v2` 真语义前，不要继续叠加新的提速技巧
- 优先做语义收敛，而不是继续追求 `15s/it`

## 9. 这份 handover 与哪些文档一起看

建议按以下顺序一起阅读：

- `docs_xzh/hand_over/phase_e_e41_native_sp_v2_quality_control_and_next_speedup_handover.md`
- `docs_xzh/add_strategy/13_phase_e_sp_quality_closure_plan.md`
- 本文档

三者之间的关系是：

- 旧 handover 记录了 `v1 / v2` 语义与早期提速判断
- `13_phase_e_sp_quality_closure_plan.md` 记录了此前失败策略与避免继续踩坑的提速计划
- 本文档补充了 `2026-06-15` 的最新事实：
  - 当前 `HEAD` 上即使重跑名义 `v2 + FA + compile`，也出现了质量回退
  - 后续主线应从“提速”切回“修复语义”

