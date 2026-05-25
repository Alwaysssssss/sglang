# STAR -> SGLang 任务交接文档（20260525_03）

本文档用于给下一位 Codex 接棒当前 `STAR` 集成与优化工作。  
这份交接重点覆盖：

1. 这轮已经完成了什么修改
2. 现在各条路径的真实状态
3. 当前最关键的问题是什么
4. 下一阶段应该优先做什么
5. 哪些方向暂时不要继续投入

---

## 1. 这轮已经完成的工作

### 1.1 运行时去掉 `STAR_mg` 代码路径依赖

已完成：

- STAR VAE 运行时不再动态扫描外部 `STAR_mg/cogvideox-based/sat`
- 将 STAR VAE 运行时真正依赖的 SAT 子树 vendor 到 `sglang` 仓库内部
- 运行时代码已改为固定从 vendored 路径加载

关键位置：

- [star_cogvideox_vae.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py:1)
- [star_sat_vendor](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/vaes/star_sat_vendor)

说明：

- 现在 `sglang` 推理运行时不再需要 `STAR_mg` 仓库代码路径
- 但原版 STAR reference 命令当然仍然需要 `STAR_mg` 仓库，因为那本来就是原版实现

### 1.2 本地模型目录已可独立准备

已完成：

- 用 `convert_star_cogvideox_sr.py` 将 STAR 资产转为 `sglang` 本地模型目录
- 已验证本地模型目录可被当前仓库加载

本地模型目录：

- [sglang_star_cogvideox_sr](/sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr)

### 1.3 推理命令文档已修正

已完成：

- 修正 [infer_STAR.md](/sgl-workspace/sglang/infer_STAR.md:1)
- 文档中的命令现在都使用绝对路径
- 原版 STAR 单-case 数据准备已写成自包含步骤
- 本地模型转换步骤已写清
- `FP8` 权重目录复制步骤已写清
- exact / FP8 / 历史 1.4x exact 复现实验命令都已更新

### 1.4 FSDP text encoder 回迁问题已修

问题：

- 当前本地模型目录 + `text_encoder_cpu_offload` + FSDP text encoder 组合下
- `TextEncodingStage` 之前会强行 `.to(cuda)` 回迁 encoder
- 这会触发 FSDP 参数存储设备不匹配错误

已完成修复：

- `TextEncodingStage` 现在会跳过 FSDP CPU-offload text encoder 的强制 `.to(cuda)` 回迁
- 普通 CPU encoder 仍保持原逻辑

关键位置：

- [text_encoding.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py:1)
- [test_text_encoding_stage.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/test/unit/test_text_encoding_stage.py:1)

### 1.5 本轮实跑结果

#### exact 本地模型目录路径

已跑通：

- 输出目录：
  [star_repro_single_fa_compile_fusedln_v2_fps8_localmodel](/sgl-workspace/sglang/outputs/star_repro_single_fa_compile_fusedln_v2_fps8_localmodel)
- summary：
  [summary.json](/sgl-workspace/sglang/outputs/star_repro_single_fa_compile_fusedln_v2_fps8_localmodel/summary.json:1)

结果：

- `avg_wall_clock_s = 181.9636`
- `warm_e2e_speedup = 1.2532x`
- baseline parity：通过
- strict parity：未通过

#### FP8 本地模型目录路径

已跑通：

- 输出目录：
  [star_fp8_full_localmodel_v1](/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1)
- summary：
  [summary.json](/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1/summary.json:1)

结果：

- `avg_wall_clock_s = 162.1279`
- `warm_e2e_speedup = 1.4065x`
- baseline parity：通过
- strict parity：未通过

---

## 2. 当前状态的核心判断

### 2.1 代码与环境层面

当前状态可以概括为：

- `sglang` 运行时已基本脱离 `STAR_mg` 代码路径
- 本地模型目录方案已跑通
- exact 路线和 FP8 路线都能在当前仓库 + 本地模型目录下完成端到端推理

### 2.2 质量层面

当前最关键的问题不是“能不能出视频”，而是：

- 我们的生成画面和原版 STAR 仍然没有对齐到 strict 阈值
- 当前 baseline parity 能过，但 strict parity 仍失败
- 主观上也仍然感觉和原版 STAR 有一点差距

这意味着：

- 现在最优先的目标应该从“继续堆加速”切回到“先把画面对齐”

### 2.3 性能层面

当前各条路径大致如下：

- 全局最快且质量通过：双卡 `dual_cfg_parallel = 1.8628x`
- 当前单卡 FP8 本地模型目录：`1.4065x`
- 当前单卡 exact 本地模型目录：`1.2532x`
- 历史单卡 exact 最佳：`single_fa_compile_fusedln_v2 = 1.4314x`

关键结论：

- `FP8` 路线虽然是当前“单卡、量化方案”里更快的一条
- 但它**不是下一阶段的主目标**
- 下一阶段主目标应该是先达成 strict quality 对齐，再回到 attention / operator 优化

---

## 3. 下一阶段明确优先级

这是本轮用户已经明确给出的新优先级，请下一位 Codex 严格按此执行：

### 第一优先级：先做画面对齐

目标：

- 达到 [phase_5_decoding_parity_and_acceptance.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_5_decoding_parity_and_acceptance.md:1) 中的 strict 阈值
- 先把生成画面和原版 STAR 做更严格的对齐

含义：

- 当前工作主线不是先追 `1.6x / 1.8x`
- 也不是先继续推量化路线
- 而是先把 strict quality 做起来

### 第二优先级：画面对齐后，再做 attention / operator 优化

在 strict 阈值满足之后，再继续：

- attention 相关优化
- kernel / operator 相关优化

这一阶段的优化目标应该优先放在：

- `sglang` 的 attention 部分
- `sglang` 的算子部分

### 暂时不要继续主攻的方向

当前用户已明确：

- 暂时先**不做量化这条加速方案**

这意味着：

- `FP8`
- `AWQ`
- `Nunchaku / SVDQuant`

这些都不是下一阶段的主线工作重点。

可以保留现有成果和命令，但不要再把主要精力放在这上面。

---

## 4. 为什么现在要先做 strict 对齐

原因很简单：

1. baseline parity 已经过了
2. 但 strict parity 还没过
3. 主观质量也和原版 STAR 还有一点差距
4. 如果这时候继续堆加速，很容易把“质量偏差”和“性能改动”混在一起

所以正确顺序应该是：

1. 先把画面与 strict 阈值对齐
2. 锁定 quality-safe 主线
3. 再围绕 attention / operators 做性能优化

---

## 5. 下一位 Codex 建议直接做的事情

### 5.1 先建立 strict 对齐调试基线

建议以当前 exact 路线为主，不要从 FP8 开始。

推荐从这条 exact 命令开始：

- 见 [infer_STAR.md](/sgl-workspace/sglang/infer_STAR.md:1) 的 exact profile 命令

重点不是速度，而是：

- 逐步把 candidate 拉近原版 STAR reference

### 5.2 重点排查画面对齐的几类源头

建议优先排查：

1. condition video latent 语义
2. decode 前 latent 语义
3. VAE encode / decode 行为
4. timestep / scheduler 细节
5. denoise 中 attention 路径是否与历史 reference 存在数值差异
6. rope / local enhancer / modulation / norm 路径是否有细微偏差

### 5.3 在 strict 对齐阶段，不要让变量过多

建议：

- 先用 exact
- 单卡
- 固定 `FA`
- 固定 `8 fps`
- 固定当前 reference case

不要同时改：

- quantization
- cache
- 多卡并行
- 新 backend

### 5.4 strict 对齐完成后，再做 attention / operator 优化

strict 过线后，再进入：

1. attention kernel 路径比较
2. fused operator 深挖
3. denoise 主循环热路径 profile

---

## 6. 当前已知不建议作为下一主线的方向

### 6.1 FP8 路线

虽然已经跑通，而且 `1.4065x` 不差，但：

- strict 仍未通过
- 用户已明确下一步先不做量化

因此：

- 保留文档和命令
- 暂不作为下一轮主线

### 6.2 双卡 cfg-parallel

虽然快，但：

- 不是用户当前最关心的问题
- 当前主问题是 strict 对齐

因此：

- 可以留作最终性能对照
- 不是下一轮主任务

---

## 7. 关键文件参考

### 核心代码

- STAR transformer：
  [star_cogvideox_sr.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/star_cogvideox_sr.py:1)
- STAR pipeline：
  [star_cogvideox_sr_pipeline.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines/star_cogvideox_sr_pipeline.py:1)
- STAR VAE：
  [star_cogvideox_vae.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/vaes/star_cogvideox_vae.py:1)
- Text encoding stage：
  [text_encoding.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/text_encoding.py:1)
- Condition video VAE stage：
  [video_condition_vae_encoding.py](/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/video_condition_vae_encoding.py:1)

### 文档与台账

- strict / baseline 验收标准：
  [phase_5_decoding_parity_and_acceptance.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_5_decoding_parity_and_acceptance.md:1)
- phase6 性能与 benchmark 口径：
  [phase_6_performance_hardening_and_upstream_sync.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_6_performance_hardening_and_upstream_sync.md:1)
- 运行命令：
  [infer_STAR.md](/sgl-workspace/sglang/infer_STAR.md:1)
- FP8 路径配置总结：
  [config_running_time.md](/sgl-workspace/sglang/docs_xzh/add_STAR/config_running_time.md:1)
- 总台账：
  [compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:1)

### 关键产物

- exact 本地模型目录复现实验：
  [star_repro_single_fa_compile_fusedln_v2_fps8_localmodel](/sgl-workspace/sglang/outputs/star_repro_single_fa_compile_fusedln_v2_fps8_localmodel)
- FP8 本地模型目录复现实验：
  [star_fp8_full_localmodel_v1](/sgl-workspace/sglang/outputs/star_fp8_full_localmodel_v1)

---

## 8. 最后一句话

下一位 Codex 不要把主要精力继续放在量化加速上。  
当前最优先任务已经明确变成：

- **先把生成画面和原版 STAR 做 strict 阈值对齐**
- **对齐完成后，再重点做 attention 和 operators 的性能优化**
