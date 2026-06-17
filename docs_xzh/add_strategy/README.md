# Vivid-VR 接入 SGLang 实施策略索引

本目录是后续把 `/home/zhiheng/Vivid-VR` 接入 `/home/zhiheng/sglang/python/sglang/multimodal_gen` 的正式实施方案。

这些文档以 `/home/zhiheng/xzh_docs/add_rules` 为最高优先级规范，目标不是复述 `Vivid-VR` 现有实现，而是回答：

> 如果按照 SGLang 现有 modular / model-specific 接入规范，Vivid-VR 应该如何被改造成可维护、可注册、可验证的原生接入。

## 总结论

- `Vivid-VR` 不是 Wan 变体，而是基于 `CogVideoX1.5-5B` 的自定义视频修复 / 恢复 pipeline。
- 它的核心不是单一 diffusers pipeline 包装，而是三层叠加：
  - `CogVideoX` 基座
  - VividVR 自定义 transformer / controlnet / scheduler 差异
  - 空间 tiling、跨 clip 时间聚合、caption、AdaIN、OCR/ESRGAN 后处理
- `sglang.multimodal_gen` 当前没有可直接复用的 `CogVideoX` runtime 组件，因此不能把实施方案写成“只加一个 pipeline 文件”。
- 推荐路线不是细粒度的通用 modular 拆分，而是：
  - `pipeline` 层负责长视频 orchestration
  - `model-specific stages` 负责单 clip 的准备、denoise、decode
  - `runtime helper` 负责 tiling / temporal merge / postprocess
- 第一版优先做单卡正确性，不优先做多卡与极限性能。
- 在 `sglang` 集成阶段，禁止实时调用 `CogVLM2` 生成 caption；统一读取现成 caption 文件：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- `CogVLM2` 在 `sglang` 环境中的接入位置只保留占位，不作为当前阶段实现目标。
- `DiffusersPipeline` generic wrapper 只能作为调试和对照工具，不能作为最终生产实现路径。

## 阅读顺序

1. [01_stage0_project_survey.md](./01_stage0_project_survey.md)
2. [02_stage1_sglang_mapping.md](./02_stage1_sglang_mapping.md)
3. [03_stage2_mvp_scope.md](./03_stage2_mvp_scope.md)
4. [04_stage3_pipeline_mod_plan.md](./04_stage3_pipeline_mod_plan.md)
5. [05_stage4_component_migration.md](./05_stage4_component_migration.md)
6. [06_stage5_acceleration_adaptation.md](./06_stage5_acceleration_adaptation.md)
7. [07_stage6_risk_analysis.md](./07_stage6_risk_analysis.md)
8. [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
9. [09_code_mod_order.md](./09_code_mod_order.md)
10. [10_grouped_stage_acceptance.md](./10_grouped_stage_acceptance.md)
11. [11_phase_e_acceleration_implementation.md](./11_phase_e_acceleration_implementation.md)
12. [12_phase_e_sp_native_acceleration_plan.md](./12_phase_e_sp_native_acceleration_plan.md)
13. [13_phase_e_sp_quality_closure_plan.md](./13_phase_e_sp_quality_closure_plan.md)

## 推荐执行分组（5 个大阶段）

正式路线图仍然按 `8` 个小 phase 维护，便于逐项验收；如果进入实际代码改造，建议合并成下面 `5` 个大阶段推进。

### 大阶段 A: 方案冻结 + 工程入口

- 对应小阶段：`Phase 1 + Phase 2`
- 目标：冻结 MVP 范围、reference 口径、checkpoint 契约，并补齐 `config / sampling / registry` 工程入口
- 核心文档：
  - [01_stage0_project_survey.md](./01_stage0_project_survey.md)
  - [02_stage1_sglang_mapping.md](./02_stage1_sglang_mapping.md)
  - [03_stage2_mvp_scope.md](./03_stage2_mvp_scope.md)
  - [04_stage3_pipeline_mod_plan.md](./04_stage3_pipeline_mod_plan.md)
  - [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
  - [09_code_mod_order.md](./09_code_mod_order.md)

### 大阶段 B: 核心模型底座迁移

- 对应小阶段：`Phase 3 + Phase 4`
- 目标：补齐 `CogVideoX` base transformer / VAE / scheduler，并叠加 `VividVR` 的 transformer / controlnet 增量
- 核心文档：
  - [02_stage1_sglang_mapping.md](./02_stage1_sglang_mapping.md)
  - [04_stage3_pipeline_mod_plan.md](./04_stage3_pipeline_mod_plan.md)
  - [05_stage4_component_migration.md](./05_stage4_component_migration.md)
  - [07_stage6_risk_analysis.md](./07_stage6_risk_analysis.md)
  - [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
  - [09_code_mod_order.md](./09_code_mod_order.md)

### 大阶段 C: 单 clip MVP + Reference 对齐

- 对应小阶段：`Phase 5`
- 目标：建立单 clip 端到端 pipeline，并把输出修到 reference 对齐达标
- 核心文档：
  - [03_stage2_mvp_scope.md](./03_stage2_mvp_scope.md)
  - [04_stage3_pipeline_mod_plan.md](./04_stage3_pipeline_mod_plan.md)
  - [05_stage4_component_migration.md](./05_stage4_component_migration.md)
  - [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
  - [09_code_mod_order.md](./09_code_mod_order.md)

### 大阶段 D: 长视频能力 + 可选增强

- 对应小阶段：`Phase 6 + Phase 7`
- 目标：在单 clip 达标后补长视频 orchestration，并接入 `caption / postprocess` 等可选能力
- 核心文档：
  - [04_stage3_pipeline_mod_plan.md](./04_stage3_pipeline_mod_plan.md)
  - [05_stage4_component_migration.md](./05_stage4_component_migration.md)
  - [06_stage5_acceleration_adaptation.md](./06_stage5_acceleration_adaptation.md)
  - [07_stage6_risk_analysis.md](./07_stage6_risk_analysis.md)
  - [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
  - [09_code_mod_order.md](./09_code_mod_order.md)

### 大阶段 E: 性能收口 + 回归验收

- 对应小阶段：`Phase 8`
- 目标：完成 `compile / offload / backend` 适配、默认参数收口与回归集建设
- 核心文档：
  - [06_stage5_acceleration_adaptation.md](./06_stage5_acceleration_adaptation.md)
  - [07_stage6_risk_analysis.md](./07_stage6_risk_analysis.md)
  - [08_stage7_execution_roadmap.md](./08_stage7_execution_roadmap.md)
  - [09_code_mod_order.md](./09_code_mod_order.md)
  - [11_phase_e_acceleration_implementation.md](./11_phase_e_acceleration_implementation.md)
  - [12_phase_e_sp_native_acceleration_plan.md](./12_phase_e_sp_native_acceleration_plan.md)
  - [13_phase_e_sp_quality_closure_plan.md](./13_phase_e_sp_quality_closure_plan.md)

## 实施原则

- 先复用，再新建。
- pipeline 只做编排，不把模型私有逻辑扩散到公共 runtime。
- 模型私有逻辑局部化在 `pipeline`、`model-specific stage`、`sampling params`、`runtime/vividvr/*`。
- 不整棵复制 `Vivid-VR/src/diffusers`。
- 不为了单一模型过早改造公共 `DenoisingStage`、公共 loader、公共 registry 主流程。
- 每个阶段都要以“可验证的 batch / runtime 合同”为中心推进，而不是只看文件是否补齐。
