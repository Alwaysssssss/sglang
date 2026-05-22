# STAR CogVideoX-SR 详细实施计划

## 1. 文档定位

本目录用于承接 [total_plan.md](/sgl-workspace/sglang/docs_xzh/add_STAR/code_plan/total_plan.md:1) 的落地实施计划。

`total_plan.md` 负责回答：

1. 为什么这样设计
2. 总体边界怎么划分
3. 为什么选择 modular pipeline

本目录下的文档负责回答：

1. 具体先改什么、后改什么
2. 每一阶段改哪些代码文件
3. 每一阶段如何验收
4. 阶段间的依赖关系和禁止事项

本目录是后续实际代码修改的执行手册。

---

## 2. 实施总原则

所有阶段都必须遵守以下原则：

1. **只接入 STAR 的 CogVideoX-SR 分支**
   不处理 `STAR_mg/video_to_video` 这条 I2VGen/VEnhancer 链路。
2. **运行时完全脱离 STAR 仓库**
   `sglang.multimodal_gen` 的运行时代码中不允许依赖 `STAR_mg` 包路径、数据集类、环境变量或 YAML 入口。
3. **坚持 modular pipeline**
   优先复用 `TextEncodingStage`、`LatentPreparationStage`、`TimestepPreparationStage`、`DenoisingStage`。
4. **不复用 `image_path` 承载视频**
   条件视频必须使用独立字段，如 `condition_video_path`。
5. **不复用现有 `TI2V` 语义**
   本模型按 `T2V 主干 + condition video latent` 方式接入。
6. **实现阶段必须先保真，再优化**
   第一目标是端到端对齐 STAR 原始推理结果；性能和并行优化在后置阶段完成。

---

## 3. 阶段拆分

建议按以下顺序执行：

1. [phase_1_model_assets_and_weight_conversion.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_1_model_assets_and_weight_conversion.md:1)
   目标：定义模型资产布局与权重转换，先切断运行时对 STAR repo 的耦合。
2. [phase_2_request_contract_and_condition_video_io.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_2_request_contract_and_condition_video_io.md:1)
   目标：定义 `SamplingParams`、`Req`、`condition_video_path` 与条件视频加载契约。
3. [phase_3_pipeline_and_stage_wiring.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_3_pipeline_and_stage_wiring.md:1)
   目标：把 pipeline、stage 顺序、registry、配置 wiring 搭起来。
4. [phase_4_model_components_dit_vae_scheduler.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_4_model_components_dit_vae_scheduler.md:1)
   目标：实现 STAR 专属 DiT、VAE、scheduler adapter。
5. [phase_5_decoding_parity_and_acceptance.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_5_decoding_parity_and_acceptance.md:1)
   目标：完成时序分块 decode、端到端结果对齐、功能验收。
6. [phase_6_performance_hardening_and_upstream_sync.md](/sgl-workspace/sglang/docs_xzh/add_STAR/detail_plan/phase_6_performance_hardening_and_upstream_sync.md:1)
   目标：补并行/显存/性能收尾，沉淀后续同步 upstream 的维护规则。

---

## 4. 推荐执行节奏

每个阶段建议都遵循统一节奏：

1. 先实现最小可运行路径
2. 先补阶段内最小测试
3. 通过阶段验收后再进入下一阶段
4. 任何阶段一旦发现上游假设错误，优先回写到文档，而不是先在代码里打补丁

---

## 5. 阶段依赖关系

### 5.1 严格前置依赖

1. 阶段 2 依赖阶段 1
   因为 `PipelineConfig` 和 loader 的设计必须建立在资产目录和权重布局稳定的前提上。
2. 阶段 3 依赖阶段 2
   因为 pipeline 组装必须基于稳定的请求契约与 `Req` 字段。
3. 阶段 4 依赖阶段 3
   因为 DiT/VAE/scheduler 的接线位置和 forward 契约需要先固定。
4. 阶段 5 依赖阶段 4
   因为 parity 验证必须在组件可运行后进行。
5. 阶段 6 依赖阶段 5
   因为性能优化必须以“结果已对齐”为前提。

### 5.2 允许并行准备的内容

以下事项可以提前准备，但不要提前合并到主实现：

1. 测试视频和 prompt 样本清单
2. checkpoint key mapping 草稿
3. parity 比对脚本草稿
4. 性能 profiling 命令模板

---

## 6. 每阶段交付要求

每个阶段的文档都必须至少包含：

1. 阶段目标
2. 作用范围
3. 不在本阶段处理的内容
4. 计划新增/修改的代码文件
5. 数据流与接口契约
6. 关键实现步骤
7. 推荐测试与验收方式
8. 风险点与止损条件

---

## 7. 全局禁止事项

在整个实施过程中，以下做法都不建议出现：

1. 在 `sglang` 运行时代码里直接 import `STAR_mg.*`
2. 让运行时去解析 STAR 原始训练 YAML
3. 用 `image_path=.mp4` 偷渡条件视频输入
4. 为了省事直接重写一个超大 `BeforeDenoisingStage`
5. 为了绕过适配问题把 STAR 推理主循环整段复制到 SGLang 里
6. 在未完成 parity 前就提前做大规模性能优化

---

## 8. 全局验收口径

整个项目最终应满足以下总验收标准：

1. 可以通过 SGLang native pipeline 接收 `prompt + condition_video_path`
2. 条件视频是整段读入和编码，而不是首帧
3. 不依赖 `STAR_mg` 原始仓库运行
4. `DenoisingStage` 仍然是主去噪循环
5. 结果与 STAR 原始实现达到可接受的对齐程度
6. 接入代码边界清晰，后续升级时可局部替换

---

## 9. 使用方式

后续实际开始写代码时，建议严格按阶段文档推进：

1. 每进入一个阶段前，先复读该阶段“本阶段不处理的内容”
2. 实现完成后，先跑该阶段文档列出的验收项
3. 验收不通过，不进入下一阶段
4. 任何与文档冲突的实现决策，都应先更新文档再继续编码

这能最大程度避免“写着写着回到 STAR 原仓库耦合模式”。
