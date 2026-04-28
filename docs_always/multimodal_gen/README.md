# SGLang `multimodal_gen` 多模态子系统分析

本文档面向 `python/sglang/multimodal_gen`，重点解释它如何把“模型识别、配置解析、组件加载、Pipeline 组装、Stage 执行、服务接口、应用集成”拼成一套统一的多模态生成框架。

## 阅读顺序

1. [01_architecture_overview.md](./01_architecture_overview.md)
2. [02_registry_and_config.md](./02_registry_and_config.md)
3. [03_runtime_execution.md](./03_runtime_execution.md)
4. [04_pipeline_and_stage.md](./04_pipeline_and_stage.md)
5. [05_loader_and_models.md](./05_loader_and_models.md)
6. [06_interfaces_and_apps.md](./06_interfaces_and_apps.md)
7. [07_disaggregation_and_optimization.md](./07_disaggregation_and_optimization.md)
8. [08_skill_add_model.md](./08_skill_add_model.md)（实战 Skill：向 `multimodal_gen` 新增一个扩散模型）
9. [09_case_study_wan2_1.md](./09_case_study_wan2_1.md)（案例实战：以 Wan2.1 为例，落地到具体文件与代码位置）
10. [10_wan2_1_end_to_end.md](./10_wan2_1_end_to_end.md)（端到端源码跟读：Wan2.1 从 CLI 启动到底层 kernel 的全流程）
11. [11_wan_model_architecture.md](./11_wan_model_architecture.md)（Wan 家族全景：所有模型型号 + DiT/VAE/Scheduler/Encoder 详细结构）

## 优化专题（深入 `multimodal_gen` 每一类优化技术）

- [optimizations/README.md](./optimizations/README.md)：并行 / 量化 / 缓存 / 注意力 / 算子融合 / 编译 / Offload / PD 分离 / 后处理共 **9 篇** 专题文档，每篇聚焦一个模块，附源码引用、默认值、环境变量、调优建议。`07_disaggregation_and_optimization.md` 可以看作它的索引版，细节都收在此目录下。

## 使用 / 部署指南

- [wan2_1_guide.md](./wan2_1_guide.md)：Wan2.1 从安装、单卡/多卡、在线服务到 Cache-DiT / LoRA / 插帧超分的完整运行文档（面向用户，而非开发者）

## 一句话理解

`multimodal_gen` 不是“某个模型的推理脚本集合”，而是一个统一的扩散/多模态生成运行时：

- 上层暴露 `DiffGenerator`、CLI、FastAPI、OpenAI 风格接口、WebUI、ComfyUI。
- 中层用 `ServerArgs`、`PipelineConfig`、`SamplingParams` 和 `registry.py` 完成模型识别与配置归一化。
- 下层用 `ComposedPipelineBase + PipelineStage + Executor` 组织具体推理。
- 更底层通过 `ComponentLoader`、`ModelRegistry`、`runtime/models`、`runtime/layers`、`runtime/platforms` 完成权重加载、算子选择、分布式与平台适配。

## 顶层模块地图

| 目录 | 作用 |
| --- | --- |
| `apps/` | WebUI、ComfyUI 集成 |
| `configs/` | PipelineConfig、SamplingParams、模型结构配置、量化配置 |
| `runtime/entrypoints/` | `DiffGenerator`、CLI、HTTP/OpenAI API |
| `runtime/managers/` | Scheduler、GPUWorker、CPUWorker |
| `runtime/pipelines/` | 具体模型 Pipeline，如 Wan / Flux / Qwen-Image |
| `runtime/pipelines_core/` | 通用 Pipeline 框架、Stage、Executor、Req/OutputBatch |
| `runtime/loader/` | 组件加载器、FSDP/权重装载、量化装载 |
| `runtime/models/` | SGLang 自定义模型实现和模型注册 |
| `runtime/disaggregation/` | 编码/去噪/解码三段式拆分服务 |
| `runtime/postprocess/` | RIFE 插帧、Real-ESRGAN 放大 |
| `runtime/platforms/` | CUDA/ROCm/MPS/NPU/MUSA/XPU 平台差异 |
| `csrc/` | 注意力与渲染相关底层扩展 |
| `benchmarks/`、`test/` | 性能基准、服务与单测 |

## 入口文件

- 包入口：`python/sglang/multimodal_gen/__init__.py`
- 统一推理入口：`python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`
- 模型识别入口：`python/sglang/multimodal_gen/registry.py`
- 服务启动入口：`python/sglang/multimodal_gen/runtime/launch_server.py`
- Pipeline 构建入口：`python/sglang/multimodal_gen/runtime/pipelines_core/__init__.py`

## 建议抓主线的方式

如果你只想先抓住主干，建议沿着下面这条链读代码：

`DiffGenerator.from_pretrained`  
-> `ServerArgs.from_kwargs`  
-> `PipelineConfig.from_kwargs`  
-> `registry.get_model_info`  
-> `launch_server` / `Scheduler` / `GPUWorker`  
-> `build_pipeline`  
-> `ComposedPipelineBase.load_modules`  
-> `create_pipeline_stages`  
-> `executor.execute_with_profiling`  
-> `PipelineStage.forward`

