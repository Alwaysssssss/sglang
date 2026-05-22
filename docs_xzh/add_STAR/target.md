

请依照 @/sgl-workspace/sglang/python/sglang/multimodal_gen/.codex/skills/sglang-diffusion-add-model skill 的方法，将 @/sgl-workspace/STAR_mg 的视频超分模型作为新模型接入，注意，只接入其中的CogVideoX分支即可，并在 @/sgl-workspace/sglang/docs_xzh/add_STAR/code_plan/total_plan.md 中详细完善一份总计划方案。注意一点，我们尽可能需要使用风格 B:Modular 组合式 Pipeline，可以尝试通过复用和修改结合，把项目逻辑调整成我们的stage，这样子组织会更加的清晰，相比 hybrid 的一大段 stage，这会更方便我们后面进行修改。完善时需重点解决以下逻辑：

1. 脱离原 STAR_mg 仓库的耦合，确保集成到 sglang 后，不依赖原 repo 的路径、数据结构和私有调用；
2. 采用 SGLang 推荐的扩展方式，优先复用已有 VAE、DiT 主体和 pipeline 设计，仅补充 STAR_mg 专属的数据组装、pipeline stage 或接口适配代码，避免冗余复制已有通用逻辑；
3. 设计清晰的模型/预处理/后处理解耦机制，以便 sglang 未来升级或 STAR_mg 仓库变更后，可便捷同步新特性并无缝合并新版，无需大改现有集成代码；
4. 方案里明确接口分层、数据流转与模块边界，便于后续维护和自动化对齐 upstream 变更。

请注意，这一轮不要做任何代码的改动，只需要你给出详细的计划文档
方案目标：实现模块边界清晰、可配置、松耦合的集成方式，使 sglang 升级与新模型并存无障碍合并。