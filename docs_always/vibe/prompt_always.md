

请依照 @python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model skill 的方法，将 @../VideoEdit-diffusers 的视频编辑模型作为新模型接入，并在 @docs_always/add_new_mode/add_videoedit_diffusers/README.md 中详细完善方案。完善时需重点解决以下逻辑：

1. 脱离原 VideoEdit-diffusers 仓库的耦合，确保集成到 sglang 后，不依赖原 repo 的路径、数据结构和私有调用；
2. 采用 SGLang 推荐的扩展方式，优先复用已有 VAE、DiT 主体和 pipeline 设计，仅补充 VideoEdit 专属的数据组装、pipeline stage 或接口适配代码，避免冗余复制已有通用逻辑；
3. 设计清晰的模型/预处理/后处理解耦机制，以便 sglang 未来升级或 VideoEdit-diffusers 仓库变更后，可便捷同步新特性并无缝合并新版，无需大改现有集成代码；
4. 方案里明确接口分层、数据流转与模块边界，便于后续维护和自动化对齐 upstream 变更。

方案目标：实现模块边界清晰、可配置、松耦合的集成方式，使 sglang 升级与新模型并存无障碍合并。

完善方案，加强输出结果的校验机制：集成后生成的视频结果需与原始 VideoEdit-diffusers 仓库的 infer.py 输出结果逐帧对齐。对齐方式为针对每一帧采用结构相似度（如 SSIM）、均方误差（MSE）等图像处理算法进行自动对比，可容忍微小容差，但不得出现大范围失配或整帧错误。请实现自动化对比和统计功能，便于回归验证与迭代优化，确保改造集成后模型输出在视觉效果和数值尺度上高度一致。建议将对齐逻辑封装为独立测试脚本或模块，便于 CICD 持续回归和 future update 跟进。