# VideoEdit Compare

本目录记录当前 SGLang `WanVideoEditPipeline` 与
`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers` 当前源码的
逐层对比。

- 完整对比文档：`wan_videoedit_pipeline_vs_videoedit_diffusers.md`
- 前后处理专项对比：`videoedit_prepost_alignment.md`
- 对比方式：源码静态阅读，多 subagent 分片比较；未运行推理或测试程序。
- 覆盖范围：顶层推理入口、stage/dataflow、模型构建与加载、Transformer/VAE layer 调用、默认参数差异和 1:1 复现检查项。
