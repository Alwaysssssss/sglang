/goal 按照'/mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang/docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh'启动了模型服务。

回答如下问题：

1. 当agent的上下文超过256k时，推理服务会发生什么
2. 每次对话都发了全部的上下文过来，他的cache如何管理的呢
3. 在 @tmp/deer-flow/deer-flow 框架，本框架会有多个地方使用模型，每个模型都占有一个并发吗
4. 在 @tmp/deer-flow/deer-flow 框架通过 Langchain/OpenAIChat 调用该模型推理服务，经常性的不回复了，详解可能的原因


/goal 详解 @docs_always/qwen3.6-27b/qwen36_27b.conf 每个参数的含义，他是否会影响前端agent没有输出


/goal 通过分析模型输出日志的片段 @docs_always/qwen3.6-27b/log.json，发现模型会输出如下内容：
        、、、
        {
          "role": "assistant",
          "content": ""
        },
        、、、
        从而导致agent也没有输出，帮我确定一下是不是这个原因
        假如是这个原因，该如何修改，才可以保证模型不会输出为空，通过配置"min_tokens"可以让模型一定输出吗

        注：采用多subagent分析，给出解决方案，不要写代码

/goal 为了达到模型服务返回值不为空，在 @tmp/deer-flow/deer-flow/config.yaml 中该如何配置
、、、
- name: qwen3-6-27b
  display_name: Other OpenAI-compatible / qwen3.6-27b
  use: langchain_openai:ChatOpenAI
  model: qwen3.6-27b
  api_key: $OPENAI_API_KEY
  base_url: http://127.0.0.1:18080/v1
、、、

      "model": "qwen3.6-27b",
      "frequency_penalty": 0,
      "logprobs": false,
      "max_completion_tokens": 32000,
      "n": 1,
      "presence_penalty": 0,
      "stream": true,
      "stream_options": {
        "include_usage": true,
        "continuous_usage_stats": false
      },
      "tool_choice": "auto",
      "parallel_tool_calls": true,
      "return_hidden_states": false,
      "return_routed_experts": false,
      "return_cached_tokens_details": false,
      "min_tokens": 8,
      "no_stop_trim": false,
      "ignore_eos": false,
      "continue_final_message": false,
      "skip_special_tokens": true,
      "separate_reasoning": true,
      "stream_reasoning": true,
      "chat_template_kwargs": {
        "enable_thinking": false
      }


请基于当前核心代码 @python/sglang/srt，系统性、持续性地进行全面深入的源码分析与架构梳理，并将分析结果以高质量文档形式持续输出到 @docs_always/srt 目录中。

## 分析目标
帮助开发者充分理解的设计理念、架构体系、核心模块、关键流程及实现细节，为后续开发、维护、扩展提供完整的技术参考。

## 分析要求

### 1. 系统性与全面性
- 从项目整体架构入手，逐层深入到模块、类、函数级别
- 覆盖核心功能模块、工具类、配置管理、依赖关系等所有关键部分
- 分析代码组织结构、设计模式、技术选型及其合理性

### 2. 持续性与可迭代性
- 采用分阶段、分模块的方式进行分析，支持长时间持续工作
- 每完成一个模块的分析后，立即输出对应文档，便于增量式理解
- 建立清晰的文档索引和导航体系，方便后续查阅和更新

### 3. 文档质量标准
- 结构清晰：采用统一的文档模板，包含概述、架构图、核心逻辑、关键代码解析、使用示例等
- 内容详实：不仅说明"是什么"，更要解释"为什么"和"如何实现"
- 表达准确：使用专业术语，配合代码片段、流程图、架构图等辅助说明
- 易于理解：面向不同技术背景的读者，提供必要的背景知识和上下文说明

### 4. 分析维度
请从以下维度展开分析：
- **项目概览**：项目定位、核心功能、技术栈、目录结构
- **架构设计**：整体架构、模块划分、层次关系、设计模式
- **核心模块**：每个核心模块的职责、接口、实现原理、关键算法
- **数据流转**：数据如何在各模块间流转，关键数据结构设计
- **配置管理**：配置文件结构、配置加载机制、可配置项说明
- **依赖关系**：外部依赖库的使用、内部模块间的依赖关系
- **扩展机制**：插件体系、钩子函数、可扩展点分析
- **最佳实践**：代码规范、错误处理、日志记录、测试策略
- **常见问题**：潜在的坑点、注意事项、troubleshooting指南

注：采用多subagent详细分析

        




        