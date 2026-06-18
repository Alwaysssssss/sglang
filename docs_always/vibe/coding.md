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

        




        