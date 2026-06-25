# Qwen3.6-27B OpenAI Chat、TokenizerManager 与工具调用解析

本文覆盖 `/v1/chat/completions` 到 `OpenAIServingChat`、chat template、`GenerateReqInput`、`TokenizerManager`、`qwen3_coder` tool call parser 的链路。当前脚本默认 `SERVED_MODEL_NAME=qwen3.6-27b`，调用时 `model` 建议与 `/v1/models` 返回值一致。

## 1. 路由与鉴权

FastAPI app 在 `python/sglang/srt/entrypoints/http_server.py:373-389` 创建。OpenAI chat 路由在 `http_server.py:1388-1395`：

```text
POST /v1/chat/completions
  -> validate_json_request()
  -> raw_request.app.state.openai_serving_chat.handle_request()
```

`validate_json_request()` 检查 `Content-Type: application/json`，见 `http_server.py:457-470`。

当前启动脚本总是传：

```bash
--api-key "$OPENAI_API_KEY"
```

普通 `/v1/*` 请求需要：

```http
Authorization: Bearer <OPENAI_API_KEY>
```

鉴权逻辑在 `python/sglang/srt/utils/auth.py:74-146`：

- `OPTIONS` 放行。
- `/health` 与 `/metrics` 放行。
- 普通 endpoint 在配置 `api_key` 后必须带匹配 Bearer token。
- admin 强制 endpoint 需要 `admin_api_key`。

## 2. `ChatCompletionRequest`

OpenAI chat 请求模型定义在 `python/sglang/srt/entrypoints/openai/protocol.py`。核心字段：

| 字段 | 说明 |
| --- | --- |
| `model` | 当前脚本默认 `qwen3.6-27b` |
| `messages` | 必填，支持 `system`、`developer`、`user`、`assistant`、`tool`、`function` |
| `max_tokens` / `max_completion_tokens` | 生成长度；后者优先 |
| `stream` | 是否返回 SSE |
| `temperature`, `top_p`, `top_k`, `min_p` | 采样参数 |
| `stop`, `stop_token_ids`, `stop_regex` | 停止条件 |
| `response_format` | JSON object / JSON schema / structural tag 等约束 |
| `tools` | OpenAI function tool 列表 |
| `tool_choice` | `auto`、`required`、`none` 或指定函数 |
| `parallel_tool_calls` | 是否允许并行工具调用 |
| `rid`, `priority`, `extra_key`, `cache_salt` | SGLang 扩展请求控制 |

`tool_choice` 默认值由 `ChatCompletionRequest.set_tool_choice_default()` 处理：没有 `tools` 时为 `none`，有 `tools` 时为 `auto`。

## 3. OpenAI serving 通用处理

通用外壳是 `OpenAIServingBase.handle_request()`，见 `python/sglang/srt/entrypoints/openai/serving_base.py:73-109`：

```text
handle_request()
  -> 记录 received_time
  -> _validate_request(request)
  -> log OpenAI raw request
  -> _convert_to_internal_request(request, raw_request)
  -> request.stream ? _handle_streaming_request : _handle_non_streaming_request
```

Chat endpoint 的类是 `OpenAIServingChat`，见 `python/sglang/srt/entrypoints/openai/serving_chat.py:89`。初始化时保存：

```text
self.tool_call_parser = tokenizer_manager.server_args.tool_call_parser
self.reasoning_parser = tokenizer_manager.server_args.reasoning_parser
```

因此脚本的 `--tool-call-parser qwen3_coder` 会进入 Chat serving 层。

## 4. Chat 请求校验

`OpenAIServingChat._validate_request()` 在 `serving_chat.py:194-240`，主要检查：

- `messages` 不能为空。
- `tool_choice="required"` 时必须提供 `tools`。
- 指定函数作为 `tool_choice` 时，该函数必须存在于 `tools`。
- `function.parameters` 必须是合法 JSON schema。
- `max_completion_tokens` 或 `max_tokens` 不能超过 server context length。
- `response_format.type="json_schema"` 时必须有 schema。

这里检查的是单独 completion 上限；`TokenizerManager` 后面还会检查 `input_tokens + max_new_tokens` 是否超过总上下文。

## 5. Messages / tools 到 chat template

转换入口是 `OpenAIServingChat._convert_to_internal_request()`，见 `serving_chat.py:242-327`：

```text
_convert_to_internal_request()
  -> _process_messages(request, is_multimodal)
  -> request.to_sampling_params(...)
  -> GenerateReqInput(...)
```

`_process_messages()` 在 `serving_chat.py:329` 附近：

1. 根据 `tools` 与 `tool_choice` 判断是否启用 tool path。
2. 若有 tools 且 `tool_choice != "none"`，设置 `skip_special_tokens=False`。
3. 指定工具时，只把匹配工具传给 chat template。
4. 如配置 parser，创建 `FunctionCallParser(request.tools, self.tool_call_parser)`。
5. 对 `required` 或指定工具的场景生成 JSON schema constrained decoding 约束。
6. 选择 Jinja chat template 或 conversation template。

默认 Jinja 路径在 `serving_chat.py:379-537`，会调用：

```text
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    tools=tools,
    return_dict=False,
    **chat_template_kwargs
)
```

普通文本 Chat 请求通常得到 `prompt_ids`，随后构造：

```text
GenerateReqInput(input_ids=processed_messages.prompt_ids, sampling_params=..., stream=...)
```

## 6. `GenerateReqInput` 与采样参数

`request.to_sampling_params()` 定义在 `protocol.py`，会把 OpenAI 字段映射到内部 sampling params：

```text
max_new_tokens = max_completion_tokens or max_tokens
stop / temperature / top_p / top_k / min_p
presence_penalty / frequency_penalty / repetition_penalty
json_schema / structural_tag
```

`GenerateReqInput` 保留 `input_ids` 或 `text`、采样参数、流式标志、LoRA、routing、priority、`rid`、`extra_key`、`return_hidden_states` 等信息。

## 7. TokenizerManager 接收请求

OpenAI Chat 层不直接和 scheduler 通信，而是调用：

```text
TokenizerManager.generate_request(adapted_request, raw_request)
```

`TokenizerManager` 初始化在 `python/sglang/srt/managers/tokenizer_manager.py:178-225`，包括 model config、tokenizer、IPC、request logging、dispatcher。

`generate_request()` 在 `tokenizer_manager.py:478-525`：

```text
auto_create_handle_loop()
normalize_batch_and_arguments()
_set_default_priority()
_validate_rid_not_in_flight()
_req_stats_init()
log_received_request()
_validate_and_resolve_lora()
_tokenize_one_request()
_send_one_request()
_wait_one_response()
```

Chat template 已经产生 `input_ids` 时，`_tokenize_one_request()` 不会重复 tokenize；它会进入 `elif obj.input_ids is not None` 分支，见 `tokenizer_manager.py:665-775`。

长度校验在 `tokenizer_manager.py:785-835`：

- 输入 token 数不能大于等于 `context_len`。
- `input_token_num + max_new_tokens` 不能超过 context length。
- 部分扩展能力需要服务端显式开启。

## 8. 发送 scheduler 与等待回包

`_create_tokenized_object()` 在 `tokenizer_manager.py:928-1003`，会创建 `TokenizedGenerateReqInput` 并保留 `rid`、`input_ids`、`sampling_params`、`priority`、`extra_key` 等字段。

发送函数在 `tokenizer_manager.py:1095-1102`：

```text
tokenized_obj.time_stats.set_api_server_dispatch_time()
wrap_shm_features(tokenized_obj)
self.send_to_scheduler.send_pyobj(tokenized_obj)
tokenized_obj.time_stats.set_api_server_dispatch_finish_time()
```

后台 `handle_loop()` 在 `tokenizer_manager.py:1523-1530` 从 detokenizer 收回 `BatchStrOutput`，然后 `_handle_batch_output()` 更新 `ReqState`，构造：

```python
{
  "text": output_text,
  "output_ids": output_token_ids,
  "meta_info": {
    "id": rid,
    "finish_reason": ...,
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "cached_tokens": ...,
    "weight_version": ...
  }
}
```

`_wait_one_response()` 在 `tokenizer_manager.py:1120-1263`：

- streaming：有新输出就 yield。
- non-streaming：完成后 yield 最终结果。
- 客户端断开时 abort。
- finished 时记录 request log 和 request metrics。

## 9. Non-stream 与 streaming response

非流式入口：

```text
OpenAIServingChat._handle_non_streaming_request()
  -> await tokenizer_manager.generate_request(...).__anext__()
  -> _build_chat_response()
```

见 `serving_chat.py:917-1050`。如果请求带 tools 且启用了 parser，会在 `_process_tool_calls()` 把模型文本解析成 OpenAI `message.tool_calls`。

流式入口：

```text
OpenAIServingChat._handle_streaming_request()
  -> _generate_chat_stream()
  -> StreamingResponse(text/event-stream)
```

见 `serving_chat.py:604-915`。流式响应顺序通常是 role chunk、content/reasoning/tool_calls delta、finish chunk、可选 usage chunk、`data: [DONE]`。

## 10. `tool_call_parser=qwen3_coder`

注册表在 `python/sglang/srt/function_call/function_call_parser.py:48-66`：

```text
"qwen3_coder" -> Qwen3CoderDetector
```

链路：

```text
--tool-call-parser qwen3_coder
  -> ServerArgs.tool_call_parser
  -> OpenAIServingChat.tool_call_parser
  -> FunctionCallParser(..., "qwen3_coder")
  -> Qwen3CoderDetector
```

`Qwen3CoderDetector` 在 `python/sglang/srt/function_call/qwen3_coder_detector.py`，识别格式：

```text
<tool_call>
<function=FUNCTION_NAME>
<parameter=PARAM_NAME>VALUE</parameter>
</function>
</tool_call>
```

非流式解析：

```text
OpenAIServingChat._process_tool_calls()
  -> FunctionCallParser.has_tool_call(text)
  -> FunctionCallParser.parse_non_stream(text)
  -> Qwen3CoderDetector.detect_and_parse(text, tools)
```

流式解析：

```text
_generate_chat_stream()
  -> _process_tool_call_stream()
  -> FunctionCallParser.parse_stream_chunk(delta)
  -> Qwen3CoderDetector.parse_streaming_increment(delta, tools)
  -> delta.tool_calls
```

如果 `tool_choice="required"` 或指定函数，SGLang 会更倾向于使用 JSON array parser 和 JSON schema constraint，预期输出形态是：

```json
[
  {"name": "tool_name", "parameters": {"arg": "value"}}
]
```

而不是 qwen3_coder XML 标签。

## 11. 最小调用示例

普通非流式：

```bash
curl --noproxy '*' -sS http://127.0.0.1:30000/v1/chat/completions \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "你好，简单介绍一下你自己。"}],
    "max_completion_tokens": 512,
    "temperature": 0.7
  }'
```

工具调用：

```json
{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "查一下北京今天的天气。"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_completion_tokens": 1024
}
```

## 12. 常见排错

| 现象 | 检查点 |
| --- | --- |
| 401 | Bearer token 是否与启动时 `OPENAI_API_KEY` 一致；`/health` 和 `/metrics` 本身应放行 |
| 400 Content-Type | 是否带 `Content-Type: application/json` |
| `Messages cannot be empty` | `messages` 是否为空 |
| 上下文超限 | 同时检查 completion 上限和 `input_tokens + max_new_tokens` |
| tool_calls 没解析 | 请求是否有 `tools`、`tool_choice` 是否不是 `none`、server 是否是 `qwen3_coder`、模型输出格式是否匹配 |
| 流式参数不完整 | 客户端应累加 `delta.tool_calls[*].function.arguments`，不要覆盖 |
| 请求断开 | `TokenizerManager._wait_one_response()` 会检测 disconnect 并 abort |
