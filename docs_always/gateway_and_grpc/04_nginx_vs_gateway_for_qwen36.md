# Qwen3.6-27B 场景下：Nginx 与 `sgl-model-gateway` 的关系

## 1. 结论

`docs_always/qwen3.6-27b/` 里的 Nginx 方案和 `sgl-model-gateway` 不是同一层东西：

- **Nginx** 是边缘入口 / 反向代理，主要解决公网暴露、域名/TLS、长超时、流式转发、访问日志、简单限流。
- **`sgl-model-gateway`** 是模型服务层网关，主要解决 worker 注册、健康检查、负载均衡、重试、熔断、PD/gRPC、多模型路由、tokenizer/parser/MCP 等模型调度能力。

因此默认不应该把二者理解成“二选一”。更合理的生产关系是：

```text
Client / OpenAI SDK / LangChain
  -> Nginx / API Gateway / Ingress
  -> sgl-model-gateway
  -> SGLang workers
```

但对当前 `Qwen3.6-27B` 单机单模型 runbook 来说，继续使用：

```text
Client
  -> Nginx :18080
  -> SGLang :30000
```

更简单，也更贴合现有文档和脚本。

## 2. 当前 Qwen3.6-27B Nginx 方案在做什么

当前 runbook 的关键事实来自 `docs_always/qwen3.6-27b/qwen3.6-27b.md` 与 `qwen36_27b.conf`：

- SGLang 绑定 `127.0.0.1:30000`，不直接暴露到公网。
- Nginx 绑定 `0.0.0.0:18080`，对外提供 OpenAI 兼容入口。
- Nginx `proxy_pass` 到 `127.0.0.1:30000`。
- `Authorization` header 被透传给 SGLang，实际模型服务仍使用 `--api-key` 鉴权。
- `proxy_buffering off` 和 `proxy_request_buffering off` 保证流式输出不会被代理层缓冲。
- `proxy_send_timeout`、`proxy_read_timeout`、`send_timeout` 被调大，用来承载 256K 长上下文和长时间 agent 请求。
- `limit_req` 提供基于客户端 IP 的轻量入口限流。

这套配置不知道模型拓扑，也不理解 worker 健康状态；它只是把 HTTP 请求可靠地送到本机 SGLang。

对应拓扑：

```text
client
  -> Nginx HTTP entrypoint on 0.0.0.0:18080
  -> SGLang HTTP server on 127.0.0.1:30000
  -> Qwen/Qwen3.6-27B local weights
  -> 4 x A100 tensor parallel runtime
```

这个设计的优点是：链路短、组件少、排障直接。对单机、单模型、单 SGLang 进程来说，这是正确的初始方案。

## 3. `sgl-model-gateway` 能补什么

`sgl-model-gateway` 的入口同样是 OpenAI 兼容 HTTP，但它不是普通反向代理。它会把后端视为一组 worker，并维护模型服务层状态：

- worker 注册、删除、更新：`POST /workers`、`GET /workers`。
- readiness 判断：根据健康 worker 数和 PD 拓扑判断是否 ready。
- 路由策略：`random`、`round_robin`、`cache_aware`、`power_of_two`、`manual`、`prefix_hash` 等。
- HTTP worker、gRPC worker、PD prefill/decode worker 的统一调度。
- 重试、熔断、健康检查、并发限制和排队。
- gRPC 模式下在 gateway 内做 tokenizer、reasoning parser、tool-call parser。
- 多模型 IGW 模式下可按 `model_id` 管理不同 worker 和策略。
- MCP、WASM、history backend、Prometheus、OpenTelemetry 等扩展能力。

换句话说，gateway 适合在“一个模型服务入口背后有多个模型执行单元”时使用。

## 4. 哪个方案更好

按场景判断：

| 场景 | 推荐方案 |
| --- | --- |
| 当前只有一个 `Qwen3.6-27B` SGLang 服务，监听 `127.0.0.1:30000` | 继续用 Nginx 直连 SGLang |
| 需要公网入口、TLS、域名、统一访问日志、长 timeout、流式转发 | Nginx / API Gateway / Ingress |
| 同一个模型有多个 SGLang worker 副本，需要健康检查和负载均衡 | Nginx -> `sgl-model-gateway` -> SGLang workers |
| 多模型统一入口，例如 qwen、embedding、rerank、其他 LLM 混部 | Nginx -> `sgl-model-gateway` |
| 需要 PD prefill/decode、gRPC worker、cache-aware routing | `sgl-model-gateway` 是核心组件，Nginx 仍可作为外层入口 |
| 想替代模型推理、KV cache、采样、forward | Nginx 和 gateway 都不做；仍由 SGLang worker 执行 |

因此：

- **当前 qwen3.6-27b 单机上线：Nginx 更好。**
- **生产化、多实例、多模型或 PD/gRPC：Nginx + `sgl-model-gateway` 更好。**
- **不建议直接把 `sgl-model-gateway` 当公网唯一入口替掉 Nginx。**

## 5. 为什么不建议当前直接替换

当前 qwen3.6-27b 服务脚本已经把模型运行参数、显存预算、ready check、API key、日志和 256K 上下文验收都绑定在单个 SGLang HTTP server 上。此时引入 gateway 会增加：

- 一个额外服务进程。
- 一层额外 timeout、错误码和日志面。
- worker 注册与 readiness 的排障成本。
- API key 在 Nginx、gateway、SGLang worker 之间的配置边界。
- 对长流式请求的端到端 timeout 重新校准。

如果后端仍只有一个 worker，这些成本很难换来实质收益。

## 6. 推荐演进路径

### 阶段 1：当前单机单模型

保持现状：

```text
Client
  -> Nginx :18080
  -> SGLang :30000
```

重点治理：

- Nginx `proxy_read_timeout`、`proxy_send_timeout`、`send_timeout` 足够长。
- `proxy_buffering off` 保持流式输出。
- 调用方显式限制 `max_tokens` / `max_completion_tokens`。
- LangChain `ChatOpenAI` 使用合理客户端 timeout，优先启用 streaming。
- Qwen reasoning 输出要读取 `reasoning_content`，不要只等 `content`。

### 阶段 2：同机或多机多个 Qwen worker

引入 gateway，但 Nginx 仍留在最外层：

```text
Client
  -> Nginx :18080
  -> sgl-model-gateway :30080
  -> SGLang worker A :30000
  -> SGLang worker B :30001
```

示例：

```bash
python -m sglang_router.launch_router \
  --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --policy cache_aware \
  --host 127.0.0.1 \
  --port 30080 \
  --request-timeout-secs 9000
```

此时 Nginx 的 upstream 改为 gateway：

```nginx
upstream qwen36_gateway {
    server 127.0.0.1:30080;
    keepalive 32;
}

location / {
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Authorization $http_authorization;
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_pass http://qwen36_gateway;
}
```

### 阶段 3：多模型 / PD / gRPC

如果需要多模型、PD 或 gRPC，把 gateway 作为模型入口：

```text
Client
  -> Nginx / Ingress
  -> sgl-model-gateway
  -> HTTP regular worker
  -> gRPC worker
  -> prefill worker
  -> decode worker
```

gRPC 模式需要给 gateway 提供 tokenizer 来源：

```bash
python -m sglang_router.launch_router \
  --worker-urls grpc://127.0.0.1:20000 \
  --model-path /mnt/shanhai-ai/wenhy/models/Qwen/Qwen/Qwen3___6-27B \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 127.0.0.1 \
  --port 30080
```

PD 模式则通过 `--pd-disaggregation`、`--prefill`、`--decode` 建立 prefill/decode worker 池。

## 7. 排障边界

引入 gateway 后，排障要按层定位：

| 症状 | 优先检查 |
| --- | --- |
| 外部访问 502/504 | Nginx access/error log、upstream 地址、proxy timeout |
| Nginx 正常但 gateway 503 | `GET /readiness`、`GET /workers`、worker 健康状态 |
| gateway 正常但请求慢 | gateway metrics、worker load、SGLang scheduler metrics |
| 流式中断 | Nginx buffering/timeout、gateway `request-timeout-secs`、客户端 timeout |
| 鉴权失败 | Nginx 是否透传 `Authorization`、gateway `--api-key`、SGLang worker `--api-key` |
| 长请求一直无 `content` | Qwen reasoning 是否输出到 `reasoning_content`，调用方是否只读取 `content` |

## 8. 最终建议

对 `docs_always/qwen3.6-27b` 当前这类单机 4 卡、单 Qwen3.6-27B、256K agent 服务：

1. 保留 Nginx 作为对外入口。
2. 暂不引入 `sgl-model-gateway`。
3. 优先修正调用侧 `max_tokens`、streaming、timeout、reasoning 字段处理。
4. 当出现多 worker、多模型、PD/gRPC 或需要统一模型控制面时，再把 gateway 插入到 Nginx 和 SGLang worker 之间。

这条路径能避免当前单模型服务过早复杂化，同时为后续扩展留下清晰落点。
