# Qwen3.6-27B 使用 Cloudflare Tunnel 暴露公网 API

本文给出一套可落地的 Cloudflare Tunnel 方案，让当前内网 Qwen3.6-27B OpenAI 兼容 API 可以通过公网 HTTPS 域名访问。

当前机器状态：

| 项目 | 当前值 |
| --- | --- |
| 系统 | Ubuntu 22.04 |
| 模型服务 | Qwen3.6-27B |
| OpenAI 模型名 | `qwen3.6-27b` |
| SGLang | `127.0.0.1:30000` |
| 本机 Nginx | `0.0.0.0:18080` |
| 本机可用 origin | `http://127.0.0.1:18080` |
| 内网地址 | `10.119.16.70/20` |
| 公网 SSH 地址 | `106.75.235.227:10036` |
| 客户端 Base URL 变量 | `OPENAI_BASE_URL` |
| API key 变量 | `OPENAI_API_KEY` |
| cloudflared | 已安装：`/usr/local/bin/cloudflared`，版本 `2026.6.0` |
| systemd/systemctl | 本机不可用：`systemctl is-system-running` 返回 `offline` |

Cloudflare Tunnel 的核心价值是：服务端只需要主动连出到 Cloudflare，不需要开放入站公网端口。公网客户端访问 `https://<your-api-domain>/v1`，Cloudflare 再通过 `cloudflared` 把请求转发到本机 `http://127.0.0.1:18080`。

官方参考：

- Cloudflare Tunnel 概览：<https://developers.cloudflare.com/tunnel/>
- Cloudflare Tunnel 设置：<https://developers.cloudflare.com/tunnel/setup/>
- Ingress 配置校验：<https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/configuration-file/>
- Tunnel 防火墙建议：<https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/tunnel-with-firewall/>

## 1. 推荐架构

```text
公网客户端
  -> https://mgtvqwen36-apiexample.com/v1
  -> Cloudflare
  -> cloudflared outbound tunnel
  -> 本机 Nginx http://127.0.0.1:18080
  -> SGLang http://127.0.0.1:30000
```

推荐使用 Cloudflare Dashboard 创建的 remotely-managed tunnel。原因：

- 不需要把公网 `18080` 放开。
- 不需要模型服务机有公网 IP。
- 证书由 Cloudflare 处理，客户端使用 HTTPS。
- `cloudflared` 已安装，且本机不能使用 `systemctl`；本方案用普通进程、PID 文件和日志文件管理。
- Cloudflare 后台可以直接管理 Public Hostname。

不要使用 Quick Tunnel 作为生产方案。Cloudflare 官方文档明确 Quick Tunnel 只适合测试，且不支持 SSE；OpenAI 流式输出使用 `data:` 分片，生产应使用命名 Tunnel。

## 2. 前置检查

以下命令在模型服务机执行。

### 2.1 确认 Qwen API 本机可用

```bash
curl --noproxy '*' -sS \
  -H "Authorization: Bearer $(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)" \
  http://127.0.0.1:18080/v1/models
```

预期：返回模型列表，包含 `qwen3.6-27b`。

### 2.2 确认监听状态

```bash
ss -ltnp | awk 'NR==1 || /:18080|:30000/'
```

预期：

```text
0.0.0.0:18080      nginx
127.0.0.1:30000   python3
```

### 2.3 确认 cloudflared 已安装

```bash
command -v cloudflared
cloudflared --version
```

当前机器预期：

```text
/usr/local/bin/cloudflared
cloudflared version 2026.6.0
```

不要在本机执行 `cloudflared service install`、`systemctl enable`、`systemctl restart` 等命令；当前环境 `systemctl is-system-running` 为 `offline`。

### 2.4 确认本机可以出公网

```bash
curl -I https://www.cloudflare.com --connect-timeout 10
```

如果网络出口有防火墙，至少需要允许：

- outbound TCP `443`
- outbound TCP/UDP `7844`，用于 Cloudflare Tunnel 的 `http2`/`quic`

## 3. 方案 A：Dashboard Token 直接运行，推荐

这套方案不需要在服务器保存 Cloudflare API Token。Cloudflare 后台创建 Tunnel 后，会给出一个 `<TUNNEL_TOKEN>`，服务器只用这个 token 启动 connector 进程。

### 3.1 在 Cloudflare Dashboard 创建 Tunnel

在浏览器进入：

```text
Cloudflare Zero Trust -> Networks -> Tunnels -> Create a tunnel
```

选择：

```text
Connector: cloudflared
Tunnel name: qwen36-api
Environment: Debian / Ubuntu
```

复制 Cloudflare 页面生成的 `<TUNNEL_TOKEN>`。如果页面给出的是包含 `cloudflared service install` 的安装命令，本机不要直接执行，只取出 token。

### 3.2 配置 Public Hostname

在同一个 Tunnel 页面添加 Public Hostname：

```text
Subdomain: qwen36-api
Domain: example.com
Path: 留空
Type: HTTP
URL: 127.0.0.1:18080
```

最终公网地址：

```text
https://mgtvqwen36-apiexample.com/v1
```

注意：

- Type 选 `HTTP`，因为本机 Nginx 是 `http://127.0.0.1:18080`。
- URL 不要填 `10.119.16.70:18080`，直接填 `127.0.0.1:18080`，减少网络绕行。
- 不要把 `127.0.0.1:30000` 直接作为 origin；继续保留本机 Nginx 的限流、超时和流式代理配置。

### 3.3 准备运行目录

在模型服务机执行：

```bash
install -d -m 700 /tmp/cloudflared-qwen36-api
```

该目录只保存本次 Tunnel 的 token 文件、PID 文件和日志文件；不要把其中任何文件提交到 Git。

### 3.4 启动 Tunnel 进程

把 `<TUNNEL_TOKEN>` 替换成 Cloudflare Dashboard 给出的 token。下面的写法使用 `--token-file`，避免把 token 暴露在进程命令行里：

```bash
read -rsp 'Cloudflare tunnel token: ' CLOUDFLARE_TUNNEL_TOKEN; echo
printf '%s' "$CLOUDFLARE_TUNNEL_TOKEN" > /tmp/cloudflared-qwen36-api/token
chmod 600 /tmp/cloudflared-qwen36-api/token
unset CLOUDFLARE_TUNNEL_TOKEN

nohup cloudflared tunnel \
  --no-autoupdate \
  --pidfile /tmp/cloudflared-qwen36-api/cloudflared.pid \
  --logfile /tmp/cloudflared-qwen36-api/cloudflared.log \
  run --token-file /tmp/cloudflared-qwen36-api/token \
  > /tmp/cloudflared-qwen36-api/stdout.log 2>&1 &
```

查看日志：

```bash
tail -f /tmp/cloudflared-qwen36-api/cloudflared.log
```

预期日志中能看到 connector 启动、连接 Cloudflare、Tunnel 状态为 active。

### 3.5 服务端本机验证

```bash
ps -fp "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
tail -n 50 /tmp/cloudflared-qwen36-api/cloudflared.log
```

如果 `cloudflared tunnel list` 因为未登录 Cloudflare CLI 而不能列出 Tunnel，不影响 token 方式运行。以 Cloudflare Dashboard 中 connector 状态、PID 进程状态和日志为准。

## 4. 方案 B：本地管理 Tunnel，适合需要文件化配置

如果希望把 Tunnel 配置落到服务器文件中审计，可以使用 locally-managed tunnel。

### 4.1 登录 Cloudflare

```bash
cloudflared tunnel login
```

按提示在浏览器授权域名。

### 4.2 创建 Tunnel

```bash
cloudflared tunnel create qwen36-api
```

记录输出的 Tunnel UUID，例如：

```text
396a68e0-e58b-45bb-9364-0e594a58fa03
```

### 4.3 写配置文件

创建 `/etc/cloudflared/config.yml`：

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /root/.cloudflared/<TUNNEL_UUID>.json

originRequest:
  connectTimeout: 30s
  noHappyEyeballs: true

ingress:
  - hostname: mgtvqwen36-apiexample.com
    service: http://127.0.0.1:18080
    originRequest:
      connectTimeout: 30s
  - service: http_status:404
```

校验配置：

```bash
cloudflared tunnel --config /etc/cloudflared/config.yml ingress validate
cloudflared tunnel --config /etc/cloudflared/config.yml ingress rule https://mgtvqwen36-apiexample.com/v1/models
```

预期：匹配到 `mgtvqwen36-apiexample.com -> http://127.0.0.1:18080`。

### 4.4 创建 DNS 路由

```bash
cloudflared tunnel route dns qwen36-api mgtvqwen36-apiexample.com
```

这会在 Cloudflare DNS 中创建 CNAME，指向 `<TUNNEL_UUID>.cfargotunnel.com`。

### 4.5 直接运行 Tunnel 进程

```bash
install -d -m 700 /tmp/cloudflared-qwen36-api

nohup cloudflared tunnel \
  --config /etc/cloudflared/config.yml \
  --no-autoupdate \
  --pidfile /tmp/cloudflared-qwen36-api/cloudflared.pid \
  --logfile /tmp/cloudflared-qwen36-api/cloudflared.log \
  run qwen36-api \
  > /tmp/cloudflared-qwen36-api/stdout.log 2>&1 &
```

查看状态：

```bash
ps -fp "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
tail -n 50 /tmp/cloudflared-qwen36-api/cloudflared.log
```

### 4.6 本机自动化脚本

仓库内已提供适配本机环境的自动化脚本：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
PUBLIC_HOSTNAME=mgtvqwen36-apiexample.com \
  TUNNEL_ID=396a68e0-e58b-45bb-9364-0e594a58fa03 \
  docs_always/qwen3.6-27b/setup_cloudflare_named_tunnel.sh
```

如果报 `tunnel with name already exists`，说明 Cloudflare 中已经有同名 Tunnel。不要重复创建；设置 `TUNNEL_ID=<已有 Tunnel UUID>` 复用已有 Tunnel 即可。

脚本不会调用 `systemctl`，而是用 `nohup cloudflared tunnel ... run ...` 启动普通进程，并把 PID 和日志写到 `/tmp/cloudflared-qwen36-api/`。

如果使用 Dashboard 生成的 remotely-managed tunnel token：

```bash
read -rsp 'Cloudflare tunnel token: ' CLOUDFLARE_TUNNEL_TOKEN; echo
export CLOUDFLARE_TUNNEL_TOKEN
PUBLIC_HOSTNAME=mgtvqwen36-apiexample.com \
  docs_always/qwen3.6-27b/setup_cloudflare_named_tunnel.sh
unset CLOUDFLARE_TUNNEL_TOKEN
```

Dashboard token 方式的 Public Hostname 仍需要在 Cloudflare Dashboard 中配置，指向 `http://127.0.0.1:18080`。

脚本默认值：

- `TUNNEL_NAME=qwen36-api`
- `ORIGIN_SERVICE=http://127.0.0.1:18080`
- `CONFIG_FILE=/etc/cloudflared/config.yml`
- `RUN_DIR=/tmp/cloudflared-qwen36-api`

## 5. 公网客户端验收

以下命令可在 Mac 或其它公网客户端执行。

### 5.1 设置环境变量

```bash
export OPENAI_BASE_URL=https://mgtvqwen36-apiexample.com/v1
export OPENAI_API_KEY=<OPENAI_API_KEY>
```

不要继续使用历史的专用 Base URL 变量；统一使用 `OPENAI_BASE_URL`。

### 5.2 TCP 和模型列表

```bash
nc -vz mgtvqwen36-apiexample.com 443

curl -sS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

预期：

- `nc` 显示 443 可连接。
- `/models` 返回 `qwen3.6-27b` 和 `max_model_len=131072`。

### 5.3 非流式 chat

```bash
curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "请用一句话介绍 Qwen3.6-27B。"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

预期：HTTP 200，`choices[0].message.content` 非空。

### 5.4 流式 chat

```bash
curl -N "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "qwen3.6-27b",
    "stream": true,
    "messages": [
      {"role": "user", "content": "请用两句话说明流式输出。"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

预期：持续返回 `data:` 分片，最后返回 `data: [DONE]`。

### 5.5 Python OpenAI SDK

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "请用一句话介绍李白。"}],
    max_tokens=128,
    temperature=0,
)
print(resp.choices[0].message.content)
```

### 5.6 完整验收脚本

在模型服务机或任意可访问公网域名的机器上执行：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export OPENAI_BASE_URL=https://mgtvqwen36-apiexample.com/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

预期输出包含：

```text
PASS health http=200
PASS models id=qwen3.6-27b max_model_len=131072
PASS bad_key http=401
PASS chat completion_tokens=...
PASS stream chunks=... done=True
PASS concurrency requests=8
PASS long_context ... prompt_tokens>=100000
PASS all requested checks
```

## 6. Cloudflare 配置建议

### 6.1 不建议启用缓存

LLM API 请求和响应都不应缓存。对 `mgtvqwen36-apiexample.com/*` 配置 Cache Rule：

```text
Cache eligibility: Bypass cache
```

### 6.2 保留 API key 鉴权

当前 SGLang 已使用 API key。公网入口必须继续要求：

```http
Authorization: Bearer <OPENAI_API_KEY>
```

不要在 Cloudflare、文档、Git 或日志中写入明文 API key。

### 6.3 Cloudflare Access 的取舍

如果启用 Cloudflare Access，普通 OpenAI SDK 请求会额外需要 Access service token header；这会增加客户端接入复杂度。

建议初始策略：

- 对内部试用：只保留 SGLang API key。
- 对固定调用方：在 Cloudflare WAF 中加 IP allowlist。
- 对公网大范围开放：再考虑 Access service token 或 mTLS。

### 6.4 限流

建议在 Cloudflare WAF 或 Rate Limiting 中限制：

```text
hostname eq "mgtvqwen36-apiexample.com"
path starts_with "/v1/"
rate: 60 requests / minute / IP
action: block or challenge
```

本机 Nginx 仍保留当前 `2r/s`、`burst=16` 限流作为第二层保护。

## 7. 运行维护

### 7.1 常用状态命令

```bash
ps -fp "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
tail -n 200 /tmp/cloudflared-qwen36-api/cloudflared.log
tail -f /tmp/cloudflared-qwen36-api/cloudflared.log
```

### 7.2 服务重启顺序

如果 Qwen 服务重启：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  docs_always/qwen3.6-27b/start_qwen36_27b.sh

kill "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
```

通常 Qwen 服务重启后不需要重启 `cloudflared`。如果需要刷新 Tunnel 连接，先 `kill` 旧进程，再重新执行 3.4 或 4.5 的 `nohup cloudflared tunnel ... run ...` 启动命令。

### 7.3 日志定位

客户端公网请求失败时按顺序查：

```bash
# 1. Cloudflare Tunnel 连接状态
ps -fp "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
tail -n 200 /tmp/cloudflared-qwen36-api/cloudflared.log

# 2. 本机 Nginx 是否收到请求
tail -n 100 /var/log/nginx/qwen36_27b_access.log
tail -n 100 /var/log/nginx/qwen36_27b_error.log

# 3. SGLang 是否可用
curl --noproxy '*' -sS \
  -H "Authorization: Bearer $(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)" \
  http://127.0.0.1:18080/v1/models
```

判断：

- Cloudflare 返回 `502/1033`：优先查 `cloudflared` 是否在线。
- Cloudflare 请求到达但 Nginx `502`：查 SGLang `127.0.0.1:30000`。
- `/v1/models` 返回 `401`：API key 错。
- 流式输出中断：查 Cloudflare/WAF/客户端超时，以及本机 Nginx 是否关闭 buffering。

## 8. 回滚

停止公网暴露不会影响本机 Qwen 服务。

### 8.1 停止 cloudflared

```bash
kill "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"
rm -f /tmp/cloudflared-qwen36-api/cloudflared.pid
```

### 8.2 删除 Cloudflare Public Hostname

在 Cloudflare Dashboard：

```text
Zero Trust -> Networks -> Tunnels -> qwen36-api -> Public Hostnames
```

删除 `mgtvqwen36-apiexample.com`。

### 8.3 删除 DNS 记录

在 Cloudflare DNS 删除：

```text
mgtvqwen36-apiexample.com CNAME <TUNNEL_ID>.cfargotunnel.com
```

## 9. 最终交付检查清单

- [ ] `cloudflared --version` 正常输出。
- [ ] `ps -fp "$(cat /tmp/cloudflared-qwen36-api/cloudflared.pid)"` 能看到 `cloudflared tunnel ... run` 进程。
- [ ] Cloudflare Dashboard 中 Tunnel connector 为 Healthy。
- [ ] Public Hostname 指向 `http://127.0.0.1:18080`。
- [ ] `OPENAI_BASE_URL=https://mgtvqwen36-apiexample.com/v1`。
- [ ] 公网客户端 `/v1/models` 返回 `qwen3.6-27b`。
- [ ] 非流式 chat HTTP 200。
- [ ] 流式 chat 返回 `data:` 并以 `[DONE]` 结束。
- [ ] `verify_qwen36_27b.py` 完整通过。
- [ ] 文档、Git、日志中没有明文 API key。
