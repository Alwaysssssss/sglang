# Qwen3.6-27B 暴露为公网可访问 API 的可行方案

本文给出把当前 Qwen3.6-27B OpenAI 兼容服务暴露为公网 API 的具体方案。当前服务已经在内网可用，但服务机只有 `10.119.16.70/20` 内网地址；这意味着不能只把客户端 URL 改成 `http://10.119.16.70:18080/v1` 就实现公网访问。要让 Mac 或任意公网客户端访问，必须引入一个公网入口。

## 1. 当前状态和结论

当前服务端状态：

| 项目 | 当前值 |
| --- | --- |
| 模型服务 | Qwen3.6-27B |
| OpenAI 模型名 | `qwen3.6-27b` |
| SGLang 后端 | `127.0.0.1:30000` |
| 本机 Nginx | `0.0.0.0:18080` |
| 服务机内网 IP | `10.119.16.70/20` |
| 当前内网 Base URL | `http://10.119.16.70:18080/v1` |
| 客户端环境变量 | `OPENAI_BASE_URL` |
| 鉴权 | `Authorization: Bearer <OPENAI_API_KEY>` |

结论：

- 当前 `10.119.16.70` 是内网地址，不是公网地址。
- 服务端本机访问 `http://10.119.16.70:18080/v1/models` 正常，只证明服务在内网侧可用。
- 公网客户端连接超时说明请求没有到达服务端，问题在网络入口，不在模型、API key、JSON payload 或 `stream=true`。
- 保持 SGLang 只监听 `127.0.0.1:30000` 是正确的；不要把 SGLang 直接暴露到公网。

## 2. 推荐方案：公网 VPS 网关 + SSH 反向隧道 + HTTPS Nginx

这是当前最可行、改动最小、风险可控的方案。它不要求模型服务机拥有公网 IP，也不要求在模型服务机打开入站公网端口；只要求模型服务机能够主动 SSH 连接一台公网 VPS。

### 2.1 架构

```text
公网客户端
  -> https://api.example.com/v1
  -> 公网 VPS Nginx 443
  -> VPS 本机 127.0.0.1:118080
  -> SSH reverse tunnel
  -> 模型服务机 127.0.0.1:18080
  -> 模型服务机 Nginx
  -> SGLang 127.0.0.1:30000
```

公网客户端只看到 HTTPS 域名；API key 仍由 SGLang 校验。公网 VPS 只承担 TLS、限流、日志和反向代理，不跑模型。

### 2.2 前置条件

需要准备：

- 一台公网 VPS，例如 `PUBLIC_VPS_IP`。
- 一个域名，例如 `api.example.com`，DNS A 记录指向 `PUBLIC_VPS_IP`。
- 模型服务机能访问公网 VPS 的 SSH 端口 `22`。
- 模型服务机当前 API 可用：

```bash
curl --noproxy '*' -sS \
  -H "Authorization: Bearer $(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)" \
  http://127.0.0.1:18080/v1/models
```

预期返回模型列表，包含 `qwen3.6-27b`。

## 3. 公网 VPS 配置

以下命令在公网 VPS 上执行。

### 3.1 安装依赖

```bash
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx
```

### 3.2 创建隧道用户

```bash
adduser --disabled-password --gecos "" qwen-tunnel
install -d -m 700 -o qwen-tunnel -g qwen-tunnel /home/qwen-tunnel/.ssh
touch /home/qwen-tunnel/.ssh/authorized_keys
chown qwen-tunnel:qwen-tunnel /home/qwen-tunnel/.ssh/authorized_keys
chmod 600 /home/qwen-tunnel/.ssh/authorized_keys
```

### 3.3 限制 SSH 用户只允许反向转发

创建 `/etc/ssh/sshd_config.d/qwen-tunnel.conf`：

```sshconfig
Match User qwen-tunnel
    AllowTcpForwarding remote
    GatewayPorts no
    X11Forwarding no
    AllowAgentForwarding no
    PermitTTY no
```

重载 SSH：

```bash
sshd -t
systemctl reload ssh
```

### 3.4 写入模型服务机公钥

先在模型服务机生成隧道专用 key，见第 4.1 节。拿到模型服务机公钥后，在 VPS 上追加到：

```bash
/home/qwen-tunnel/.ssh/authorized_keys
```

建议使用 `permitlisten` 限制只能监听 VPS 本机 `127.0.0.1:118080`：

```text
permitlisten="127.0.0.1:118080",no-agent-forwarding,no-X11-forwarding,no-pty ssh-ed25519 <模型服务机公钥>
```

如果 VPS 的 OpenSSH 版本不支持 `permitlisten`，保留第 3.3 节的 `Match User` 限制，并确保反向隧道只绑定 `127.0.0.1`。

### 3.5 配置公网 Nginx

先创建 HTTP 版本，证书签发后再自动升级 HTTPS。创建 `/etc/nginx/conf.d/qwen36_public.conf`：

```nginx
limit_req_zone $binary_remote_addr zone=qwen36_public_limit:10m rate=2r/s;

upstream qwen36_reverse_tunnel {
    server 127.0.0.1:118080;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    client_max_body_size 128m;
    proxy_connect_timeout 60s;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    send_timeout 900s;

    location / {
        limit_req zone=qwen36_public_limit burst=16 nodelay;

        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Authorization $http_authorization;

        proxy_buffering off;
        proxy_request_buffering off;
        proxy_pass http://qwen36_reverse_tunnel;
    }
}
```

检查并加载：

```bash
nginx -t
systemctl reload nginx
```

### 3.6 签发 HTTPS 证书

DNS A 记录生效后执行：

```bash
certbot --nginx -d api.example.com
```

证书签发后确认 Nginx 监听：

```bash
ss -ltnp | awk 'NR==1 || /:80|:443|:118080/'
```

公网最终 Base URL：

```text
https://api.example.com/v1
```

## 4. 模型服务机配置

以下命令在模型服务机 `10.119.16.70` 上执行。

### 4.1 生成隧道专用 SSH key

```bash
ssh-keygen -t ed25519 -f /etc/sglang/qwen36_tunnel_ed25519 -N "" -C "qwen36-public-api-tunnel"
cat /etc/sglang/qwen36_tunnel_ed25519.pub
```

把输出的公钥写入公网 VPS 的 `/home/qwen-tunnel/.ssh/authorized_keys`。

### 4.2 手工验证反向隧道

把 `PUBLIC_VPS_IP` 替换为公网 VPS IP：

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -i /etc/sglang/qwen36_tunnel_ed25519 \
  -R 127.0.0.1:118080:127.0.0.1:18080 \
  qwen-tunnel@PUBLIC_VPS_IP
```

保持该命令运行，在公网 VPS 上测试：

```bash
curl -sS \
  -H "Authorization: Bearer <OPENAI_API_KEY>" \
  http://127.0.0.1:118080/v1/models
```

预期返回模型列表，包含 `qwen3.6-27b`。

### 4.3 配置 systemd 常驻反向隧道

创建 `/etc/systemd/system/qwen36-public-api-tunnel.service`：

```ini
[Unit]
Description=Qwen3.6 public API reverse tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o StrictHostKeyChecking=accept-new \
  -i /etc/sglang/qwen36_tunnel_ed25519 \
  -R 127.0.0.1:118080:127.0.0.1:18080 \
  qwen-tunnel@PUBLIC_VPS_IP
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动：

```bash
systemctl daemon-reload
systemctl enable --now qwen36-public-api-tunnel
systemctl status qwen36-public-api-tunnel --no-pager
```

日志：

```bash
journalctl -u qwen36-public-api-tunnel -f
```

## 5. 公网验收

以下命令可在 Mac 或任意公网客户端执行。

### 5.1 环境变量

```bash
export OPENAI_BASE_URL=https://api.example.com/v1
export OPENAI_API_KEY=<从管理员获取的 key>
```

### 5.2 TCP 和模型列表

```bash
nc -vz api.example.com 443

curl -sS "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

预期：

- `nc` 显示 443 端口可连接。
- `/models` 返回 `qwen3.6-27b`。

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

预期：连续返回 `data:` 分片，并以 `data: [DONE]` 结束。

### 5.5 完整验收脚本

公网 HTTPS 入口打通后，可以在模型服务机或其它能访问公网域名的机器执行：

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export OPENAI_BASE_URL=https://api.example.com/v1
docs_always/qwen3.6-27b/verify_qwen36_27b.py
```

预期输出包含：

```text
PASS health http=200
PASS models id=qwen3.6-27b max_model_len=131072
PASS bad_key http=401
PASS chat ...
PASS stream ...
PASS concurrency requests=8
PASS long_context ... prompt_tokens>=100000
PASS all requested checks
```

## 6. 安全要求

公网暴露必须满足以下要求：

- 只暴露公网 VPS 的 `443`，不要把 SGLang `30000` 暴露到公网。
- 公网入口必须使用 HTTPS。
- 业务请求必须带 `Authorization: Bearer <OPENAI_API_KEY>`。
- 不要把 API key 写入文档、Git、Nginx 配置或命令历史。
- Nginx 必须关闭 `proxy_buffering` 和 `proxy_request_buffering`，否则流式响应和长上下文请求可能异常。
- 保留公网入口限流，初始建议 `2r/s`、`burst=16`。
- 如调用方固定，建议在公网 VPS Nginx 增加 IP allowlist：

```nginx
allow <client-public-ip>;
deny all;
```

## 7. 回滚

关闭公网入口不会影响模型服务本身。

在模型服务机关闭反向隧道：

```bash
systemctl disable --now qwen36-public-api-tunnel
```

在公网 VPS 下线 Nginx 入口：

```bash
rm -f /etc/nginx/conf.d/qwen36_public.conf
nginx -t
systemctl reload nginx
```

删除 DNS A 记录或改回维护页。

## 8. 备选方案

### 8.1 云负载均衡或安全组 DNAT

如果当前环境所属云平台能给 `10.119.16.70` 绑定公网负载均衡或 DNAT，可以使用：

```text
公网 LB 443
  -> 10.119.16.70:18080
  -> 模型服务机 Nginx
  -> SGLang 127.0.0.1:30000
```

要求：

- LB 负责 TLS 证书。
- 安全组只放通 `443`。
- 后端目标端口为 `10.119.16.70:18080`。
- LB idle timeout 至少 `900s`，否则长上下文和流式响应可能被中断。

这是最标准的生产方案，但需要云平台权限。

### 8.2 Cloudflare Tunnel

如果没有公网 VPS，但模型服务机可以主动访问公网，可以使用 Cloudflare Tunnel：

```text
公网客户端
  -> https://api.example.com
  -> Cloudflare
  -> cloudflared tunnel
  -> 127.0.0.1:18080
```

示例配置 `/etc/cloudflared/config.yml`：

```yaml
tunnel: <tunnel-id>
credentials-file: /etc/cloudflared/<tunnel-id>.json

ingress:
  - hostname: api.example.com
    service: http://127.0.0.1:18080
    originRequest:
      connectTimeout: 60s
      noHappyEyeballs: true
  - service: http_status:404
```

启动：

```bash
cloudflared tunnel run <tunnel-name>
```

这个方案不需要自建 VPS，但依赖 Cloudflare 账号和域名托管，且公网流量经过第三方网络。

## 9. 推荐落地顺序

1. 准备公网 VPS 和域名。
2. 在 VPS 配置 `qwen-tunnel` 用户、SSH 限制和 Nginx。
3. 在模型服务机配置 systemd 反向隧道。
4. 在 VPS 本机验收 `http://127.0.0.1:118080/v1/models`。
5. 签发 HTTPS 证书。
6. 从 Mac 使用 `OPENAI_BASE_URL=https://api.example.com/v1` 验收 `/models`、非流式、流式。
7. 跑完整 `verify_qwen36_27b.py`。
8. 观察 Nginx access/error log 和 SGLang 日志，再开放给外部用户。
