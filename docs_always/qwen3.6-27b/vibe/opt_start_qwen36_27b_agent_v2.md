# Feature: 实际agent在使用过程中，经常出现qwen-27b停止工作的情况，需要点击继续，qwen-27b才会继续运行

> 本文是实施方案，不包含实现代码。后续如执行，需要按本文范围修改脚本、测试和文档，并在真实 GPU 环境完成验收。

## 0. 需求归纳

- 检查当前docs_always/qwen3.6-27b/start_qwen36_27b_agent.sh的配置
- 发出请求的主要是 langchain 的 ChatOpenAI

找到具体的原因，并提出解决办法

## 1. Nginx `proxy_read_timeout` 调大流程

### 1.1 适用场景

如果 Nginx error log 出现类似：

```text
upstream timed out while reading response header from upstream
```

并且 access log 对应请求返回 `504`，说明 Nginx 在等待 SGLang 返回响应头时超过了 `proxy_read_timeout`。当前 `docs_always/qwen3.6-27b/nginx_qwen36_27b.conf` 与线上 `/etc/nginx/conf.d/qwen36_27b.conf` 的默认值是 `900s`。如果业务确实允许单次长请求运行超过 15 分钟，可以调大该值。

注意：调大 Nginx timeout 只能避免代理层过早断开，不能解决 `max_new_tokens=128000`、Qwen3 thinking 过长、客户端只读取 `content` 不读取 `reasoning_content` 等请求侧根因。根因治理仍应限制调用方的 `max_tokens/max_completion_tokens`，普通请求建议先收敛到 `1024~4096`。

### 1.2 确认当前线上配置

```bash
grep -n "proxy_.*timeout\|send_timeout" /etc/nginx/conf.d/qwen36_27b.conf
```

预期当前类似：

```nginx
proxy_connect_timeout 60s;
proxy_send_timeout 900s;
proxy_read_timeout 900s;
send_timeout 900s;
```

### 1.3 备份线上配置

```bash
sudo cp /etc/nginx/conf.d/qwen36_27b.conf \
  /etc/nginx/conf.d/qwen36_27b.conf.bak.$(date +%Y%m%dT%H%M%S)
```

### 1.4 修改线上配置

编辑：

```bash
sudo vim /etc/nginx/conf.d/qwen36_27b.conf
```

建议同步调大 `proxy_send_timeout`、`proxy_read_timeout` 和 `send_timeout`。例如调到 30 分钟：

```nginx
proxy_connect_timeout 60s;
proxy_send_timeout 1800s;
proxy_read_timeout 1800s;
send_timeout 1800s;
```

如果明确允许 1 小时长请求，可以改为：

```nginx
proxy_connect_timeout 60s;
proxy_send_timeout 3600s;
proxy_read_timeout 3600s;
send_timeout 3600s;
```

不建议只改 `proxy_read_timeout`，否则客户端上传、上游发送或下游发送链路仍可能先被其它 timeout 截断。

### 1.5 检查 Nginx 语法

```bash
sudo nginx -t
```

必须看到类似：

```text
syntax is ok
test is successful
```

如果语法检查失败，不要 reload；先恢复备份或修正配置。

### 1.6 平滑 reload

```bash
sudo nginx -s reload
```

或：

```bash
sudo systemctl reload nginx
```

### 1.7 验证配置已生效

查看 Nginx 实际加载后的配置：

```bash
sudo nginx -T | grep -A20 -B5 "listen 18080" | grep "timeout"
```

验证 OpenAI 兼容入口仍可用：

KEY=sk-qwen36-KGf3k3ocLnAzsMuvcLep4rXyPdWWIDJ0vep3opqriug

curl --noproxy '*' -i \
  -H "Authorization: Bearer ${KEY}" \
  http://106.75.235.227:10069/v1/models

```bash
KEY=$(tr -d '[:space:]' < /etc/sglang/qwen36_openai_api_key)

curl --noproxy '*' -i \
  -H "Authorization: Bearer ${KEY}" \
  http://127.0.0.1:18080/v1/models
```

再发一个有明确短输出上限的 chat 请求，确认代理链路可正常返回：

```bash
curl --noproxy '*' -i \
  -H "Authorization: Bearer ${KEY}" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:18080/v1/chat/completions \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Answer exactly: pong"}],
    "max_tokens": 16,
    "temperature": 0,
    "reasoning_effort": "none"
  }'
```

### 1.8 观察日志

```bash
tail -f /var/log/nginx/qwen36_27b_error.log
```

```bash
tail -f /var/log/nginx/qwen36_27b_access.log
```

调大后，同类长请求在新 timeout 前不应再出现：

```text
upstream timed out while reading response header from upstream
```

如果仍出现该错误，说明请求运行时间已经超过新的 `proxy_read_timeout`，需要继续从调用方的 `max_tokens/max_completion_tokens`、是否默认 thinking、是否非流式调用、客户端 timeout 等方向收敛。

### 1.9 同步仓库配置副本

线上配置变更后，需要同步仓库副本，避免文档和线上漂移：

```bash
vim docs_always/qwen3.6-27b/nginx_qwen36_27b.conf
```

把其中的：

```nginx
proxy_send_timeout 900s;
proxy_read_timeout 900s;
send_timeout 900s;
```

同步改成线上值，例如：

```nginx
proxy_send_timeout 1800s;
proxy_read_timeout 1800s;
send_timeout 1800s;
```

同步后至少执行：

```bash
git diff -- docs_always/qwen3.6-27b/nginx_qwen36_27b.conf \
  docs_always/qwen3.6-27b/vibe/opt_start_qwen36_27b_agent_v2.md
```

确认只包含预期的 timeout 和文档变更。
