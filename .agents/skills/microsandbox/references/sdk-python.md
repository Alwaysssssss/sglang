# Python SDK Reference

Install: `pip install microsandbox`

## Sandbox Lifecycle

```python
from microsandbox import Sandbox

# Create (attached — stops when process exits)
sb = await Sandbox.create("worker", image="python:3.12")

# Create with full options
sb = await Sandbox.create(
    "worker",
    image="python:3.12",
    memory=1024,        # MiB
    cpus=2,
    workdir="/app",
    shell="/bin/bash",
    env={"DEBUG": "true"},
    volumes={"/data": {"type": "named", "name": "my-data"}},
    ports={8080: 80},
    network="public_only",  # "public_only" | "none" | "allow_all"
    replace=True,
    detached=False,
    idle_timeout=300,   # seconds
    max_duration=3600,  # seconds
)

# Context manager (recommended — auto-cleanup)
async with await Sandbox.create("worker", image="alpine") as sb:
    output = await sb.shell("echo hello")
    print(output.stdout_text)

# Restart a stopped sandbox
sb = await Sandbox.start("worker")

# Get handle to existing sandbox
handle = await Sandbox.get("worker")
sb = await handle.connect()

# List all sandboxes
handles = await Sandbox.list()
```

## Command Execution

```python
# Shell command (pipes, redirects, && chains)
output = await sb.shell("ls -la /app && echo done")
print(output.stdout_text)   # str
print(output.exit_code)     # int

# Run a binary directly
output = await sb.exec("python", ["-c", "print('hello')"])

# Streaming output
handle = await sb.shell_stream("tail -f /var/log/app.log")
# handle emits stdout, stderr, exit events

# Streaming exec
handle = await sb.exec_stream("python", ["-c", "import time; [print(i) or time.sleep(1) for i in range(5)]"])
```

## Filesystem

Access via `sb.fs` (no `await` — methods are async but called on the property):

```python
# Read
content = await sb.fs.read_text("/app/config.json")     # str
raw = await sb.fs.read("/app/data.bin")                  # bytes

# Write
await sb.fs.write("/app/data.json", b'{"key": "value"}')

# List directory
entries = await sb.fs.list("/app")
for e in entries:
    print(e.path, e.kind, e.size)  # kind: "file"|"directory"|"symlink"|"other"

# Create directory (recursive)
await sb.fs.mkdir("/app/output")

# Check existence
exists = await sb.fs.exists("/app/config.json")

# File metadata
meta = await sb.fs.stat("/app/data.json")
# meta.kind, meta.size, meta.mode, meta.modified, meta.readonly

# Copy / rename / remove
await sb.fs.copy("/app/a.txt", "/app/b.txt")
await sb.fs.rename("/app/old.txt", "/app/new.txt")
await sb.fs.remove("/app/temp.txt")
await sb.fs.remove_dir("/app/cache")

# Host <-> guest transfer
await sb.fs.copy_from_host("./local.txt", "/app/local.txt")
await sb.fs.copy_to_host("/app/result.txt", "./result.txt")

# Streaming (large files)
async with sb.fs.write_stream("/data/output.bin") as sink:
    await sink.write(chunk1)
    await sink.write(chunk2)

async with sb.fs.read_stream("/data/large.bin") as stream:
    async for chunk in stream:
        process(chunk)
```

### FsEntry properties

| Property | Type | Description |
|----------|------|-------------|
| `path` | `str` | File path |
| `kind` | `str` | `"file"`, `"directory"`, `"symlink"`, `"other"` |
| `size` | `int` | Size in bytes |
| `mode` | `int` | Unix permission bits |
| `modified` | `float \| None` | Last modified (ms since epoch) |

## Rootfs Patches (before boot)

```python
from microsandbox import Sandbox, Patch

sb = await Sandbox.create("worker", image="alpine", patches=[
    Patch.text("/app/config.json", '{"debug": true}', mode=0o644, replace=False),
    Patch.mkdir("/app/logs", mode=0o755),
    Patch.copy_file("./cert.pem", "/etc/ssl/cert.pem"),
    Patch.copy_dir("./src", "/app/src"),
    Patch.append("/etc/hosts", "10.0.0.1 myhost"),
    Patch.symlink("/target", "/link"),
    Patch.remove("/tmp/junk"),
])
```

## Lifecycle Control

```python
await sb.stop()                # Graceful (SIGTERM)
await sb.stop_and_wait()       # Stop + wait → (exit_code, success)
await sb.kill()                # Force (SIGKILL)
await sb.drain()               # Graceful drain (SIGUSR1) — finish current, reject new
await sb.detach()              # Release handle, sandbox keeps running
await sb.wait()                # Block until exit → (exit_code, success)
await sb.remove_persisted()    # Delete sandbox + state from disk

# Static removal
await Sandbox.remove("worker") # Must be stopped first
```

## Metrics & Logs

```python
# Point-in-time metrics
m = await sb.metrics()
# m.cpu_percent, m.memory_bytes, m.memory_limit_bytes,
# m.disk_read_bytes, m.disk_write_bytes, m.net_rx_bytes, m.net_tx_bytes,
# m.uptime_ms, m.timestamp_ms

# Streaming metrics
async for snapshot in sb.metrics_stream(interval=2.0):
    print(f"CPU: {snapshot.cpu_percent}%, Mem: {snapshot.memory_bytes}")

# Read logs (works on running and stopped sandboxes)
entries = await sb.logs(tail=50, sources=["stdout", "stderr", "output", "system"])
for e in entries:
    print(f"[{e.timestamp_ms / 1000:.3f}] {e.source}: {e.text()}")
```

## SandboxHandle (from Sandbox.get / Sandbox.list)

| Property/Method | Type | Description |
|----------------|------|-------------|
| `name` | `str` | Sandbox name |
| `status` | `str` | `"running"`, `"stopped"`, `"crashed"`, `"draining"`, `"paused"` |
| `config_json` | `str` | Raw JSON configuration |
| `created_at` | `float \| None` | Creation timestamp (ms) |
| `updated_at` | `float \| None` | Last update timestamp (ms) |
| `connect()` | → `Sandbox` | Connect to running sandbox |
| `start(*, detached=False)` | → `Sandbox` | Start sandbox |
| `stop()` | → `None` | Graceful shutdown |
| `kill()` | → `None` | Force terminate |
| `remove()` | → `None` | Delete sandbox and state |
| `metrics()` | → `SandboxMetrics` | Resource metrics |
| `logs()` | → `list[LogEntry]` | Read captured logs |

## Key Types

### ExecOutput

| Property | Type | Description |
|----------|------|-------------|
| `stdout_text` | `str` | Stdout as UTF-8 string |
| `stderr_text` | `str` | Stderr as UTF-8 string |
| `exit_code` | `int` | Process exit code |

### SandboxMetrics

| Field | Type |
|-------|------|
| `cpu_percent` | `float` |
| `memory_bytes` | `int` |
| `memory_limit_bytes` | `int` |
| `disk_read_bytes` | `int` |
| `disk_write_bytes` | `int` |
| `net_rx_bytes` | `int` |
| `net_tx_bytes` | `int` |
| `uptime_ms` | `int` |
| `timestamp_ms` | `float` |

### Enums

- **PullPolicy**: `"always"`, `"if-missing"` (default), `"never"`
- **LogLevel**: `"trace"`, `"debug"`, `"info"` (default)`, `"warn"`, `"error"`
- **NetworkPolicy**: `"public_only"` (default), `"none"`, `"allow_all"`
