# VideoEdit 原始/DMD 双服务 + 单队列修改方案

## 1. 结论

方案一在架构上可行，目标形态为：

```text
                         ┌─ normal 后端：127.0.0.1:31100 ─┐
客户端 ──> 队列网关 :30000 ┤                               ├─> 同一组 GPU 0,1
                         └─ DMD 后端：127.0.0.1:32100 ────┘
                                  全局并发恒为 1
```

- 两个 SGLang 服务始终加载各自的 Transformer，不再发生请求时权重切换。
- 两个后端都使用 GPU 0、1，并尽可能把 DiT、文本编码器、图像编码器和 VAE
  offload 到 CPU。
- 客户端只访问 `:30000`。网关只有一个队列消费者，在当前视频到达终态之前，
  绝不向另一个后端提交下一条视频。
- 两个后端只监听 `127.0.0.1`，避免调用方绕过全局队列直接造成双视频并发。
- 两个后端严格串行启动；第一服务完成 offload 和 CUDA cache 清理并通过资源
  闸门后，才允许启动第二服务。

这不是“大幅修改模型推理代码”。核心推理流程和权重加载逻辑不改，修改范围主要是：

1. 在 layerwise offload 初始化结束后增加一次 CUDA allocator cache 清理；
2. 新增一个轻量队列网关；
3. 新增双服务启动、停止、状态和资源检查脚本；
4. 新增队列互斥及 L20 资源验收测试。

但是，当前 L20 的资源不能预先判定为一定足够：

- A100 实测单服务完成清理后约占每卡 `7.3 GiB`，单服务启动峰值约
  `38.6 GiB`。当前 L20 的 `nvidia-smi` 总显存为 `46068 MiB`，即约
  `44.99 GiB`；第二服务启动时粗略叠加 `7.3 + 38.6 = 45.9 GiB/卡`，
  已超过物理容量。加上 `2 GiB` 安全余量后，按该组 A100 数据推算会直接被
  第二服务硬闸门拒绝，只有 L20 实测启动峰值明显更低时才可能通过。
- A100 实测单服务 cgroup `memory.current` 约为 `121.5 GiB`。若第二个完整
  服务增量接近第一服务，两服务约为 `243 GiB`。本次检查机器当前约有
  `763 GiB MemAvailable`、cgroup `memory.max=max`，因此这台机器的首要风险
  是 GPU 启动峰值，不是 host memory；部署脚本仍保留 host/cgroup 闸门以兼容
  其他机器。

因此本方案是“带资源闸门的可验证实施方案”。任何闸门不通过时都不强行启动第二
服务，而是保留 normal 单服务并回退到热切换方案或增加机器资源。

## 2. 目标与边界

### 2.1 必须满足

- 原始 Transformer 和 DMD Transformer 分别常驻在两个独立后端；
- 两个后端都使用同一组 `CUDA_VISIBLE_DEVICES=0,1`；
- 全局同时最多执行一个 VideoEdit 视频任务；
- 请求可排队，而不是返回当前后端的 `code=2, A task is running`；
- 模型选择和排队是一个原子提交操作；
- 原有 `/v1/videos/repairs` 请求字段、输出路径和 callback 能力保持可用；
- 后端启动失败、OOM 或资源不足时可自动停止第二服务并恢复到可用状态。

### 2.2 本阶段不做

- 不在请求间调用 `/update_weights_from_disk`；
- 不让两个服务同时执行视频推理；
- 不实现跨进程共享 PyTorch module/tensor；
- 不默认修改 DMD 的推理步数、TeaCache 或其他业务参数；
- 不引入 Redis。单机先使用 SQLite 持久化队列，后续多网关实例再切换 Redis。

## 3. 为什么不能只起两个原始服务

当前 `video_api.py` 中的 `_VIDEOEDIT_SEMAPHORE` 是进程内对象。两个 SGLang 服务
各自持有一个 semaphore：

```text
normal semaphore != DMD semaphore
```

因此同时向 `31100` 和 `32100` 提交请求时，两个请求都能获得自己的 semaphore，
最终会同时占用两张 GPU。仅设置 `VIDEOEDIT_QUEUE_CAPACITY=1` 只能限制单个后端，
不能建立跨进程的全局互斥。

此外，后端 `POST /v1/videos/repairs` 在注册后台任务后就返回，视频尚未执行完成。
所以网关不能以“POST 已返回”作为释放队列锁的条件，必须继续轮询：

```text
GET /v1/videos/{task_id}
```

直到状态为 `completed` 或 `failed`，才能调度下一条任务。

## 4. 模型与目录约定

建议集中在实际启动项目下保存日志、PID、队列库和默认输出：

```text
PROJECT_ROOT=/mnt/nas/models/test/sglang
PYTHON_BIN=python3
SGLANG_BIN=/usr/local/bin/sglang

BASE_MODEL=/mnt/nas/models/DifusserEdit/pretrain_models/VideoEdit-diffusers-model
NORMAL_TRANSFORMER=/mnt/nas/models/latest_edit/step-55000-diffusers-lh/transformer
DMD_TRANSFORMER=/mnt/nas/models/latest_edit/merged_dit_lightx2v_lora_scale_1p0

RUNTIME_DIR=${PROJECT_ROOT}/runtime/videoedit-dual
LOG_DIR=${RUNTIME_DIR}/logs
PID_DIR=${RUNTIME_DIR}/pids
QUEUE_DB=${RUNTIME_DIR}/queue.sqlite3
OUTPUT_DIR=${PROJECT_ROOT}/outputs/videoedit-dual
INPUT_DIR=${RUNTIME_DIR}/inputs
```

当前两个 `/mnt/nas/models/latest_edit/...` 目录为 `root:root 700`；非 root
服务用户启动前必须获得只读/遍历权限。2026-07-28 的只读检查显示 DMD 目录现有
7 个分片及 index，`MERGE_VERIFIED.txt` 记录全部分片重载验证通过；完整模型目录也
通过相同结构检查。启动脚本仍必须在每次启动前验证 `config.json`、全部
safetensors 分片和 index 引用，且确认 `_class_name` 为
`WanVideoEditTransformer3DModel`、`in_channels=36`、`out_channels=16`，防止后续
传输或替换产生截断文件。

两个后端必须使用不同的 input/output 子目录，避免文件名和清理逻辑互相影响：

```text
${OUTPUT_DIR}/normal
${OUTPUT_DIR}/dmd
${INPUT_DIR}/normal
${INPUT_DIR}/dmd
```

## 5. 后端启动参数

### 5.1 Offload 组合

两个后端统一使用：

```text
--dit-layerwise-offload true
--dit-offload-prefetch-size 0
--dit-cpu-offload false
--text-encoder-cpu-offload true
--image-encoder-cpu-offload true
--vae-cpu-offload true
--pin-cpu-memory true
```

说明：

- `--dit-layerwise-offload true` 与整个 DiT 的 `--dit-cpu-offload true` 是互斥
  方案，不能同时开启。因此这里显式设置 `--dit-cpu-offload false`。
- `--dit-offload-prefetch-size 0` 表示只预取一层，是当前最低 GPU 显存模式。
- 其他可 offload 组件全部启用 CPU offload。
- `pin_cpu_memory=true` 有利于 H2D 传输，但会增加 pinned/shmem 压力。它是
  首选配置；若 CPU/cgroup 闸门失败，再单独 A/B 测试
  `--pin-cpu-memory false` 的内存和延迟，不能未经测试直接用于生产。
- 初次 L20 验证应分别记录
  `SGLANG_USE_RUNAI_MODEL_STREAMER=true/false` 的启动峰值。建议先用 `false`
  测试，避免 streamer `clone()` 带来的临时 CPU tensor；最终选择以 L20
  实测的 GPU 峰值、cgroup 峰值和启动时间为准。

### 5.2 端口规划

| 用途 | normal | DMD |
| --- | ---: | ---: |
| HTTP | 31100 | 32100 |
| HTTP broker（HTTP+1） | 31101 | 32101 |
| distributed master | 31105 | 32105 |
| scheduler | 31755 | 32755 |
| NCCL | 31955 | 32955 |

2026-07-28 预检发现原草案端口 `31000/31555/32000` 已被当前运行环境占用；下表改用已逐个 bind 验证空闲的端口组。

统一网关监听：

```text
0.0.0.0:30000
```

后端传入 `--strict-ports true`，启动脚本还需在拉起进程前检查上表全部端口。
本次实现已给 multimodal CLI 暴露已有的 `ServerArgs.nccl_port`，两个后端使用
固定且不重叠的 NCCL 端口，避免随机端口在双服务部署中造成不可复现的冲突。

### 5.3 normal 后端命令模板

```bash
VIDEOEDIT_QUEUE_CAPACITY=1 \
SGLANG_USE_RUNAI_MODEL_STREAMER=false \
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="${PROJECT_ROOT}/python" "${SGLANG_BIN}" serve \
  --model-type diffusion \
  --model-path "${BASE_MODEL}" \
  --transformer-path "${NORMAL_TRANSFORMER}" \
  --host 127.0.0.1 \
  --port 31100 \
  --master-port 31105 \
  --scheduler-port 31755 \
  --nccl-port 31955 \
  --scheduler-response-timeout -1 \
  --strict-ports true \
  --num-gpus 2 \
  --sp-degree 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --dit-layerwise-offload true \
  --dit-offload-prefetch-size 0 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload true \
  --image-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --pin-cpu-memory true \
  --warmup false \
  --output-path "${OUTPUT_DIR}/normal" \
  --input-save-path "${INPUT_DIR}/normal"
```

### 5.4 DMD 后端命令模板

DMD 后端参数完全相同，只替换：

```text
--transformer-path "${DMD_TRANSFORMER}"
--port 32100
--master-port 32105
--scheduler-port 32755
--nccl-port 32955
--output-path "${OUTPUT_DIR}/dmd"
--input-save-path "${INPUT_DIR}/dmd"
```

启动阶段关闭自动 warmup，防止第二服务刚启动就产生额外显存峰值。两个服务均
通过资源闸门后，再由验收脚本按 normal、DMD 的顺序分别执行一次受控 smoke
请求。

## 6. 需要修改和新增的文件

### 6.1 必做：offload 后主动清理 CUDA cache

修改：

```text
python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
```

位置：`GPUWorker.init_device_and_model()` 中所有
`configure_layerwise_offload()` 完成之后、worker ready 日志之前。

逻辑：

```python
if self.server_args.dit_layerwise_offload:
    gc.collect()
    torch.get_device_module().empty_cache()
```

同时记录每个 rank 清理前后的：

```text
memory_allocated
memory_reserved
```

该修改的作用不是释放仍在使用的 tensor，而是归还模型初始化期间已经无引用、
但仍被 PyTorch CUDA allocator 保留的显存。现有代码已导入 `gc` 和 `torch`，
修改量很小。

验收要求：

- 两个 rank 都执行清理，而不是只在 rank 0 执行；
- `/health` 只能在两个 rank 都完成清理后变为 ready；
- 单服务 idle GPU 占用应接近 A100 实测的 `7.3 GiB/卡`，L20 以实际值为准；
- 不通过“对同一 checkpoint 做一次权重更新”来间接清 cache，避免额外读取
  `30.6 GiB` 权重和数十秒至数分钟延迟。

### 6.2 新增：持久化单消费者队列网关

建议新增：

```text
python/sglang/multimodal_gen/runtime/videoedit/dual_service_gateway.py
python/sglang/multimodal_gen/runtime/videoedit/dual_service_store.py
```

网关仍提供原接口：

```text
POST   /v1/videos/repairs
GET    /v1/videos/{task_id}
GET    /v1/videos/{task_id}/progress
DELETE /v1/videos/{task_id}
GET    /health
GET    /admin/queue
```

队列采用标准库 `sqlite3`，开启 WAL。核心字段：

| 字段 | 含义 |
| --- | --- |
| task_id | 全局唯一任务 ID |
| variant | `normal` 或 `dmd` |
| backend_url | 被选中的后端 |
| request_json | 原请求 |
| status | `queued/dispatching/running/completed/failed/cancelled` |
| created_at/started_at/completed_at | 排队与执行时间 |
| backend_response | 后端状态快照 |
| error | 调度或后端错误 |

只启动一个 dispatcher 协程，状态机为：

```text
queued
  -> dispatching
  -> POST 对应后端 /v1/videos/repairs
  -> running
  -> 循环 GET /v1/videos/{task_id}
  -> completed / failed
  -> 才读取下一条 queued 任务
```

SQLite 事务使用 `BEGIN IMMEDIATE` 认领最早任务，并增加 partial unique index：

```sql
CREATE UNIQUE INDEX tasks_one_active ON tasks ((1))
WHERE status IN ('dispatching', 'running', 'cancelling');
```

事务负责 FIFO 认领，数据库约束负责保证最多一个 active task；即使误启动两个
dispatcher，也不能同时认领两条任务。部署层仍规定 gateway 使用单 worker。

后端 POST 返回 `code=0` 只表示任务已注册，不能释放 dispatcher。只有后端 GET
返回终态才释放。

### 6.3 模型选择协议

复用现有 `VideoRepairRequest.model` 字段，不修改 VideoEdit 请求模型：

```json
{
  "task_id": "normal-001",
  "model": "videoedit-normal"
}
```

或：

```json
{
  "task_id": "dmd-001",
  "model": "videoedit-dmd"
}
```

路由规则：

| `model` 值 | 后端 |
| --- | --- |
| 缺省、`videoedit`、`videoedit-normal`、`normal` | normal |
| `videoedit-dmd`、`dmd` | DMD |
| 其他值 | `400`，不进入队列 |

normal 路由不改写采样参数。DMD 路由在进入持久队列前统一强制为
`num_inference_steps=4`、`guidance_scale=1.0`、`dynamic_cfg=false`，并清空
`negative_prompt`。因此旧请求端只需增加 `"model": "videoedit-dmd"`，即使仍携带
完整模型的 CFG 参数也不会执行反向提示词编码或 unconditional Transformer 分支。

`task_id` 必须全局唯一，不能在 normal 和 DMD 各自重复。网关在入队事务中检查
重复 ID。

### 6.4 callback、查询和取消

- 网关保存 `task_id -> backend_url` 映射。
- 任务在 `queued` 时，GET 直接返回网关的排队状态。
- 任务进入 `running` 后，GET/progress 代理对应后端并同步 SQLite 快照。
- 第一阶段将原 `callback_url` 原样传给后端，由实际执行任务的后端回调用户；
  网关的轮询仅用于全局互斥，不重复发送 callback。
- queued 任务取消时直接改为 `cancelled`，不提交后端。
- running 任务取消复用 `online_videoedit` 的跨进程 cancel marker：DELETE 写入
  marker 后等待 scheduler/GPU worker 在现有检查点实际退出，后端任务达到终态后
  才返回；gateway 在此之前保持 `cancelling` active slot，不会派发另一模型。
- 网关重启后重新加载 queued 任务；对 dispatching/running/cancelling 任务先查询
  原后端，无法确认时暂停队列，禁止不加判断地自动重跑。
- 当前 v1 约定排队时间不计入后端 `timeout`；带签名的输入 URL 必须覆盖最大排队
  时间。SQLite 会保存原请求，因此队列目录权限必须为 `0700`、数据库为 `0600`，
  `/admin/queue` 不返回 `request_json`。

### 6.5 新增：生命周期和资源脚本

建议新增：

```text
scripts/videoedit_dual_service/config.env.example
scripts/videoedit_dual_service/start.sh
scripts/videoedit_dual_service/stop.sh
scripts/videoedit_dual_service/status.sh
scripts/videoedit_dual_service/resource_probe.py
scripts/videoedit_dual_service/smoke_and_queue_test.py
```

要求：

- `start.sh` 使用 `flock` 防止两个启动流程并发；
- PID 写入独立 pidfile，停止时只向 pidfile 中经过校验的 PID 发 `SIGTERM`；
- 不使用宽泛的 `pkill sglang`；
- normal、DMD、gateway 使用独立日志；
- 第二服务失败时只停止 DMD，保留已经健康的 normal；
- normal 健康后就启动 gateway；DMD 失败或资源闸门不通过时 gateway 进入
  `degraded_normal_only`，normal 请求继续可用，DMD 请求返回明确 `503`；
- `status.sh` 同时显示三进程状态、两个 `/health`、队列深度、活动任务、
  GPU 使用量、`MemAvailable`、cgroup `memory.current/max/events`。

## 7. 串行启动和资源闸门

### 7.1 启动前基线

记录：

```text
nvidia-smi 每卡 total/used/free
/proc/meminfo 的 MemAvailable
cgroup v2 memory.current
cgroup v2 memory.max
cgroup v2 memory.events
```

前置条件：

- GPU 0、1 上没有其他计算进程；
- 每卡初始可用显存不低于总显存减 `2 GiB`；
- `memory.events` 中没有正在增长的 `oom`/`oom_kill`；
- `memory.max` 若不是 `max`，必须同时用它计算余量，不能只看 `free -h`；
- 所有模型、输入和输出目录可读写；
- 规划端口全部空闲。

### 7.2 启动 normal

1. 启动 normal，最长等待 15 分钟；
2. 每秒记录 GPU used、进程 RSS、cgroup memory.current；
3. 等待 `GET 127.0.0.1:31100/health` 成功；
4. 等待显存连续 10 秒不再下降，记录：
   - 第一服务启动 GPU 峰值增量 `P1`；
   - 清理后 idle GPU 增量 `I1`；
   - 第一服务 cgroup 增量 `C1`；
5. 检查日志中两个 rank 都出现 offload cache trim 记录。

### 7.3 启动 DMD 前的硬闸门

对每张 GPU：

```text
free_after_normal >= P1 + 2 GiB
```

CPU/cgroup：

```text
MemAvailable_after_normal >= 1.15 * C1 + 40 GiB
```

若 `memory.max` 有限，还必须：

```text
memory.max - memory.current >= 1.15 * C1 + 40 GiB
```

这里 `15%` 用于第二服务差异和启动临时 tensor，`40 GiB` 是无 Swap 环境的最低
系统/请求工作区余量。该门槛在当前 L20 上可能不通过；不通过就停止本次双服务
尝试，不能靠反复重启碰运气。

### 7.4 启动 DMD

1. 启动 DMD，继续每秒监控；
2. 等待 `GET 127.0.0.1:32100/health` 成功；
3. 等待显存稳定；
4. 检查 `memory.events` 的 `oom`、`oom_kill`、`max` 没有新增；
5. 要求双服务 idle 后：
   - 每卡至少保留 `4 GiB` 显存；
   - `MemAvailable >= 40 GiB`；
   - 有限 cgroup 至少保留 `40 GiB`；
6. 任一条件失败，向 DMD pidfile 对应进程发送 `SIGTERM`，等待退出，确认
   normal 仍健康。

`4 GiB` 只是进入推理验收的最低门槛，不代表已经满足真实视频峰值。

### 7.5 受控 warmup 和真实请求验收

双服务 idle 闸门通过后，严格串行执行：

1. normal：`80/81 frames`、`1 step` smoke；
2. DMD：同样的 `1 step` smoke；
3. normal：生产分辨率和生产步数请求；
4. DMD：生产分辨率和 DMD 实际步数请求。

每次请求期间都记录另一闲置服务的占用以及系统总峰值。要求：

- 每卡峰值时仍至少保留 `2 GiB`；
- `MemAvailable` 和有限 cgroup headroom 峰值时仍至少保留 `24 GiB`；
- 无 CUDA OOM、host OOM、cgroup `oom_kill`；
- 两个模型输出文件均成功生成；
- 两个服务在任务结束后显存回到可重复的 idle 水位。

完成上述验证后才启动外部 `:30000` 网关。

## 8. 队列正确性验收

### 8.1 两请求同时提交

同时向 gateway 提交：

```text
task A -> videoedit-normal
task B -> videoedit-dmd
```

必须观察到：

```text
A: running
B: queued
```

只有 A 终态之后，B 才能从 queued 进入 running。

### 8.2 反向顺序

再测试：

```text
task C -> videoedit-dmd
task D -> videoedit-normal
```

要求完全相同，证明互斥不是某个后端的本地 semaphore 偶然实现的。

### 8.3 自动化断言

测试脚本至少断言：

- 任意时间 `running + dispatching <= 1`；
- normal 与 DMD 后端的运行时间区间没有交集；
- FIFO 顺序正确；
- task_id 到 backend 的映射正确；
- queued/running 取消正确；
- 后端返回 `code=2` 时不会错误释放队列；
- 后端 5xx、超时、网关重启后不会同时派发第二任务；
- gateway 不发送 callback；实际后端可按进度变化发送多次进度 callback，但每个
  进度状态和最终 callback 不因 gateway 轮询/重启而重复；
- 客户端仍使用原来的 `http://<host>:30000/v1/videos/repairs`。

### 8.4 稳定性测试

在 L20 上交替提交至少 20 个短任务：

```text
normal -> DMD -> normal -> DMD ...
```

记录：

- queue_wait_s；
- backend_submit_s；
- inference_s；
- 每卡 GPU used 峰值和 idle；
- host/cgroup 内存峰值；
- 两个服务是否出现长期内存增长；
- callback 成功率。

稳定性测试中不调用 `/update_weights_from_disk`。

## 9. 故障处理和回滚

### 9.1 启动失败

- normal 启动失败：停止 normal，gateway 不启动；
- DMD 启动失败或资源闸门失败：只停止 DMD，保留 normal；
- gateway 启动失败：两个后端继续只监听 localhost，不对外接收业务；
- 禁止在资源闸门失败后自动循环重启 DMD。

### 9.2 运行期后端异常

- 当前 running 任务失败并记录明确 reason；
- 网关暂停取下一任务，先确认故障后端没有残留 GPU 工作；
- 健康后端不自动接管另一套权重，因为本方案不做运行期权重切换；
- 人工恢复或自动健康恢复满足资源闸门后，再继续 queued 任务。

### 9.3 一键回退

回退顺序：

1. 停止 gateway，阻止新任务；
2. 等待或取消当前任务；
3. 停止 DMD；
4. 保留 normal 作为单服务；
5. 如仍需两权重能力，恢复单服务权重热切换流程。

SQLite 队列库和日志保留，不删除，便于核对未完成任务。

## 10. 关于跨服务共享非 Transformer 权重

两个独立进程只能自然共享 checkpoint 的 Linux 文件页缓存，不能共享已经实例化的
PyTorch text encoder、image encoder、VAE 或 pipeline 对象。offload 到 CPU
仍然意味着每个服务有自己的参数 tensor/buffer，不能把 `free -h` 中的 page cache
等同于模型实例共享。

真正共享 Transformer 之外的 module，需要改成“单进程、一个共享 pipeline、挂两
套 Transformer 实例、请求时切换 module 引用”的架构。这会影响 pipeline 状态、
TeaCache、offload hook、分布式 worker 同步、异常回滚和线程安全，修改和验证量
明显大于本方案。

所以本阶段不做跨进程 module 共享。先用资源闸门验证两完整服务是否能在 L20
安全常驻；只有 CPU 内存不通过且业务又必须零切换延迟时，才单独评估单进程双
Transformer 方案。

## 11. 实施顺序

1. 给 `GPUWorker` 增加 layerwise offload 后的 CUDA cache trim 和日志；
2. 增加对应单元测试，确认每个 rank 都执行；
3. 实现 SQLite 队列 store 和单 dispatcher gateway；
4. 实现 model 路由、状态代理、取消和重启恢复；
5. 实现 start/stop/status/resource probe；
6. 在当前 L20 上先启动 normal，记录清 cache 后 idle 与真实启动峰值；
7. 执行第二服务硬闸门，不通过时验证 `degraded_normal_only`；
8. 闸门通过后执行两模型真实视频峰值验收；
9. 最后进行 20 个交替任务稳定性测试。

## 12. 最终上线验收标准

只有以下条件全部满足才判定方案一可上线：

- 两个后端都健康且各自 checksum/输出证明权重正确；
- 服务启动期间没有 GPU OOM、host OOM 或 cgroup OOM；
- 双服务 idle 和真实视频峰值都满足第 7 节余量；
- 任意时刻最多一个视频处于 dispatching/running；
- 20 个交替任务无重叠、无丢失、无重复 callback；
- 网关重启可以恢复 queued 任务并安全核对 running 任务；
- 外部无法直接访问 `31100/32100`；
- 停止第二服务后 normal 可以独立继续工作；
- 全程没有请求时权重转换耗时。

若 GPU 启动闸门或 CPU/cgroup 闸门未通过，应把结论记录为“当前双 L20 资源不足以
安全常驻两个完整服务”，而不是通过降低安全余量强行上线。
