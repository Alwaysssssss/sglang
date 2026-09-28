# VideoEdit 本机服务重启手册（L40S / 5402，双卡 → 四卡）

> 编写日期：2026-09-24。仓库：`/home/zhouhao6/VideoEdit/sglang`，分支：`cos_l40s`。
>
> 场景：宿主源码自容器启动后已有改动，需要重启服务；同时把 GPU 从 `4,5` 换成 `4,5,6,7`。
>
> 配套文档：[videoedit_service_quickstart.v2.l40.md](./videoedit_service_quickstart.v2.l40.md)（部署与验收边界）、
> [videoedit_service_quickstart.v2.md](./videoedit_service_quickstart.v2.md)（算法与请求参数规则）。

**本手册的重启命令尚未执行过，四卡配置也从未运行过。** 下述“核对结果”是 2026-09-24 的静态检查记录，
不是运行结论；两卡时代的验收（§0、§0.1 的 105 帧 / 40 步 / THP 检查）不能外推到四卡。

## 1. 结论：为什么不能只 `docker restart`

三个约束同时成立：

1. **卡数是容器级参数。** `CONTAINER_CUDA_VISIBLE_DEVICES` 通过 `docker run -e` 传入，
   属于容器环境变量。按 quickstart §8，改 GPU / 挂载 / 端口 / 环境变量必须**重建**容器，
   `docker restart` 只会复用旧参数。
2. **只改卡列表不够。** 并行度硬编码在 `scripts/videoedit_dual_service/start.sh` 里
   （不在 `config.*.env` 中）。quickstart §2.1 已注明：“不能只改 GPU 列表就切换到单卡运行”，
   四卡同理——`--num-gpus 2` 配 4 张可见卡不会自动扩到四卡。
3. **代码改动确实是重启范畴。** 宿主仓库以同路径挂载进容器，线上跑的是宿主源码；
   纯 Python 改动重启即生效，只有依赖声明变化才要重建镜像（quickstart §3.2）。

因此正确的动作是：**改 3 处配置 → 停容器 → 删容器 → 用四卡重建启动**。

## 2. 2026-09-24 核对结果

| 项目 | 核对结果 |
| --- | --- |
| 队列 | `dispatching=0 running=0 queued=0`；`completed=8 failed=1 cancelled=0`，可安全重启 |
| GPU 4 / 5 | 被 `videoedit_l40s` 占用，各 11634 MiB / 46068 MiB |
| GPU 6 / 7 | 空闲，各 0 MiB used / 45460 MiB free |
| GPU 0–3 | 空闲（本手册不使用） |
| 拓扑 | 8 卡两两间均为 `PHB`（PCIe，无 NVLink），全部 `CPU Affinity 0-127`、NUMA 0 → 4,5,6,7 的分组不比其他四卡组合差 |
| 宿主内存 | 1.0 Ti，`available` 584 Gi；容器当前 233.8 GiB |
| 代码改动 | 容器启动（2026-09-22 02:54 UTC，镜像 `sha256:da597cfcbcfa...`）之后改动集中在 `python/sglang/multimodal_gen/runtime/videoedit/`（`stream_io.py`、`streaming.py`、`composite.py`、`mask_stabilize.py`、`postprocess.py` 等）与 `start.sh` 的 THP 分支 |
| 依赖 | `python/pyproject.toml`、`docker/` 无改动 → **不需要重建镜像** |

并行度参数按 `python/sglang/multimodal_gen/runtime/server_args.py:1196-1240` 的校验推导：

- `num_gpus(4) % sp_degree(4) == 0` 且 `sp_degree <= num_gpus` ✔
- `sp_degree(4) == ring_degree(1) * ulysses_degree(4)` ✔
- Ulysses all-to-all 要求 `h_global % world_size == 0`（`runtime/layers/usp.py:88`）：
  transformer `num_attention_heads = 40`，`40 % 4 == 0` ✔
- `hsdp_shard_dim` 默认为 `num_gpus`，`4 % 4 == 0` ✔

**这是静态推导，不等于已验证可行。**

## 3. 需要改动的 3 处

| 文件 | 配置项 | 原值 | 新值 |
| --- | --- | --- | --- |
| `scripts/videoedit_dual_service/config.l40s.env` | `CUDA_DEVICES` | `0,1` | `0,1,2,3` |
| `scripts/videoedit_dual_service/start.sh` | `--num-gpus` / `--sp-degree` / `--ulysses-degree` | `2` / `2` / `2` | `4` / `4` / `4` |
| `.local/videoedit-l40s/start-container.sh` | `HOST_GPUS`、`CONTAINER_CUDA_VISIBLE_DEVICES` | `4,5`、未设置（默认 `0,1`） | `4,5,6,7`、`0,1,2,3` |

`--ring-degree` 保持 `1`。注意 `config.l40s.env` 里写的是**容器内**卡号（0-3），
`HOST_GPUS` 写的是**宿主**卡号（4-7）：`--gpus "device=4,5,6,7"` 会按给定顺序在容器内重编号为 0-3。

两个后端（normal / DMD）共用同一份 `CUDA_DEVICES`，也就是**四张卡同时被两个后端各自全量使用**，
与原先两卡时“两个后端共用两张卡”的拓扑一致。Gateway 串行调度请求，因此不存在把 4 张卡拆成
normal 2 张 + DMD 2 张的收益。

## 4. 执行命令

命令在具备 Docker 权限的**宿主 Bash 终端**执行。

### 4.1 改配置

> **2026-09-24 已执行。** 三处改动均已落到文件（`git status` 可见 `config.l40s.env`、`start.sh` 为 modified）。
> 本节命令保留作为回滚参考，**不要重复执行**：`sed` 匹配不到会静默跳过，Python 片段的 `assert` 会报错退出。
> 已确认通用模板 `config.env` / `config.env.example` 未被改动，仍为 `CUDA_DEVICES=0,1`。

```bash
cd /home/zhouhao6/VideoEdit/sglang

# 4.1a 容器内后端用的卡号
sed -i 's/^CUDA_DEVICES=0,1$/CUDA_DEVICES=0,1,2,3/' scripts/videoedit_dual_service/config.l40s.env

# 4.1b 并行度（硬编码在 start.sh，不在 config 里）
python3 - <<'PY'
from pathlib import Path
p = Path('scripts/videoedit_dual_service/start.sh')
s = p.read_text()
old = """      --num-gpus 2 \\
      --sp-degree 2 \\
      --ulysses-degree 2 \\
      --ring-degree 1 \\"""
new = """      --num-gpus 4 \\
      --sp-degree 4 \\
      --ulysses-degree 4 \\
      --ring-degree 1 \\"""
assert s.count(old) == 1, f'unexpected match count: {s.count(old)}'
p.write_text(s.replace(old, new))
print('start.sh: num-gpus/sp-degree/ulysses-degree -> 4')
PY

# 4.1c 宿主卡分配 4,5,6,7 -> 容器内 0,1,2,3
sed -i 's/^export HOST_GPUS=4,5 HOST_PORT=5402/export HOST_GPUS=4,5,6,7 CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3 HOST_PORT=5402/' \
  .local/videoedit-l40s/start-container.sh
```

### 4.2 核对改动

```bash
grep -n 'CUDA_DEVICES=' scripts/videoedit_dual_service/config.l40s.env
grep -nE '\-\-num-gpus|\-\-sp-degree|\-\-ulysses-degree|\-\-ring-degree' scripts/videoedit_dual_service/start.sh
grep -n 'HOST_GPUS' .local/videoedit-l40s/start-container.sh
```

期望分别看到 `CUDA_DEVICES=0,1,2,3`、四个 `--xxx 4/4/4/1`、以及
`export HOST_GPUS=4,5,6,7 CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3 HOST_PORT=5402 ...`。

### 4.3 确认队列空闲

```bash
curl --noproxy '*' -fsS 'http://127.0.0.1:5402/admin/queue?limit=5' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["counts"])'
```

要求 `queued=0 running=0 dispatching=0`。重启会中断在途任务，且**不能从内存结果断点续跑**；
有在途任务时先等它跑完。不要删除 `queue.sqlite3` 来绕过问题。

### 4.4 停容器并重建

```bash
docker stop -t 120 videoedit_l40s
docker rm videoedit_l40s
bash .local/videoedit-l40s/start-container.sh
```

- `docker stop`（而非 `rm -f`）是刻意的：容器 entrypoint 的 trap 会调用 `stop.sh`，
  顺带清掉 `.local/videoedit-l40s/dual/pids/*.pid` 并 SIGTERM 两个后端。
  `-t 120` 给 `stop.sh` 的每个后端 60 秒退出窗口留足时间。
- `docker rm` 不能省：`.local/videoedit-l40s/start-container.sh` 有同名容器保护
  （`docker ps -aq` 会匹配到已停止的容器），不删就会直接退出。
- 不要用 `docker restart` 代替，它不会重新读取 `-e CUDA_VISIBLE_DEVICES`。

### 4.5 启动观察

```bash
docker logs -f videoedit_l40s
```

启动顺序 normal → DMD → Gateway。`STARTUP_TIMEOUT=900` 是**每个后端**的启动监测超时，
不是整个服务的总超时。quickstart §0 记录过 normal 首次加载超过 900 秒、容器自动重启一次的情况；
加载慢时先看日志，不要反复重建。

## 5. 验收

```bash
curl --noproxy '*' -fsS --max-time 10 http://127.0.0.1:5402/health | python3 -m json.tool
docker exec videoedit_l40s bash scripts/videoedit_dual_service/status.sh
docker exec videoedit_l40s nvidia-smi
```

| `status` | 含义 |
| --- | --- |
| `ok` | normal 和 DMD 均健康，可进入推理验收 |
| `degraded_normal_only` | 仅 normal 健康，四卡部署未完成 |
| `unavailable` | normal 不健康，不可验收 |

三种状态都返回 HTTP 200，不能只看 curl 退出码。`nvidia-smi` 应看到 4 张卡都有占用。
若 quickstart §0.1 的 THP 修复仍需保持，另需确认两个后端的 4 个 scheduler 实测 `THP_enabled: 0`
（本机 `config.l40s.env` 已设 `VIDEOEDIT_DISABLE_THP=true`）。

四卡是否被真正吃满，可看后端日志：

```bash
docker exec videoedit_l40s tail -n 100 .local/videoedit-l40s/dual/logs/normal.log
docker exec videoedit_l40s tail -n 100 .local/videoedit-l40s/dual/logs/dmd.log
docker exec videoedit_l40s tail -n 100 .local/videoedit-l40s/dual/logs/gateway.log
```

## 6. 已知边界与回滚

### 6.1 边界

- **四卡未经验收。** 健康检查通过只表示两个后端起来了，不证明推理正确、不证明提速、
  不证明 105 帧 / 40 步 / 并发队列已经跑通。要下结论必须按 quickstart §7 重新提交请求并检查输出。
- **`--ulysses-degree 4` 是推导值，不是实测优选值。** 若四卡启动或推理异常，
  下一个候选是 `--sp-degree 4 --ulysses-degree 2 --ring-degree 2`（`docs_tyx/optimizer.md` §9
  出现过 ring/ulysses 的并行对照写法），但一次只改并行参数，不要同时动 attention backend、
  compile、offload 或 cache。
- **冷启动耗时不可预测。** quickstart §0 记录过权重顺序读耗时 9.28 秒与 142.27 秒的差异，
  根因未证实，本手册未做任何缓解。
- **`docker stop` 若超时被 SIGKILL**，entrypoint 的 trap 不会执行，pid 文件会残留。
  该目录 `.local/videoedit-l40s/dual/pids/` 属主是 root、权限 700，宿主用户读不到；
  只有新进程 PID 恰好撞上旧号时才需要清：
  ```bash
  docker run --rm -v /home/zhouhao6/VideoEdit/sglang/.local/videoedit-l40s/dual:/d alpine rm -f /d/pids/*.pid
  ```

### 6.2 回滚到两卡（4,5）

把 §4.1 的三处改回原值，然后重跑 §4.4 的停 / 删 / 启：

```bash
cd /home/zhouhao6/VideoEdit/sglang
sed -i 's/^CUDA_DEVICES=0,1,2,3$/CUDA_DEVICES=0,1/' scripts/videoedit_dual_service/config.l40s.env
python3 - <<'PY'
from pathlib import Path
p = Path('scripts/videoedit_dual_service/start.sh')
s = p.read_text()
old = """      --num-gpus 4 \\
      --sp-degree 4 \\
      --ulysses-degree 4 \\
      --ring-degree 1 \\"""
new = """      --num-gpus 2 \\
      --sp-degree 2 \\
      --ulysses-degree 2 \\
      --ring-degree 1 \\"""
assert s.count(old) == 1, f'unexpected match count: {s.count(old)}'
p.write_text(s.replace(old, new))
print('start.sh: 回滚到两卡')
PY
sed -i 's/^export HOST_GPUS=4,5,6,7 CONTAINER_CUDA_VISIBLE_DEVICES=0,1,2,3 HOST_PORT=5402/export HOST_GPUS=4,5 HOST_PORT=5402/' \
  .local/videoedit-l40s/start-container.sh
```

### 6.3 顺带清理

`docker ps -a` 中存在残留容器 `blissful_shtern`（镜像 `sglang-videoedit-src:l40s`，状态 `Created`，
不占卡）。确认无用后可 `docker rm blissful_shtern`。历史容器 `videoedit_reset`（Exited）按 quickstart
§0 的说明保留，未删除。

### 6.4 运行数据位置

队列、日志、pid、启停锁、输出、请求都在宿主 `.local/videoedit-l40s/`（部分属主为 root），
不随容器删除。该目录是运行数据，不应整体提交到 Git。
