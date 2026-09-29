# L40S 单卡预取层数对比（2026-09-29）

## 结论

保留 GPU 0 单卡、`DIT_OFFLOAD_PREFETCH_SIZE=1`。2/4 层没有显示稳定的延迟收益，显存占用反而上升。本次未测试关闭 DiT offload、双卡 Ring 或 Ulysses。

| 实际预取层数 | 三次正式端到端耗时（秒） | 端到端中位数（秒） | 去噪中位数（秒） | GPU 使用量采样峰值（GiB） | PyTorch 分配峰值（GiB） | 容器内存采样峰值（GiB） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 80.671 / 79.118 / 78.206 | 79.118 | 45.573 | 18.640 | 12.589 | 61.064 |
| 2 | 85.784 / 79.367 / 79.275 | 79.367 | 46.611 | 20.144 | 13.341 | 60.368 |
| 4 | 78.987 / 79.947 / 78.750 | 78.987 | 45.972 | 23.151 | 14.845 | 60.559 |

4 层与 1 层的中位数相差约 0.13 秒，不足以证明提速；GPU 使用量采样峰值增加约 4.51 GiB。全部 12 次输出（含预热）的解码视频 framemd5 清单哈希完全一致。

## 测量方法与局限

- 同一历史请求的本地视频、mask 和参考图，输入文件 SHA-256 在各组一致。
- 单卡、40 步、seed 42、TeaCache 开启（阈值 0.3）、动态 CFG；各组只调整预取层数。
- 每组先预热一次，再串行运行三次；输出仅写入仓库内的本地基准目录，关闭云端上传及回调。
- 端到端时间来自网关 `completed_at - started_at`，包括本地处理及网关轮询；去噪时间来自后端 perf report。
- GPU 使用量与 cgroup 内存每秒采样，因此不保证捕获瞬时峰值。PyTorch 分配峰值来自后端请求级统计，两者计量范围不同。
- 这不是整机隔离测试。测试期间其他服务发生启停，4 层启动前还处理了一次共享运行目录冲突；小幅差异不能归因于预取。结论仅为本输入未观察到增大预取的稳定收益，不推广到其他分辨率、视频长度或缓存配置。
- 没有运行持续负载测试；相同输出帧不代表其他输入也必然完全一致。

原始报告、视频及本次复现脚本位于仓库 `.local/prefetch-benchmark/`，汇总为 `summary.json`（本地文件，未纳入 Git）。

## 当前部署

- 容器：`videoedit_l40s-v1`
- 物理 GPU：`0`；容器可见 GPU：`0`
- 端口：`5403 -> 30000`
- 配置：`scripts/videoedit_dual_service/config.l40s.0.prefetch.env`
- CPU 内存硬上限：300 GiB，容器额外 swap 为 0。
- normal-only；DMD 由资源检查拦截。

测试期间 `config.l40s.0.env` 被外部改为 GPU 7 服务使用的 `.local/videoedit-l40s.7` 运行目录。为了避免启动锁、PID 文件及队列冲突，GPU 0 服务改用独立的 `config.l40s.0.prefetch.env`，保留原 `.local/videoedit-l40s.0123` 队列及输入输出目录。不要让两个运行中的容器共用这套运行目录。

重建当前服务：

```bash
IMAGE_NAME=sglang-videoedit-dev-v2:l40s \
CONTAINER_NAME=videoedit_l40s-v1 \
HOST_PORT=5403 CONTAINER_PORT=30000 \
HOST_GPUS=0 CONTAINER_CUDA_VISIBLE_DEVICES=0 \
CONTAINER_MEMORY=0g CONTAINER_MEMORY_SWAP=0g \
DUAL_SERVICE_CONFIG_HOST="$PWD/scripts/videoedit_dual_service/config.l40s.0.prefetch.env" \
DUAL_SERVICE_CONFIG_CONTAINER=/sgl-workspace/sglang/scripts/videoedit_dual_service/config.l40s.0.prefetch.env \
RECREATE=1 bash scripts/start_videoedit_container.sh
```

`start.sh` 现在读取配置项 `DIT_OFFLOAD_PREFETCH_SIZE`；未设置时保持原默认值 `0`。当前实现中 `0` 和 `1` 都表示一层，整数 `2`、`4` 分别表示两层、四层。修改配置需重启后端才能生效。
