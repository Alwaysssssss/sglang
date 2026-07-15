# VividVR 加速实验自动运行

## 固定行为

| 项目 | 行为 |
| --- | --- |
| 实验顺序 | `R0`—`R9`、`R99`、`R100`，严格串行 |
| 服务生命周期 | Moto S3、callback receiver、固定 caption mock 每批启动一次；主推理服务每个可执行方案重启一次 |
| Warmup | 仅启用 `torch.compile` 的方案执行一次完整 warmup；eager 方案直接执行 formal |
| Formal | 每个可执行方案执行一次完整正式请求，并立即原子写入 JSON |
| 不支持方案 | `R7`、`R8`、`R9` 不启动服务，写入带明确原因的 `unsupported` JSON |
| 运行环境 | `/home/zhiheng/sglang/.venv`，`PYTHONPATH=python` |
| 推理承载 | 公开命令自动启动独立 tmux session |
| 失败策略 | 请求失败记录 JSON 后继续下一方案；owned service 清理失败则终止批次 |
| 恢复策略 | `--resume --batch-id ...` 只跳过 fingerprint 一致且 formal 成功的方案 |

## 命令

| 用途 | 命令 |
| --- | --- |
| 只检查路径并打印 12 个方案 | `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py dry-run` |
| 顺序运行全部方案 | `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py run-all` |
| 只运行一个方案 | `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py run-one --scheme R3` |
| 恢复指定批次 | `PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_acceleration_benchmark.py run-all --batch-id <BATCH_ID> --resume` |
| 只读查看 | 使用启动结果中的 `tmux attach -r -t <SESSION>` |

## 默认输入

| 项目 | 路径 |
| --- | --- |
| 输入视频 | `/home/zhiheng/input/test_video_long_960x720_130f.mp4` |
| 固定 caption | `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/quad-test-video-long-960x720-130f-run2-20260708T060202Z.txt` |
| Reference | `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4` |
| 模型 | `/home/zhiheng/ckpts/CogVideoX1.5-5B` |
| VividVR checkpoint | `/home/zhiheng/ckpts/Vivid-VR` |
| 批次输出 | `/home/zhiheng/sglang/Vivid_Acceptance/acceleration_benchmark/<BATCH_ID>` |

## 输出结构

| 路径 | 内容 |
| --- | --- |
| `batch_summary.json` | 批次状态及每个方案的记录路径 |
| `records/<SCHEME>_warmup.json` | compile 方案 warmup 记录 |
| `records/<SCHEME>_formal.json` | 可执行方案 formal 记录 |
| `records/<SCHEME>_unsupported.json` | 不支持方案及原因 |
| `requests/<TASK_ID>/perf.json` | pipeline perf 原始数据 |
| `requests/<TASK_ID>/downloaded.mp4` | 下载后的生成视频 |
| `requests/<TASK_ID>/compare.json` | formal 质量对比 |
| `logs/` | 批次、主服务、Moto、callback、caption mock 日志 |

## JSON 覆盖字段

| 分区 | 内容 |
| --- | --- |
| `inputs` | 输入、caption、reference、帧数、步数、seed、guidance、upscale、dtype |
| `runtime` | requested/effective backend、compile、并行拓扑、fusion、cache、量化 |
| `timings` | 总耗时、模型推理、八个 VividVR stage、denoise、step 统计、通信不可观测原因 |
| `gpu_memory` | 逐卡峰值、最大单卡峰值、采样后端与采样错误 |
| `quality` | `pass_compare`、SSIM、failed frame ratio；warmup 明确记为未评估 |
| `derived` | 相对 R0 累计加速、模块增量、GPU·秒、资源效率 |
| `artifacts` | perf、视频、compare、服务日志路径 |
| `reproducibility` | 命令、环境、请求、路径及 config fingerprint |

