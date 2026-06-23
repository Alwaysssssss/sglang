# Vivid-VR Dual-Worker Caption Sidecar 实现与验收交接

日期：`2026-06-23 UTC`

## 1. 本轮结论

本轮已经把 Vivid-VR caption sidecar 升级为“同机双 worker 并行 + 串行回退”的正式实现，并完成了两层验收：

- dual-worker caption sidecar 独立 benchmark
- `sglang serve` 自动 caption bridge 接入验收

当前可以确认：

- `caption.txt` 对同一输入与单卡基线逐行逐字完全一致
- sidecar 在 `2 clip` 场景下真实命中了双 worker 并行
- sidecar 生成的 caption 能被 `sglang` 服务自动接入并正常进入正式推理
- 之前双 worker 跑完后 GPU 上残留约 `25 GB` 显存的问题已经修复；当前请求结束后显存会回落到 sidecar 常驻水平

## 2. 本轮边界

本轮继续遵守以下边界：

- 主推理仍然只使用 `/home/zhiheng/sglang/.venv`
- caption sidecar 仍然只使用 `/home/zhiheng/Vivid-VR/.venv`
- 主服务仍然只调用 `POST /v1/vividvr/captions`
- sidecar 文件契约仍然是一行一个 temporal clip caption
- `expected_caption_count` 继续表示 temporal clip 数
- 失败时仍然允许回退到当前已验收的串行 sidecar 路径

## 3. 本轮代码范围

### 3.1 新增文件

- `python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_runtime.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_runtime.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_benchmark.py`

### 3.2 修改文件

- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py`
- `python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py`
- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`
- `docs_xzh/run_vivid_benchmark.md`

## 4. 当前实现形态

当前 sidecar 已经是下面这条路径：

- 一个 controller HTTP 入口
- 两个常驻 worker 执行器
- `clip_index` 级 round-robin 分发
- 按 `clip_index` 顺序聚合写回
- worker 失败时自动回退到串行路径
- sidecar 额外返回并行元数据与 timing 信息

当前对外响应中新增的可选元数据包括：

- `mode`
- `worker_count`
- `fallback_used`
- `request_id`
- `total_clip_count`
- `assigned_clip_indices_by_worker`
- `timing`

主服务日志已经透传这些信息，便于在 `serve` 端直接确认 bridge 是否真正命中双 worker 并行路径。

## 5. 显存问题修复

本轮中途发现一个关键问题：

- dual-worker caption 请求完成后，两张 GPU 上仍残留约 `25 GB` 显存

根因是：

- worker 路径和串行 fallback 路径虽然都把 caption 模型 `to(cpu)` 了
- 但没有显式执行 CUDA allocator cache 清理

本轮已在 `run_vividvr_caption_sidecar.py` 中补了统一的 CUDA 释放 helper：

- `gc.collect()`
- `torch.cuda.empty_cache()`
- `torch.cuda.ipc_collect()`

修复后实测：

- sidecar benchmark 跑完后，GPU 显存从执行中的约 `24-25 GB / GPU` 回落到常驻水平
- 当前稳定观测值约为：
  - `GPU0 = 559 MiB`
  - `GPU1 = 575 MiB`

这说明现在没有再把整份 caption 模型残留在 GPU 上。

## 6. 独立 benchmark 验收

### 6.1 tmux 启动命令

dual-worker sidecar session：

```bash
tmux new-session -d -s vividvr_caption_sidecar_dual_0623_v6 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && export PYTHONPATH=python && /home/zhiheng/Vivid-VR/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31204 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_dual_20260623T_v6.log'
```

查看命令：

```bash
tmux attach -r -t vividvr_caption_sidecar_dual_0623_v6
```

独立 benchmark session：

```bash
tmux new-session -d -s vividvr_caption_bench_0623_v6 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/caption_sidecar_benchmark && export PYTHONPATH=python && /home/zhiheng/sglang/.venv/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar_benchmark.py --video-path /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4 --baseline-caption-path /home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt --sidecar-base-url http://127.0.0.1:31204 --manifest-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/manifest_20260623T_v6.json --output-caption-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/generated_20260623T_v6.txt --metrics-json-path /home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/metrics_20260623T_v6.json 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_bench_20260623T_v6.log'
```

查看命令：

```bash
tmux attach -r -t vividvr_caption_bench_0623_v6
```

### 6.2 benchmark 结果

当前最新 benchmark 已确认：

- `expected_caption_count = 2`
- `generated_caption_count = 2`
- `baseline_caption_count = 2`
- `captions_match = true`
- `first_mismatch_index = null`
- `sidecar_mode = "parallel"`
- `sidecar_worker_count = 2`
- `sidecar_fallback_used = false`
- `sidecar_total_clip_count = 2`
- `sidecar_assigned_clip_indices_by_worker = {"0": [0], "1": [1]}`
- `elapsed_seconds = 33.32912730053067`
- `sidecar_timing.total_seconds = 33.28468017280102`
- `sidecar_timing.read_seconds = 0.5065147653222084`
- `sidecar_timing.write_seconds = 0.00023739784955978394`

每个 worker 的 timing 如下：

- worker `0`
  - `total_seconds = 16.572346460074186`
  - `clip 0 inference_seconds = 6.035266324877739`
- worker `1`
  - `total_seconds = 16.931224197149277`
  - `clip 1 inference_seconds = 6.263736702501774`

对应产物：

- benchmark JSON：`/home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/metrics_20260623T_v6.json`
- 输出 caption：`/home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/generated_20260623T_v6.txt`
- manifest：`/home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/manifest_20260623T_v6.json`
- 日志：`/home/zhiheng/sglang/Vivid_Acceptance/logs/vividvr_caption_bench_20260623T_v6.log`

额外人工核对：

- `cmp -s` 退出码为 `0`
- `wc -l` 确认生成 caption 与基线都是 `2` 行
- benchmark JSON 明确记录两个 worker 都真正分到了 clip，不是“挂着 2 个 worker 实际只跑 1 个”

## 7. `serve` 自动 bridge 验收

### 7.1 tmux 启动命令

双卡 `serve` session：

```bash
tmux new-session -d -s vividvr_serve_dual_bridge_0623_v6 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR --attention-backend fa --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-torch-compile --dist-timeout 3600 --host 127.0.0.1 --port 31195 --master-port 30195 --scheduler-port 56195 --strict-ports --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31204 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_dual_bridge_20260623T_v6.log'
```

查看命令：

```bash
tmux attach -r -t vividvr_serve_dual_bridge_0623_v6
```

### 7.2 提交请求

这次验收使用了轻量 `POST /v1/videos/repairs` 请求，不手工传入 `caption_file_path`，强制走自动 bridge：

```bash
TASK_ID=vividvr-bridge-smoke-20260623T0800Z
curl --noproxy '*' -sS -X POST 'http://127.0.0.1:31195/v1/videos/repairs' \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "model": "VividVR",
  "task_id": "${TASK_ID}",
  "video_input_path": "/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4",
  "num_inference_steps": 1,
  "seed": 42,
  "num_temporal_process_frames": 121,
  "output_path": "/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/${TASK_ID}.mp4",
  "perf_dump_path": "/home/zhiheng/sglang/Vivid_Acceptance/indicator/${TASK_ID}_perf.json"
}
JSON
```

### 7.3 bridge 命中结果

当前已从主服务日志确认，这次请求确实走了双 worker 自动 bridge：

- `mode=parallel`
- `worker_count=2`
- `fallback_used=False`
- `total_clip_count=2`
- `bridge_elapsed_s=25.132`
- `worker_assignments={'0': [0], '1': [1]}`
- `sidecar_request_id=12f5e74fb9e74e689663b4c76d8abe5c`

这次 `serve` 内部 sidecar timing 为：

- `read_seconds = 2.6820039711892605`
- `write_seconds = 0.0002240687608718872`
- `total_seconds = 25.087545461952686`

worker 细分 timing：

- worker `0`
  - `total_seconds = 15.684556499123573`
  - `clip 0 inference_seconds = 4.897501360625029`
- worker `1`
  - `total_seconds = 15.490313481539488`
  - `clip 1 inference_seconds = 4.854467075318098`

当前已确认自动生成的 sidecar 文件：

- manifest：`/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/vividvr-bridge-smoke-20260623T0800Z.manifest.json`
- caption：`/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/vividvr-bridge-smoke-20260623T0800Z.txt`

当前已确认：

- `caption.txt` 行数为 `2`
- 与 `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt` 的 `cmp -s` 退出码为 `0`

## 8. 本轮实际输出

虽然这次 `serve` 验收最初只要求“caption 能交给主服务并正常进入推理”，但这条 smoke 请求实际上已经完整跑完，额外得到了一份完整产物：

- 输出视频：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/vividvr-bridge-smoke-20260623T0800Z_0.mp4`
- perf JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-bridge-smoke-20260623T0800Z_perf.json`

`ffprobe` 已确认：

- `codec_name = h264`
- `width = 960`
- `height = 720`
- `nb_frames = 130`
- `avg_frame_rate = 25/1`
- `duration = 5.200000`
- `size = 10869021`

perf JSON 已确认主要阶段耗时：

- `total_duration_ms = 249538.1236039102`
- `VividVRLongClipPreparationStage = 63174.11056905985 ms`
- `VividVRMultiClipDenoisingStage = 84622.80783429742 ms`
- `VividVRMultiClipDecodeTrimStage = 98232.82378166914 ms`

这说明 sidecar 生成的 caption 不只是“被主服务接收”，而是已经真实支撑了一次完整的正式推理。

## 9. 当前验收结论

按当前任务的最新范围，本轮验收已经完成，而且证据强于最低要求：

- `caption` 独立 benchmark 已确认双 worker 真实并行，且结果与单卡基线逐字一致
- `serve` 自动 bridge 已确认 sidecar 结果能正常接入主服务
- 实际 smoke 请求最终还完整跑完并落了视频与 perf JSON
- 显存残留问题已经修复，请求结束后 GPU 会回到 sidecar 常驻水平

## 10. 已知限制与下一步

- 当前双卡加速来自不同 clip 的任务级并行，不是同一个 clip 的 TP / 模型并行。
- 当前 benchmark 主要覆盖 `130f -> 2 clip` 场景；如果后续要覆盖 `>2 clip` 的正式性能口径，需要补更长样本。
- 当前 sidecar 仍依赖原版 caption 环境；后续如原版环境升级，需要先重跑独立 benchmark，再重跑 `serve` 验收。
- 主推理仍然是主要耗时项；caption 双 worker 只能缩短 bridge 阶段，不会改变正式双卡主链是重耗时项这一事实。

## 11. 一句话交接

现在的 Vivid-VR caption bridge 已经是可用的 dual-worker sidecar：`2 clip` 输入会被分发到 `GPU0/GPU1` 并行生成，`caption.txt` 与单卡基线逐字一致，自动 bridge 生成的 sidecar 文本已经被 `sglang serve` 成功消费并支撑一次完整正式推理；最新双卡 caption benchmark 用时约 `33.33s`，`serve` 内真实 bridge 用时约 `25.13s`，且显存残留问题已修复。
