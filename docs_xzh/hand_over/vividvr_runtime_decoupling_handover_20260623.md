# Vivid-VR Runtime Decoupling 交接

日期：`2026-06-23 UTC`

## 1. 本轮目标

本轮工作的目标是继续推进 `Vivid-VR` caption 运行时与原版 `/home/zhiheng/Vivid-VR` Python 代码解耦，让 `sglang` 内部的 caption sidecar 不再依赖原版仓库的运行时代码，同时保持当前 `Phase C / D / E` 主推理基线不被破坏。

这条线当前的边界是：

- `sglang` 运行时不再 import 原版 `/home/zhiheng/Vivid-VR` 的 Python 模块。
- checkpoint、输入视频、`prompt.txt`、baseline caption 文件和 reference 视频等静态资源，仍允许继续复用原版仓库路径。
- 主推理环境仍固定为 `/home/zhiheng/sglang/.venv`。
- caption sidecar 单独使用独立环境，避免为了 caption 去改坏主推理依赖。

## 2. 本轮已确认结论

当前已经确认的结论如下：

1. caption 代码已经迁移到：
   - `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_backend/`
2. sidecar 运行时已经不再 import：
   - `/home/zhiheng/Vivid-VR`
3. caption sidecar 独立环境已经固定为：
   - `/home/zhiheng/sglang/.venv-vividvr-caption`
4. caption env 安装脚本已经固定为：
   - `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`
5. sidecar 在 caption env 中的视频读取卡住问题已经通过 `decord==0.6.0` 修复。
6. 当前 `_load_video_tensor(...)` 的读取策略已经改为：
   - 优先走 `decord`
   - `decord` 失败时 fallback 到 `cv2`
7. 独立 caption sidecar benchmark 已通过。
8. 定向单测已通过。
9. `serve` 自动 bridge 端到端验收已补齐并通过。

## 3. 本轮代码与运行时改动

### 3.1 Caption backend 已迁入 `sglang`

当前 caption sidecar backend 已迁入：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/vividvr/caption_sidecar_backend/`

这一步的意义是：

- caption sidecar 后续可以直接从 `sglang.multimodal_gen.runtime.vividvr.caption_sidecar_backend` 加载本地 backend。
- 运行时 import 路径已经从“依赖原版仓库 Python 模块”切到“依赖 `sglang` 仓库内 vendor 后的本地实现”。
- 后续 bridge、sidecar HTTP 服务和 `serve` 自动 caption 路径可以建立在这套本地 backend 之上，而不是继续注入原版 `sys.path`。

当前约束仍然成立：

- 不要创建与现有 `python/sglang/multimodal_gen/runtime/vividvr/captioning.py` 冲突的同名目录。
- 当前新 backend 包路径统一使用：
  - `sglang.multimodal_gen.runtime.vividvr.caption_sidecar_backend`

### 3.2 Caption env 已独立

当前 caption sidecar 的独立环境固定为：

- `/home/zhiheng/sglang/.venv-vividvr-caption`

创建和维护入口固定为：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/setup_vividvr_caption_env.sh`

已确认该脚本承担的职责包括：

- 创建 `.venv-vividvr-caption`
- 安装 `python/requirements-vividvr-caption.txt`
- 把仓库 `/home/zhiheng/sglang/python` 写入 sidecar env 的 `.pth`
- 在无 `PYTHONPATH` 条件下做 sidecar 导入自检

这条独立 env 路径的意义是：

- caption sidecar 依赖与主推理环境隔离
- 不需要为了 caption 回退或替换 `/home/zhiheng/sglang/.venv` 中的核心依赖
- 后续 `serve` 自动 bridge 可以稳定把 caption 运行时固定在单独环境中

### 3.3 视频读取路径已收口到 `decord first`

当前 sidecar 的视频读取逻辑已经明确：

- `_load_video_tensor(...)` 优先走 `decord`
- 如果 `decord` 解码失败，再 fallback 到 `cv2`

相关位置：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`

已确认的背景是：

- sidecar 在 caption env 中曾出现视频读取卡住问题
- 通过 `decord==0.6.0` 后，该问题已得到修复

当前仓库内可见的相关依赖和说明包括：

- `/home/zhiheng/sglang/python/requirements-vividvr-caption.txt`
- `/home/zhiheng/sglang/docs_xzh/run_vivid_benchmark.md`
- `/home/zhiheng/sglang/docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

## 4. 已验证结果

### 4.1 独立 caption sidecar benchmark

当前已通过的 benchmark 产物位于：

- `/home/zhiheng/sglang/Vivid_Acceptance/caption_sidecar_benchmark/metrics_decouple_v3_20260623.json`

本轮已确认的核心结果如下：

- `captions_match = true`
- `elapsed_seconds = 47.08574205636978`
- `sidecar_mode = parallel`
- `sidecar_worker_count = 2`
- `fallback_used = false`
- `assigned_clip_indices_by_worker = {0:[0],1:[1]}`

补充说明：

- 当前指标文件中的字段前缀存在 `sidecar_*` 命名，例如 `sidecar_fallback_used`、`sidecar_assigned_clip_indices_by_worker`。
- 如果后续有脚本或文档需要引用这些字段，默认以实际 JSON 字段名为准，不要把交接文档里的简写口径误当成固定 schema。

当前这组 benchmark 结果说明：

- decoupled sidecar 已能稳定跑通独立 caption 生成
- 当前 `130f` / 2 clip 基准下，生成的 sidecar caption 与 baseline caption 逐字一致
- parallel sidecar 双 worker 路径已经被实际 benchmark 覆盖
- 本次 benchmark 中没有走 fallback 路径

### 4.2 定向单测

当前已确认的定向单测执行命令为：

```bash
/home/zhiheng/sglang/.venv/bin/pytest \
  python/sglang/multimodal_gen/test/unit/test_vividvr_captioning_loader.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_sidecar_tool.py \
  python/sglang/multimodal_gen/test/unit/test_vividvr_caption_bridge.py \
  -v
```

本轮结果为：

- `24 passed, 8 warnings in 11.55s`

这组单测覆盖的关键点包括：

- 新包可导入
- tokenizer 加载时明确使用 `trust_remote_code=False`
- 模型加载走本地 `CogVLMVideoForCausalLM.from_pretrained(...)`
- `sys.path` 中没有注入 `/home/zhiheng/Vivid-VR`
- sidecar loader 会优先走 `decord`，失败时再 fallback 到 `cv2`
- bridge 请求会解析并保留并行 worker 元数据

### 4.3 Serve 自动 bridge 端到端验收

本轮 `serve` 自动 bridge 验收已经完成，关键会话和命令如下。

sidecar 会话：

- `tmux session`: `vividvr_caption_sidecar_decouple_v3`
- 环境：`/home/zhiheng/sglang/.venv-vividvr-caption`
- 启动命令：

```bash
tmux new-session -d -s vividvr_caption_sidecar_decouple_v3 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs && /home/zhiheng/sglang/.venv-vividvr-caption/bin/python python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py --host 127.0.0.1 --port 31206 --parallel-workers 2 --worker-devices cuda:0,cuda:1 2>&1 | tee Vivid_Acceptance/logs/vividvr_caption_sidecar_decouple_v3_$(date -u +%Y%m%dT%H%M%SZ).log'
```

`serve` 会话：

- `tmux session`: `vividvr_serve_decouple_v1`
- 环境：`/home/zhiheng/sglang/.venv`
- 启动命令：

```bash
tmux new-session -d -s vividvr_serve_decouple_v1 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/captions/service_sidecars && export PYTHONUNBUFFERED=1 && export PYTHONPATH=python && export SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global && export SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1 && CUDA_VISIBLE_DEVICES=0,1 /home/zhiheng/sglang/.venv/bin/sglang serve --model-path /home/zhiheng/Vivid-VR/ckpts/CogVideoX1.5-5B --model-id VividVR --pipeline-class-name CogVideoXVividVRControlNetPipeline --component-paths.vividvr /home/zhiheng/Vivid-VR/ckpts/Vivid-VR --attention-backend fa --num-gpus 2 --tp-size 1 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-torch-compile --dist-timeout 3600 --host 127.0.0.1 --port 31196 --master-port 30196 --scheduler-port 56196 --strict-ports --output-path /home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark --prompt-file-path /home/zhiheng/Vivid-VR/input/720p/prompt.txt --vividvr-caption-bridge --vividvr-caption-sidecar-url http://127.0.0.1:31206 --vividvr-caption-work-dir /home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars --vividvr-caption-sidecar-timeout 1800 2>&1 | tee Vivid_Acceptance/logs/vividvr_serve_decouple_v1_$(date -u +%Y%m%dT%H%M%SZ).log'
```

smoke 请求会话：

- `tmux session`: `vividvr_serve_smoke_decouple_v2`
- 请求命令：

```bash
tmux new-session -d -s vividvr_serve_smoke_decouple_v2 \
  'cd /home/zhiheng/sglang && mkdir -p Vivid_Acceptance/logs Vivid_Acceptance/result_videos/service_benchmark Vivid_Acceptance/indicator && TASK_ID=vividvr-bridge-decouple-smoke-20260623T1447Z && NO_PROXY=* curl -sS -X POST http://127.0.0.1:31196/v1/videos/repairs -H "Content-Type: application/json" --data-binary @- <<JSON 2>&1 | tee Vivid_Acceptance/logs/${TASK_ID}.log
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
JSON'
```

本轮已确认的验收结果如下：

- `serve` 端成功自动触发 caption bridge，没有手工传 `caption_file_path`
- sidecar 自动生成的 caption 文件路径：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/vividvr-bridge-decouple-smoke-20260623T1447Z.txt`
- sidecar 自动生成的 manifest 路径：
  - `/home/zhiheng/sglang/Vivid_Acceptance/captions/service_sidecars/vividvr-bridge-decouple-smoke-20260623T1447Z.manifest.json`
- 该 caption 文件与 baseline `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt` 做 `diff -u` 无差异，逐字一致
- `serve` 日志明确记录：
  - `mode=parallel`
  - `worker_count=2`
  - `fallback_used=False`
  - `bridge_elapsed_s=27.604`
  - `worker_assignments={'0': [0], '1': [1]}`
- sidecar 日志中可见新的 `POST /v1/vividvr/captions` `200 OK`
- 最终视频产物已落盘：
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/vividvr-bridge-decouple-smoke-20260623T1447Z_0.mp4`
- 最终 perf JSON 已落盘：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-bridge-decouple-smoke-20260623T1447Z_perf.json`

这次 `serve` 验收还确认了一个接口行为细节：

- `POST /v1/videos/repairs` 当前会先返回一个 `status=queued` 的对象
- 真正的生成继续在后端执行
- 验收时需要继续检查日志和最终落盘文件，不能只看 HTTP `200`

本轮可直接引用的关键耗时有两类：

- caption bridge 阶段：
  - `bridge_elapsed_s = 27.604`
- 主推理 perf JSON：
  - `total_duration_ms = 251785.6687232852`
  - `VividVRLongClipPreparationStage = 68983.44086110592 ms`
  - `VividVRMultiClipDenoisingStage = 82810.75962260365 ms`
  - `VividVRMultiClipDecodeTrimStage = 97891.78502559662 ms`
  - `VividVRTemporalStitchPostprocessStage = 439.8234635591507 ms`

## 5. 当前仍未完成的部分

### 5.1 当前剩余限制

本轮验收完成后，当前剩余限制主要有：

- 当前正式覆盖的是 `130f -> 2 clip` 的场景，更长视频仍需要补 `>2 clip` 自动 bridge 样本
- caption sidecar 的代码与环境虽然已经迁入 `sglang`，但 checkpoint、输入视频、`prompt.txt`、baseline caption 等静态资源仍暂时来自原版仓库
- `POST /v1/videos/repairs` 当前是“先返回 queued，再在后端继续生成”的模式，运维侧需要知道不能只看首个 HTTP 返回
- 当前 smoke 用的是 `num_inference_steps=1` 轻量请求，它证明的是运行时解耦和自动 bridge 打通，不是新的质量基准

## 6. 对当前状态的判断

截至 `2026-06-23 UTC`，可以把当前状态判断为：

- caption sidecar runtime decoupling 的“本地 backend + 独立 env + 独立 benchmark + loader/bridge 单测 + serve 自动 bridge 验收”这一层已经落地
- sidecar 已不再依赖原版 `/home/zhiheng/Vivid-VR` 的 Python 运行时代码
- 当前最关键的未收尾项已经从“运行时代码迁移”转为“更长视频样本覆盖”和“静态资源是否继续收口”

换句话说，本轮已经把风险最大的“运行时代码耦合”拆掉，并且验证了 decoupled sidecar 可以真实支撑 `serve` 自动 bridge 主链。

## 7. 下一轮建议执行顺序

建议下一轮按下面顺序继续：

1. 先补 `>2 clip` 长视频样本，确认自动 bridge 在更多 temporal split 形态下仍保持行数、顺序和 baseline 一致性。
2. 继续把与原版仓库耦合的静态资源入口收口成更清晰的可配置参数，但暂时不要把任务膨胀成 checkpoint 资产重组。
3. 如需新的正式 benchmark，继续保持当前 `Phase E` 默认配置口径，不要让 caption 解耦改动回归主推理基线。
4. 如果后续 sidecar 依赖再次调整，优先复跑独立 benchmark、24 条单测和一条 `serve` 自动 bridge smoke。

## 8. 一句话交接

当前 `Vivid-VR` caption runtime decoupling 已完成核心拆耦并通过验收：caption backend 已迁入 `sglang`、sidecar 已不再 import 原版 `Vivid-VR` Python 代码、独立 caption env 与 `decord first` 视频读取路径已稳定，且已经通过独立 benchmark、24 条定向单测以及一条真实 `serve` 自动 bridge 端到端 smoke，当前后续重点转为 `>2 clip` 覆盖和静态资源入口收口。
