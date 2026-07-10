# VividVR CFG parallel v2 等价实现与验收交接

日期：2026-07-09

## 背景

本轮目标是在不改变 VividVR v2 语义的前提下，引入 CFG parallel 加速：

- 两卡 `CFG-only`：`CFG=2, SP=1`
- 四卡组合：`CFG=2 x SP=2`
- 保留既有纯 SP 路径，不删除、不替换当前 Phase E 默认双卡 `SP=2 + eager_global + fa + compile` 配置

并行组合由服务启动参数决定：

```bash
--vividvr-parallel-mode auto|single|sp|cfg|cfg_sp
```

该参数只做意图表达和校验；底层拓扑仍由 `--enable-cfg-parallel`、`--sp-degree`、`--ulysses-degree`、`--ring-degree`、`--num-gpus` 初始化。请求级不能动态切换并行拓扑。

## 已完成实现

- `python/sglang/multimodal_gen/runtime/server_args.py`
  - 新增 `vividvr_parallel_mode: str = "auto"`
  - 新增 CLI 参数 `--vividvr-parallel-mode`

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
  - 新增 `_resolve_vividvr_parallel_mode`
  - `VividVRInputValidationStage` 写入 `vividvr_debug["vividvr_parallel_mode"]`
  - `VividVRDenoisingStage.parallelism_type` 在 `--enable-cfg-parallel` 时返回 `StageParallelismType.CFG_PARALLEL`
  - CFG parallel 下按 rank 拆分 prompt 分支：
    - CFG rank 0：positive / cond
    - CFG rank 1：negative / uncond
  - 串行路径继续保持原来的 `negative + positive` batch=2 顺序
  - CFG parallel 合成使用与串行公式等价的 all-reduce：

```python
serial = uncond + guidance_scale * (cond - uncond)
serial = guidance_scale * cond + (1 - guidance_scale) * uncond
```

- `test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py`
  - 覆盖 mode 解析、非法组合拒绝、stage parallelism 声明、prompt branch、model input batch shape、CFG combine 公式

- `docs_xzh/run_command/mock_test.md`
  - 补充 CFG-only 两卡服务命令
  - 补充四卡 `CFG=2 x SP=2` 服务命令
  - 明确第一次推理为 compile warmup，第二次才记录正式耗时
  - 明确 reference 视频和 SSIM 门槛

## 已完成轻量验证

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py -q
```

结果：`26 passed`

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m py_compile \
  python/sglang/multimodal_gen/runtime/server_args.py \
  python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py \
  test/srt/multimodal_gen/test_vividvr_cfg_parallel_stage.py
```

结果：通过。

## 正式 mock 验收口径

正式验收必须按 `/home/zhiheng/sglang/docs_xzh/run_command/mock_test.md`：

- 启动 moto S3
- 启动 callback receiver
- 启动 caption sidecar
- 启动 FlowCut bridge service
- 第一次完整请求仅作为 torch compile warmup
- 第二次完整请求记录正式 `total_runtime_seconds` 和 `model_inference_runtime_seconds`
- 从 mock S3 下载结果视频
- 与上一轮四卡 reference 做 SSIM 对比

reference：

```bash
/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/quad-test-video-long-960x720-130f-run2-20260708T060202Z.bridge-downloaded.mp4
```

本机可用输入视频：

```bash
/home/zhiheng/input/test_video_long_960x720_130f.mp4
```

质量门槛：

- `summary.ssim_mean > 0.98`
- `summary.ssim_min >= 0.98`
- `summary.pass_compare == true`

## 正式验收记录

### CFG-only 两卡闭环

- 服务口径：`--num-gpus 2 --enable-cfg-parallel --vividvr-parallel-mode cfg --enable-torch-compile --attention-backend fa`
- warmup 任务：`vividvr-cfg-only-warmup-20260709T090628Z`
  - 用途：torch compile warmup，不作为正式耗时
  - denoising：`840.4626s`
  - pixel data generated：`1008.71s`
- formal 任务：`vividvr-cfg-only-formal-20260709T092339Z`
  - S3 URL：`http://127.0.0.1:4566/flowcut/bridge-semantic-check/vividvr-cfg-only-formal-20260709T092339Z.mp4`
  - perf JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg-only-formal-20260709T092339Z_perf.json`
  - 下载视频：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/vividvr-cfg-only-formal-20260709T092339Z.mp4`
  - compare JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg-only-formal-20260709T092339Z_compare.json`
  - denoising：`493.5108s`
  - pixel data generated：`674.34s`
  - SSIM：`mean=0.9845679069864866`，`min=0.9805834440842233`，`compared_frames=130`，`pass_compare=true`

说明：CFG-only formal 服务启动早于 `gpu_worker.py` 的 `vividvr_debug` perf dump 补丁，因此该 perf JSON 中没有 `meta.vividvr_debug`。质量和耗时记录有效，但调试字段以后续四卡 formal 为准。

### 四卡 `CFG=2 x SP=2` 闭环

- 服务口径：`--num-gpus 4 --sp-degree 2 --ulysses-degree 2 --ring-degree 1 --enable-cfg-parallel --vividvr-parallel-mode cfg_sp --enable-torch-compile --attention-backend fa`
- 有效服务 session：`vividvr_flowcut_cfg2_sp2_mock_service`
- attach 命令：`tmux attach -r -t vividvr_flowcut_cfg2_sp2_mock_service`
- warmup 任务：`vividvr-cfg2-sp2-warmup3-20260710T013628Z`
  - 用途：torch compile warmup，不作为正式耗时
  - S3 URL：`http://127.0.0.1:4566/flowcut/bridge-semantic-check/vividvr-cfg2-sp2-warmup3-20260710T013628Z.mp4`
  - denoising：`194.5758s`
- formal 任务：`vividvr-cfg2-sp2-formal-20260710T014318Z`
  - S3 URL：`http://127.0.0.1:4566/flowcut/bridge-semantic-check/vividvr-cfg2-sp2-formal-20260710T014318Z.mp4`
  - perf JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg2-sp2-formal-20260710T014318Z_perf.json`
  - compare JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg2-sp2-formal-20260710T014318Z_compare.json`
  - 验收 summary JSON：`/home/zhiheng/sglang/Vivid_Acceptance/indicator/service_benchmark/vividvr-cfg2-sp2-formal-20260710T014318Z_acceptance_summary.json`
  - 下载视频：`/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/downloads/vividvr-cfg2-sp2-formal-20260710T014318Z.mp4`
  - `total_runtime_seconds=356.31927194446325`
  - `model_inference_runtime_seconds=194.2424`
  - `denoising_runtime_seconds=194.2424`
  - SSIM：`mean=0.984805283930388`，`min=0.9802324220300491`，`compared_frames=130`，`pass_compare=true`

formal perf debug 关键字段：

```json
{
  "vividvr_parallel_mode": "cfg_sp",
  "cfg_parallel_enabled": true,
  "cfg_rank": 0,
  "cfg_world_size": 2,
  "cfg_branch": "cond",
  "sp_world_size": 2,
  "sp_rank": 0,
  "prompt_embed_shape": [1, 226, 4096],
  "output_num_frames": 130,
  "cfg_combine_formula": "guidance_scale * cond + (1 - guidance_scale) * uncond"
}
```

四卡正式验收满足计划门槛：

- 第一次完整请求作为 compile warmup，第二次 formal 才记录正式耗时。
- formal 输出与上一轮四卡 reference 比较 `130` 帧。
- `summary.ssim_mean > 0.98`。
- `summary.ssim_min >= 0.98`。
- `summary.pass_compare == true`。

### 验收过程中的环境修正

- 服务进程必须显式执行 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，并设置 `NO_PROXY=127.0.0.1,localhost`、`no_proxy=127.0.0.1,localhost`。否则 mock S3 上传会被外部代理劫持并返回 `502`。
- `minioConfig.endpoint` 必须传不带 scheme 的 `host:port`，例如 `127.0.0.1:4566`。如果传 `http://127.0.0.1:4566`，服务会拼成 `http://http:/127.0.0.1%3A4566/...` 并导致上传失败。
- 上述两点已同步进 `/home/zhiheng/sglang/docs_xzh/run_command/mock_test.md`。

## 风险与注意事项

- `--vividvr-parallel-mode` 是服务启动级参数，不是请求字段。
- 未传 `--enable-cfg-parallel` 时，VividVR denoising 仍走原串行 CFG batch=2 或纯 SP 路径。
- 两卡不能同时跑 `SP=2` 和 `CFG=2`；四卡才允许 `CFG=2 x SP=2`。
- CFG parallel 的 all-reduce 必须覆盖每个 tile 的 noise pred，不能做近似同步。
- 服务 perf JSON 的 `total_duration_ms` 是服务侧全链路耗时毫秒；本轮额外生成的 formal acceptance summary JSON 包含 AGENTS 要求的 `total_runtime_seconds` 与 `model_inference_runtime_seconds`。
- 当前环境没有安装 `ruff`，本轮未运行 `ruff format`，已用 `py_compile` 和 pytest 做基础验证。
