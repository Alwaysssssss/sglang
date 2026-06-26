# Vivid-VR compile 稳定化与服务器迁移前环境打包交接

日期：`2026-06-25 UTC`

## 背景

本交接承接以下最近文档：

- `docs_xzh/hand_over/vividvr_origin_resolution_alignment_and_newspaper_compile_handover_20260625.md`
- `docs_xzh/hand_over/vividvr_flowcut_minio_progress_acceptance_handover_20260625.md`
- `docs_xzh/hand_over/vividvr_service_boundary_alignment_handover_20260624.md`

上一份交接里已经把问题缩小到：

- `newspaper.mp4` 的失败不是 FlowCut submit、callback、caption bridge、MinIO 或写视频 warning
- 失败点在 `VividVRDenoisingStage -> transformer -> torch.compile / torch._inductor autotune`
- 报错形态是：
  - `TypeError: randn() received an invalid combination of arguments - got (Add, device=torch.device, dtype=torch.dtype)`

本轮工作的目标有两个：

1. 先用最小改动验证这是不是 `torch.compile` 动态 shape autotune 导致的问题。
2. 在同一个仍然运行的 compile 服务实例上，再发起一个不同 shape 的请求，确认连续不同 shape 请求也能正常通过推理。

## 本轮结论

结论已经可以先收口为：

- 本轮 `newspaper` compile 故障的主根因，基本可以定为 `torch.compile` 在 Vivid-VR edge tile shape 上触发的 inductor autotune 动态 shape 问题。
- 把 Vivid-VR transformer 的 `torch.compile` 策略从 `dynamic=None` 收口到 `dynamic=False` 后，问题被修复。
- 在同一个双卡 compile 服务实例上，已经连续成功处理了两个不同 shape 的请求：
  - `newspaper` 单 clip 请求
  - `960x720 / 130f` 长视频请求
- 这说明“compile 默认主路径 + 连续不同 shape 请求进入同一服务实例”当前已经可以稳定工作。

需要单独说明的是：

- 这两次验证都没有使用 `MinIO`
- 但这不影响对 compile 根因的判断，因为原始报错发生在模型生成阶段，早于结果上传阶段
- 当前没有证据表明 `MinIO` 与这次 compile 故障有直接关系

## 本轮实现

### 1. compile 策略最小修复

修改点：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`

当前逻辑：

- Vivid-VR 的 `torch.compile` 仍然默认开启
- compile mode 仍然沿用 `SGLANG_TORCH_COMPILE_MODE`
- 但显式把 `compile_kwargs["dynamic"] = False`

含义是：

- 仍然保留 compile 作为默认主路径
- 但改成按每个实际 tile shape 做静态图编译
- 避免 edge tile 的符号尺寸流入 inductor autotune 的 benchmark buffer 分配

对应代码位置：

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:308-315`

### 2. 单测断言补强

修改点：

- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py`

新增断言：

- 明确检查 compile helper 传入的是 `dynamic=False`

对应代码位置：

- `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py:542-553`

## 本轮验证

### 1. 聚焦单测

已跑命令：

```bash
PYTHONPATH=python /home/zhiheng/sglang/.venv/bin/python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_attention_backend.py \
  -k torch_compile -q
```

结果：

- `1 passed`

这个用例先在旧行为下红掉，确认 `dynamic` 原来确实不是 `False`
- 之后在修复后转绿

### 2. 同一服务实例上的 `newspaper` 复测

服务：

- tmux session：`vividvr_serve_dual_default`
- base URL：`http://127.0.0.1:31191`

任务：

- `task_id`：`vividvr-newspaper-compilefix-20260625T082532Z`

结果：

- 之前的 `randn(Add, ...)` 没有再出现
- 服务完整跑完 denoising、decode 和 postprocess
- callback 最终成功

关键产物：

- callback：
  - `Vivid_Acceptance/logs/vividvr-newspaper-compilefix-20260625T082532Z_callback.jsonl`
- perf：
  - `Vivid_Acceptance/indicator/vividvr-newspaper-compilefix-20260625T082532Z_perf.json`
- 实际输出视频：
  - `/home/zhiheng/sglang/inputs/uploads/vividvr-newspaper-compilefix-20260625T082532Z/outputs/vividvr-newspaper-compilefix-20260625T082532Z_0.mp4`

性能摘要：

- 总耗时约 `478.96s`
- `VividVRDenoisingStage` 约 `420.32s`

### 3. 同一服务实例上的第二个不同 shape 请求复测

为了验证“连续不同 shape 请求进入同一 compile 服务实例也能正常推理”，本轮又向同一服务发送了长视频请求：

- 输入视频：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.mp4`
- caption file：
  - `/home/zhiheng/Vivid-VR/input/720p_long/test_video_long_960x720_130f.txt`
- reference video：
  - `/home/zhiheng/Vivid-VR/result/720p_long_up1_result_vivid_ori_20step/videos/test_video_long_960x720_130f.mp4`

任务：

- `task_id`：`vividvr-long130f-compilefix-20260625T084030Z`

结果：

- 请求被同一服务实例成功 accept
- 进入 `caption_ready`
- 顺利进入 `denoising`
- 20 个 denoising step 全部完成
- 最终进入 `uploading_result`
- callback 最终 `succeeded`

关键产物：

- callback：
  - `Vivid_Acceptance/logs/vividvr-long130f-compilefix-20260625T084030Z_callback.jsonl`
- perf：
  - `Vivid_Acceptance/indicator/vividvr-long130f-compilefix-20260625T084030Z_perf.json`
- 实际输出视频：
  - `/home/zhiheng/sglang/inputs/uploads/vividvr-long130f-compilefix-20260625T084030Z/outputs/vividvr-long130f-compilefix-20260625T084030Z_0.mp4`

性能摘要：

- 总耗时约 `600.37s`
- `VividVRLongClipPreparationStage` 约 `70.15s`
- `VividVRMultiClipDenoisingStage` 约 `422.81s`
- `VividVRMultiClipDecodeTrimStage` 约 `101.57s`

这一步的意义是：

- 前一个请求已经在这个服务实例里触发过 compile
- 第二个不同 shape 请求继续进入同一服务实例
- 没有再出现 inductor compile 崩溃

因此当前可以认为：

- `dynamic=False` 的修复不只是“恰好救活了 newspaper”
- 它也覆盖了“同一 compile 服务连续接收不同 shape 请求”的核心场景

## 当前边界与未决事项

### 1. compile 问题已收口，但不是所有服务问题都已收口

当前已经解决的是：

- Vivid-VR 模型阶段的 compile 崩溃

当前还没有一起解决的是：

- `output_path` 请求字段与最终实际落盘路径之间的契约差异

现象仍然是：

- 请求里传入了 `Vivid_Acceptance/result_videos/...mp4`
- 服务最终实际输出仍然落在 `inputs/uploads/<task_id>/outputs/..._0.mp4`

这不是本轮 compile 根因，但后续如果要做 release gate 或服务迁移，仍然应该单独收口。

### 2. 本轮没有重新验证 MinIO 上传链路

需要明确：

- 这两次 compile 修复后的服务复测都没有带 `minioConfig`
- 因此它们验证的是“本地路径模式下 compile 是否稳定”

但这不影响对根因的判断，因为：

- 之前失败点在模型生成阶段
- `MinIO` 介入是在更后面的 `uploading_result`

所以当前合理结论是：

- `MinIO` 不是这次 compile 故障的原因
- 但如果后续需要把“MinIO 模式也稳定”写成正式验收结论，仍然应该单独补一轮 MinIO 请求

### 3. caption 仍然保持 sidecar 独立环境路线

当前项目的大方向没有变化：

- 主推理环境继续使用 `/home/zhiheng/sglang/.venv`
- caption sidecar 继续保持独立环境 `/home/zhiheng/sglang/.venv-vividvr-caption`

不要为了服务器迁移准备，反过来把 caption 依赖揉进主推理环境。

## 下一阶段主任务：环境打包，为迁移到服务器主机做准备

当前从项目状态看，下一步不应该继续追 `newspaper` compile 问题，因为它已经收口。更合理的主任务是：

- 把当前已经可用的 Vivid-VR 运行环境、依赖、资源和服务拓扑打包清楚
- 为迁移到服务器主机做可重复部署准备

### 下一阶段目标

目标不是“重新设计运行方式”，而是：

- 把当前本机已经验证通过的运行形态，整理成一套可以迁移到服务器主机的环境包和启动说明

### 建议的打包范围

建议至少覆盖以下内容：

1. 主推理环境打包

- 固定 `/home/zhiheng/sglang/.venv` 的依赖集合
- 输出可重建的依赖清单
- 记录 CUDA / PyTorch / flash-attn / compile 相关关键版本

2. caption sidecar 环境打包

- 固定 `/home/zhiheng/sglang/.venv-vividvr-caption`
- 明确它和主推理环境的边界
- 给出独立启动命令和健康检查方式

3. 模型与资源清单

- checkpoint 所在位置
- VAE / transformer / text encoder / tokenizer 等依赖资源
- 参考视频、默认 prompt、caption 文件、测试输入视频的目录约定

4. 服务拓扑与启动顺序

- 主 Vivid-VR serve
- caption sidecar
- 如果需要 acceptance 或演示链路，再补 callback receiver / MinIO 模拟服务

5. 环境变量清单

- `PYTHONPATH=python`
- `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE=eager_global`
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`
- `SGLANG_TORCH_COMPILE_MODE`
- 其他双卡正式配置所需变量

6. 路径迁移风险清单

- 当前很多命令和资源路径都写死为 `/home/zhiheng/...`
- 迁移前需要明确哪些路径必须抽象成服务器侧可配置参数

### 建议的实际产出

建议把下一阶段最少产出收口成下面几项：

1. 一份服务器迁移环境说明文档

- 写清依赖环境
- 写清目录布局
- 写清启动顺序
- 写清 smoke test 命令

2. 一套环境导出材料

- 主推理环境依赖导出
- caption sidecar 环境依赖导出
- 关键环境变量模板

3. 一套最小迁移验收脚本或命令

- 单卡或双卡健康检查
- 一条最小 Vivid-VR 本地推理 smoke test
- 一条服务模式 smoke test

### 推荐执行顺序

建议下一位同学按这个顺序推进：

1. 先盘点当前真实依赖和运行目录，不要先写容器或大重构。
2. 导出主推理环境与 caption sidecar 环境的依赖清单。
3. 整理当前默认正式配置对应的 serve 启动命令。
4. 识别并收口所有迁移时会出问题的绝对路径。
5. 在本机先按“冷启动新环境”的方式做一次重建演练。
6. 最后再考虑是否需要容器化或更强的自动化部署包装。

## 给下一位同学的直接建议

- 不要再把精力放在 `newspaper` compile 故障本身，这个问题当前已经有修复和双请求验证。
- 下一步优先级应该切到“环境打包 + 迁移准备”，让当前 Phase E 的可运行状态能够被复制到服务器主机。
- 打包时要把主推理环境和 caption sidecar 环境继续分开。
- 如果后续顺手补验证，优先补的是“迁移后 smoke test”和“路径配置化”，不是重新深挖这次 compile 崩溃。

