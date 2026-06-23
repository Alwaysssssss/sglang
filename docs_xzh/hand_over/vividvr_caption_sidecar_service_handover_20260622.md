# Vivid-VR Caption Sidecar Service 交接

日期：`2026-06-22 UTC`

## 1. 目标与边界

本轮完成的是 Vivid-VR `serve` 自动生成 caption sidecar 的服务化接线。用户请求仍然只需要提供 `video_input_path` 或 `video_url`；当请求没有显式传 `caption_file_path`，且服务端启用了 caption bridge 时，主服务会先生成 manifest，再请求本机 caption sidecar 产出 sidecar 文本，最后把 `caption_file_path` 写回 `VividVRSamplingParams`，继续走 `sglang` 原生 Vivid-VR 推理链。

保留的 override 语义：

- `caption_file_path` 仍然是可选 override。
- 用户显式传入 `caption_file_path` 时优先，不会再触发自动 caption。
- bridge 只负责 sidecar 生成，不把原版 Vivid-VR 推理代码带入主服务运行时。

## 2. 进程职责

caption sidecar 必须由 `/home/zhiheng/Vivid-VR/.venv/bin/python` 启动。它负责读取 manifest、按 temporal clip 顺序生成 caption，并写出 sidecar 文本。manifest 中的 spatial tile 信息继续保留，只用于调试和与 Phase D 语义对齐，不参与 sidecar 行数契约。

如果原版环境声明的 `INCLUDEPY` 指向宿主机上不存在的系统目录，sidecar 现在会额外探测 `~/tmp_py310dev/extracted/usr/include/python3.10` 或 `~/tmp_py310_headers/extracted/libpython3.10-dev/usr/include/python3.10` 这类已解压 Python 3.10 dev headers，并把它们注入 `CPATH` / `C_INCLUDE_PATH`，避免 Triton/CUDA 扩展编译时直接因为 `Python.h` 缺失而失败。

主服务必须由 `/home/zhiheng/sglang/.venv/bin/python` 或对应 `sglang serve` 入口启动。它负责下载/定位输入视频、构造 manifest、请求 sidecar、校验输出，并继续执行 Vivid-VR 推理。

这条边界的目的是隔离 caption 模型依赖，避免为了让原版 caption 在 `sglang/.venv` 中运行而破坏 Phase C/D/E 主推理环境。

## 3. 当前协议

主服务在 bridge 启用时会为每个请求生成：

- `${work_dir}/${task_id}.manifest.json`
- `${work_dir}/${task_id}.txt`

manifest 中的 `expected_caption_count` 是强约束。当前它表示 temporal clip 数，而不是 spatial tile 总数。sidecar 输出文件固定为一行一个 temporal clip caption，行数必须等于 manifest 的 `expected_caption_count`。如果 sidecar 返回成功但文件不存在、为空，或行数不匹配，主服务会把它当作 bridge 失败处理。

当前失败映射：

- 普通 `/v1/videos/repairs`：返回 HTTP `500`，detail 包含 `caption bridge failed`
- FlowCut `/v1/videos/repairs/flowcut`：返回 `code=1`，不进入推理队列

## 4. 部署顺序

重型验收必须在 `tmux` 中运行，推荐顺序：

1. 先在 `tmux` 中启动 `vividvr_caption_sidecar`
2. 再在 `tmux` 中启动启用了 `--vividvr-caption-bridge` 的 Vivid-VR 主服务
3. 最后发起不带 `caption_file_path` 的 `repair` 或 FlowCut 请求

建议查看命令：

```bash
tmux attach -r -t vividvr_caption_sidecar
tmux attach -r -t vividvr_serve_caption_bridge
```

如果要做 FlowCut 端到端 bridge 验收，仓库内 `python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py` 现在已经支持：

- 不传 `caption_file_path`
- 不传 `reference_video_path`
- 通过 `--callback-log` 在本机自建 callback receiver，并把 `running/succeeded/failed` 回调写成 JSONL
- 通过 `--submit-timeout-s` 放大首次提交超时，覆盖“服务端同步等待 caption sidecar 先生成 sidecar 文本”的长提交阶段

## 5. 本轮落地范围

代码侧已新增/接通：

- `python/sglang/multimodal_gen/runtime/vividvr/caption_manifest.py`
- `python/sglang/multimodal_gen/runtime/vividvr/caption_bridge.py`
- `python/sglang/multimodal_gen/tools/run_vividvr_caption_sidecar.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `python/sglang/multimodal_gen/runtime/server_args.py`

文档入口已更新：

- `docs_xzh/run_command/vividvr_default_run_and_serve_commands.md`

辅助验收工具已更新：

- `python/sglang/multimodal_gen/tools/run_flowcut_vividvr_service_acceptance.py`

## 6. 已知风险

- caption sidecar 仍依赖原版视频 caption 模型和 `/home/zhiheng/Vivid-VR/.venv`，如果原版环境漂移，bridge 会直接失败。
- sidecar 输出文件目前只在全部 caption 生成完成后一次性 `rename` 到最终 `${task_id}.txt`；对 `130f / 121 temporal frames` 这类长视频，虽然最终 sidecar 只会写出 2 行 temporal clip caption，但首次 bridge 提交仍会同步等待原版 caption 模型把两个 clip 都跑完，因此可能阻塞数分钟到十几分钟，这属于当前设计预期。
- manifest 生成逻辑必须继续与 Phase D temporal clip / spatial tile 语义保持一致；后续如果改 `num_temporal_process_frames`、tile 配置或 windowing 逻辑，需要同步复查 `expected_caption_count` 是否仍然等于实际 temporal clip 数。
- 当前只实现了本机 HTTP sidecar；如果未来拆到远端机器，需要额外处理网络、路径共享和 sidecar 产物回收。

## 7. 一句话交接

现在的 Vivid-VR 服务已经支持“请求只给视频，服务端自动补 caption sidecar”；显式 `caption_file_path` 仍优先，bridge 失败时 FlowCut 走 `code=1` 且不排队，所有重型验收继续要求在 `tmux` 中执行。
