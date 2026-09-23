# VSR 音频保留

2026-09-23：CLI（含 `--via-pipeline`）和 `POST /v1/videos/restorations` 默认保留源视频的全部音轨。本需求取代第一阶段文档中“不处理音频”的范围限制。

## 使用

CLI 原命令无需增加参数。显式开启使用 `--preserve-audio`，关闭使用 `--no-preserve-audio`。

HTTP 请求增加可选布尔字段 `preserve_audio`，默认 `true`：

```json
{
  "video_input_path": "/data/input.mp4",
  "target_resolution": "1080x1920",
  "preserve_audio": true
}
```

本地路径、URL 下载、单卡和多卡 tile 共享相同输出流程。服务返回结构不变，状态完成、下载、上传和成功回调均在合并结束后执行。

运行环境需在 `PATH` 中提供 `ffmpeg` 和 `ffprobe`。Dockerfile 已增加系统 `ffmpeg` 包，镜像构建检查同时验证两个命令；已有镜像需要重建。`imageio-ffmpeg` 的内置编码器不能代替 `ffprobe`。显式关闭音频保留不需要这两个额外的系统命令。

## 行为

- 无音轨：正常生成无声视频。
- 有音轨：保留全部音轨的顺序、语言、标题（容器支持时）和默认标记。
- 编码兼容输出容器：音轨直接复制，不重新编码。视频始终复制超分后已编码的码流。
- 编码不兼容：仅将不兼容音轨转为 AAC，码率 192 kbit/s，日志记录音轨序号及原编码。I/O、权限、损坏数据等其他错误不会触发转码兜底，也不会降级成无声成功。
- 时间轴以源视频首个视频流的起始时间为基准，保留音轨相对偏移；源视频起点前的音频由输出容器的时间戳／编辑列表处理。按超分视频时长限制输出，音频短时不截短画面、不补静音。音轨包有时间粒度，尾部可能有一个编码包量级的边界误差。
- 合并子进程每 0.2 秒检查请求取消和期限，退出时回收子进程。合并时间计入任务完成耗时。
- 编码和合并在输出目录下独立临时目录中进行；成功后原子替换最终文件，失败或取消清理临时目录，保留已有结果。禁止输入输出指向同一文件。

严格同步的验收范围是恒定帧率输入。现有视频编码路径仍使用平均 FPS，未实现 VFR 逐帧时间戳保留；可变帧率输入可能出现局部音画偏移。此功能不处理字幕、附件或视频中其他数据流，也不改变 VSR 模型。

## 实现位置

- `runtime/vsr/audio.py`：媒体探测、可取消的合并、按音轨回退、原子发布。
- `runtime/vsr/stream.py`：公共 `stream_restore()` 包装原有视频流式恢复，所有调用方共享音频处理。
- CLI、`WanVRSamplingParams`、VSR stage、HTTP 请求模型：传递 `preserve_audio`。

## 验证命令

在仓库根目录执行：

```bash
PYTHONPATH=python OMP_NUM_THREADS=4 \
output_results/vsr/migration_env210/bin/python -m pytest -q \
  test/registered/multimodal_gen/vsr/test_audio.py \
  test/registered/multimodal_gen/vsr/test_server_api.py \
  test/registered/multimodal_gen/vsr/test_request_parameters.py
```

真实模型 CLI 验收（包含直连默认保留、显式关闭、原生 pipeline 默认保留）：

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 PYTHONPATH=python \
output_results/vsr/migration_env210/bin/python \
  docs_always/add_new_mode/add_vsr/scripts/test_audio_e2e.py \
  --output-dir output_results/vsr/audio_20260923 \
  --checkpoint-dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan-root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers
```

按 `server_api.md` 启动原生服务后，验证本地输入、URL 下载、音频关闭、结果下载和成功回调：

```bash
PYTHONPATH=python output_results/vsr/migration_env210/bin/python \
  docs_always/add_new_mode/add_vsr/scripts/test_audio_e2e.py \
  --output-dir output_results/vsr/audio_20260923 \
  --base-url http://127.0.0.1:30179
```

脚本生成 17 帧、64×64、10 FPS 的双音轨视频：AAC + 延迟开始的短 PCM 音轨。检查帧数、分辨率、音轨数、语言和默认标记、起始偏移，以及开启／关闭音频后的画面逐帧哈希一致性。测试结果、媒体和日志保存在指定目录。重复验收请更换目录。

## 2026-09-23 实测结果

- 音频媒体测试 12 项通过：默认复制、无音轨、主动关闭、多音轨、语言／默认标记、PCM 转 AAC、正负起始偏移、长短音频、画面哈希不变、AAC 包哈希不变、取消／合并失败保护旧结果、子进程回收、缺少 ffprobe 的错误与关闭开关。
- HTTP 参数与请求参数回归 18 项通过。首次合并运行是 28 通过、2 失败；失败来自旧多卡测试夹具遗漏已存在的 CPU 卸载字段，补齐夹具后这 18 项重跑全部通过。累计 30 项独立用例通过。
- GPU6 真实模型 CLI 3 项通过：直连默认保留 2 条音轨、直连关闭为 0 条、`--via-pipeline` 默认保留 2 条。
- GPU6 原生 HTTP 服务 3 项通过：本地／URL 输入均保留 2 条，关闭时为 0 条；三次下载和三次 completed 回调均验证通过。
- 6 个真实模型输出均为 17 帧、64×64。CLI 三个输出的逐帧哈希文件 SHA256 均为 `f053bf88858cc7cc039bfeaf800f321079e761248abfee2f245856aca1b6a520`；HTTP 三个均为 `e21ddbc91034e458ecec8e21819a0b740a2ab8c41c2444e4a8350dec9cad4b02`。两组推理配置不同（服务开启编译与 GPU 后处理），仅比较各组音频开关前后的画面一致性。
- 新增文件 Ruff 检查及 `git diff --check` 通过。Docker 依赖和构建检查已更新，本轮没有重建镜像，也没有实测外部云存储或多卡推理。

证据目录：`output_results/vsr/audio_20260923/`，包括 `cli_results.json`、`api_results.json`、`callbacks.json`、`api_unit_tests.log`、各 CLI 日志、`server.log` 及测试媒体。
