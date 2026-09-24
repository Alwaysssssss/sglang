# VideoEdit 流式推理与音频保留验证

日期：2026-09-23。改动只位于 sglang 仓库，两个算法仓作为只读参考。

## 实现范围

- 按窗口解码、预处理、推理、贴回、编码；文件返回模式不再累计整段视频的输出张量。
- 对齐当前原始仓的窗口 bbox：`tight`（默认）、`fixed_size`、`global`；支持窗口内 mask union/shape 稳定化。
- overlap/bridge 传递全分辨率贴回帧，逆向片段分块恢复时间顺序；crop sidecar 按公共画布居中。
- 实际执行 AdaIN 边界处理和可选 crop 边缘羽化；保留原始编码/色彩元数据的能力。
- 最终主视频默认复制输入音轨，不重新编码音频；保留相对起始偏移。短音频不会截短视频，长音频按编辑片段时长裁切。crop sidecar 仅用于图像对比，不附音轨。
- 临时目录随成功/异常清理，主视频完成后原子发布；记录窗口 bbox、帧归属、音频处理结果到 `.videoedit.json`。

新增参数通过 sampling params、CLI、HTTP repair 请求传递：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `chunk_bbox_mode` | `tight` | 窗口 bbox 策略 |
| `stabilize_mask_union` | `false` | 窗口 mask 合并稳定化 |
| `stabilize_mask_shape` | `false` | 窗口 mask 形状稳定化，与 union 互斥 |
| `stabilize_smooth_window` | `5` | 形状平滑窗口 |
| `crop_edge_feather` | `0` | crop 边缘羽化 |
| `preserve_audio` | `true` | 保留主输出原始音轨；CLI 可用 `--no-preserve-audio` 关闭 |

sglang 现有 `bbox_expand_scale=0.3` 为单边比例，对应原始仓最终倍数 `1.6`，没有直接改成不兼容的默认值。

## 实际可用环境（不安装或升级依赖）

文档原路径 `/home/root/uv-envs/sglang-llm-diffusion/bin/python` 在本容器不可用。按用户确认使用：

```bash
VE_PACKAGES=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages
```

解释器为现有 `/opt/conda/bin/python`（Python 3.11），主依赖为上述目录（Torch 2.9.1+cu128）。算法原仓用现有 `/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/bin/python`。

验证脚本只读追加两处搜索路径：

- `$VE_PACKAGES/nvidia_cutlass_dsl/python_packages`：已有 CUTLASS 的 `.pth` 内容，普通 PYTHONPATH 不会自动处理。
- `edit_ruidi/lib/python3.11/site-packages`：已有 `ftfy` 等缺失依赖的后备路径。标准库路径放在前面，避免旧 argparse 包覆盖标准库。

`scripts/videoedit_existing_env.py` 还显式加载现有 `/opt/conda` 的 filelock 3.18.0（包括 spawn worker）。指定目录中的新版 filelock 在当前挂载盘上先 unlink 再 unlock，会触发 `FileNotFoundError`；独立锁测试可复现，3.18 不提前 unlink。没有修改 site-packages，也没有改变模型算子。可用 `VE_GPU=7 bash scripts/videoedit_stream_validation.sh check-runtime` 独立验证文件锁、融合 LayerNorm 和 FlashInfer RoPE。

测试前曾创建仓库内 `.venv` 用于工具和测试探索；收到“不修改环境”要求后停止安装，后续运行使用上述已有依赖。没有升级原有环境。

## 可复现命令

```bash
# 从 sglang 仓库根目录执行；先自行确认 GPU 空闲。
VE_GPU=6 bash scripts/videoedit_stream_validation.sh reference
VE_GPU=7 bash scripts/videoedit_stream_validation.sh sglang
bash scripts/videoedit_stream_validation.sh compare

# 多窗口/bridge/逆向合并模型冒烟测试（2 步不是画质验收）
VE_GPU=6 VE_FRAMES=110 VE_STEPS=2 VE_REF=20 \
VE_MASTER_PORT=30105 VE_SCHEDULER_PORT=5665 \
bash scripts/videoedit_stream_validation.sh sglang
```

默认 case0008、step_47500、48 输入帧、40 步、seed42、ref0、infer_len49/overlap5、dynamic CFG、关闭 TeaCache。可通过 `VE_FRAMES`、`VE_STEPS`、`VE_REF` 改变测试规模。输出、日志、缓存均在仓库 `outputs/videoedit-stream-validation/`。

注意：当前原始 case0008 的 video/mask 均为 **210 帧**，不是旧记录的 210/209；本轮直接使用当前原始输入，不复用旧的归一化数据。

### 原始仓在测试期间发生外部更新

只读检查发现原始仓 `utils/video_io_ffmpeg.py` 于 10:35:04 UTC、`infer.py` 于 10:41:17 UTC 更新；新增源音轨复制、全程无损中间片段及最终编码流程。不是本轮 sglang 修改的文件。

48 帧/40 步原始基线在更新前生成，110 帧/2 步/ref20 原始对比在更新后启动，**二者不能合称同一个固定算法快照**。窗口规划文件未随本次外部更新改变。后续 reference 命令自动保存关键源文件 SHA256 清单；若要求严格统一的最新原始仓验收，应冻结其版本再重新生成全部基线。

收尾再次检查 `reference-sources-observed-after-110f-start.sha256` 时，`utils/video_io_ffmpeg.py` 的摘要再次变化，其余五个记录文件未变化。原始仓仍在被外部更新，因此本报告描述的是已生成产物之间的对比，不是对收尾时原始仓最新文件的完整验收。

## 验证记录

全量 VideoEdit 单元/集成测试汇总：**151 passed，53 subtests passed**（898.70 秒），日志 `outputs/videoedit-stream-validation/unit-tests.log`。另补的 HTTP 互斥校验定向回归为 1 passed、6 subtests passed。测试中的 Pillow/FastAPI 等弃用警告未做环境升级处理。

```bash
export PYTHONDONTWRITEBYTECODE=1
export VE_PACKAGES=/mnt/shanhai-ai/shanhai-workspace/zhouhao6/env/sglang/lib/python3.11/site-packages
export PYTHONPATH="$PWD/python:/opt/conda/lib/python3.11:$VE_PACKAGES:/mnt/shanhai-ai/envs/conda/envs/edit_ruidi/lib/python3.11/site-packages"
export TMPDIR="$PWD/outputs/videoedit-stream-validation/tmp"
/opt/conda/bin/python -B -m pytest -q python/sglang/multimodal_gen/test/unit/test_videoedit_*.py
```

- 原始仓 48 帧/40 步推理已生成 full/crop 成片。
- 原始仓函数逐像素对比：3 组通过，覆盖动态 bbox、union/shape、overlap/bridge、逆向分支、AdaIN；比较真实前处理张量及贴回 RGB，不仅比较调用次数。
- 音频专项：6 项通过，包括压缩音频 packet SHA256 一致、无音频、短音频、长音频、起始偏移和 VFR 拒绝。
- case0008 真实 AAC 音轨：48/210 帧像素流输出分别保留 46/198 个音频 packet；SHA256 和 packet PTS 与原始音轨对应前缀完全一致。
- 序列规划、bbox、pipeline helper、参数、mask、frame provider 加窗口输出矩阵及异常清理：合并运行 **88 项通过，44 个 subtest 通过**。其中包括 9 组窗口/参考位置组合，覆盖 eager 输入分支。
- 逆向合并后音频 packet 一致及关闭音频：**2 项通过**；新增参数别名/透传/约束：**2 项通过，5 个 subtest 通过**。
- 指定依赖环境的融合 LayerNorm 最小测试通过；没有改用另一套模型数学分支。
- sglang 48 帧/40 步成片及数值对比通过（更新前原始仓基线）：crop SSIM 均值 **0.993563**、MSE **1.045261**；full SSIM 均值 **0.993285**、MSE **1.581353**。全部 48 帧分别满足 SSIM≥0.97/0.98、MSE≤25、MAE≤2.5，失败帧为 0。分辨率、帧数、帧率、像素格式和 bt709 色彩标记一致；成片的 46 个音频 packet 内容/PTS 与输入前缀完全一致。
- sglang 110 帧/2 步/ref20 真实模型冒烟测试通过：3 个窗口（含正向 overlap、bridge、逆向合并），帧归属恰好覆盖 0–109；实际成片 1920×1080、50 fps、110 帧，105 个音频 packet 的 SHA256/PTS 与原始输入前缀一致。**与更新后原始仓的严格逐帧画质比较未通过**，详见下节。
- HTTP 互斥稳定化参数的早期拒绝已通过先失败、后修复的定向回归，避免提交后才失败。

48 帧两次模型命令及 110 帧原始仓命令都完成了成片，但外层 shell 脚本在运行期间被维护，继续读取脚本尾部时出现语法错误、返回 2；不是模型推理失败。当前脚本已通过 `bash -n`，并在模型命令成功后显式退出，避免再次读取脚本尾部。48 帧独立 `compare` 命令返回 0；110 帧 sglang 完整启动脚本返回 0。运行脚本时不要原地编辑该脚本。

### 未解决：110 帧严格数值比较

此组使用 2 步作为跨窗口功能冒烟测试，不是完整 40 步画质验收。额外沿用 48 帧的严格阈值进行比较，结果如下；**未放宽阈值**。

| 输出 | SSIM 均值 / 最低 | MSE 均值 | SSIM 不通过帧 |
| --- | --- | ---: | --- |
| crop | 0.970131 / 0.942795 | 7.808463 | 0–19、68–109，共 62 帧 |
| full | 0.986075 / 0.977662 | 4.091973 | 69–89，共 21 帧 |

全部帧的 MSE/MAE 在阈值内，失败项为 SSIM；分辨率、帧数、帧率、像素格式、色彩标记一致。报告为 `case0008_110f_2s_ref20_tight_{crop_only,full}_compare.json`。

已确认两个实现均逐窗口重置 CPU seed42，采样时间表也重新设置；窗口输入/贴回的独立逐像素测试通过。差异集中在第二个窗口和逆向分支，不能据此直接归因于编码或模型中的某一项。更新后原始仓与本轮移植版的最终编码策略不同，实测 full 码率约 9.89/14.77 Mbps、crop 约 5.67/12.02 Mbps（原始仓/sglang），这一差异已确认，但没有证明它能解释全部 SSIM 差异。

下一轮应先冻结原始仓版本，再在同一最终编码配置下抓取各窗口编码前 RGB、模型条件和 latent 边界，区分跨窗口数值传播与最终编码影响；最后补正式 40 步多窗口验收。当前不能宣称所有长视频画质均已与最新原始仓对齐。

### 真实 1080p 像素流内存

`scripts/videoedit_stream_memory_check.py` 使用真实视频/mask，模型回调为 identity，独立进程测量；不含生成模型、CUDA 内存或 ffmpeg 子进程 RSS。

| 输入帧数 | Python 进程增量 RSS 峰值 | tracemalloc 峰值 |
| --- | ---: | ---: |
| 48 | 1,846,267,904 bytes | 1,201,241,946 bytes |
| 210 | 2,403,880,960 bytes | 1,274,553,579 bytes |

输入长度增加 4.375 倍，受跟踪分配峰值增加约 6.1%，进程 RSS 增量峰值增加约 30.2%。这支持窗口级像素缓存的实现，不等同于严格恒定 RSS 或完整模型长视频性能结论。报告为 `io48.memory.json`、`io210.memory.json`。

## 限制与后续验收

- 带音轨的 VFR 输入目前明确拒绝，避免 RGB 恒帧率重编码后静默音画错位；需要先规范时间轴，尚未实现逐帧 PTS 输出。
- 音频编码不兼容目标容器时明确失败，不会静默丢音或擅自有损转码。
- 返回张量、插帧、超分和显式 `decode_mode=eager` 不承诺整段像素内存有界；NPZ/object mask 保留兼容性的 eager 加载。
- 210 帧当前只完成像素流测试；没有把它当成 210 帧完整神经网络推理验收。
- HTTP 服务、实际多卡推理、所有容器/音频编码组合尚未穷尽验证。
