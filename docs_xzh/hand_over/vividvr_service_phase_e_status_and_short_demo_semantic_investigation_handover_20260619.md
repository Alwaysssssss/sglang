# VividVR 服务化当前状态与短视频 Demo SSIM 排查交接

更新时间：`2026-06-19 UTC`

## 1. 文档目的

本文档面向下一位继续推进 `Phase E` 的 Codex，目标是把当前项目完成状态、最近几轮正式服务验收结果，以及下一步需要排查的核心问题一次性讲清楚。

当前主判断先压缩成四句话：

1. `VividVR` 已经原生集成到当前仓库 `sglang` 的服务框架中，`/v1/videos/repairs` 真实可用。
2. 服务输出视频的编码 profile 已经基本对齐原版 `Vivid-VR`，短视频与长视频的正式服务验收都能通过当前灰度 SSIM 门限。
3. 长视频 `130f / 20 step` 的服务结果已经稳定在 `0.984+`，双卡 `SP(pool=1) + fa_sp + compile` 的纯推理加速比也已经接近 `2.0x`。
4. 当前真正需要继续排查的问题不是“服务不可用”或“编码不对”，而是：**为什么短视频 demo 的服务结果并没有达到 `0.98+`，这到底是目标口径误解，还是当前 `sglang` 部署的 `VividVR` 与原版短视频语义存在差异。**

---

## 2. 仓库与环境锚点

| 项 | 值 |
|----|-----|
| 分支 | `sglang_Vivid` |
| 当前 HEAD | `60c5a26f7` |
| Python 环境 | `/home/zhiheng/sglang/.venv/bin/python` |
| 服务仓库 | `/home/zhiheng/sglang` |
| 原版仓库 | `/home/zhiheng/Vivid-VR` |
| 当前仍可查看的双卡服务 tmux | `vividvr_serve_long_acceptance_sp2_v2` |

只读查看双卡服务：

```bash
tmux attach -r -t vividvr_serve_long_acceptance_sp2_v2
```

当前工作区仍是脏的，主要是前几轮服务化、编码对齐、reference 透传和双卡验收留下的运行时代码改动与验收产物。  
**下一轮默认不要清理工作区，也不要回退当前未提交改动。**

---

## 3. 当前已经完成到什么程度

### 3.1 服务化已经完成

当前仓库内部已经打通：

- `POST /v1/videos/repairs`
- `GET /v1/videos/{id}`
- `GET /v1/videos/{id}/progress`
- `GET /v1/videos/{id}/content`

`VividVR` 不再依赖仓库外单独服务进程，而是原生接入到 `sglang.multimodal_gen`。

### 3.2 编码参数对齐已经完成到可验收状态

这轮之前的核心遗留问题是服务输出视频编码 profile 未对齐原版，导致短视频服务口径 SSIM 偏低。  
该问题已经收口到“reference-profile 写出”路径，当前正式服务产物与原版 reference 已做到：

- `codec_name = h264`
- `profile = High`
- `level = 3.1`
- `pix_fmt = yuv420p`
- `width/height = 960x720`
- `fps = 25`
- `frame_count` 对齐
- 下载链路不再发生二次转码

短视频服务输出码率当前约 `16.41 Mbps`，原版 reference 约 `17.02 Mbps`；  
长视频服务输出码率当前约 `8.69 Mbps`，原版 reference 约 `8.78 Mbps`。  
这已经不再是当前最主要的质量阻塞点。

### 3.3 长视频服务验收与双卡提速已经跑通

当前 `Phase E` 日常 benchmark 主口径 `130f / 20 step` 已完成：

- 单卡服务正式验收通过
- 双卡服务正式验收通过
- 双卡 warmup 后复跑的纯推理加速比约 `1.99x`

因此，当前项目状态已经不再是“性能线不成立”，而是“性能线基本成立，但短视频 demo 质量目标需要重新澄清并排查”。

---

## 4. 当前最关键的正式结果

### 4.1 短视频历史正式基线：`Phase C`

指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json`

关键结果：

- `ssim_mean = 0.967716215299506`
- `ssim_min = 0.9473462237832677`

这条结果非常重要，因为它说明：

- 当前仓库历史上“短视频正式基线”本来就不是 `0.98+`
- 下一轮排查前，必须先确认“短视频期待 `0.98+`”这个目标本身是否成立

### 4.2 当前单卡短视频服务正式结果

指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-encodealign-v3-50step-20260618T175003Z.json`

关键结果：

- `ssim_mean = 0.9669889352908251`
- `ssim_min = 0.9471865035211443`
- `total_runtime_seconds = 795.977483`
- `model_inference_runtime_seconds = 795.932328`

和历史 `Phase C` 基线对比：

- `ssim_mean` 只低约 `0.00073`
- `ssim_min` 只低约 `0.00016`

这说明：

- **当前单卡短视频服务结果实际上已经非常接近历史正式短视频基线**
- 如果下一轮要追问“为什么没有 `0.98+`”，首先要区分：
  - 是不是目标值本来就设错了
  - 还是当前短视频 reference / metric / 推理语义与原版另有差异

### 4.3 历史本地双卡短视频 `SP(pool=1)` 对照

指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/ctrl_pool1_720p_50steps_phasecprompt_metrics_seed42_20260617T161102Z.json`

关键结果：

- `ssim_mean = 0.9679797357829025`
- `ssim_min = 0.9566068425373722`
- `total_runtime_seconds = 457.620024`
- `model_inference_runtime_seconds = 349.72472`

这条结果说明：

- 仓库内曾经已经存在一条更好的本地双卡短视频 `SP(pool=1)` 结果
- 所以当前“服务双卡短视频结果偏低”不能直接归结为“短视频双卡天然做不到”

### 4.4 当前双卡短视频服务结果

指标文件：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-short-70f-50step-sp2-20260619T041059Z.json`

关键结果：

- `ssim_mean = 0.9610210946062645`
- `ssim_min = 0.9508794853069598`
- `total_runtime_seconds = 370.4253195002675`
- `model_inference_runtime_seconds = 360.23482835292816`
- `pass_compare = true`

与单卡短视频服务结果对比：

- `ssim_mean` 下降约 `0.00597`
- `ssim_min` 反而高约 `0.00369`

与历史本地双卡短视频 `SP(pool=1)` 对比：

- `ssim_mean` 下降约 `0.00696`
- `ssim_min` 下降约 `0.00573`

这说明：

- 当前双卡短视频服务质量确实存在值得继续追的回退
- 但它和“短视频为什么没有 `0.98+`”并不是完全同一个问题

### 4.5 长视频服务结果

单卡长视频正式结果：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-long-130f-20step-20260619T021333Z.json`
- `ssim_mean = 0.9848321152151382`
- `ssim_min = 0.979075011104728`
- `model_inference_runtime_seconds = 1087.390713777393`

双卡长视频 warmup 后复跑结果：

- `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-long-130f-20step-20260619T030729Z.json`
- `ssim_mean = 0.9846671910503901`
- `ssim_min = 0.9788592185342503`
- `model_inference_runtime_seconds = 545.1444808579981`

双卡长视频加速比：

- 端到端：`1.97x`
- 纯推理：`1.99x`

这条线的意义是：

- 当前 `VividVR` 服务化、编码对齐、双卡 `SP(pool=1)` 的主线已经基本闭环
- “`0.98+`”这个数值更像当前长视频 formal 结果，而不是已知短视频 formal 基线

---

## 5. 当前最重要的判断

### 5.1 不要把“短视频没到 0.98+”直接当成已证实回归

历史短视频 `Phase C` 正式基线就是：

- `0.967716 / 0.947346`

当前单卡短视频服务结果是：

- `0.966989 / 0.947187`

因此从当前正式基线口径看，单卡短视频服务结果几乎已经打平。  
下一轮首先要排除的是：**当前对短视频的 `0.98+` 期待，本身可能不是这个口径下的历史真目标。**

### 5.2 但双卡短视频服务结果确实值得继续深挖

虽然单卡短视频服务已经接近历史基线，但双卡短视频服务结果：

- 低于单卡短视频服务
- 也低于仓库内已有的历史本地双卡 short result

所以当前更精确的问题应该拆成两层：

1. **短视频单卡 service 与原版短视频主语义是否还存在未识别差异？**
2. **双卡 short service 是否在 `fa_sp + compile + SP(pool=1)` 路径上又引入了额外回退？**

### 5.3 长视频已经高于 `0.984`，说明“编码问题”已不是主因

如果编码仍是主因，那么长视频 formal 结果不会稳定到 `0.984+`。  
所以当前更该优先怀疑的是：

- reference / metric 目标理解错误
- 短视频路径的某个语义点与原版不完全等价
- 或者双卡 short service 在服务默认值、prompt 路径、attention/backend、compile、SP 语义上与历史本地结果不完全一致

---

## 6. 下一步任务定义

下一轮任务不要再泛化成“继续优化 SSIM”，而应严格收缩为：

1. **先确认短视频 demo 的正式目标值到底应不应该是 `0.98+`。**
2. **如果 `0.98+` 不是当前灰度 SSIM + 当前 reference 下的正确目标，就要明确记录这一点，避免后续一直追错目标。**
3. **如果目标确认无误，就继续排查当前 `sglang` 部署的 `VividVR` 与原版短视频主语义是否仍有差异。**
4. **在单卡短视频主语义问题厘清之后，再单独排查双卡 short service 的额外回退。**

---

## 7. 建议的排查顺序

### 7.1 H0：先确认“`0.98+` 目标”是否成立

第一步不要先改代码，而是先确认：

- 这个 `0.98+` 是不是来自长视频 formal 结果
- 还是来自彩色 SSIM、不同 reference、不同 compare 口径
- 还是来自某条历史本地双卡 short result 的误读

当前仓库正式 compare 实现是灰度 SSIM：

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/compare.py`

而当前短视频正式 reference 是：

- `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

在这套固定口径下，历史基线就是 `0.9677`，不是 `0.98+`。

### 7.2 H1：确认当前单卡 short service 与当前本地 short path 是否仍完全一致

建议先做最小 A/B：

- 本地当前 HEAD 单卡 short 路径
- 当前单卡 service short 路径

目标不是先追求更高分，而是先确认：

- 当前 service 是否真的和当前本地短视频主链完全等价
- 是否存在服务侧默认值覆盖，例如：
  - `prompt` / `prompt_file_path`
  - `caption_file_path`
  - `reference_video_path`
  - `output_quality` / `output_compression`
  - `dtype`
  - `attention_backend`

如果这一步不能打平，再去怀疑更深层语义就会浪费时间。

### 7.3 H2：如果 service 和本地已一致，再排查“当前短视频主语义 vs 原版”差异

下一优先级应回到 `Phase C` 语义红线逐项核对：

- prompt 默认是否仍来自 `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- 是否仍然不走 live `CogVLM2`
- `prompt_embed_shape` 是否仍保持 `226`
- VAE tiling 默认是否仍是 `240 / 360`
- preprocess 是否仍保留未 padding 的 `reference_video`
- decode / postprocess 是否仍保持：
  - `drop first 3 frames`
  - `crop padding`
  - `AdaIN / reference color fix`

如果短视频单卡始终只能稳定在 `0.9669` 附近，而确认目标确实应更高，那么最可能的根因仍然会落在这些语义点上，而不是编码层。

### 7.4 H3：单独排查双卡 short service 回退

当单卡 short 语义问题澄清后，再做双卡短视频回退排查。  
当前更合理的 A/B 维度是：

- 本地双卡 short `SP(pool=1)` 历史好结果
- 当前双卡 short service
- 是否开启 `torch.compile`
- 是否走 `fa_sp`
- `SGLANG_VIVIDVR_CONNECTOR_SP_CONTEXT_MODE`
- `SGLANG_VIVIDVR_CONNECTOR_CONTROL_POOL_SIZE=1`

当前双卡 short service 已经明确是：

- `SP(pool=1)`
- `fa_sp`
- `compile`

因此如果下一轮再复现实验，应避免把 `pool=2` 或其他性能实验变量混进来。

---

## 8. 下一轮建议优先看的文件

### 8.1 服务入口与默认值

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`

### 8.2 短视频主链语义

- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/preprocess.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/io.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/ffmpeg_io.py`
- `/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/compare.py`

### 8.3 已有验收脚本与产物

- 短视频 service 单卡结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-encodealign-v3-50step-20260618T175003Z.json`
- 短视频 service 双卡结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-short-70f-50step-sp2-20260619T041059Z.json`
- 长视频 service 单卡结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-long-130f-20step-20260619T021333Z.json`
- 长视频 service 双卡结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-long-130f-20step-20260619T030729Z.json`
- 历史本地双卡 short 好结果：
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/ctrl_pool1_720p_50steps_phasecprompt_metrics_seed42_20260617T161102Z.json`
- 临时短视频 service 验收脚本：
  - `/home/zhiheng/sglang/Vivid_Acceptance/tmp/run_vividvr_short_service_acceptance.py`

---

## 9. 本轮结论

截至 `2026-06-19`，可以把当前项目状态总结为：

1. `VividVR` 服务化已经完成，编码参数对齐也已经进入可正式验收状态。
2. 长视频 `130f / 20 step` 单卡与双卡服务结果都很好，双卡纯推理加速比约 `2.0x`。
3. 当前单卡短视频服务结果已经基本贴近历史 `Phase C` 正式基线，并不能简单定性为失败。
4. 真正需要下一轮重点排查的是：
   - 短视频 `0.98+` 目标是否本身就不适用于当前灰度 SSIM formal 口径
   - 如果目标确认成立，当前 `sglang` 部署的 `VividVR` 在短视频语义上是否仍与原版存在差异
   - 以及双卡 short service 为什么比仓库内已有的历史本地双卡 short result 更低

下一轮建议先做“目标值澄清 + 单卡 short service/local A/B”，不要一开始就直接改动大段运行时代码。
