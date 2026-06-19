# VividVR 服务接入完成与编码参数对齐交接

更新时间：`2026-06-18 UTC`

## 1. 文档目的

本文档面向下一位接手 `VividVR` 服务化与质量验收的 Codex，记录当前已经完成的部分、尚未闭环的问题，以及下一步明确要做的事情。

当前结论可以先压缩成一句话：

1. `VividVR` 已经成功接入当前仓库 `sglang` 内部的视频服务框架，`/v1/videos/repairs` 可真实拉起服务并完成端到端推理。
2. 服务语义入口已经按要求改成“用户不传 `prompt` / `prompt_file_path` / `mask_*`，服务端内部读取 demo prompt”。
3. 当前剩余的核心问题不是服务不可用，而是**服务输出视频的编码参数仍未完全对齐原版 `Vivid-VR` 输出 profile**，因此灰度 SSIM 仍低于历史 `Phase C` 基线。
4. 后续任务已经收口为：**继续把编码参数对齐原版 `Vivid-VR`，然后按固定 reference + 灰度 SSIM 口径完成正式推理验收。**

---

## 2. 仓库与环境锚点

| 项 | 值 |
|----|-----|
| 分支 | `sglang_Vivid` |
| 当前 HEAD | `60c5a26f7` |
| Python 环境 | `/home/zhiheng/sglang/.venv/bin/python` |
| 服务仓库 | `/home/zhiheng/sglang` |
| 原版 Vivid-VR 仓库 | `/home/zhiheng/Vivid-VR` |
| 当前服务 tmux session | `vividvr_serve_acceptance_encodealign` |

只读查看当前服务：

```bash
tmux attach -r -t vividvr_serve_acceptance_encodealign
```

停止当前服务：

```bash
tmux kill-session -t vividvr_serve_acceptance_encodealign
```

---

## 3. 当前已经完成的服务功能

### 3.1 已完成的服务能力

`VividVR` 已经接入当前仓库 `sglang` 内部的视频服务框架，而不是项目外的 `sglang_serve`。

当前已打通的入口：

- `POST /v1/videos/repairs`
- `GET /v1/videos/{id}`
- `GET /v1/videos/{id}/progress`
- `GET /v1/videos/{id}/content`

相关核心文件：

- [protocol.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py:120)
- [video_api.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py:344)
- [http_server.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py:281)

### 3.2 当前 `VividVR` 服务契约

用户最小请求体已经收敛为：

```json
{
  "model": "VividVR",
  "video_input_path": "/abs/path/input.mp4"
}
```

允许用户额外传的常用字段：

- `task_id`
- `num_frames`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `dtype`
- `negative_prompt`
- `num_temporal_process_frames`
- `restoration_guidance_scale`
- `output_path`
- `callback_url`

不再要求用户传：

- `prompt`
- `prompt_file_path`
- `mask_input_path`
- `mask_url`

当前服务端内部默认读取的 demo prompt：

- `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`

### 3.3 已覆盖的基础测试

当前已补的服务相关测试主要在：

- [test_video_api_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_video_api_vividvr.py:21)
- [test_entrypoints_utils.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_entrypoints_utils.py:10)

这部分已经覆盖：

- `VividVR` 请求不传 `prompt` / `mask` 也能提交
- `Wan` 原有 repair 入口仍保留兼容
- 视频保存辅助逻辑的新分支行为

---

## 4. 当前固定的验收口径

### 4.1 Reference 视频

短视频 demo 的标准 reference 固定为原版 `Vivid-VR` 输出：

- `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`

这也是历史 `Phase C` 正式基线使用的 reference，见：

- [phase_c_metrics_seed42_20260604T090642Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json:1)

### 4.2 SSIM 口径

当前仓库正式验收使用的是 **灰度 SSIM**，不是彩色 SSIM。

实现位置：

- [compare.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/compare.py:45)

关键事实：

- `_load_frames()` 读出 RGB 帧
- `_ssim()` 里先做 `cv2.cvtColor(..., cv2.COLOR_RGB2GRAY)`
- 所以 `ssim_mean / ssim_min` 都是灰度口径

### 4.3 历史短视频正式基线

当前短视频 `Phase C` 正式基线指标是：

- `ssim_mean = 0.967716215299506`
- `ssim_min = 0.9473462237832677`
- `mse_mean = 39.878108160836355`
- `mae_mean = 3.3365604979651313`

对应文件：

- [phase_c_metrics_seed42_20260604T090642Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T090642Z.json:1)

后续任何“服务正式验收是否达标”的判断，都应该继续对齐这一 reference 和这一灰度 SSIM 口径。

---

## 5. 这几轮真实服务验收发生了什么

### 5.1 第一轮真实服务验收：20 step，结果明显偏低

产物：

- [vividvr_service_acceptance_20260618T091617Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr_service_acceptance_20260618T091617Z.json:1)
- [vividvr_service_acceptance_20260618T091617Z_framewise_ssim.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr_service_acceptance_20260618T091617Z_framewise_ssim.json:1)

结果：

- `ssim_mean = 0.613979`
- `ssim_min = 0.580767`

结论：

- 这轮主要问题不是服务链断掉，而是请求用了 `20 step`
- 它和 `Phase C` 的 `50 step` 基线不公平比较，质量当然会严重偏低

### 5.2 第二轮真实服务验收：50 step，服务已正常，但仍低于基线

产物：

- [vividvr_service_acceptance_50step_20260618T094535Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr_service_acceptance_50step_20260618T094535Z.json:1)
- [vividvr_service_acceptance_50step_20260618T094535Z_framewise_ssim.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr_service_acceptance_50step_20260618T094535Z_framewise_ssim.json:1)

结果：

- `ssim_mean = 0.945633`
- `ssim_min = 0.930447`
- `pass_compare = true`

结论：

- 服务推理主链已经回到“正常质量簇”
- 但仍低于历史 `Phase C` 基线 `0.967716 / 0.947346`

### 5.3 第三轮：只做 reference path 对齐，没有改善

产物：

- [vividvr-service-refalign-50step-20260618T154459Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-refalign-50step-20260618T154459Z.json:1)
- [vividvr-service-refalign-50step-20260618T154459Z_framewise_ssim.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-refalign-50step-20260618T154459Z_framewise_ssim.json:1)

结果：

- `ssim_mean = 0.945633`
- `ssim_min = 0.930447`

结论：

- 只把 `reference_video_path` 切到原版输出视频，不足以改变最终落盘编码结果
- 说明问题不只在 reference 选择

### 5.4 第四轮：当前工作区里的 encodealign 尝试，有提升但仍未打平

当前目录里已经有一轮 `encodealign` 真实验收产物：

- [vividvr-service-encodealign-50step-20260618T163306Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-encodealign-50step-20260618T163306Z.json:1)
- [vividvr-service-encodealign-50step-20260618T163306Z_framewise_ssim.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/vividvr-service-encodealign-50step-20260618T163306Z_framewise_ssim.json:1)

结果：

- `ssim_mean = 0.9494933450343498`
- `ssim_min = 0.9341837736255159`
- `pass_compare = true`

profile 对照：

- `reference_profile.bit_rate = 17019260`
- `service_output_profile.bit_rate = 2752702`
- `download_profile.bit_rate = 2752702`

结论：

- 编码链路已经有部分改善，SSIM 从 `0.9456` 提升到了 `0.9495`
- 但最终输出码率仍明显低于原版 reference 的 `17.02 Mbps`
- 因此这轮也还不能宣称“编码参数已经完全对齐原版”

---

## 6. 当前最核心的技术判断

截至这份文档为止，最重要的判断是：

1. 当前 `VividVR` 服务功能本身已经打通。
2. 当前 `50 step / seed=42 / 同一输入 / 同一 prompt` 下，主链质量已经不再是“明显跑歪”的状态。
3. 当前残余差距最可疑的方向仍然是**最终视频保存/编码链路**，而不是先下结论说 `sglang` 集成后的 `Vivid-VR` 已发生显著语义漂移。
4. 但是，编码链路即使做了第一轮 `encodealign`，输出码率仍只有 `2.75 Mbps`，距离 reference `17.02 Mbps` 还很远，所以问题仍未解决。

也就是说，下一步不应该重新发散去改服务请求协议，也不应该先去做新的模型语义大排查。  
当前任务已经收口为：

**把服务输出视频的编码参数继续对齐原版 `Vivid-VR` 输出，然后按固定 reference + 灰度 SSIM 再跑一次正式验收。**

---

## 7. 当前工作区的未提交改动

当前工作区是脏的，不要误以为这些改动已经提交。  
`git diff --stat` 显示当前主要未提交改动是：

- `python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/utils.py`
- `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`
- `python/sglang/multimodal_gen/runtime/videoedit/io.py`
- `python/sglang/multimodal_gen/test/unit/test_entrypoints_utils.py`
- `Vivid_Acceptance/` 目录下还有新的未跟踪验收产物

其中最关键的当前未提交逻辑有两块：

### 7.1 Reference/encoding 路径分发

主要在：

- [utils.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:75)

当前新增了这些概念：

- `resolve_video_reference_path(...)`
- `resolve_video_encoding_mode(...)`
- `resolve_video_encoding_quality(...)`
- `VIDEO_ENCODING_MODE_REFERENCE_PROFILE`
- `VIDEO_ENCODING_MODE_VIVIDVR_ORIGINAL`

### 7.2 `save_video_frames()` 的编码写出方式调整

主要在：

- [io.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/videoedit/io.py:14)

当前改动把原来的 `imageio.mimsave(...)` 改成了 `imageio.get_writer(...).append_data(...)` 风格，这正是编码对齐尝试的一部分。

注意：

- 不要直接丢弃这些改动
- 也不要默认相信“这些改动已经足够了”
- 当前事实是：它们支撑出了 `encodealign` 那轮 `0.9495` 的结果，但仍没达到正式目标

---

## 8. 下一位接手者的明确任务

下一步任务不是泛泛地“继续优化质量”，而是下面这件非常具体的事：

### 8.1 主任务

继续把服务输出视频的编码参数对齐到原版 `Vivid-VR` 输出风格，目标至少包括：

- `codec_name`
- `pix_fmt`
- `profile`
- `bit_rate`
- 可能还包括影响压缩质量的 writer/ffmpeg 选项

重点排查方向：

1. `save_video_frames()` 现在到底用了哪些默认 ffmpeg/imageio 写出参数。
2. 为什么 reference profile 已经是 `17.02 Mbps`，服务输出仍落在 `2.75 Mbps`。
3. 当前 `quality=8` 是否只是 imageio 抽象层面的“质量提示”，并不能等价复现原版 Vivid-VR 的真实编码参数。
4. 是否还存在某个更下游的封装，把最终码率重新压低了。

### 8.2 完成后的正式验收要求

修完编码参数后，重新跑一次固定口径的真实服务验收：

- 输入视频固定：
  - `/home/zhiheng/Vivid-VR/input/720p/test_video_960x720.mp4`
- prompt 固定：
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- reference 固定：
  - `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`
- 推理参数固定：
  - `num_inference_steps = 50`
  - `seed = 42`
- 指标口径固定：
  - 当前仓库 `compare.py` 的灰度 SSIM

### 8.3 目标判断标准

最低要求：

- 服务正常提交、运行、完成、下载
- `pass_compare = true`
- 输出 profile 明显比当前 `2.75 Mbps` 更接近 reference profile

更理想的目标：

- 灰度 `ssim_mean` 尽量逼近历史 `Phase C` 基线 `0.967716`
- 灰度 `ssim_min` 尽量逼近历史 `Phase C` 基线 `0.947346`

---

## 9. 建议的下一步执行顺序

建议严格按这个顺序做，不要跳步：

1. 先检查当前未提交的 `encodealign` 改动到底如何控制写出参数。
2. 用 `ffprobe` 对比：
   - 原版 reference 输出视频
   - 当前 `encodealign` 服务输出视频
3. 只改编码写出相关逻辑，不碰服务请求 schema、pipeline 语义链路、compare 指标实现。
4. 跑一轮新的真实服务验收。
5. 保存：
   - summary JSON
   - framewise SSIM JSON
   - 输出视频
   - 下载视频
   - `ffprobe` profile 对照结果
6. 再判断是否已经达到“正式验收通过”的结论。

---

## 10. 一句话结论

当前项目状态不是“服务还没做完”，而是：

**服务已经做完并跑通了，剩下的关键收尾是把输出视频编码参数继续对齐原版 `Vivid-VR`，再用原版 reference + 灰度 SSIM 完成最后一轮正式推理验收。**
