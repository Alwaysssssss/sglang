# VideoEdit 前后处理源码对齐差异

本文继续比较 SGLang `WanVideoEditPipeline` 与
`/mnt/shanhai-ai/shanhai-workspace/zhouhao6/VideoEdit-diffusers` 的非推理流程。
范围只覆盖输入读取、全局预处理、窗口物化、条件输入构造、窗口提交、贴回和保存输出；
不比较 denoising/Transformer 推理数值。

对比方式：主线程源码静态阅读，并用 4 个 subagent 分片核查
全局预处理、窗口物化、后处理输出、CLI/API 参数映射。未运行测试或推理。

## 结论

默认配置下两边没有 1:1 对齐。最主要的不对齐在窗口物化和输出阶段：

- SGLang 默认 `overlap_commit_mode="weighted"`、`tail_padding_mode="reflect"`，
  VideoEdit-diffusers 是 native skip + reverse mirror。
- 即使把 SGLang 设为 `native_skip`，SGLang 也只用上一窗口生成结果替换
  新窗口的 `frames[0]`；VideoEdit-diffusers 会替换前 `overlap` 帧。
- 有参考图时，VideoEdit-diffusers 当前源码保留 prepended reference frame；
  SGLang 默认会在最终输出和 crop sidecar 中丢弃第 0 帧。
- SGLang 没有 VideoEdit-diffusers 默认保存的 `_color.mp4` AdaIN 颜色校正输出。
- 普通同尺寸 video mask 的 eager 全局预处理较接近：mask expand、bbox、crop、
  16 对齐、OpenCV resize 和 tensor packing 主体是对齐的。

## 具体不对齐点

| # | 范围 | VideoEdit-diffusers 当前源码 | SGLang 当前源码 | 影响 |
| --- | --- | --- | --- | --- |
| 1 | CLI 默认输入 | `infer.py:104-119` 有 demo `video_path/mask_path/img_path/output_dir/output_name` 默认值 | `runtime/videoedit/cli.py:33-45` 要求显式传 model/prompt/video/mask/output，reference 默认空 | 两边直接跑默认值没有可比性 |
| 2 | `num_frames` 默认 | `infer.py:168` 默认为 `None`，`utils/preprocess.py:440-447` 读取全视频后取 `min(video, mask)` | CLI `cli.py:45` 默认 81；API `protocol.py:159` 默认 `-1`，再由 `preprocess.py:88-111` 解析为 `min(video, mask)` | SGLang CLI 默认会截断 81 帧；要对齐 VideoEdit 全视频需传 `--num-frames -1` |
| 3 | 视频读取容错 | `utils/preprocess.py:24-47` 直接 `VideoCapture` 读帧，FPS 不兜底 | `preprocess.py:24-47` 检查 `cap.isOpened()`，FPS falsey 时用 24.0；还支持 frame cache | 坏路径、空 FPS metadata、缓存输入时行为不同 |
| 4 | mask 输入格式 | `utils/preprocess.py:440-441` 把 mask 当视频读取 | `mask_io.py:22-33` 自动检测 video/NumPy/COCO JSON，`mask_io.py:267-290` 统一二值化 | `.npy/.npz/COCO` 是 SGLang-only 能力；同一语义 mask 可能得到不同 bbox |
| 5 | mask 尺寸 | VideoEdit-diffusers 不把 mask resize 到 video size 后再进入全局预处理 | `prepare_global_inputs()` 调 `load_mask_frames(..., target_size=original_frames[0].size)`，`mask_io.py:36-39` 最近邻 resize | 不同分辨率 mask 下几何不对齐；同尺寸 video mask 才接近 |
| 6 | 非视频 mask 数值语义 | Video mask 经 PIL 灰度后按 `gray > 0.5` 扩展 | `mask_io.py:68-77` 对 integer mask 用 `> 0`，float mask 用 `> 0.5` | 低正整数 mask 在 SGLang 会成为前景，Video 视频灰度路径不会 |
| 7 | reference image resize | `utils/preprocess.py:452-461` prepend reference，并在尺寸不同时显式 `Image.BICUBIC` resize | `preprocess.py:479-485`、`frame_provider.py:117-122` prepend reference，但 resize 未显式指定 resampling | reference 尺寸不一致时依赖 Pillow 默认值，像素可能不同 |
| 8 | reference 输出帧 | `infer.py:390-392` 删除 reference 的代码被注释 | `wan_videoedit_pipeline.py:514-515`、`557-558` 在 `drop_reference_frame=True` 时删除第 0 帧；CLI 默认 true，API 在有 reference 时默认 true | 有 reference 时，VideoEdit 输出 raw+1 帧，SGLang 默认输出 raw 帧 |
| 9 | stream/eager 路径 | `infer.py:240-251` 只走 eager `prepare_global_inputs()`，完整加载帧列表 | SGLang 默认 `decode_mode="stream"`；`wan_videoedit_pipeline.py:172-205` 先扫描 bbox，再由 `WindowFrameProvider` 懒解码 | 正常文件几何应接近，但坏 frame count metadata、decoder 行为和缓存路径可能不同；源码路径也不同 |
| 10 | window 默认参数 | `infer.py:137-145` 默认 `infer_len=81, overlap=9`，stride 固定 `infer_len-overlap` | sampling 默认 `overlap=10`、CLI 默认 9；`configs/sample/videoedit_wan.py:77-78` 默认 `weighted/reflect` | 默认窗口 start 集合和窗口内容不一致 |
| 11 | tail padding | `utils/preprocess.py:567-578` reverse mirror，越界第一帧重复最后一帧 `N-1` | `windowing.py:4-20` 同时有 `reflect` 和 `native_reverse_mirror`，默认 `reflect` 不重复边界 | 尾窗口 conditioning frames/masks 不一致；需设 `tail_padding_mode="native_reverse_mirror"` |
| 12 | previous-window reference | `infer.py:301-310` 取上一窗口 `stride:stride+overlap`；`utils/preprocess.py:589-597` 替换新窗口前 `overlap` 帧并清零这些 mask | `wan_videoedit_pipeline.py:286-308` 只替换 `frames[0]`；`310-326` 可清零多个 mask | 即使 `native_skip`，SGLang 的 `1..overlap-1` 仍是源帧加黑 mask，VideoEdit 是上一窗口生成帧 |
| 13 | 非首窗口 frame 0 | `prepare_window_inputs()` 内部每个窗口都会通过 `create_masked_video/create_mask_video` 保留 local frame 0 并置黑 mask | `VideoEditConditionEncodingStage` 调 `prepare_window_inputs(..., preserve_first_frame=(runtime_window_index in (None, 0)))` | 当 `overlap=0` 或 reference 替换失败时，非首窗口 frame 0 预处理语义不同 |
| 14 | weighted commit | VideoEdit 只做 native skip，后续窗口提交 `local_idx >= overlap`，见 `infer.py:378-386` | `windowing.py:70-97` 的 weighted 模式使用 `[start-1] + range(...)`，`wan_videoedit_pipeline.py:369-427` 按权重累积 | weighted 是 SGLang-only 行为，不能期待帧级对齐 |
| 15 | repaired context | VideoEdit 只有显式 overlap reference | `wan_videoedit_pipeline.py:254-275` 可用累计输出替换任意已修复 global frame | 开启 `use_repaired_context` 后 conditioning 历史不同 |
| 16 | `chunks` | `infer.py:86-89`、`273` 支持只跑前 N 个 window chunks | `wan_videoedit_pipeline.py:591-628` 始终运行全部 window specs | VideoEdit 的 partial-window debug run 无 SGLang 参数等价物 |
| 17 | mask tensor downsample | `utils/preprocess.py:619-635` hardcode `F.interpolate(..., mode="nearest")` 后 `<0.5` invert/pack | `preprocess.py:549-577` 支持 `nearest` 或 `nearest-exact`，默认 `nearest` | 默认对齐；改成 `nearest-exact` 会破坏当前 VideoEdit 源码 parity |
| 18 | CLIP 图像预处理 | `infer.py:159-160` 默认 `clip_preprocess="diffuser"`；`pipeline_wan_edit.py:729-732` 可选 diffuser 或 diffsynth | `VideoEditImageEncodingStage` 只实现 DiffSynth 风格 resize/interpolate/mean-std 路径，见 `pipelines_core/stages/model_specific_stages/videoedit_wan.py:287-312` | 默认 reference CLI 与 SGLang 的 image conditioning 不对齐；要接近需在 reference 侧传 `--clip_preprocess diffsynth` |
| 19 | crop-only 输出 | `infer.py:174-179` 默认保存 pasted、crop-only、color 三个输出 | `configs/sample/videoedit_wan.py:68-70` 默认只启用主 paste-back，不保存 crop sidecar | 默认 artifact 集合不同 |
| 20 | AdaIN/color 输出 | `utils/postprocess.py:31-69` 有 `_adain_boundary`，`infer.py:432-446` 默认保存 `_color.mp4` | `postprocess.py:45-47` 和 `frame_provider.py:274-281` 接收但删除/忽略 `adain_boundary_dilate` | SGLang 无法生成 VideoEdit 的 color-corrected 输出 |
| 21 | regular paste-back | `utils/postprocess.py:85-134` resize generated crop、mask 二值化、feather blend、写回 bbox | `postprocess.py:37-88` 主流程基本一致，但无 AdaIN，`zip(..., strict=False)` | 普通 pasted result 更接近；color result 不对齐 |
| 22 | 保存编码 | `infer.py:80-83` MoviePy `libx264` 固定 `bitrate="10M"`，使用源 fps | `gpu_worker.py:270-292` 交给 shared saver；`entrypoints/utils.py:490-504` 优先 `save_video_frames_like_reference()`，`ffmpeg_io.py:129-166` 继承参考视频 codec/pix_fmt/bitrate 等 | 输出文件字节、codec、bitrate、pix_fmt、29.97 类 fps 都可能不同 |
| 23 | progress/metadata/storage | VideoEdit 只打印本地保存路径 | `progress.py:10-28` 写 progress JSON；`wan_videoedit_pipeline.py:444-498` 写 `.videoedit.json`；API 支持 URL/MinIO 下载上传 | API 可见 artifact 与本地文件生命周期不同，不属于像素后处理对齐 |

## 当前对齐或基本对齐的部分

- 同尺寸 video mask、eager 路径下，mask expand 的主逻辑对齐：
  first frame mask forced zero，后续灰度阈值、dilate、scale，见两边
  `expand_mask_frames/expand_mask_frame`。
- 全局 bbox 主流程对齐：基于扩展 mask 的 union bbox、`bbox_padding`、
  `bbox_expand_scale`、小 bbox 扩到短边约 480、crop、16 对齐、OpenCV resize。
  SGLang 对极小/线状 bbox 用 `short_side=max(1, ...)`，VideoEdit 直接除以
  `short_side`，这是边界输入差异。
- 当物化出的 window PIL frames/masks 完全一致，tensor 构造基本对齐：
  `frames_to_tensor` 归一化、首 mask repeat 4 帧、1/8 nearest downsample、`<0.5`
  invert、view/transpose packing。
- VAE decode 后转 PIL 的边界基本对齐：VideoEdit 在 `infer.py:362-374` clamp
  到 `[0,1]` 后乘 255 转 `uint8`；SGLang 在
  `pipelines_core/stages/model_specific_stages/videoedit_wan.py:711-723` 做等价流程。

## 接近 1:1 非推理流程的最低参数约束

若只想尽量贴近当前 VideoEdit-diffusers `infer.py` 的前后处理源码，SGLang 至少需要：

- CLI 使用 `--num-frames -1`，或 API 使用默认 `num_frames=-1` 后让入口解析为
  `min(video, mask)`。
- 使用同尺寸 video mask，不使用 SGLang-only 的 `.npy/.npz/COCO` mask。
- `--decode-mode eager`，避免默认 stream 源码路径。
- `--overlap 9 --overlap-commit-mode native_skip --tail-padding-mode native_reverse_mirror`。
- `--mask-downsample-mode nearest`。
- reference 对齐时谨慎设置 `--no-drop-reference-frame`；否则输出帧数与 VideoEdit 当前源码不同。
- reference 侧若要对齐 SGLang 的 CLIP conditioning，传 `--clip_preprocess diffsynth`。
- 如需 crop artifact，SGLang 显式打开 `--save-crop-only`；但 `_color.mp4` AdaIN 输出当前无等价实现。

即便设置上述参数，仍有一个关键未对齐点：SGLang `native_skip` 只替换新窗口
`frames[0]`，VideoEdit-diffusers 替换前 `overlap` 帧。因此 overlap 窗口的
conditioning frames 仍无法做到严格源码级一致。
