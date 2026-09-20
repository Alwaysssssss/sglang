# VSR 验收工具

这套工具用于证明 SGLang 的 VSR 实现与参考实现一致。判据定义在
[`requirements.md`](../../../../../../docs_always/add_new_mode/add_vsr/requirements.md) §4，
**本目录不另立标准**。

参考实现位于只读仓库 `$VSR_REPO`，因此所有观察都通过 monkey-patch 完成，不修改其源码。

## 一个硬约束：生产者工具不依赖 `sglang`

`dump_baseline.py`、`encode_frames.py`、`dumps.py` **不导入 `sglang`**，可以直接按路径用任一环境的
解释器运行：

```bash
/path/to/some/python -B runtime/vsr/verify/dump_baseline.py ...
```

原因：参考侧运行在 conda `swiftvr` 环境里，那里**没有安装 sglang**。第一次实现时用
`python -m sglang...` 调用，在 swiftvr 侧直接以 `ModuleNotFoundError: No module named 'sglang'` 失败。
这条约束对 `compare_frames.py` / `compare_tensors.py` / `structural_check.py`（裁判工具）不适用——
它们只在一个环境里跑，可以用 `-m` 调用。

## 工具

### `dump_baseline.py` — 跑参考实现并 dump 中间量

把参考 CLI 交给它，`--` 之后原样透传，所以 dump 的那次运行和普通运行由完全相同的命令行驱动。

```bash
$PY -B runtime/vsr/verify/dump_baseline.py \
    --vsr-repo "$VSR_REPO" --dump-root /path/to/dump \
    --dump-frames [--dump-tiles none|first|all] [--dump-chunks] \
    -- --input in.mp4 --output out.mp4 --checkpoint_dir ... --wan_root ... \
       --target_resolution 3840x2160 --tile_h 320 --tile_w 640 --tile_t 33 \
       --color_ref global --read_queue 2
```

| 开关 | 作用 | 体积提示 |
| --- | --- | --- |
| （默认） | 只写 `manifest.json` | — |
| `--dump-frames` | 编码前 uint8 帧，按退休批次分文件写 `retired/part_NNN.pt` | 4K/53 帧约 `1.3 GB` |
| `--dump-tiles first\|all` | 每 tile 的 `window` / `latent` / `velocity` / `decoded` | 4K 下 `all` 会非常大 |
| `--dump-chunks` | 整 chunk 的 `chunk_input` / `spatial_fused` | 4K 下每 chunk `3.3 GB` |

采样点与 `requirements.md` §4.1.3 一一对应；`manifest.json` 记录形状、dtype、参考仓库的 git 状态与配置。

**dump 目录里的 `retired/` 已被端到端验证**：把这些帧重新编码，产物与参考实现自己写出的 mp4
逐字节相同。见 `requirements.md` §4.2.3。

### `compare_frames.py` — 帧层比对（算法对齐主判据）

```bash
python -m sglang.multimodal_gen.runtime.vsr.verify.compare_frames \
    --reference-dir DUMP_A --candidate-dir DUMP_B --report-json R.json
```

SSIM 复用 `runtime/videoedit/compare.py` 的 `_ssim`，保证与 mp4 层口径一致。
与 `compare_videos` 的唯一差别：这里的帧已经是 RGB（`to_uint8_hwc` 直接产出 `[T,H,W,C]`），
所以不做 BGR→RGB 转换。

形状不一致会直接抛错，**不会**像 `compare_videos` 那样静默 `cv2.resize`——结构一致性是
`structural_check.py` 的零容差职责。

### `compare_tensors.py` — 张量层比对

```bash
python -m sglang.multimodal_gen.runtime.vsr.verify.compare_tensors \
    --reference-dir DUMP_A --candidate-dir DUMP_B --report-json R.json \
    [--atol X --rtol Y]        # 省略则只测量、不判定
```

分两类报告，对应 `requirements.md` §4.1.1：

- **结构性**（shape、dtype、文件是否成对）——零容差，不一致即缺陷；
- **数值性**——`max_abs` / `mean_abs` / `rel_mean` / `rel_max`，后两者按参考张量的
  `|ref|.max()` 归一化。

**门限用 `rel_mean`，不要用 `rel_max`。** `rel_max` 由极少数离群元素支配：实测中 `velocity` 超过
动态范围 10% 的元素只占 `0.004%`，`decoded` 的最坏元素误差达 `1.47`（超出 `[-1,1]` 本身）。
以它设门限会宽到没有意义；它只作诊断记录。

刻意不报告逐元素 `|diff|/|ref|`：它被参考值接近 0 的元素支配，不反映漂移。

### `structural_check.py` — 结构前置门（§4.3）

```bash
python -m sglang.multimodal_gen.runtime.vsr.verify.structural_check \
    --reference A.mp4 --candidate B.mp4 --report-json R.json
```

检查分辨率、逐帧几何、帧数、帧率、帧序。**必须在运行比较器之前独立完成**，不能交给
`videoedit.compare`：

- 形状不一致时它会静默 `cv2.resize(candidate, ...)`，把错误分辨率抹成高 SSIM；
- 它用 `cv2.VideoCapture` 读帧，不看容器帧率，fps 不一致会静默通过；
- 帧数差超限时它抛 `ValueError`（且 `main()` 未捕获），与"门限不过"无法区分。

报告里 `resolution` 是 **W×H**、`frame_geometry_uniform` 是 **H×W**，两者都带 `units` 字段——
这正是管道参数（H×W）最容易写反的地方。

### `encode_frames.py` — 编码器确定性检查（§4.2.3）

```bash
/path/to/python -B runtime/vsr/verify/encode_frames.py \
    --dump-dir DUMP --output out.mp4 --fps 25 --crf 5
```

用**参考实现自己的 writer**（`infer.utils.video_io.write_video`）把帧 dump 重新编码，在两端各跑一次
并比对 md5。若相同，则"mp4 差异 ⇒ 帧差异"成立，§4.2.2 的 mp4 门限才可解释。

## 配置扫描

```bash
bash runtime/vsr/verify/sweep_eps_torch.sh <GPU> [配置过滤]     # 过滤如 "A3,C2,C3"
SEQUENTIAL=1 ...                                                # 串行，省显存
```

对 `requirements.md` §3.2 的每个配置在两个环境下各跑一次参考实现，再两两比对，
产出结构/帧层/mp4 层三份报告和一张汇总表。

**A1 排在最前是当对照组**：它的数值已由 M1 独立测得，若扫描复现不出来，说明扫描本身有问题。

`ε_torch` 是**配置相关**的（见 `requirements.md` §4.1.2），所以每个配置都要各自测量，
不能用一个数字外推。

## 运维注意事项

**GPU 争用是常态。** 这些卡是共享的，邻居进程会让显存读数在几分钟内从 60 GiB 空闲掉到 0。
脚本会在启动前等待 `GPU_NEED_MIB`（默认 20 GiB）空闲、并在命中 `OutOfMemoryError` 时重试
（`RETRIES`，默认 4 次）。2026-09-18 的首次扫描就因此损失了 A3/C2/C3 三个配置。

**不要用 `cmd; echo "... exit=$?"`。** 如果 `$?` 和命令替换出现在同一个词里，取到的是替换的
状态而不是命令的。首次扫描的崩溃进程因此报了 `exit=0`。现在的写法是先存 `st=$?` 再打印。

**生产者工具必须能在无 `sglang` 的环境里运行**，见上文。

## 产物位置

```
output_results/vsr/
├── EPS_<配置>_<环境>.mp4        # 扫描产物
├── EPS_<配置>_<环境>.log
├── dumps/EPS_<配置>_<环境>/     # 帧与张量 dump
└── reports/                     # JSON + 文本报告
```
