# VSR 的 SGLang PyTorch 2.10/cu126 环境

2026-09-20，按用户要求将 VSR 的默认 SGLang 运行环境切换到 PyTorch 2.10/cu126。
状态：**切换完成并通过验证**。新环境导入、原生 pipeline、6项回归测试、61项纯函数对拍、逐阶段张量以及全矩阵9组/649帧均通过。

## 使用

在仓库根目录：

```bash
source docs_always/add_new_mode/add_vsr/scripts/activate_runtime210.sh
# 本任务使用物理 GPU7；若当前会话之前设置过其他卡，显式覆盖。
export CUDA_VISIBLE_DEVICES=7
"$VE_SGLANG_PYTHON" -m sglang.multimodal_gen.runtime.vsr.cli restore \
  --via-pipeline \
  --checkpoint_dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300 \
  --wan_root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers \
  --input /mnt/shanhai-ai/shanhai-workspace/zhouhao6/vsr/input/input.mp4 \
  --output "$PWD/output_results/vsr/sglang_stream210.mp4" \
  --target_resolution 3840x2160 --dtype bfloat16 --color_ref global
```

解释器：`output_results/vsr/migration_env210/bin/python`。
也可用 `bash docs_always/add_new_mode/add_vsr/scripts/python210.sh <Python参数>`，无需激活 shell。
`3840x2160` 是 H×W，输出为竖屏4K。

## 依赖与兼容性

| 组件 | 生效版本/位置 |
| --- | --- |
| Python | 3.11.12 |
| torch | 2.10.0+cu126 |
| torchvision | 0.25.0+cu126 |
| diffusers | 0.37.0 |
| cuDNN | 91002 |
| SGLang | 当前仓库源码 |
| sgl-kernel | 0.4.1，使用 `output_results/vsr/kernel210-runtime` 的 torch2.10兼容构建 |
| numpy / decord | 2.4.6 / 0.6.0 |
| imageio / imageio-ffmpeg | 2.36.0 / 0.6.0 |

环境通过 `vsr_runtime.pth` 按顺序复用已有目录：torch2.10/cu126、匹配的 kernel、先前迁移环境中的补充依赖、已有 SGLang 依赖及当前仓库源码。
这是本工作区的 VSR 专用环境，依赖这些目录持续存在，不是可直接搬移的独立发行包。

仓库通用 `python/pyproject.toml` 仍声明 torch2.9.1；本次为用户指定的 VSR 环境覆盖，没有改变整个仓库的依赖约束。
kernel构建记录在 `output_results/vsr/kernel210_wheel_provenance.json`，其 FA3/FlashMLA 被关闭；VSR 第一阶段使用 diffusers 默认 torch SDPA，与验收设定一致。
本次验证覆盖 VSR，不代表其他 SGLang 模型或所有可选 CUDA 扩展也已通过。
旧 `migration_env`（torch2.9.1/cu128）保留，可复现历史验收结果。

重建本地环境入口（CPython3.11，要求上述依赖目录已存在）：

```bash
python docs_always/add_new_mode/add_vsr/scripts/setup_runtime210.py
```

## 验证与产物

产物目录：`output_results/vsr/runtime210_validation/`。

- `native.log` / `native.mp4`：实际经过 SGLang 调度器和 WanVSRPipeline 完成推理。
- `tests.log`：请求参数、精度一致性和无效张量检查。
- `benchmark.json`：2次预热、6次计时，33×320×640真实 tile 中位数 **1.312秒**；旧环境为3.277秒，耗时约降低60%，约2.50倍吞吐。该数字不是整视频加速比。
- `matrix.log` / `matrix/results.json`：9配置逐帧与mp4精确对照，参考为原始swiftvr环境产物。

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 "$VE_SGLANG_PYTHON" \
  docs_always/add_new_mode/add_vsr/scripts/run_acceptance.py \
  --output-dir output_results/vsr/runtime210_validation/matrix \
  --reference-environment swiftvr
```

## 最终验收结果

机器报告：`output_results/vsr/runtime210_validation/report.json`（`pass=true`）。

| 配置 | 帧数 | 与原始 swiftvr 的编码前像素 | mp4文件 |
| --- | ---: | --- | --- |
| A1 | 53 | 完全一致 | SHA256一致 |
| A2 | 53 | 完全一致 | SHA256一致 |
| A3 | 53 | 完全一致 | SHA256一致 |
| B1 | 54 | 完全一致 | SHA256一致 |
| B2 | 54 | 完全一致 | SHA256一致 |
| B3 | 54 | 完全一致 | SHA256一致 |
| C1 | 64 | 完全一致 | SHA256一致 |
| C2 | 200 | 完全一致 | SHA256一致 |
| C3 | 64 | 完全一致 | SHA256一致 |

全矩阵无失败帧，未放宽原验收门限。当前运行栈消除了旧torch2.9.1环境下的数值漂移；旧`ε_torch`数据保留为历史记录。
单窗口另做了全部采样点验证：6类中间张量加1个编码前帧块，共7个文件逐元素完全相同，shape/dtype/设备/有限值及颜色统计也相同。原生pipeline产物与独立运行的原始VSR产物逐字节一致。

实际环境版本、模块导入路径、kernel wheel SHA256、GPU和Git状态见 `environment.json`。
主交付视频：`output_results/vsr/runtime210_validation/matrix/A1.mp4`。
本次全矩阵从16:04:03到16:29:12，约25分09秒，包含加载、dump和精确比对；这不是纯推理性能基准。

汇总验证可复现命令：

```bash
"$VE_SGLANG_PYTHON" docs_always/add_new_mode/add_vsr/scripts/finish_runtime210.py
```
