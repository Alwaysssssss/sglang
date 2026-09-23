# VSR 原始实现与单卡、双卡、四卡测试命令

以下保留原始 SwiftVR 单卡与当前 SGLang VSR 单卡、双卡、四卡的单次 CLI 命令。正式性能测试已按用户最终要求完成单卡、双卡、三卡，四卡取消；结果见 [性能报告](performance_20260923.md)。当前实现开启全部 7 个已接入 CLI 的加速选项；多卡使用空间 tile 并行。

模型卸载默认关闭；需要降低显存占用时，见 [模型与 DiT 卸载开关及实测](offload.md)。

## 执行命令

### 进程内预热的正式性能测试（2026-09-23）

最终测试范围按用户要求改为单卡、双卡、三卡。使用 GPU 3 跑当前单卡、GPU 6/7 跑当前双卡，两组同时运行；全部成功退出后使用 GPU 3/6/7 跑当前三卡，最后在 GPU 3 跑原始基线。GPU 2 测试期间出现外部计算任务，因此避开该卡；最初的 GPU 2 基线仅保留为受干扰记录，四卡测试已按用户要求停止。

```bash
bash docs_always/add_new_mode/add_vsr/scripts/run_benchmark_all.sh
```

每组完整视频预热一次，再收集 3 次无新增编译的完整视频计时，最终取中位数。全部 7 个 CLI 加速选项已开启。结果和日志在 `output_results/vsr/all_optimizations_20260923/`，各组 `results.json` 保存环境、参数、GPU 状态、预热和计时记录。脚本也接受第一个位置参数指定新的输出目录，重复执行时应换目录以保留历史结果。

本机没有 `/usr/bin/time`，因此正式测试使用 `benchmark_all.py` 的 Python 计时，不依赖下面冷启动示例中的外部 `time` 程序。并发测试共享 CPU、内存带宽和存储；此运行方式遵照本次指定调度，结果应连同该条件报告。

### 单次 CLI 运行（包含冷启动）

在 `sglang` 仓库根目录用 Bash 执行。GPU 编号按实际空闲卡调整，四组任务顺序运行。

```bash
cd /mnt/shanhai-ai/shanhai-workspace/zhouhao6/sglang

export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=4

REF_PY=/mnt/shanhai-ai/envs/conda/envs/swiftvr/bin/python
OPT_PY="$PWD/output_results/vsr/migration_env210/bin/python"
OUT="$PWD/output_results/vsr/all_optimizations_1_2_4gpu"
mkdir -p "$OUT"

COMMON=(
  --input /mnt/shanhai-ai/shanhai-workspace/zhouhao6/vsr/input/input.mp4
  --checkpoint_dir /mnt/shanhai-ai/shanhai-workspace/zhouyang/SwiftVR/outputs/best/checkpoint-1300
  --wan_root /mnt/shanhai-ai/liuh/Wan2.2-TI2V-5B-Diffusers
  --target_resolution 3840x2160
  --tile_t 33 --tile_h 320 --tile_w 640
  --temporal_overlap 5 --spatial_overlap 32
  --dtype bfloat16
  --color_ref global --color_ref_samples 64
  --read_queue 2 --write_queue 4 --crf 5
)

ACCEL=(
  --cudnn-benchmark
  --channels-last-3d
  --compile-encoder
  --compile-decoder
  --decoder-implicit-padding
  --cache-dit-condition
  --gpu-postprocess
)

# 原始实现：单卡基线
CUDA_VISIBLE_DEVICES=6 \
/usr/bin/time -f 'wall_seconds=%e' -o "$OUT/reference.time" \
  "$REF_PY" ../vsr/run_inference_stream.py \
  "${COMMON[@]}" --output "$OUT/reference.mp4" \
  > "$OUT/reference.log" 2>&1

# 当前实现：单卡，全部加速
CUDA_VISIBLE_DEVICES=6 \
/usr/bin/time -f 'wall_seconds=%e' -o "$OUT/single.time" \
  "$OPT_PY" -m sglang.multimodal_gen.runtime.vsr.cli restore \
  "${COMMON[@]}" "${ACCEL[@]}" \
  --output "$OUT/single.mp4" \
  > "$OUT/single.log" 2>&1

# 当前实现：双卡，全部加速 + 空间 tile 并行
CUDA_VISIBLE_DEVICES=6,7 \
/usr/bin/time -f 'wall_seconds=%e' -o "$OUT/dual.time" \
  "$OPT_PY" -m sglang.multimodal_gen.runtime.vsr.cli restore \
  "${COMMON[@]}" "${ACCEL[@]}" \
  --tile-devices cuda:0 cuda:1 \
  --output "$OUT/dual.mp4" \
  > "$OUT/dual.log" 2>&1

# 当前实现：四卡，全部加速 + 空间 tile 并行
CUDA_VISIBLE_DEVICES=4,5,6,7 \
/usr/bin/time -f 'wall_seconds=%e' -o "$OUT/quad.time" \
  "$OPT_PY" -m sglang.multimodal_gen.runtime.vsr.cli restore \
  "${COMMON[@]}" "${ACCEL[@]}" \
  --tile-devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --output "$OUT/quad.mp4" \
  > "$OUT/quad.log" 2>&1
```

`3840x2160` 表示高 × 宽。四组保持输入、权重、tile、重叠、精度和编码参数一致。`--tile-devices` 的编号相对于 `CUDA_VISIBLE_DEVICES`；单卡不传此选项。

## 计时口径

上述 `.time` 是进程总耗时，包含模型加载、首次编译和完整视频读写；编译缓存状态也会影响耗时，不能直接将其解释为稳态推理性能。

正式比较稳态加速比，应在同一模型进程内预热后重复计时，排除模型加载、编译和预热，保留完整视频解码、推理、融合、颜色校正和编码耗时。计时前后应确认无新增编译。反复启动上述 CLI 不能替代进程内预热。

分别计算：

- 单卡加速比 = 原始单卡稳态耗时 / 当前单卡稳态耗时。
- 双卡加速比 = 原始单卡稳态耗时 / 当前双卡稳态耗时。
- 四卡加速比 = 原始单卡稳态耗时 / 当前四卡稳态耗时。

原始实现使用 `swiftvr` 环境，当前实现使用 `migration_env210` 环境，因此该比较包含运行环境差异；若需隔离代码优化收益，应补充同环境原始实现对照。

现有 `scripts/benchmark_parallel.py` 写死了单卡/双卡且未开启固定 DiT 条件缓存，不能直接覆盖本次全部加速的四组稳态比较。已有双卡报告也不能作为本次全部加速的结果，四卡伸缩性尚未实测。参见 [多卡并行记录](parallel_optimization.md) 和 [第四轮优化记录](optimization_round4.md)。
