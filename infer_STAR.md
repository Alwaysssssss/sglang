# STAR Inference Commands

本文档给出两套命令：

1. 原版 `STAR_mg` 的参考推理命令
2. `sglang` 集成版 STAR 的推理命令

目标是让两边都在同一个固定 case、同一组采样参数下运行，方便做主观对比和验收。

## 1. 固定测试 case

统一使用 phase5/phase6/phase7 一直在用的 reference case：

- prompt: `023_klingai_reedit.txt`
- condition video: `023_klingai_reedit.mp4`
- seed: `1234`
- resolution: `480x720`
- sampling num frames: `7`
- output video frames: `25`
- output fps: `8`
- num inference steps: `50`
- guidance scale: `6.0`

为了让原版 STAR 只跑这一条 case，可以先把测试资产复制到 `sglang` 仓库内：

```bash
export STAR_CASE_DIR=/sgl-workspace/sglang/test_assets/star_case_023
mkdir -p "${STAR_CASE_DIR}/text" "${STAR_CASE_DIR}/lq"

cp /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  "${STAR_CASE_DIR}/text/023_klingai_reedit.txt"
cp /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  "${STAR_CASE_DIR}/lq/023_klingai_reedit.mp4"
```

不过如果你只是想直接复现实验，不想额外准备 `test_assets`，后面的 `sglang` 命令也都直接使用原始输入路径：

- prompt: `/sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt`
- condition video: `/sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4`

## 2. 原版 STAR 参考命令

原版 STAR 仍然从 `STAR_mg` 仓库运行。下面这组命令是自包含的，会先准备单-case 数据目录，再只读取这一条 case：

```bash
export STAR_CASE_DIR=/sgl-workspace/sglang/test_assets/star_case_023
rm -rf "${STAR_CASE_DIR}"
mkdir -p "${STAR_CASE_DIR}/text" "${STAR_CASE_DIR}/lq"

cp /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  "${STAR_CASE_DIR}/text/023_klingai_reedit.txt"
cp /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  "${STAR_CASE_DIR}/lq/023_klingai_reedit.mp4"

cd /sgl-workspace/STAR_mg/cogvideox-based/sat

export STAR_COG_TEST_DATA_DIR="${STAR_CASE_DIR}"
export STAR_COG_OUTPUT_DIR=/sgl-workspace/sglang/outputs/star_original_case023

CUDA_VISIBLE_DEVICES=0 bash inference_sr.sh
```

运行完成后，原版 STAR 的候选视频一般会落在：

```bash
/sgl-workspace/sglang/outputs/star_original_case023/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4
```

说明：

- 原版 `sample_sr.py` 会从数据目录枚举 case，所以这里用单-case 数据目录来保证口径一致。
- 原版 reference 路径如果你已经提前跑过，也可以直接复用已有的：

```bash
/sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4
```

## 3. SGLang 集成版 STAR

### 3.1 一次性准备本地模型目录

如果你希望 `sglang` 侧在运行时完全不依赖 `STAR_mg` 仓库代码，建议先把转换后的模型目录放到 `sglang` 仓库内：

```bash
mkdir -p /sgl-workspace/sglang/model_artifacts
```

如果这个目录还不存在，可以用下面命令一次性生成：

```bash
python -m sglang.multimodal_gen.tools.convert_star_cogvideox_sr \
  --src-transformer /sgl-workspace/STAR_mg/pretrained_weight/CogVideoX-5B-based/1/mp_rank_00_model_states.pt \
  --src-vae /sgl-workspace/STAR_mg/pretrained_weight/cogvideox/vae/3d-vae.pt \
  --src-text-encoder /sgl-workspace/STAR_mg/pretrained_weight/cogvideox/t5-v1_1-xxl \
  --src-tokenizer /sgl-workspace/STAR_mg/pretrained_weight/cogvideox/t5-v1_1-xxl \
  --src-config /sgl-workspace/STAR_mg/cogvideox-based/sat/configs/cogvideox_5b/cogvideox_5b_infer_sr.yaml \
  --output-dir /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --overwrite
```

完成这一步之后，`sglang` 运行时不再需要 `STAR_mg` 仓库代码路径；当前 STAR VAE 所需的 SAT 实现已经 vendored 到 `sglang` 仓库内部。

如果你还要跑 `FP8` 命令，再补一步把现有 `FP8` transformer 导出复制到本地模型目录：

```bash
mkdir -p /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128
cp -a /sgl-workspace/STAR_mg/pretrained_weight/sglang_star_cogvideox_sr/transformer-fp8-block128/. \
  /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128/
```

### 3.2 Exact 对齐推理命令

如果你要和原版 STAR 做“同标准”的主观/算法对齐，优先用 exact 路线，不要先上 FP8。  
这条命令优先保证稳定可跑，不强调速度：

```bash
python -m sglang.multimodal_gen.test.manual.run_star_cogvideox_sr_smoke \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt "$(cat /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt)" \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_sglang_exact_case023 \
  --output-file-name candidate.mp4 \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --dit-cpu-offload \
  --text-encoder-cpu-offload
```

这个命令的输出主要看：

- candidate: `/sgl-workspace/sglang/outputs/star_sglang_exact_case023/candidate.mp4`
- summary: `/sgl-workspace/sglang/outputs/star_sglang_exact_case023/star_smoke_summary.json`

### 3.3 Exact 验收命令

如果你希望同时拿到分 stage 统计和 parity 报告，用下面这条 exact profile 命令。  
这是当前仓库下已经实际跑通过的一条单卡 exact compile 命令：

```bash
python -m sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt-path /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_sglang_exact_case023_profile \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --warmup-runs 0 \
  --measured-runs 1
```

### 3.4 Phase7 FP8 验收命令

如果你要复现当前 phase7 的单卡量化验收口径，再额外跑一遍 FP8。  
这条命令依赖上面已经把 `transformer-fp8-block128` 复制到本地模型目录：

```bash
python -m sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --transformer-weights-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr/transformer-fp8-block128 \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt-path /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_sglang_fp8_case023_profile \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --condition-video-vae-peak-memory-mode text_encoder_and_transformer \
  --keep-transformer-gpu-resident-between-requests \
  --warmup-runs 1 \
  --measured-runs 1
```

### 3.5 历史 `1.4x` Exact Compile 复现实验

下面两条命令对应的是历史上“未量化、仅靠 `torch.compile` 就到 `1.4x` 左右”的两条 exact 路线。

它们和历史台账有两个区别：

1. 现在统一显式写 `--fps 8`，不再沿用旧脚本时期错误的 `24 fps` 输出。
2. 现在建议把模型目录放在 `sglang` 仓库内，例如 `${SGLANG_STAR_MODEL_DIR}`，从而直接在当前仓库下复现。

#### 3.5.1 `single_fa_compile_warm_v2` 复现命令

这是 phase6 时期的单卡 exact compile 路线，历史台账约为 `1.4153x`。

```bash
python -m sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt-path /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_repro_single_fa_compile_warm_v2_fps8 \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --warmup-runs 1 \
  --measured-runs 1 \
  --original-star-cold-e2e-s 302.334 \
  --original-star-warm-e2e-s 228.0
```

对应历史台账：

- run label: `single_fa_compile_warm_v2`
- 历史速度：`warm_e2e_speedup = 1.4153x`
- 台账位置：[compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:212)

#### 3.5.2 `single_fa_compile_fusedln_v2` 复现命令

这是 phase7 时期更好的单卡 exact compile 路线，历史台账约为 `1.4314x`。

注意：这条命令和上面几乎一样，差别主要来自当前代码版本已经包含 fused layernorm/modulation v2 热路径。

```bash
python -m sglang.multimodal_gen.test.manual.profile_star_cogvideox_sr \
  --model-path /sgl-workspace/sglang/model_artifacts/sglang_star_cogvideox_sr \
  --condition-video-path /sgl-workspace/STAR_mg/input/cogvideox_test/lq/023_klingai_reedit.mp4 \
  --prompt-path /sgl-workspace/STAR_mg/input/cogvideox_test/text/023_klingai_reedit.txt \
  --reference-video /sgl-workspace/STAR_mg/cogvideox-based/sat/output/ref_seed1234/0_A_serene_scene_of_a_panda_bear_playing_a_guitar_at_sunset_unfolds_by_a_tranquil_lake._The_panda,_with_its_black-and-whit/000000.mp4 \
  --output-dir /sgl-workspace/sglang/outputs/star_repro_single_fa_compile_fusedln_v2_fps8 \
  --seed 1234 \
  --num-frames 7 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --condition-video-num-frames 25 \
  --attention-backend fa \
  --num-gpus 1 \
  --enable-torch-compile \
  --dit-cpu-offload \
  --text-encoder-cpu-offload \
  --warmup-runs 1 \
  --measured-runs 1 \
  --original-star-cold-e2e-s 302.334 \
  --original-star-warm-e2e-s 228.0
```

对应历史台账：

- run label: `single_fa_compile_fusedln_v2`
- 历史速度：`warm_e2e_speedup = 1.4314x`
- 台账位置：[compare.json](/sgl-workspace/sglang/docs_xzh/add_STAR/compare.json:402)

说明：

- 如果你在当前代码上执行这两条命令，得到的结果不一定和历史值一模一样，因为当前仓库状态、VAE 路径、本地化修复、fps 修正都已经变化。
- 但这两条就是当前仓库下最接近历史 `1.4x exact` 路线、且已经修正为正确 `8 fps` 输出的直接复现实验命令。

## 4. 对比建议

建议至少保留下面四个路径：

- 原版 STAR candidate
- SGLang exact candidate
- SGLang FP8 candidate
- 原版 STAR reference mp4

推荐优先做两组比较：

1. 原版 STAR vs SGLang exact  
   这是“同标准算法对齐”比较。
2. 原版 STAR vs SGLang FP8  
   这是“phase7 量化验收”比较。

## 5. 当前代码状态说明

当前 `sglang` 侧 STAR 已经具备以下性质：

- STAR pipeline / DiT / stage 均在 `sglang` 仓库内
- STAR VAE 运行时所需的 SAT 代码已经 vendor 到 `sglang` 仓库内
- `sglang` 推理运行时不再需要 `STAR_mg` 仓库代码路径

注意：

- 原版 STAR 参考命令当然仍然需要 `STAR_mg` 仓库，因为它本身就是原版实现。
- 如果你要完全移除 `STAR_mg` 仓库，至少要先把 `SGLANG_STAR_MODEL_DIR` 这个转换后模型目录保存在 `sglang` 仓库或其他独立位置。
