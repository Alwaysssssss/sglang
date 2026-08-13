# VideoEdit FP8 Quality Evaluation

## Purpose

This evaluation measures how closely the optimized FP8 + SageAttention output
matches the BF16 output. It does not measure semantic correctness against a
ground-truth edited video, because no ground-truth target is available.

Both pairs use the same input video, mask, reference image, prompt, seed,
resolution, frame count, guidance settings, and inference-step count. Each video
contains 80 frames at 1920x1080. The binary mask covers 7.4406% of the frame on
average.

## Results

Metrics are calculated per frame and then averaged. BF16 is the reference and
the optimized FP8 + SageAttention output is the candidate.

| Steps | Region | PSNR mean (dB) | SSIM mean | SSIM P05 | MAE mean |
|---:|---|---:|---:|---:|---:|
| 4 | Full frame | 39.7255 | 0.967792 | 0.963598 | 0.003923 |
| 4 | Mask | 32.2020 | 0.884215 | 0.879586 | 0.015861 |
| 4 | Background | 41.7713 | 0.974694 | 0.970452 | 0.002963 |
| 4 | 16 px mask boundary | 32.6632 | 0.851606 | 0.843601 | 0.011744 |
| 40 | Full frame | 41.1961 | 0.971689 | 0.968675 | 0.003517 |
| 40 | Mask | 34.9285 | 0.914801 | 0.910102 | 0.012104 |
| 40 | Background | 42.5158 | 0.976415 | 0.973523 | 0.002827 |
| 40 | 16 px mask boundary | 36.6219 | 0.875754 | 0.869664 | 0.008785 |

The 40-step pair has a temporal delta MAE of 0.000629. This is the mean absolute
difference between the BF16 and FP8 consecutive-frame residuals; lower is
better.

## Interpretation

- The 40-step result passes the initial same-seed SSIM gates in the quantization
  plan: full-frame mean SSIM is above 0.95 and P05 is above 0.90.
- The edited area is the more meaningful region because paste-back and the small
  mask make full-frame metrics optimistic. Mask SSIM is 0.9148 with P05 0.9101.
- The mask boundary is the weakest region at 0.8758 SSIM. This should be checked
  visually for edge shimmer, color discontinuity, and blending artifacts.
- These numbers support continuing the current optimization work, but one video
  is not enough for a general quality claim. The next quality gate should run the
  same comparison over the 20-case calibration/evaluation set and report the
  distribution across cases.

## Inputs And Artifacts

- 4-step BF16:
  `videoedit_phase15_diagnostics/phase15_20260721_034005/bf16_layerwise/benchmark/phase15_bf16_layerwise_profile81_20260721_042227_run00.mp4`
- 4-step FP8 + SageAttention:
  `videoedit_phase15_diagnostics/phase15_20260722_063759/fp8_layerwise/benchmark/phase15_fp8_layerwise_profile81_20260722_063912_run00.mp4`
- 40-step BF16:
  `videoedit_phase0_outputs/phase0_bf16_1080_single81_20260717_043315_run00.mp4`
- 40-step FP8 + SageAttention:
  `videoedit_phase15_diagnostics/phase15_20260722_073221/fp8_layerwise/benchmark/phase15_fp8_layerwise_single81_20260722_073659_run00.mp4`
- Raw metrics:
  `videoedit_quality_evaluation/fp8_sage_vs_bf16_4steps.json` and
  `videoedit_quality_evaluation/fp8_sage_vs_bf16_40steps.json`
- Reusable evaluator: `scripts/videoedit_evaluate_quality.py`

## Metric Definitions

- Full-frame SSIM is `skimage.metrics.structural_similarity` on RGB frames with
  `data_range=1.0`.
- Mask/background/boundary SSIM is the corresponding weighted mean of the SSIM
  similarity map.
- PSNR, MAE, and RMSE are calculated in normalized RGB space `[0, 1]`.
- The boundary region is the morphological band formed by dilating and eroding
  the binary mask by 16 pixels.
