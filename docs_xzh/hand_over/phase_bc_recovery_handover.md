# Vivid-VR In SGLang Phase B/C Recovery Handover

## 1. Scope And Purpose

This document summarizes the previous Phase B and Phase C implementation work for integrating `Vivid-VR` into `sglang.multimodal_gen`, with emphasis on:

- which files were added or changed,
- which problems were encountered during recovery and alignment,
- which semantic mismatches had to be corrected to match the original `Vivid-VR`,
- which tests and acceptance commands were used,
- what order should be followed when restoring the code again.

This document is for recovery guidance only. It does not claim that the current worktree still contains the Phase B/C code.

`Phase A` was implemented in another conversation. If Phase A code is missing, restore Phase A first or use its implementation as the baseline before attempting to recover Phase B/C.


## 2. High-Level Project Constraints Inherited From Earlier Planning

These constraints were already fixed by the strategy docs and by the earlier Phase A work. Phase B/C must remain consistent with them:

- `Vivid-VR` is not treated as a generic `diffusers` wrapper.
- The integration path is a `hybrid / model-specific pipeline` inside `sglang.multimodal_gen`.
- Do not copy `/home/zhiheng/Vivid-VR/src/diffusers` wholesale into `sglang`.
- During integration and acceptance, prompt text must be read from:
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- Do not run live `CogVLM2` caption generation inside `sglang`.
- The single-clip acceptance reference video is fixed at:
  - `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`
- Correctness and reference alignment come before performance features such as compile/offload/backend tuning.


## 3. Source Of Truth For Recovery

The main implementation session for Phase B and Phase C was:

- `/home/zhiheng/.codex/sessions/2026/06/03/rollout-2026-06-03T08-44-22-019e8ca7-b803-7b41-a68b-a133925ddd80.jsonl`

The strategy and staged planning docs already restored into this repo are:

- [docs_xzh/add_strategy/README.md](/home/zhiheng/sglang/docs_xzh/add_strategy/README.md:1)
- [docs_xzh/add_strategy/03_stage2_mvp_scope.md](/home/zhiheng/sglang/docs_xzh/add_strategy/03_stage2_mvp_scope.md:1)
- [docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md](/home/zhiheng/sglang/docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md:1)
- [docs_xzh/add_strategy/05_stage4_component_migration.md](/home/zhiheng/sglang/docs_xzh/add_strategy/05_stage4_component_migration.md:1)
- [docs_xzh/add_strategy/09_code_mod_order.md](/home/zhiheng/sglang/docs_xzh/add_strategy/09_code_mod_order.md:1)
- [docs_xzh/add_strategy/10_grouped_stage_acceptance.md](/home/zhiheng/sglang/docs_xzh/add_strategy/10_grouped_stage_acceptance.md:1)

When recovering code, use the session file above as the concrete patch source and the `add_strategy` docs as the design contract.


## 4. Phase B Goal

Phase B was the component migration phase. The goal was not yet a full end-to-end reference-aligned video. The goal was to establish the core model substrate inside `sglang` so that later pipeline work could run on native SGLang components.

The practical Phase B target was:

- add a working `CogVideoX` base transformer runtime,
- add a working `CogVideoX` VAE runtime,
- add a working scheduler path for `VividVR`,
- add the `VividVR`-specific transformer/controlnet pieces,
- register the components through SGLang config/registry contracts,
- verify component-level forward/load/shape/dtype behavior through tests.


## 5. Phase B Main Files

### 5.1 Runtime model files added or implemented

- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/runtime/models/schedulers/cogvideox_dpm_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py`
- `python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py`

### 5.2 Config files added or updated

- `python/sglang/multimodal_gen/configs/models/dits/cogvideox.py`
- `python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`

### 5.3 Tests added or updated

- `python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py`


## 6. Phase B Implementation Method

Phase B was implemented by following the SGLang native contracts instead of wrapping the original project externally.

The implementation style was:

- define config entries that can be discovered by the existing SGLang registry,
- implement runtime modules under `runtime/models/...`,
- keep VividVR-specific code localized instead of changing the generic execution core,
- load `VividVR` controlnet-side private weights explicitly,
- validate each model family piece independently before attempting a full clip run.

The key architectural choice was to separate:

- reusable `CogVideoX` base behavior,
- `VividVR`-specific additions such as control features and connector weights.


## 7. Phase B Main Problems And Resolutions

### 7.1 Problem: there was no direct one-step runtime mapping

The original `Vivid-VR` code assumes its own runtime composition and does not fit directly into the existing SGLang multimodal generation contracts.

Resolution:

- implement native `sglang.multimodal_gen` runtime/config entries instead of forcing a generic wrapper path.

### 7.2 Problem: private `VividVR` side weights are not covered by standard base model loading

The original pipeline depends on extra files such as connector and control projection weights that are not part of a plain base transformer checkpoint.

Resolution:

- explicitly support loading:
  - `connectors.pt`
  - `control_feat_proj.pt`
  - `control_patch_embed.pt`

### 7.3 Problem: shape/dtype compatibility across controlnet and transformer forward path

Even after the files load, the combined path can fail because of:

- shape mismatch,
- dtype mismatch,
- `NaN` / `Inf` propagation,
- mismatched expectations between controlnet output and transformer input.

Resolution:

- add component-level forward tests that exercise the real combined path and fail if any of those conditions appear.


## 8. Phase B Test Coverage

The Stage B tests were written to validate the component migration contract, not just file existence.

The test coverage included:

- `ModelRegistry` discovery,
- initialization from real configs,
- VAE `encode/decode`,
- scheduler single-step semantics,
- independent loading of:
  - `connectors.pt`
  - `control_feat_proj.pt`
  - `control_patch_embed.pt`
- controlnet checkpoint shape alignment,
- combined controlnet + transformer forward without:
  - `NaN`
  - `Inf`
  - shape mismatch
  - dtype mismatch


## 9. Phase B Acceptance Result

The reported acceptance command for Stage B was:

```bash
uv run pytest \
  python/sglang/multimodal_gen/test/unit/test_registry_vividvr.py \
  python/sglang/multimodal_gen/test/unit/test_sampling_params.py \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py -q
```

The reported result was:

```text
38 passed, 237 subtests passed
```

At that point, Stage B was considered passed and Phase C work started on top of that baseline.


## 10. Phase C Goal

Phase C was the single-clip end-to-end reference alignment phase.

This phase was not just about making the pipeline run. The actual goal was to reproduce the original `Vivid-VR` semantics closely enough that the generated clip passed the reference comparison thresholds.

This required:

- completing the SGLang-side VividVR pipeline assembly,
- reproducing original preprocessing and decode semantics,
- fixing latent shape mismatches,
- fixing text encoder behavior mismatches,
- restoring original wrapper behavior after pipeline decode.


## 11. Phase C Main Files

### 11.1 Pipeline and stage files

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

### 11.2 VividVR helper files

- `python/sglang/multimodal_gen/runtime/vividvr/__init__.py`
- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/postprocess.py`
- `python/sglang/multimodal_gen/runtime/vividvr/tiling.py`

### 11.3 Config updates used during Phase C refinement

- `python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py`
- `python/sglang/multimodal_gen/configs/vividvr_defaults.py`

### 11.4 Tests

- `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py`


## 12. Phase C Initial Implementation Direction

Phase C began by adding the missing pipeline-side pieces:

- a `VividVR` pipeline wrapper in SGLang,
- VividVR-specific preprocess logic,
- VividVR-specific tiling helpers,
- a model-specific stage to run the denoise/decode path,
- a single-clip acceptance test.

That initial build was enough to execute the pipeline path, but not enough to pass reference alignment. The main work in Phase C was then iterative semantic correction.


## 13. Critical Phase C Semantic Alignments

These were the most important correctness fixes. They are the main things that must be preserved if code is restored again.

### 13.1 VAE tiling defaults had to match original CogVideoX behavior

This turned out to be one of the key root causes.

Observed mismatch:

- the original `AutoencoderKLCogVideoX.enable_tiling()` behavior effectively produced a `90x120` latent for `960x720` input,
- the earlier SGLang-side default path used a generic `256x256` threshold,
- that caused the latent to become `91x122`,
- once latent geometry drifted here, downstream output could not align with the original reference video.

The fix applied in Phase C was to set the VAE config defaults to:

- `tile_sample_min_height = 240`
- `tile_sample_min_width = 360`

This was recorded as the critical config correction in:

- `python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py`

### 13.2 Prompt input had to come from prompt.txt, not runtime captioning

The original integration constraints required fixed prompt reuse rather than online caption generation.

The preprocess logic had to:

- read prompt text from `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`,
- construct the positive prompt suffix in the same way as the original flow,
- avoid introducing live `CogVLM2` caption behavior into SGLang.

This logic lived in:

- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`

### 13.3 Text encoder wrapper semantics had to match original Vivid-VR expectations

One important mismatch was around how the text encoder was called.

The Phase C alignment for the SGLang-side wrapper was:

- ignore `attention_mask`,
- call T5 using `input_ids` only,
- disable learned positional embeddings,
- use the rotary-compatible path expected by the original Vivid-VR integration.

This alignment was done in:

- `python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py`

### 13.4 The original wrapper kept an unpadded reference video for postprocess alignment

The original Vivid-VR behavior was not just "decode latent -> write result".

It preserved a reference video tensor before padding, then used it after decode to restore appearance semantics.

The SGLang-side implementation had to preserve:

- the unpadded `reference_video`,
- the exact padding metadata,
- the final postprocess usage of that reference.

This required updates in:

- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`

### 13.5 The original flow had frame dropping and padding crop semantics that mattered

Another important mismatch was that the original wrapper behavior was not frame-neutral.

The aligned SGLang logic had to support:

- dropping the first `3` frames when required,
- cropping padded frames after decode,
- exposing padding metadata such as `num_padding_frames`.

This mattered both for visual alignment and for test contracts.

The relevant logic lived in:

- `python/sglang/multimodal_gen/runtime/vividvr/preprocess.py`
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`
- `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py`

### 13.6 Output AdaIN against the reference video was required

One of the most important "last mile" wrapper semantics was that the original Vivid-VR path applied adaptive instance normalization on the decoded output using the reference video.

Without this, the pipeline could be structurally correct but still drift in appearance statistics and fail comparison thresholds.

The SGLang-side recovery added:

- `adaptive_instance_normalization(...)`

in:

- `python/sglang/multimodal_gen/runtime/vividvr/postprocess.py`

and then applied it in the decode/postprocess stage:

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py`


## 14. Phase C Main Problems And Root Causes

### 14.1 Problem: the pipeline ran, but reference metrics still failed

This was the main Phase C debugging pattern. The code could execute end-to-end while still being wrong at the semantic level.

Root causes included:

- latent shape mismatch,
- wrong prompt path semantics,
- missing frame handling logic,
- missing post-decode appearance normalization,
- text encoder wrapper behavior drift.

### 14.2 Problem: generic defaults from unrelated paths leaked into VividVR behavior

The `256x256` VAE tiling threshold is the best example. It is a plausible generic default, but it is not the original Vivid-VR behavior.

Resolution:

- replace generic defaults with original-flow-aligned values where they materially affect outputs.

### 14.3 Problem: correctness and performance were easy to mix together

At this point it would have been easy to start changing acceleration or backend settings while the semantic alignment was still wrong.

Resolution:

- hold performance work until the single-clip reference contract is passed.


## 15. Phase C Test And Acceptance Design

The Phase C acceptance test was centered on the single-clip reference comparison.

The key contract points included:

- same-seed determinism,
- correct padding behavior,
- correct reference comparison metrics.

The reported acceptance thresholds that were enforced were:

- `ssim_min >= 0.90`
- `mse_max <= 150.0`
- `mae_max <= 8.0`
- `pass_compare == True`

The padding-related contract also included:

- `num_padding_frames == 3`


## 16. Phase C Acceptance Commands And Reported Results

### 16.1 Lightweight regression

```bash
uv run pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q
```

Reported result:

```text
12 passed, 1 skipped, 229 subtests passed
```

### 16.2 Heavy acceptance

```bash
SGLANG_RUN_VIVIDVR_ACCEPTANCE=1 \
uv run pytest python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q
```

Reported result:

```text
3 passed, 5 warnings in 1584.40s (0:26:24)
```

This was the run that was treated as the real Phase C acceptance result.


## 17. Output Video Location Behavior During Acceptance

During the Phase C acceptance test, the generated candidate video was not written to a fixed persistent artifact directory.

The behavior was:

- the test created a temporary directory via `tempfile.TemporaryDirectory()`,
- the candidate video filename was `candidate.mp4`,
- the directory was deleted after the test completed.

So if code is restored and acceptance is rerun, do not expect the candidate output to remain on disk unless the test is intentionally changed to write to a persistent location.

The fixed comparison reference remained:

- `/home/zhiheng/Vivid-VR/result/720p_up1_result_vivid_ori/videos/test_video_960x720.mp4`


## 18. Recommended Recovery Order

If the code must be restored again, use this order.

### 18.1 Restore or verify Phase A baseline first

Phase B/C assume the earlier baseline already exists, including:

- family recognition,
- prompt path policy,
- no live `CogVLM2`,
- reference path conventions,
- config/registry/sampling contract stabilization.

Do not start Phase C pipeline recovery on top of a missing or inconsistent Phase A baseline.

### 18.2 Restore Phase B component files before any Phase C pipeline files

Recover these first:

- base VAE
- base transformer
- scheduler
- VividVR transformer/controlnet/common helpers
- configs
- Stage B tests

Then rerun Stage B tests before proceeding.

### 18.3 Restore Phase C in semantic order, not just chronological patch order

The most practical recovery order is:

1. `vividvr_pipeline.py`
2. `runtime/vividvr/preprocess.py`
3. `runtime/vividvr/tiling.py`
4. `model_specific_stages/vividvr.py`
5. `runtime/vividvr/postprocess.py`
6. `test_stage_c_vividvr_single_clip.py`
7. final config corrections, especially VAE tiling defaults

### 18.4 Re-check the known critical semantics before rerunning heavy acceptance

Before running the long acceptance test, explicitly verify:

- prompt source is `prompt.txt`,
- no live `CogVLM2`,
- VAE tiling defaults are `240` and `360`,
- `reference_video` is preserved before padding,
- first `3` frames are dropped when required,
- padding frames are cropped after decode,
- AdaIN postprocess is applied using reference video,
- same-seed determinism contract is still covered by the test.


## 19. Session Anchors For Patch Extraction

If a future recovery step needs to extract code from `.codex/sessions`, these patch clusters were the useful anchors from the main implementation session:

- initial Phase B config additions around line `722`
- initial Phase B runtime/test additions around line `963`
- early Phase B refinements around lines `1017`, `1021`, `1025`
- initial Phase C pipeline/stage/helper additions around line `3656`
- iterative Phase C refinements around lines:
  - `3706`
  - `3837`
  - `3944`
  - `4100`
  - `4212`
  - `4216`
  - `4220`
  - `4680`
  - `4685`
  - `4700`
  - `5041`
  - `5047`
  - `5223`
  - `5227`
  - `5696`
  - `8411`
  - `8823`
  - `8827`
  - `8831`
  - `8835`
  - `8860`
  - `8864`

These anchors are especially useful because several files were created once and then updated multiple times. Recovering only the first patch is not sufficient.


## 20. What Must Not Be Forgotten During Recovery

If only one section of this document is remembered, it should be this one.

The most important recovery points are:

- Phase B was native SGLang component migration, not an external wrapper.
- Phase C passed only after aligning original `Vivid-VR` semantics, not just after making the pipeline executable.
- The VAE tiling default correction to `240x360` was critical.
- Preserving `reference_video`, handling the first `3` frames, cropping padding, and applying AdaIN after decode were critical.
- The text encoder wrapper behavior had to match original expectations.
- Heavy acceptance was the real success criterion, not just unit tests.


## 21. Final Recovery Guidance

For future recovery work:

- use this document as the narrative map,
- use `docs_xzh/add_strategy/*` as the design contract,
- use the `2026-06-03` session JSONL as the code extraction source,
- restore Phase A baseline before Phase B/C if needed,
- rerun Stage B tests before attempting Phase C heavy acceptance.

If the next Codex needs to continue from here, it should first recover code by patch extraction, then verify Stage B, then verify Phase C, in that order.
