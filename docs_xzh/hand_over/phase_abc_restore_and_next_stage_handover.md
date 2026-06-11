# Vivid-VR In SGLang Restore Handover

Last updated: `2026-06-04 UTC`

## 1. Current Repository State

- Repo path: `/home/zhiheng/sglang`
- Active branch: `sglang_Vivid`
- Current HEAD: `853a791f0a58b17d99396d8fddd5d75bf15463af`
- Remote branch: `origin/sglang_Vivid`
- Current code status:
  - Phase A restored, tested, committed, pushed
  - Phase B restored, tested, committed, pushed
  - Phase C restored, single-run real acceptance passed, committed, pushed
- Current worktree status:
  - only local untracked artifacts remain under `/home/zhiheng/sglang/Vivid_Acceptance/`
  - these artifacts were intentionally not committed

This file is the main handover for the next Codex. The earlier recovery-only note is still available at:

- [phase_bc_recovery_handover.md](/home/zhiheng/sglang/docs_xzh/hand_over/phase_bc_recovery_handover.md:1)


## 2. What Happened In This Conversation

This conversation covered the full recovery path after `/home/zhiheng/sglang` was accidentally emptied.

Main sequence:

1. Confirmed `/home/zhiheng/sglang` had become an empty directory.
2. Verified there was no simple undelete path from the container filesystem.
3. Checked VS Code Remote Server history.
4. Found that VS Code history mostly preserved docs and prompts, but not the critical Phase B/C Python source files.
5. Scanned `.codex/sessions` and confirmed the real recovery source was:
   - `/home/zhiheng/.codex/sessions/2026/05/29/...` for `add_strategy` docs
   - `/home/zhiheng/.codex/sessions/2026/06/03/rollout-2026-06-03T08-44-22-019e8ca7-b803-7b41-a68b-a133925ddd80.jsonl` for Phase B/C code
6. Restored `docs_xzh/add_strategy/`.
7. Wrote the recovery guidance doc for Phase B/C.
8. Restored Phase A and pushed it.
9. Restored Phase B and pushed it.
10. Reviewed Phase C semantics before editing.
11. Wrote a dedicated prompt to guide Phase C recovery:
   - [phaseC.md](/home/zhiheng/sglang/docs_xzh/prompts/phaseC.md:1)
12. Restored Phase C code.
13. Ran single-run real end-to-end acceptance in `tmux` with terminal progress bar.
14. Diagnosed an initial Phase C failure.
15. Fixed the last major semantic mismatch in the text path.
16. Re-ran single-run acceptance and passed.
17. Committed and pushed Phase C.


## 3. Restored Planning And Guidance Docs

These are already restored and should be treated as the planning contract for later stages:

- [docs_xzh/add_strategy/README.md](/home/zhiheng/sglang/docs_xzh/add_strategy/README.md:1)
- [docs_xzh/add_strategy/03_stage2_mvp_scope.md](/home/zhiheng/sglang/docs_xzh/add_strategy/03_stage2_mvp_scope.md:1)
- [docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md](/home/zhiheng/sglang/docs_xzh/add_strategy/04_stage3_pipeline_mod_plan.md:1)
- [docs_xzh/add_strategy/05_stage4_component_migration.md](/home/zhiheng/sglang/docs_xzh/add_strategy/05_stage4_component_migration.md:1)
- [docs_xzh/add_strategy/09_code_mod_order.md](/home/zhiheng/sglang/docs_xzh/add_strategy/09_code_mod_order.md:1)
- [docs_xzh/add_strategy/10_grouped_stage_acceptance.md](/home/zhiheng/sglang/docs_xzh/add_strategy/10_grouped_stage_acceptance.md:1)

Important rule carried through all recovery work:

- `Vivid-VR` must run inside `sglang.multimodal_gen` as a native, model-specific integration.
- Do not depend on `/home/zhiheng/Vivid-VR` runtime source code at inference time.
- It is acceptable to use `/home/zhiheng/Vivid-VR` for:
  - checkpoints
  - prompt file
  - input test video
  - reference video


## 4. Phase A Status

Phase A was restored first as the baseline contracts layer.

Main scope:

- `config / sampling / registry` contracts
- `VividVR` family detection
- fixed `prompt.txt` path
- fixed reference video path
- explicit ban on live `CogVLM2`

Key files:

- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:1)
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/sample/vividvr.py:1)
- [vividvr_defaults.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/vividvr_defaults.py:1)
- [registry.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/registry.py:1)
- [test_sampling_params.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_sampling_params.py:1)
- [test_registry_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_registry_vividvr.py:1)

Committed and pushed:

- `a39995d82` `Restore VividVR recovery docs`
- `4b2156acf` `Add VividVR Phase A baseline contracts`


## 5. Phase B Status

Phase B restored the component substrate.

Main scope:

- `CogVideoX` base transformer
- `CogVideoX` VAE
- `VividVR` scheduler
- `VividVR` transformer/controlnet/private weight loading
- Stage B component contract tests

Key files:

- [cogvideox.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox.py:1)
- [cogvideox.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/vaes/cogvideox.py:1)
- [cogvideox_dpm_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/schedulers/cogvideox_dpm_vividvr.py:1)
- [cogvideox_vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr.py:1)
- [cogvideox_vividvr_controlnet.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_controlnet.py:1)
- [cogvideox_vividvr_common.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/models/dits/cogvideox_vividvr_common.py:1)
- [test_stage_b_vividvr_components.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py:1)

Key pitfalls already handled during restore:

- scheduler config must tolerate extra HF metadata like `_diffusers_version`
- `control_feat_proj` path must be batch-safe
- component smoke tests should exercise the rotary path, not a misleading toy learned-pos path
- VAE toy tests should validate the real Stage B contract, not over-assume arbitrary exact spatial reconstruction

Committed and pushed:

- `d45bf7a36` `Restore VividVR Phase B components`


## 6. Phase C Status

Phase C is now restored and accepted in the requested single-run mode.

Main scope:

- native SGLang `VividVR` pipeline
- preprocess / tiling / denoising / decoding chain
- `prompt.txt` driven caption path
- runtime decoupling from original repo code
- real saved acceptance video and metrics

Key files:

- [vividvr_pipeline.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines/vividvr_pipeline.py:1)
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:1)
- [preprocess.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/vividvr/preprocess.py:1)
- [tiling.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/vividvr/tiling.py:1)
- [postprocess.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/vividvr/postprocess.py:1)
- [cogvideox.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/models/vaes/cogvideox.py:1)
- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:1)
- [test_stage_c_vividvr_single_clip.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py:1)
- [run_vividvr_phase_c_single.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py:1)

Committed and pushed:

- `853a791f0` `Restore VividVR Phase C pipeline`


## 7. Critical Phase C Semantic Alignments

These are the most important details to preserve. If later regression appears, check these first.

### 7.1 Prompt source

- Prompt must come from:
  - `/home/zhiheng/Vivid-VR/input/720p/prompt.txt`
- No live `CogVLM2` path in `sglang`

### 7.2 Text encoder wrapper

- `VividVR` T5 wrapper must behave like the original pipeline:
  - call T5 with `input_ids`
  - do not rely on Wan-style postprocess
- The final restored text postprocess is `VividVR`-specific and keeps the sequence length equal to tokenizer output.

### 7.3 The Phase C failure that mattered most

The first real single-run acceptance after restoration failed badly.

Failure artifact:

- [phase_c_metrics_seed42_20260604T064630Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T064630Z.json:1)

Key failed metrics:

- `ssim_min = 0.5536798159438543`
- `mse_max = 978.3599853515625`
- `mae_max = 18.970243453979492`
- `failed_frame_ratio = 1.0`

The decisive clue was:

- `prompt_embed_shape = [1, 512, 4096]`

Root cause:

- `VividVRPipelineConfig` was still reusing Wan `t5_postprocess_text`
- that function pads prompt embeddings to length `512`
- original `Vivid-VR` expects the `226`-token path

Fix:

- added `vividvr_t5_postprocess_text(...)` in:
  - [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/vividvr.py:1)
- added contract coverage in:
  - [test_stage_c_vividvr_single_clip.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py:1)

After the fix, the accepted run recorded:

- `prompt_embed_shape = [1, 226, 4096]`

### 7.4 VAE tiling

The `CogVideoX VAE` tiling defaults must remain:

- `tile_sample_min_height = 240`
- `tile_sample_min_width = 360`

This avoids the earlier latent geometry drift.

### 7.5 Reference video preservation

- preprocess must preserve the unpadded `reference_video`
- decode/postprocess must use it later for alignment

### 7.6 Decode-side wrapper semantics

Decode side must still include:

- drop first `3` frames when required
- crop padded frames
- apply `AdaIN(reference_video, generated_video)` style postprocess

These are correctness features, not optional polish.


## 8. Real Phase C Acceptance Artifacts

The accepted single-run artifacts are:

- Metrics:
  - [phase_c_metrics_seed42_20260604T070647Z.json](/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T070647Z.json:1)
- Result video:
  - [phase_c_candidate_seed42_20260604T070647Z.mp4](/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_c_candidate_seed42_20260604T070647Z.mp4)
- Log:
  - [phase_c_single_20260604T070639Z.log](/home/zhiheng/sglang/Vivid_Acceptance/logs/phase_c_single_20260604T070639Z.log:1)

Accepted summary:

- `pass_compare = true`
- `ssim_mean = 0.967716215299506`
- `ssim_min = 0.9473462237832677`
- `mse_mean = 39.878108160836355`
- `mse_max = 81.55096435546875`
- `mae_mean = 3.3365604979651313`
- `mae_max = 3.9912755489349365`
- `failed_frame_ratio = 0.0`
- `reference_frame_count = 70`
- `candidate_frame_count = 70`
- `frame_count_delta = 0`

Debug section from the accepted run:

- `prompt_embed_shape = [1, 226, 4096]`
- `control_latent_shape = [1, 20, 16, 90, 120]`
- `latents_shape = [1, 20, 16, 90, 120]`
- `tile_count = 1`
- `padded_input_frames = 73`
- `timestep_count = 50`
- `output_shape = [3, 70, 720, 960]`

Important precision note:

- this conversation only re-ran the real single-run acceptance
- the heavy double-run determinism pytest path was not re-run after the final fix
- the older deleted workspace had already passed that heavier mode earlier, but this restored workspace was validated here by:
  - light unit/contract tests
  - one real full end-to-end single inference
  - one real saved frame-by-frame comparison against reference


## 9. Commands Used And Known-Good Validation Path

### 9.1 Light regression

```bash
PYTHONPATH=python uv run \
  --with pytest \
  --with diffusers==0.37.0 \
  --with imageio==2.36.0 \
  --with imageio-ffmpeg==0.5.1 \
  --with addict==2.4.0 \
  --with PyYAML==6.0.1 \
  --with av==16.1.0 \
  --with scikit-image==0.25.2 \
  --with cache-dit==1.3.0 \
  --with opencv-python-headless==4.10.0.84 \
  --with trimesh \
  python -m pytest \
  python/sglang/multimodal_gen/test/unit/test_stage_b_vividvr_components.py \
  python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py -q
```

Observed result in this conversation:

- `14 passed, 1 skipped, 1143 subtests passed`

### 9.2 Single-run real acceptance

```bash
PYTHONPATH=python uv run \
  --with pytest \
  --with diffusers==0.37.0 \
  --with imageio==2.36.0 \
  --with imageio-ffmpeg==0.5.1 \
  --with addict==2.4.0 \
  --with PyYAML==6.0.1 \
  --with av==16.1.0 \
  --with scikit-image==0.25.2 \
  --with cache-dit==1.3.0 \
  --with opencv-python-headless==4.10.0.84 \
  --with trimesh \
  python python/sglang/multimodal_gen/tools/run_vividvr_phase_c_single.py
```

### 9.3 tmux workflow

The acceptance run was executed in:

- session: `vividvr_phase_c`

Useful command for a later operator:

```bash
tmux attach -t vividvr_phase_c
```

The denoising stage now shows a terminal progress bar from inside:

- [vividvr.py](/home/zhiheng/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/vividvr.py:1)


## 10. Runtime Decoupling Status

Current integration is decoupled from original `/home/zhiheng/Vivid-VR` runtime source code.

The restored pipeline does not depend on importing the original repository Python runtime modules during inference.

What still intentionally points to `/home/zhiheng/Vivid-VR`:

- checkpoint roots
- prompt file path
- input video path
- reference video path

This is acceptable under the current staged integration contract.


## 11. What The Next Codex Should Do Next

The next stage after current work is Phase D / grouped Stage D in:

- [10_grouped_stage_acceptance.md](/home/zhiheng/sglang/docs_xzh/add_strategy/10_grouped_stage_acceptance.md:112)

That means the next Codex should focus on:

- long-video clip split / merge / temporal orchestration
- optional caption/postprocess module wiring
- keeping optional modules independently disable-able
- not regressing the single-clip accepted baseline

Recommended execution order:

1. Re-read:
   - [phase_abc_restore_and_next_stage_handover.md](/home/zhiheng/sglang/docs_xzh/hand_over/phase_abc_restore_and_next_stage_handover.md:1)
   - [phase_bc_recovery_handover.md](/home/zhiheng/sglang/docs_xzh/hand_over/phase_bc_recovery_handover.md:1)
   - [phaseC.md](/home/zhiheng/sglang/docs_xzh/prompts/phaseC.md:1)
   - [10_grouped_stage_acceptance.md](/home/zhiheng/sglang/docs_xzh/add_strategy/10_grouped_stage_acceptance.md:1)
2. Keep Phase C green before modifying Phase D logic.
3. Preserve the accepted single-run artifacts as the current baseline.
4. When editing later stages, re-check the six known Phase C correctness points:
   - prompt source from `prompt.txt`
   - no live `CogVLM2`
   - `226`-length prompt embeddings
   - VAE tiling `240 / 360`
   - reference video preservation
   - `drop first 3 frames + crop padding + AdaIN`
5. After each new stage passes, commit and push immediately to `origin/sglang_Vivid`.


## 12. Fast Sanity Checklist Before New Work

Before the next Codex starts Phase D or later, verify:

- branch is still `sglang_Vivid`
- `git status` is clean except local `Vivid_Acceptance/`
- accepted Phase C artifacts still exist:
  - `/home/zhiheng/sglang/Vivid_Acceptance/indicator/phase_c_metrics_seed42_20260604T070647Z.json`
  - `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/phase_c_candidate_seed42_20260604T070647Z.mp4`
- light regression still passes

If any Phase C regression appears later, check the text postprocess path first. That was the last major hidden mismatch in this restore.
