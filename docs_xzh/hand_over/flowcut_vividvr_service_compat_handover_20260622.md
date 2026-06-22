# FlowCut Vivid-VR Service Compatibility Handover - 2026-06-22

## Scope

- Adds FlowCut-compatible endpoint `POST /v1/videos/repairs/flowcut`.
- Keeps existing `POST /v1/videos/repairs` behavior unchanged.
- Uses only `/home/zhiheng/sglang` code; no direct dependency on `/home/zhiheng/sglang_serve`.
- Reuses the native Vivid-VR repair sampling path under `sglang.multimodal_gen`.

## Contract

- `code:0`: task accepted and runs asynchronously.
- `code:1`: permanent business failure, returned as HTTP 200 JSON for handled FlowCut business errors.
- `code:2`: queue full, returned as HTTP 200 JSON.
- `timeout:-1`: no service-side inference timeout for Vivid-VR long-running generation.
- Invalid JSON or invalid FlowCut request objects are mapped to HTTP 200 JSON with `code:1`.
- Callback statuses: `running`, `succeeded`, `failed`.
- `running` callback is sent before model dispatch, and periodic progress heartbeat follows while the job is still running.
- If `minioConfig` is provided, the final success callback uploads the local output and returns the per-request MinIO `result_url`.

## Verification

- Unit tests:
  - `python/sglang/multimodal_gen/test/unit/test_flowcut_protocol.py`
  - `python/sglang/multimodal_gen/test/unit/test_flowcut_video_repair_api.py`
  - Latest result: `15 passed, 6 warnings in 15.03s`.
- Vivid-VR lightweight regression:
  - `python/sglang/multimodal_gen/test/unit/test_stage_c_vividvr_single_clip.py::TestStageCVividVRContracts::test_control_video_padding_contract_matches_reference_wrapper`
  - `python/sglang/multimodal_gen/test/unit/test_stage_e_vividvr_inference_tool.py::TestVividVRInferenceTool::test_load_video_frames_reuses_compare_cache`
  - Latest result: `2 passed, 5 warnings in 72.11s`.
- Static checks:
  - `py_compile` passed for FlowCut protocol/API files.
  - `git diff --check` passed for the touched FlowCut files.
- Dual-card serve E2E:
  - tmux session: `vividvr_serve_dual_default`.
  - Endpoint: `POST /v1/videos/repairs/flowcut`.
  - Task ID: `flowcut-e2e-dual-20260622T060712Z`.
  - Result video: `/home/zhiheng/sglang/Vivid_Acceptance/result_videos/service_benchmark/flowcut-e2e-dual-20260622T060712Z_0.mp4`.
  - Perf dump: `/home/zhiheng/sglang/Vivid_Acceptance/indicator/flowcut-e2e-dual-20260622T060712Z_perf.json`.
  - Callback log: `/home/zhiheng/sglang/Vivid_Acceptance/logs/flowcut_callback_20260622.jsonl`.
  - Observed callbacks: 22 total, starting with `running/progress=1.0/reason=accepted` and ending with `succeeded/progress=100.0`.
  - Service status endpoint returned `status=completed`, `progress=100`, `inference_time_s=618.3000870645046`.

## Notes

- The FlowCut endpoint is Vivid-VR-only; non-Vivid-VR repair pipelines return `code:1`.
- Existing OpenAI-style callback payloads for `/v1/videos/repairs` are not changed.
- Heavy serve smoke should use the default single-card `single_gpu_fa_compile` command unless explicitly validating the dual-card path.
- During the first dual-card E2E attempt, FlowCut callback posts to `127.0.0.1` were intercepted by environment proxy settings and returned `502 Bad Gateway`; the mock callback server received no request. `post_flowcut_callback` now constructs `httpx.AsyncClient(..., trust_env=False)` and has a unit regression for this behavior.
- The generated perf dump for the serve E2E exists, but it is the runtime perf dump format and does not include the Vivid_Acceptance standard `total_runtime_seconds` / `model_inference_runtime_seconds` fields. If this run is promoted to an official acceptance JSON, convert or supplement the metrics before archiving it as a standard indicator.
