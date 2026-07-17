#!/usr/bin/env python3
from __future__ import annotations

from run_videoedit_dual_gpu_bench import (
    HttpClient,
    parse_args,
    run_variant,
    run_warmup_request,
    wait_for_server,
    write_json,
    write_stage_summary_csv,
    write_summary_csv,
)


def main() -> int:
    args = parse_args(
        default_task_prefix="videoedit_bench_4step_gs1_",
        default_native_variant_name="dual_gpu_4step_gs1",
        default_teacache_variant_name="dual_gpu_teacache_4step_gs1",
        description=(
            "Run one VideoEdit sample against an existing service with "
            "num_inference_steps=4 and guidance_scale=1.0, collecting wall-time, "
            "per-stage perf data, and nvidia-smi memory samples."
        ),
    )
    args.num_inference_steps = 4
    args.guidance_scale = 1.0
    args.dynamic_cfg = False

    client = HttpClient(args.base_url, timeout=args.http_timeout_s)
    wait_for_server(client, args.wait_server_timeout_s)

    variants = []
    if args.run_mode in ("native", "both"):
        variants.append((args.native_variant_name, False))
    if args.run_mode in ("teacache", "both"):
        variants.append((args.teacache_variant_name, True))

    run_warmup_request(
        args,
        client,
        enable_teacache=variants[0][1] if variants else False,
    )

    records = [
        run_variant(
            args,
            client,
            variant_name=variant_name,
            enable_teacache=enable_teacache,
        )
        for variant_name, enable_teacache in variants
    ]
    write_json(args.output_dir / "summary_4step_gs1.json", records)
    write_summary_csv(args.output_dir / "summary_4step_gs1.csv", records)
    write_stage_summary_csv(args.output_dir / "stage_summary_4step_gs1.csv", records)
    failures = sum(1 for record in records if record.get("status") != "completed")
    print(f"[summary] {args.output_dir / 'summary_4step_gs1.csv'}")
    print(f"[stage-summary] {args.output_dir / 'stage_summary_4step_gs1.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
