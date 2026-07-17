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
        default_task_prefix="videoedit_bench_torchcompile_native_",
        default_native_variant_name="dual_gpu_torchcompile",
        description=(
            "Run one VideoEdit sample against an existing dual-GPU torch.compile "
            "service with TeaCache disabled, collecting wall-time, per-stage "
            "perf data, and nvidia-smi memory samples."
        ),
    )
    client = HttpClient(args.base_url, timeout=args.http_timeout_s)
    wait_for_server(client, args.wait_server_timeout_s)

    run_warmup_request(args, client, enable_teacache=False)

    record = run_variant(
        args,
        client,
        variant_name=args.native_variant_name,
        enable_teacache=False,
    )
    records = [record]
    write_json(args.output_dir / "torchcompile_native_summary.json", records)
    write_summary_csv(args.output_dir / "torchcompile_native_summary.csv", records)
    write_stage_summary_csv(
        args.output_dir / "torchcompile_native_stage_summary.csv", records
    )
    print(f"[summary] {args.output_dir / 'torchcompile_native_summary.csv'}")
    print(
        "[stage-summary] "
        f"{args.output_dir / 'torchcompile_native_stage_summary.csv'}"
    )
    return 0 if record.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
