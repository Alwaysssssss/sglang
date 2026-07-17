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
        default_task_prefix="videoedit_bench_torchcompile_",
        description=(
            "Run one VideoEdit sample twice against an existing dual-GPU "
            "torch.compile service: no TeaCache and TeaCache, with wall-time "
            "and nvidia-smi memory sampling."
        ),
    )
    client = HttpClient(args.base_url, timeout=args.http_timeout_s)
    wait_for_server(client, args.wait_server_timeout_s)

    run_warmup_request(args, client, enable_teacache=False)

    records = [
        run_variant(
            args,
            client,
            variant_name="dual_gpu_torchcompile",
            enable_teacache=False,
        ),
        run_variant(
            args,
            client,
            variant_name="dual_gpu_torchcompile_teacache",
            enable_teacache=True,
        ),
    ]
    write_json(args.output_dir / "torchcompile_summary.json", records)
    write_summary_csv(args.output_dir / "torchcompile_summary.csv", records)
    write_stage_summary_csv(args.output_dir / "torchcompile_stage_summary.csv", records)
    failures = sum(1 for record in records if record.get("status") != "completed")
    print(f"[summary] {args.output_dir / 'torchcompile_summary.csv'}")
    print(f"[stage-summary] {args.output_dir / 'torchcompile_stage_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
