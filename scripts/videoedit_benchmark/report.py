#!/usr/bin/env python3
"""Aggregate ranks without treating partial/failed measurements as results."""
import json
from pathlib import Path
import statistics
import sys


def summarize(target):
    variant = json.loads((target / "variant.json").read_text())
    events = []
    for path in sorted(target.glob("rank*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    requests = []
    for index in range(1, 4):
        result_path = target / f"timed{index}-result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result["status"] != "completed":
            continue
        suffix = f"{variant['id']}-timed{index}.mov"
        rows = [e for e in events if (e.get("output") or "").endswith(suffix)]
        request_rows = [e for e in rows if e["kind"] == "request" and e["status"] == "ok"]
        totals = [e["seconds"] for e in request_rows]
        if len(totals) != variant["num_gpus"] or {e["rank"] for e in request_rows} != set(range(variant["num_gpus"])):
            continue
        denoise = [sum(e["seconds"] for e in rows if e["rank"] == rank and e.get("stage") == "VideoEditDenoisingStage")
                   for rank in range(variant["num_gpus"])]
        samples = [json.loads(line) for line in (target / f"timed{index}-samples.jsonl").read_text().splitlines()]
        requests.append(dict(index=index, e2e_seconds=max(totals), denoise_seconds=max(denoise),
                             cpu_memory_peak_bytes=max(int(s["memory_bytes"]) for s in samples),
                             validation=result.get("validation"), ranks=rows))
    row = dict(id=variant["id"], num_gpus=variant["num_gpus"], runs=requests, status="incomplete")
    row["configuration"] = variant
    error = target / "error.json"
    exit_path = target / "exit.json"
    exit_data = json.loads(exit_path.read_text()) if exit_path.exists() else {}
    if error.exists():
        row.update(status="failed", error=json.loads(error.read_text()))
    elif exit_data.get("error") or exit_data.get("returncode", 0) != 0:
        row.update(status="failed", error=exit_data)
    elif len(requests) == 3 and (target / "completion.json").exists():
        times = [r["e2e_seconds"] for r in requests]
        median = statistics.median(times)
        row.update(status="completed", e2e_median_s=median, e2e_min_s=min(times), e2e_max_s=max(times),
                   denoise_median_s=statistics.median(r["denoise_seconds"] for r in requests),
                   gpu_seconds=median * variant["num_gpus"])
        validations = [r.get("validation") for r in requests]
        if all(v and v.get("decoded_sha256") for v in validations):
            row["decoded_repeats_identical"] = len({v["decoded_sha256"] for v in validations}) == 1
        latent_traces = [
            [(e["window"], e["sha256"]) for e in r["ranks"]
             if e["kind"] == "latent" and e["rank"] == 0] for r in requests
        ]
        row["latent_repeats_identical"] = bool(latent_traces[0]) and all(
            trace == latent_traces[0] for trace in latent_traces)
        row["latent_ranks_identical"] = all(
            [(e["window"], e["sha256"]) for e in r["ranks"] if e["kind"] == "latent" and e["rank"] == rank]
            == latent_traces[index]
            for index, r in enumerate(requests) for rank in range(variant["num_gpus"]))
        row["cpu_memory_peak_gib"] = max(r["cpu_memory_peak_bytes"] for r in requests) / 2**30
        row["stage_peaks_gib"] = {
            str(rank): {
                stage: {field: max(e[field] for r in requests for e in r["ranks"]
                                  if e.get("stage") == stage and e["rank"] == rank) / 2**30
                        for field in ("peak_allocated_bytes", "peak_reserved_bytes")}
                for stage in {e["stage"] for r in requests for e in r["ranks"]
                              if e["rank"] == rank and e["kind"] == "stage" and "peak_allocated_bytes" in e}
            } for rank in range(variant["num_gpus"])}
    return row


def main(root):
    rows = [summarize(path.parent) for path in sorted(root.glob("*/variant.json"))]
    baseline = next((r for r in rows if r["id"] == "single_prefetch1" and r["status"] == "completed"), None)
    for row in rows:
        if baseline and row["status"] == "completed":
            row["e2e_speedup"] = baseline["e2e_median_s"] / row["e2e_median_s"]
            row["denoise_speedup"] = baseline["denoise_median_s"] / row["denoise_median_s"]
            base_validation = baseline["runs"][0].get("validation") or {}
            validation = row["runs"][0].get("validation") or {}
            row["decoded_matches_baseline"] = bool(validation.get("decoded_sha256")) and (
                validation.get("decoded_sha256") == base_validation.get("decoded_sha256"))
            row["quality_status"] = "decoded_hash_identical" if row["decoded_matches_baseline"] else "requires_error_psnr_ssim_comparison"
            baseline_trace = [e.get("decisions") for e in baseline["runs"][0]["ranks"]
                              if e.get("stage") == "VideoEditDenoisingStage" and e["rank"] == 0]
            row["teacache_matches_baseline"] = all(
                [e.get("decisions") for e in r["ranks"] if e.get("stage") == "VideoEditDenoisingStage" and e["rank"] == rank]
                == baseline_trace
                for r in row["runs"] for rank in range(row["num_gpus"]))
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| Configuration | Status | E2E median / min / max (s) | Denoise median (s) | GPU seconds |",
             "|---|---|---|---|---|"]
    for row in rows:
        times = " / ".join(f"{row[k]:.3f}" for k in ("e2e_median_s", "e2e_min_s", "e2e_max_s")) if row["status"] == "completed" else "—"
        lines.append(f"| {row['id']} | {row['status']} | {times} | {row.get('denoise_median_s', '—')} | {row.get('gpu_seconds', '—')} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
