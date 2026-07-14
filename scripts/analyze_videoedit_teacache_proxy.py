#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BRANCH_TO_DENOISE_FIELD = {
    "cond": "noise_pred_cond_change",
    "uncond": "noise_pred_uncond_change",
    "guided": "noise_pred_guided_change",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def latest_completed_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "completed" or row.get("id") is None:
            continue
        rows[str(row["id"])] = row
    return rows


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return None if not clean else float(statistics.mean(clean))


def median(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return None if not clean else float(statistics.median(clean))


def quantile(values: list[float | None], q: float) -> float | None:
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(clean[lo])
    frac = pos - lo
    return float(clean[lo] * (1.0 - frac) + clean[hi] * frac)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return None
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    output = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            output[order[k]] = rank
        i = j + 1
    return output


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def auc_lower_skipped(skipped: list[float], computed: list[float]) -> float | None:
    if not skipped or not computed:
        return None
    wins = 0
    ties = 0
    for s in skipped:
        for c in computed:
            if s < c:
                wins += 1
            elif s == c:
                ties += 1
    return float((wins + 0.5 * ties) / (len(skipped) * len(computed)))


def load_denoise_actuals(
    trace_dir: Path,
    ids: set[str],
) -> dict[tuple[str, int, int, str], float]:
    actuals: dict[tuple[str, int, int, str], float] = {}
    for video_id in sorted(ids):
        path = trace_dir / f"{video_id}.jsonl"
        if not path.exists():
            continue
        for record in read_jsonl(path):
            if record.get("event") != "videoedit_denoise_step":
                continue
            window_index = record.get("window_index")
            step = record.get("step")
            if window_index is None or step is None:
                continue
            for branch, field in BRANCH_TO_DENOISE_FIELD.items():
                metrics = record.get(field)
                if not isinstance(metrics, dict):
                    continue
                if not metrics.get("available") or not metrics.get("has_previous"):
                    continue
                if metrics.get("shape_changed"):
                    continue
                relative_l1 = safe_float(metrics.get("relative_l1"))
                if relative_l1 is None:
                    continue
                actuals[(video_id, int(window_index), int(step), branch)] = relative_l1
    return actuals


def load_teacache_decisions(
    trace_path: Path,
    task_to_video_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(trace_path):
        if record.get("event") != "teacache_decision":
            continue
        task_id = str(record.get("request_id") or "")
        video_id = task_to_video_id.get(task_id)
        if video_id is None:
            continue
        branch = str(record.get("branch") or "")
        if branch not in ("cond", "uncond"):
            continue
        window_index = record.get("window_index")
        step = record.get("denoise_step")
        if window_index is None or step is None:
            continue
        rows.append(
            {
                "video_id": video_id,
                "task_id": task_id,
                "window_index": int(window_index),
                "branch": branch,
                "step": int(step),
                "skipped": bool(record.get("skipped")),
                "is_boundary_step": bool(record.get("is_boundary_step")),
                "threshold": safe_float(record.get("threshold")),
                "rel_l1": safe_float(record.get("rel_l1")),
                "rescaled_l1": safe_float(record.get("rescaled_l1")),
                "accumulated_before": safe_float(record.get("accumulated_before")),
                "candidate_accumulated": safe_float(record.get("candidate_accumulated")),
                "accumulated_after": safe_float(record.get("accumulated_after")),
            }
        )
    return rows


def build_decision_rows(
    decisions: list[dict[str, Any]],
    normal_actuals: dict[tuple[str, int, int, str], float],
    teacache_actuals: dict[tuple[str, int, int, str], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        key = (
            decision["video_id"],
            decision["window_index"],
            decision["step"],
            decision["branch"],
        )
        row = dict(decision)
        row["normal_actual_relative_l1"] = normal_actuals.get(key)
        row["teacache_actual_relative_l1"] = teacache_actuals.get(key)
        rows.append(row)
    return rows


def reference_stats(
    rows: list[dict[str, Any]],
    *,
    reference_computed_policy: str,
) -> dict[tuple[str, str], dict[str, float | int | None]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["branch"])].append(row)

    output: dict[tuple[str, str], dict[str, float | int | None]] = {}
    for key, items in grouped.items():
        computed_items = [
            row
            for row in items
            if not row["skipped"]
            and row["normal_actual_relative_l1"] is not None
            and (
                reference_computed_policy == "all"
                or not row["is_boundary_step"]
            )
        ]
        skipped_items = [
            row
            for row in items
            if row["skipped"] and row["normal_actual_relative_l1"] is not None
        ]
        computed_values = [row["normal_actual_relative_l1"] for row in computed_items]
        skipped_values = [row["normal_actual_relative_l1"] for row in skipped_items]
        output[key] = {
            "computed_n": len(computed_values),
            "computed_mean": mean(computed_values),
            "computed_median": median(computed_values),
            "computed_p90": quantile(computed_values, 0.9),
            "skipped_n": len(skipped_values),
            "skipped_mean": mean(skipped_values),
            "skipped_median": median(skipped_values),
            "skipped_p90": quantile(skipped_values, 0.9),
            "auc_lower_skipped": auc_lower_skipped(skipped_values, computed_values),
        }
    return output


def classify_decision(
    row: dict[str, Any],
    stats: dict[str, float | int | None],
) -> str:
    actual = row.get("normal_actual_relative_l1")
    if actual is None:
        return "unknown"
    if not row["skipped"]:
        return "compute"
    computed_median = stats.get("computed_median")
    computed_p90 = stats.get("computed_p90")
    if computed_p90 is not None and actual > computed_p90:
        return "high_risk_skip"
    if computed_median is not None and actual > computed_median:
        return "medium_risk_skip"
    return "ok_skip"


def add_risk_columns(
    rows: list[dict[str, Any]],
    refs: dict[tuple[str, str], dict[str, float | int | None]],
) -> None:
    for row in rows:
        stats = refs.get((row["video_id"], row["branch"]), {})
        row["normal_computed_median"] = stats.get("computed_median")
        row["normal_computed_p90"] = stats.get("computed_p90")
        row["risk_class"] = classify_decision(row, stats)
        actual = row.get("normal_actual_relative_l1")
        median_ref = stats.get("computed_median")
        p90_ref = stats.get("computed_p90")
        row["actual_minus_computed_median"] = (
            None if actual is None or median_ref is None else actual - median_ref
        )
        row["actual_over_computed_median"] = (
            None if actual is None or median_ref in (None, 0.0) else actual / median_ref
        )
        row["actual_over_computed_p90"] = (
            None if actual is None or p90_ref in (None, 0.0) else actual / p90_ref
        )


def summarize_video_branch(
    rows: list[dict[str, Any]],
    video_id: str,
    branch: str,
    stats: dict[str, float | int | None],
) -> dict[str, Any]:
    items = [row for row in rows if row["video_id"] == video_id and row["branch"] == branch]
    with_actual = [row for row in items if row.get("normal_actual_relative_l1") is not None]
    skipped_actual = [
        row["normal_actual_relative_l1"]
        for row in with_actual
        if row["skipped"]
    ]
    computed_actual = [
        row["normal_actual_relative_l1"]
        for row in with_actual
        if not row["skipped"]
    ]
    proxy = [
        row["rel_l1"]
        for row in with_actual
        if row.get("rel_l1") is not None
    ]
    actual_for_proxy = [
        row["normal_actual_relative_l1"]
        for row in with_actual
        if row.get("rel_l1") is not None
    ]
    risks = Counter(row.get("risk_class") for row in items)
    skipped_steps = sorted({row["step"] for row in items if row["skipped"]})
    computed_steps = sorted({row["step"] for row in items if not row["skipped"]})
    return {
        "video_id": video_id,
        "branch": branch,
        "num_decisions": len(items),
        "num_with_actual": len(with_actual),
        "num_skipped": sum(1 for row in items if row["skipped"]),
        "num_computed": sum(1 for row in items if not row["skipped"]),
        "skip_ratio": (
            None if not items else sum(1 for row in items if row["skipped"]) / len(items)
        ),
        "ok_skip": risks.get("ok_skip", 0),
        "medium_risk_skip": risks.get("medium_risk_skip", 0),
        "high_risk_skip": risks.get("high_risk_skip", 0),
        "computed_actual_mean": mean(computed_actual),
        "computed_actual_median": median(computed_actual),
        "computed_actual_p90": quantile(computed_actual, 0.9),
        "skipped_actual_mean": mean(skipped_actual),
        "skipped_actual_median": median(skipped_actual),
        "skipped_actual_p90": quantile(skipped_actual, 0.9),
        "auc_lower_skipped": stats.get("auc_lower_skipped"),
        "proxy_vs_actual_pearson": pearson(proxy, actual_for_proxy),
        "proxy_vs_actual_spearman": spearman(proxy, actual_for_proxy),
        "skipped_steps": " ".join(str(x) for x in skipped_steps),
        "computed_steps": " ".join(str(x) for x in computed_steps),
    }


def summarize_per_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["branch"], row["step"])].append(row)

    output: list[dict[str, Any]] = []
    for (video_id, branch, step), items in sorted(grouped.items()):
        skipped_count = sum(1 for row in items if row["skipped"])
        computed_count = len(items) - skipped_count
        if skipped_count and computed_count:
            action = "mixed"
        elif skipped_count:
            action = "skip"
        else:
            action = "compute"
        risks = Counter(row.get("risk_class") for row in items)
        normal_values = [row.get("normal_actual_relative_l1") for row in items]
        candidate_values = [row.get("candidate_accumulated") for row in items]
        rel_values = [row.get("rel_l1") for row in items]
        output.append(
            {
                "video_id": video_id,
                "branch": branch,
                "step": step,
                "action": action,
                "num_windows": len(items),
                "skipped_count": skipped_count,
                "computed_count": computed_count,
                "risk_class": (
                    "high_risk_skip"
                    if risks.get("high_risk_skip")
                    else "medium_risk_skip"
                    if risks.get("medium_risk_skip")
                    else "ok_skip"
                    if risks.get("ok_skip")
                    else "compute"
                    if computed_count
                    else "unknown"
                ),
                "normal_actual_mean": mean(normal_values),
                "normal_actual_median": median(normal_values),
                "normal_actual_max": max(
                    [v for v in normal_values if v is not None],
                    default=None,
                ),
                "candidate_mean": mean(candidate_values),
                "candidate_min": min(
                    [v for v in candidate_values if v is not None],
                    default=None,
                ),
                "candidate_max": max(
                    [v for v in candidate_values if v is not None],
                    default=None,
                ),
                "proxy_rel_l1_mean": mean(rel_values),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def top_risk_steps(per_step_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    risky = [
        row
        for row in per_step_rows
        if row.get("risk_class") in ("high_risk_skip", "medium_risk_skip")
    ]
    return sorted(
        risky,
        key=lambda row: (
            1 if row.get("risk_class") == "high_risk_skip" else 0,
            row.get("normal_actual_mean") or -1.0,
        ),
        reverse=True,
    )[:limit]


def write_markdown_report(
    path: Path,
    *,
    matched_ids: list[str],
    video_branch_rows: list[dict[str, Any]],
    per_step_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    video_ids = sorted(set(matched_ids))
    high_total = sum(int(row.get("high_risk_skip") or 0) for row in video_branch_rows)
    medium_total = sum(int(row.get("medium_risk_skip") or 0) for row in video_branch_rows)
    lines: list[str] = []
    lines.append("# TeaCache Proxy Batch Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Matched videos: `{len(video_ids)}`")
    lines.append(f"- Decisions with normal actual: `{aggregate['num_with_normal_actual']}`")
    lines.append(f"- Skip ratio: `{fmt_float(aggregate['skip_ratio'])}`")
    lines.append(f"- Replay accuracy: `{fmt_float(aggregate['replay_accuracy'])}`")
    lines.append(f"- High-risk skipped decisions: `{high_total}`")
    lines.append(f"- Medium-risk skipped decisions: `{medium_total}`")
    lines.append(
        f"- Risk reference: per-video/branch computed-step normal actual "
        f"`median` and `p90` (`{args.reference_computed_policy}` computed policy)"
    )
    lines.append("")
    lines.append("Risk rule: skipped step with normal actual > computed median is medium risk; skipped step with normal actual > computed p90 is high risk.")
    lines.append("")
    lines.append("## Top Risky Skipped Steps")
    lines.append("")
    lines.append("| video | branch | step | risk | normal actual mean | candidate mean | windows |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for row in top_risk_steps(per_step_rows, args.markdown_top_k):
        lines.append(
            "| {video_id} | {branch} | {step} | {risk_class} | {actual} | {candidate} | {windows} |".format(
                video_id=row["video_id"],
                branch=row["branch"],
                step=row["step"],
                risk_class=row["risk_class"],
                actual=fmt_float(row.get("normal_actual_mean"), 6),
                candidate=fmt_float(row.get("candidate_mean"), 6),
                windows=row["num_windows"],
            )
        )
    lines.append("")
    lines.append("## Per Video")
    lines.append("")
    lines.append("| video | branch | skip ratio | high risk | medium risk | skipped mean | computed mean | AUC | Pearson |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(video_branch_rows, key=lambda r: (r["video_id"], r["branch"])):
        lines.append(
            "| {video_id} | {branch} | {skip_ratio} | {high} | {medium} | {skip_mean} | {compute_mean} | {auc} | {pearson} |".format(
                video_id=row["video_id"],
                branch=row["branch"],
                skip_ratio=fmt_float(row.get("skip_ratio")),
                high=row.get("high_risk_skip", 0),
                medium=row.get("medium_risk_skip", 0),
                skip_mean=fmt_float(row.get("skipped_actual_mean"), 6),
                compute_mean=fmt_float(row.get("computed_actual_mean"), 6),
                auc=fmt_float(row.get("auc_lower_skipped")),
                pearson=fmt_float(row.get("proxy_vs_actual_pearson")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze TeaCache proxy decisions against no-TeaCache VideoEdit denoise traces."
        )
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=Path("outputs/erase_data_case_repair_bbox10/manifest.jsonl"),
    )
    parser.add_argument(
        "--baseline-denoise-dir",
        type=Path,
        default=Path("outputs/erase_data_case_repair_bbox10/denoise_traces"),
    )
    parser.add_argument(
        "--teacache-manifest",
        type=Path,
        default=Path(
            "outputs/teacache_sweep_bbox1_aligned/"
            "tc_default_batch_bbox1_ms1_dp0/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--teacache-trace-path",
        type=Path,
        default=Path("outputs/teacache_sweep/teacache_trace_gpu0.jsonl"),
    )
    parser.add_argument(
        "--teacache-denoise-dir",
        type=Path,
        default=Path(
            "outputs/teacache_sweep_bbox1_aligned/"
            "tc_default_batch_bbox1_ms1_dp0/runs/"
            "teacache_thr0p3_start5_end1/denoise_traces"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs_tyx/teacache_proxy_analysis_bbox1_ms1_dp0"),
    )
    parser.add_argument(
        "--reference-computed-policy",
        choices=("all", "non-boundary"),
        default="all",
        help="Which computed steps form the normal-actual reference distribution.",
    )
    parser.add_argument("--markdown-top-k", type=int, default=40)
    args = parser.parse_args()

    baseline_manifest = latest_completed_manifest(args.baseline_manifest.expanduser())
    teacache_manifest = latest_completed_manifest(args.teacache_manifest.expanduser())
    matched_ids = sorted(set(baseline_manifest) & set(teacache_manifest))
    if not matched_ids:
        raise RuntimeError("No matched completed ids between baseline and TeaCache manifests.")

    task_to_video_id = {
        str(teacache_manifest[video_id]["task_id"]): video_id
        for video_id in matched_ids
    }

    normal_actuals = load_denoise_actuals(
        args.baseline_denoise_dir.expanduser(), set(matched_ids)
    )
    teacache_actuals = load_denoise_actuals(
        args.teacache_denoise_dir.expanduser(), set(matched_ids)
    )
    decisions = load_teacache_decisions(
        args.teacache_trace_path.expanduser(), task_to_video_id
    )
    decision_rows = build_decision_rows(decisions, normal_actuals, teacache_actuals)
    refs = reference_stats(
        decision_rows,
        reference_computed_policy=args.reference_computed_policy,
    )
    add_risk_columns(decision_rows, refs)

    video_branch_rows: list[dict[str, Any]] = []
    for video_id in matched_ids:
        for branch in ("cond", "uncond"):
            video_branch_rows.append(
                summarize_video_branch(
                    decision_rows,
                    video_id,
                    branch,
                    refs.get((video_id, branch), {}),
                )
            )

    per_step_rows = summarize_per_step(decision_rows)
    total_decisions = len(decision_rows)
    total_skipped = sum(1 for row in decision_rows if row["skipped"])
    replay_total = sum(
        1
        for row in decision_rows
        if row.get("candidate_accumulated") is not None or row.get("is_boundary_step")
    )
    replay_matches = 0
    for row in decision_rows:
        threshold = row.get("threshold")
        candidate = row.get("candidate_accumulated")
        if row.get("is_boundary_step"):
            expected_skip = False
        elif threshold is None or candidate is None:
            continue
        else:
            expected_skip = candidate < threshold
        replay_matches += int(expected_skip == row["skipped"])

    aggregate = {
        "matched_ids": matched_ids,
        "num_matched_ids": len(matched_ids),
        "num_decisions": total_decisions,
        "num_skipped": total_skipped,
        "skip_ratio": None if not total_decisions else total_skipped / total_decisions,
        "num_with_normal_actual": sum(
            1 for row in decision_rows if row.get("normal_actual_relative_l1") is not None
        ),
        "replay_total": replay_total,
        "replay_matches": replay_matches,
        "replay_accuracy": None if not replay_total else replay_matches / replay_total,
        "risk_counts": dict(Counter(row.get("risk_class") for row in decision_rows)),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_fields = [
        "video_id",
        "task_id",
        "window_index",
        "branch",
        "step",
        "skipped",
        "is_boundary_step",
        "threshold",
        "rel_l1",
        "rescaled_l1",
        "accumulated_before",
        "candidate_accumulated",
        "accumulated_after",
        "normal_actual_relative_l1",
        "teacache_actual_relative_l1",
        "normal_computed_median",
        "normal_computed_p90",
        "actual_minus_computed_median",
        "actual_over_computed_median",
        "actual_over_computed_p90",
        "risk_class",
    ]
    per_step_fields = [
        "video_id",
        "branch",
        "step",
        "action",
        "num_windows",
        "skipped_count",
        "computed_count",
        "risk_class",
        "normal_actual_mean",
        "normal_actual_median",
        "normal_actual_max",
        "candidate_mean",
        "candidate_min",
        "candidate_max",
        "proxy_rel_l1_mean",
    ]
    video_fields = [
        "video_id",
        "branch",
        "num_decisions",
        "num_with_actual",
        "num_skipped",
        "num_computed",
        "skip_ratio",
        "ok_skip",
        "medium_risk_skip",
        "high_risk_skip",
        "computed_actual_mean",
        "computed_actual_median",
        "computed_actual_p90",
        "skipped_actual_mean",
        "skipped_actual_median",
        "skipped_actual_p90",
        "auc_lower_skipped",
        "proxy_vs_actual_pearson",
        "proxy_vs_actual_spearman",
        "skipped_steps",
        "computed_steps",
    ]

    write_csv(output_dir / "per_decision.csv", decision_rows, decision_fields)
    write_csv(output_dir / "per_step_summary.csv", per_step_rows, per_step_fields)
    write_csv(output_dir / "per_video_summary.csv", video_branch_rows, video_fields)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "aggregate": aggregate,
                "video_branch_summary": video_branch_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        output_dir / "summary.md",
        matched_ids=matched_ids,
        video_branch_rows=video_branch_rows,
        per_step_rows=per_step_rows,
        aggregate=aggregate,
        args=args,
    )

    print(f"matched_ids={len(matched_ids)} decisions={total_decisions}")
    print(f"skip_ratio={aggregate['skip_ratio']:.4f} replay_accuracy={aggregate['replay_accuracy']:.4f}")
    print(f"risk_counts={aggregate['risk_counts']}")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
