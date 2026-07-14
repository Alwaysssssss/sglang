from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _step_list(values: list[int]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(str(v) for v in sorted(set(values))) + "]"


def summarize_trace(path: Path) -> list[dict[str, Any]]:
    groups: OrderedDict[tuple[str, int | None, str], dict[str, Any]] = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if record.get("event") != "teacache_decision":
                continue

            key = (
                str(record.get("request_id")),
                record.get("window_index"),
                str(record.get("branch")),
            )
            group = groups.setdefault(
                key,
                {
                    "request_id": key[0],
                    "window_index": key[1],
                    "branch": key[2],
                    "num_decisions": 0,
                    "num_skipped": 0,
                    "skipped_steps": [],
                    "computed_steps": [],
                    "boundary_steps": [],
                    "raw_forward_indices": [],
                },
            )
            denoise_step = int(record["denoise_step"])
            group["num_decisions"] += 1
            group["raw_forward_indices"].append(int(record["raw_forward_index"]))
            if record.get("is_boundary_step"):
                group["boundary_steps"].append(denoise_step)
            if record.get("skipped"):
                group["num_skipped"] += 1
                group["skipped_steps"].append(denoise_step)
            else:
                group["computed_steps"].append(denoise_step)

    return list(groups.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TeaCache JSONL trace.")
    parser.add_argument("trace_path", type=Path)
    parser.add_argument(
        "--show-computed",
        action="store_true",
        help="Also print denoise steps that ran the transformer forward.",
    )
    args = parser.parse_args()

    for group in summarize_trace(args.trace_path):
        print(
            "request={request_id} window={window_index} branch={branch} "
            "skipped={num_skipped}/{num_decisions} skipped_steps={steps}".format(
                request_id=group["request_id"],
                window_index=group["window_index"],
                branch=group["branch"],
                num_skipped=group["num_skipped"],
                num_decisions=group["num_decisions"],
                steps=_step_list(group["skipped_steps"]),
            )
        )
        if args.show_computed:
            print(
                "  computed_steps={computed} boundary_steps={boundary}".format(
                    computed=_step_list(group["computed_steps"]),
                    boundary=_step_list(group["boundary_steps"]),
                )
            )


if __name__ == "__main__":
    main()
