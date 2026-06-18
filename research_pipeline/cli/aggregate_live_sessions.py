from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_pipeline.evaluation.live_protocol import aggregate_session_reports
from research_pipeline.evaluation.live_sessions import summarize_task_records

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DIR = PROJECT_ROOT / "artifacts/live_sessions"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/reports/live_protocol_summary.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate logged live Gesture AR sessions into one reproducible live-evaluation report."
    )
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR, help="Directory of *.jsonl session logs.")
    parser.add_argument("--scenarios", type=Path, default=None, help="Optional JSON of task scenarios for ground-truth metrics.")
    parser.add_argument("--tolerance-ms", type=int, default=350)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    session_paths = sorted(args.session_dir.glob("*.jsonl"))
    if not session_paths:
        raise FileNotFoundError(f"No live session logs found in {args.session_dir}")

    scenarios = None
    if args.scenarios is not None:
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))

    reports: list[dict[str, Any]] = []
    for path in session_paths:
        records = read_jsonl(path)
        report = summarize_task_records(records, scenarios=scenarios, tolerance_ms=args.tolerance_ms)
        report["input"] = str(path)
        reports.append(report)

    aggregate = aggregate_session_reports(reports)
    aggregate["inputs"] = [str(path) for path in session_paths]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)

    overall = aggregate["overall"]
    print(
        "sessions={n} scored_runs={runs} task_success_rate={succ:.3f} "
        "action_precision={prec:.3f} action_recall={rec:.3f} output={out}".format(
            n=aggregate["num_sessions"],
            runs=overall["scored_task_runs"],
            succ=overall["task_success_rate"],
            prec=overall["action_precision_mean"],
            rec=overall["action_recall_mean"],
            out=args.output,
        )
    )


if __name__ == "__main__":
    main()
