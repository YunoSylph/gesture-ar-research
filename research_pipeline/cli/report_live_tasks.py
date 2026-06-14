from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import load_yaml, project_path
from research_pipeline.evaluation.live_sessions import summarize_task_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DIR = PROJECT_ROOT / "artifacts/live_sessions"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/reports/live_task_report.json"
DEFAULT_SCENARIO = PROJECT_ROOT / "configs/interaction/ar_task_scenarios.yaml"


def latest_session_path() -> Path:
    paths = sorted(DEFAULT_SESSION_DIR.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"No live session logs found in {DEFAULT_SESSION_DIR}")
    return paths[0]


def read_jsonl_tail(path: Path, *, tail_records: int) -> list[dict[str, Any]]:
    records: deque[dict[str, Any]] = deque(maxlen=tail_records if tail_records > 0 else None)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return list(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a task-level report from a Gesture AR live JSONL session.")
    parser.add_argument("--input", type=Path, default=None, help="Path to a live session JSONL. Defaults to latest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--no-scenario", action="store_true")
    parser.add_argument("--tolerance-ms", type=int, default=350)
    parser.add_argument(
        "--tail-records",
        type=int,
        default=20000,
        help="Use only the latest N JSONL records. Set 0 to read the full file.",
    )
    args = parser.parse_args()

    input_path = args.input or latest_session_path()
    records = read_jsonl_tail(input_path, tail_records=args.tail_records)
    scenario_payload: dict[str, Any] = {}
    if not args.no_scenario and args.scenario.exists():
        scenario_payload = load_yaml(project_path(args.scenario))
    scenarios = scenario_payload.get("tasks", {}) if scenario_payload else {}
    tolerance_ms = int(scenario_payload.get("tolerance_ms", args.tolerance_ms)) if scenario_payload else args.tolerance_ms

    report = summarize_task_records(records, scenarios=scenarios, tolerance_ms=tolerance_ms)
    report["input"] = str(input_path)
    report["tail_records"] = args.tail_records
    report["scenario"] = str(args.scenario) if scenario_payload else ""
    report["tolerance_ms"] = tolerance_ms

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    session = report["session"]
    task_names = ", ".join(report["task_order"]) or "none"
    print(
        "frames={frames} tasks={tasks} fps_mean={fps:.2f} output={output}".format(
            frames=session["frames"],
            tasks=task_names,
            fps=session["fps"]["mean"],
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
