from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.cli.common import project_path, write_json_report
from research_pipeline.evaluation.error_analysis import analyze_recognition_risk


DEFAULT_REPORTS = {
    "c1t_tcn_full": "artifacts/reports/ipn_c1t_tcn_full_recognition.json",
    "c1t_tcn_full_validated": "artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json",
    "c3_hybrid_clean": "artifacts/reports/c3_hybrid_clean_recognition.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze recognition confusion matrices for AR interaction risks.")
    parser.add_argument("--output", default="artifacts/reports/recognition_risk_analysis.json")
    parser.add_argument("--report", action="append", help="Optional name=path entry. Can be passed multiple times.")
    args = parser.parse_args()

    entries = _parse_entries(args.report) if args.report else DEFAULT_REPORTS
    analysis = {}
    for name, path in entries.items():
        report_path = project_path(path)
        if not report_path.exists():
            analysis[name] = {"status": "missing", "path": str(report_path)}
            continue
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        risk = analyze_recognition_risk(payload)
        risk["path"] = str(report_path)
        analysis[name] = risk

    write_json_report(args.output, {"variants": analysis, "selection_note": _selection_note(analysis)})
    print(f"wrote recognition risk analysis to {project_path(args.output)}")


def _parse_entries(values: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            path = Path(value)
            entries[path.stem] = value
            continue
        name, path = value.split("=", 1)
        entries[name] = path
    return entries


def _selection_note(analysis: dict[str, dict]) -> str:
    ready = {
        name: item
        for name, item in analysis.items()
        if item.get("status") == "ready"
    }
    if not ready:
        return "No complete reports available."
    safest_name, safest = min(
        ready.items(),
        key=lambda item: (
            item[1].get("no_gesture_false_action_rate", 1.0),
            item[1].get("no_gesture_false_swipe_rate", 1.0),
            item[1].get("no_gesture_false_action_total", 10**9),
        ),
    )
    return (
        f"{safest_name} has the lowest no_gesture false-action risk among available reports "
        f"({safest.get('no_gesture_false_action_rate', 0.0):.4f}; "
        f"false-swipe rate {safest.get('no_gesture_false_swipe_rate', 0.0):.4f})."
    )


if __name__ == "__main__":
    main()
