from __future__ import annotations

import argparse
from typing import Any

from research_pipeline.cli.common import project_path, write_json_report


DEFAULT_REPORT = "artifacts/reports/c4_task_benchmark.json"
DEFAULT_OUTPUT_JSON = "artifacts/reports/c4_task_failure_analysis.json"
DEFAULT_OUTPUT_MD = "artifacts/reports/c4_task_failure_analysis.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze task-level C4 benchmark failure modes.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    import json

    report = json.loads(project_path(args.report).read_text(encoding="utf-8"))
    evaluation = report.get("evaluation", {})
    method_summary = evaluation.get("summary", {})
    by_task_rows = _rows_from_summary(evaluation.get("by_task", {}), ["task", "method"])
    task_aware_rows = [row for row in by_task_rows if row.get("method") == "c4_task_aware"]
    c2_rows = {row["task"]: row for row in by_task_rows if row.get("method") == "c3_c2_default"}
    direct_rows = {row["task"]: row for row in by_task_rows if row.get("method") == "c1t_direct"}

    weak_tasks = sorted(
        task_aware_rows,
        key=lambda row: (
            float(row.get("task_success_rate_mean", 0.0)),
            -float(row.get("false_action_cost_rate_mean", 0.0)),
            -float(row.get("missed_action_cost_rate_mean", 0.0)),
        ),
    )
    comparisons = []
    for row in task_aware_rows:
        task = str(row["task"])
        c2 = c2_rows.get(task, {})
        direct = direct_rows.get(task, {})
        comparisons.append(
            {
                "task": task,
                "task_aware_success": _round(row.get("task_success_rate_mean")),
                "task_aware_false_cost": _round(row.get("false_action_cost_rate_mean")),
                "task_aware_unintended": _round(row.get("unintended_action_rate_mean")),
                "delta_success_vs_c2": _round(float(row.get("task_success_rate_mean", 0.0)) - float(c2.get("task_success_rate_mean", 0.0))),
                "delta_false_cost_vs_c2": _round(float(row.get("false_action_cost_rate_mean", 0.0)) - float(c2.get("false_action_cost_rate_mean", 0.0))),
                "delta_false_cost_vs_direct": _round(float(row.get("false_action_cost_rate_mean", 0.0)) - float(direct.get("false_action_cost_rate_mean", 0.0))),
            }
        )
    comparisons = sorted(comparisons, key=lambda row: float(row["task_aware_false_cost"]), reverse=True)
    recommendations = _recommendations(method_summary, weak_tasks, comparisons)
    payload = {
        "source_report": args.report,
        "method_summary": method_summary,
        "weakest_task_aware_tasks": weak_tasks[:8],
        "task_aware_comparisons": comparisons,
        "recommendations": recommendations,
    }
    write_json_report(args.output_json, payload)
    _write_markdown(project_path(args.output_md), payload)
    print(f"wrote C4 task failure analysis to {project_path(args.output_json)} and {project_path(args.output_md)}")


def _rows_from_summary(summary: dict[str, Any], split_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, metrics in summary.items():
        parts = key.split(" / ")
        row = {column: parts[index] if index < len(parts) else "" for index, column in enumerate(split_columns)}
        row.update(metrics)
        rows.append(row)
    return rows


def _recommendations(
    method_summary: dict[str, Any],
    weak_tasks: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[str]:
    recommendations = [
        "Keep C4 task-aware as the main thesis variant because it preserves C3+C2 task success while reducing false action cost.",
        "Do not present C3 as the central contribution; its recognition-level improvement is too small.",
    ]
    if weak_tasks:
        names = ", ".join(str(row["task"]) for row in weak_tasks[:4])
        recommendations.append(f"Prioritize live tuning and per-step diagnostics for the weakest task-aware scenarios: {names}.")
    if comparisons:
        high_cost = ", ".join(str(row["task"]) for row in comparisons[:4])
        recommendations.append(f"Inspect high residual false-cost tasks first: {high_cost}.")
    c4 = method_summary.get("c4_task_aware", {})
    if float(c4.get("latency_abs_p95_ms_mean", 0.0)) > 250:
        recommendations.append("Investigate latency introduced by stable-frame gating in task-aware mode before final demo recording.")
    return recommendations


def _write_markdown(path, payload: dict[str, Any]) -> None:
    rows = payload["weakest_task_aware_tasks"]
    comparisons = payload["task_aware_comparisons"]
    lines = [
        "# C4 Task Failure Analysis",
        "",
        "This report summarizes where the task-aware AR interaction layer still fails and which tasks should be improved first.",
        "",
        "## Weakest C4 Task-Aware Tasks",
        "",
        _markdown_table(
            [
                {
                    "task": row["task"],
                    "success": _round(row.get("task_success_rate_mean")),
                    "precision": _round(row.get("action_precision_mean")),
                    "recall": _round(row.get("action_recall_mean")),
                    "false_cost": _round(row.get("false_action_cost_rate_mean")),
                    "missed_cost": _round(row.get("missed_action_cost_rate_mean")),
                }
                for row in rows
            ]
        ),
        "",
        "## Task-Aware Deltas",
        "",
        _markdown_table(comparisons),
        "",
        "## Recommendations",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["recommendations"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0])
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(table)


def _round(value: Any) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, int | str | bool):
        return value
    return round(float(value), 4)


if __name__ == "__main__":
    main()
