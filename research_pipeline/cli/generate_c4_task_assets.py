from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import project_path


DEFAULT_REPORT = "artifacts/reports/c4_task_benchmark.json"
DEFAULT_TABLE_DIR = "artifacts/reports/c4_task_tables"
DEFAULT_FIGURE_DIR = "artifacts/figures"
DEFAULT_MARKDOWN = "artifacts/reports/c4_task_benchmark_tables.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task-level C4 benchmark tables and figures.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report = json.loads(project_path(args.report).read_text(encoding="utf-8"))
    table_dir = project_path(args.table_dir)
    figure_dir = project_path(args.figure_dir)
    markdown_path = project_path(args.markdown)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = _rows_from_summary(report["evaluation"]["summary"], ["method"])
    task_rows = _rows_from_summary(report["evaluation"]["by_task"], ["task", "method"])
    scenario_rows = _rows_from_summary(report["evaluation"]["by_scenario"], ["scenario", "method"])
    ci_rows = _ci_rows(report["evaluation"].get("bootstrap_ci", {}))

    _write_csv(table_dir / "c4_task_summary.csv", summary_rows)
    _write_csv(table_dir / "c4_task_by_task.csv", task_rows)
    _write_csv(table_dir / "c4_task_by_scenario.csv", scenario_rows)
    _write_csv(table_dir / "c4_task_bootstrap_ci.csv", ci_rows)
    figures = _write_figures(figure_dir, summary_rows, task_rows)
    _write_markdown(markdown_path, report, summary_rows, task_rows, scenario_rows, ci_rows, figures)
    print(
        json.dumps(
            {
                "summary_csv": str(table_dir / "c4_task_summary.csv"),
                "by_task_csv": str(table_dir / "c4_task_by_task.csv"),
                "by_scenario_csv": str(table_dir / "c4_task_by_scenario.csv"),
                "bootstrap_ci_csv": str(table_dir / "c4_task_bootstrap_ci.csv"),
                "markdown": str(markdown_path),
                **{key: str(value) for key, value in figures.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _rows_from_summary(summary: dict[str, Any], split_columns: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, metrics in summary.items():
        parts = key.split(" / ")
        row = {column: parts[index] if index < len(parts) else "" for index, column in enumerate(split_columns)}
        row.update(
            {
                "task_success_rate_mean": _round(metrics.get("task_success_rate_mean")),
                "action_precision_mean": _round(metrics.get("action_precision_mean")),
                "action_recall_mean": _round(metrics.get("action_recall_mean")),
                "required_action_recall_mean": _round(metrics.get("required_action_recall_mean")),
                "unintended_action_rate_mean": _round(metrics.get("unintended_action_rate_mean")),
                "weighted_action_precision_mean": _round(metrics.get("weighted_action_precision_mean")),
                "weighted_action_recall_mean": _round(metrics.get("weighted_action_recall_mean")),
                "false_action_cost_rate_mean": _round(metrics.get("false_action_cost_rate_mean")),
                "missed_action_cost_rate_mean": _round(metrics.get("missed_action_cost_rate_mean")),
                "latency_abs_median_ms_mean": _round(metrics.get("latency_abs_median_ms_mean")),
                "latency_abs_p95_ms_mean": _round(metrics.get("latency_abs_p95_ms_mean")),
            }
        )
        rows.append(row)
    return rows


def _ci_rows(bootstrap_ci: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in bootstrap_ci.items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "p2_5": _round(values.get("p2_5")),
                    "mean": _round(values.get("mean")),
                    "p97_5": _round(values.get("p97_5")),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(figure_dir: Path, summary_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    figures: dict[str, Path] = {}
    methods = [row["method"] for row in summary_rows]
    success = [float(row["task_success_rate_mean"]) for row in summary_rows]
    cost = [float(row["false_action_cost_rate_mean"]) for row in summary_rows]
    unintended = [float(row["unintended_action_rate_mean"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(methods, success)
    ax.set_title("Task-level AR completion rate")
    ax.set_ylabel("Mean task success rate")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_task_success_rate.png"
    fig.savefig(path)
    plt.close(fig)
    figures["task_success_rate_png"] = path

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(methods, cost)
    ax.set_title("Task-level weighted false action cost")
    ax.set_ylabel("Mean false action cost rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_task_false_action_cost_rate.png"
    fig.savefig(path)
    plt.close(fig)
    figures["task_false_action_cost_rate_png"] = path

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(methods, unintended)
    ax.set_title("Task-level unintended AR action rate")
    ax.set_ylabel("Mean unintended action rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_task_unintended_action_rate.png"
    fig.savefig(path)
    plt.close(fig)
    figures["task_unintended_action_rate_png"] = path

    safety_rows = [row for row in task_rows if row.get("method") == "c4_safety"]
    if safety_rows:
        labels = [row["task"] for row in safety_rows]
        values = [float(row["task_success_rate_mean"]) for row in safety_rows]
        fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
        ax.bar(labels, values)
        ax.set_title("C4 safety task completion by AR scenario")
        ax.set_ylabel("Task success rate")
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = figure_dir / "c4_task_success_by_scenario.png"
        fig.savefig(path)
        plt.close(fig)
        figures["task_success_by_scenario_png"] = path
    return figures


def _write_markdown(
    path: Path,
    report: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
    figures: dict[str, Path],
) -> None:
    lines = [
        "# C4 Task-Level AR Benchmark Tables",
        "",
        report["method"]["description"],
        "",
        "## Summary",
        "",
        _markdown_table(summary_rows),
        "",
        "## By Task",
        "",
        _markdown_table(task_rows),
        "",
        "## By Perturbation Scenario",
        "",
        _markdown_table(scenario_rows),
        "",
        "## Bootstrap 95% CI",
        "",
        _markdown_table(ci_rows),
        "",
    ]
    if figures:
        lines.extend(["## Figures", ""])
        for name, figure_path in figures.items():
            lines.append(f"- `{name}`: `{figure_path}`")
        lines.append("")
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
