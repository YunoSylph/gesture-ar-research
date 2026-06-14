from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import project_path


DEFAULT_REPORT = "artifacts/reports/c4_action_safe_research.json"
DEFAULT_TABLE_DIR = "artifacts/reports/c4_tables"
DEFAULT_FIGURE_DIR = "artifacts/figures"
DEFAULT_MARKDOWN = "artifacts/reports/c4_action_safe_tables.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C4 action-safe research tables and figures.")
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

    summary_rows = _summary_rows(report["evaluation"]["summary"])
    scenario_rows = _scenario_rows(report["evaluation"]["by_scenario"])
    calibration_rows = _calibration_rows(report["calibration"])
    ci_rows = _ci_rows(report["evaluation"].get("bootstrap_ci", {}))
    _write_csv(table_dir / "c4_summary.csv", summary_rows)
    _write_csv(table_dir / "c4_by_scenario.csv", scenario_rows)
    _write_csv(table_dir / "c4_calibration.csv", calibration_rows)
    _write_csv(table_dir / "c4_bootstrap_ci.csv", ci_rows)
    figures = _write_figures(figure_dir, summary_rows, scenario_rows)
    _write_markdown(markdown_path, report, summary_rows, scenario_rows, calibration_rows, ci_rows, figures)
    print(
        json.dumps(
            {
                "summary_csv": str(table_dir / "c4_summary.csv"),
                "scenario_csv": str(table_dir / "c4_by_scenario.csv"),
                "calibration_csv": str(table_dir / "c4_calibration.csv"),
                "bootstrap_ci_csv": str(table_dir / "c4_bootstrap_ci.csv"),
                "markdown": str(markdown_path),
                **{key: str(value) for key, value in figures.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in summary.items():
        rows.append(
            {
                "method": method,
                "action_precision_mean": _round(metrics.get("action_precision_mean")),
                "action_recall_mean": _round(metrics.get("action_recall_mean")),
                "unintended_action_rate_mean": _round(metrics.get("unintended_action_rate_mean")),
                "weighted_action_precision_mean": _round(metrics.get("weighted_action_precision_mean")),
                "weighted_action_recall_mean": _round(metrics.get("weighted_action_recall_mean")),
                "false_action_cost_rate_mean": _round(metrics.get("false_action_cost_rate_mean")),
                "missed_action_cost_rate_mean": _round(metrics.get("missed_action_cost_rate_mean")),
                "false_trigger_rate_per_minute_mean": _round(metrics.get("false_trigger_rate_per_minute_mean")),
                "num_events_mean": _round(metrics.get("num_events_mean")),
            }
        )
    return rows


def _scenario_rows(by_scenario: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scenario, methods in by_scenario.items():
        for method, metrics in methods.items():
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "action_precision": _round(metrics.get("action_precision")),
                    "action_recall": _round(metrics.get("action_recall")),
                    "unintended_action_rate": _round(metrics.get("unintended_action_rate")),
                    "weighted_action_precision": _round(metrics.get("weighted_action_precision")),
                    "weighted_action_recall": _round(metrics.get("weighted_action_recall")),
                    "false_action_cost_rate": _round(metrics.get("false_action_cost_rate")),
                    "missed_action_cost_rate": _round(metrics.get("missed_action_cost_rate")),
                    "false_trigger_rate_per_minute": _round(metrics.get("false_trigger_rate_per_minute")),
                    "num_events": metrics.get("num_events", ""),
                }
            )
    return rows


def _calibration_rows(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in ("best_balanced", "best_safety"):
        item = calibration[name]
        config = item["config"]
        summary = item["summary"]
        rows.append(
            {
                "operating_point": name,
                "default_threshold": config.get("default_threshold"),
                "label_thresholds": json.dumps(config.get("label_thresholds", {}), ensure_ascii=False, sort_keys=True),
                "default_stable_frames": config.get("default_stable_frames"),
                "label_stable_frames": json.dumps(config.get("label_stable_frames", {}), ensure_ascii=False, sort_keys=True),
                "action_precision_mean": _round(summary.get("action_precision_mean")),
                "action_recall_mean": _round(summary.get("action_recall_mean")),
                "unintended_action_rate_mean": _round(summary.get("unintended_action_rate_mean")),
                "false_action_cost_rate_mean": _round(summary.get("false_action_cost_rate_mean")),
            }
        )
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_figures(figure_dir: Path, summary_rows: list[dict[str, Any]], scenario_rows: list[dict[str, Any]]) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    figures: dict[str, Path] = {}
    methods = [row["method"] for row in summary_rows]
    precision = [float(row["action_precision_mean"]) for row in summary_rows]
    recall = [float(row["action_recall_mean"]) for row in summary_rows]
    risk = [float(row["unintended_action_rate_mean"]) for row in summary_rows]
    weighted_risk = [float(row["false_action_cost_rate_mean"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(methods, risk)
    ax.set_title("C4 action-safety risk reduction")
    ax.set_ylabel("Mean unintended action rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_unintended_action_rate.png"
    fig.savefig(path)
    plt.close(fig)
    figures["unintended_action_rate_png"] = path

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(methods, weighted_risk)
    ax.set_title("C4 weighted AR action risk")
    ax.set_ylabel("Mean false action cost rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_false_action_cost_rate.png"
    fig.savefig(path)
    plt.close(fig)
    figures["false_action_cost_rate_png"] = path

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    ax.scatter(recall, precision, s=80)
    for method, x, y in zip(methods, recall, precision):
        ax.annotate(method, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_title("C4 precision-recall operating points")
    ax.set_xlabel("Mean action recall")
    ax.set_ylabel("Mean action precision")
    ax.set_xlim(min(recall) - 0.02, 1.0)
    ax.set_ylim(min(precision) - 0.02, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "c4_precision_recall_tradeoff.png"
    fig.savefig(path)
    plt.close(fig)
    figures["precision_recall_tradeoff_png"] = path
    return figures


def _write_markdown(
    path: Path,
    report: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    ci_rows: list[dict[str, Any]],
    figures: dict[str, Path],
) -> None:
    lines = [
        "# C4 Action-Safe Research Tables",
        "",
        report["method"]["description"],
        "",
        "## Summary",
        "",
        _markdown_table(summary_rows),
        "",
        "## Calibration Operating Points",
        "",
        _markdown_table(calibration_rows),
        "",
        "## Bootstrap 95% CI",
        "",
        _markdown_table(ci_rows),
        "",
        "## Action Cost Matrix",
        "",
        _markdown_table([{"action": action, "cost": cost} for action, cost in report.get("action_costs", {}).items()]),
        "",
        "## By Scenario",
        "",
        _markdown_table(scenario_rows),
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
