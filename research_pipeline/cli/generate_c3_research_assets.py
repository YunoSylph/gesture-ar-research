from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import project_path


DEFAULT_ROBUSTNESS = "artifacts/reports/c3_hybrid_robustness.json"
DEFAULT_ABLATION = "artifacts/reports/c3_research_ablation.json"
DEFAULT_TABLE_DIR = "artifacts/reports/c3_tables"
DEFAULT_FIGURE_DIR = "artifacts/figures"
DEFAULT_MARKDOWN = "artifacts/reports/c3_research_tables.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready C3 research tables and figures.")
    parser.add_argument("--robustness", default=DEFAULT_ROBUSTNESS)
    parser.add_argument("--ablation", default=DEFAULT_ABLATION)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    robustness = _read_json(project_path(args.robustness))
    ablation = _read_json(project_path(args.ablation))
    table_dir = project_path(args.table_dir)
    figure_dir = project_path(args.figure_dir)
    markdown_path = project_path(args.markdown)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    robust_summary_rows = _summary_rows(robustness.get("summary", {}))
    robust_scenario_rows = _robustness_scenario_rows(robustness.get("robustness", {}))
    ablation_rows = _summary_rows(ablation.get("ablation", {}).get("summary", {}))
    policy_rows = _policy_rows(ablation.get("policy_ablation", {}))
    calibration_rows = _calibration_rows(ablation.get("calibration", {}).get("top_candidates", []))

    outputs = {
        "robustness_summary_csv": str(table_dir / "c3_robustness_summary.csv"),
        "robustness_scenarios_csv": str(table_dir / "c3_robustness_by_scenario.csv"),
        "ablation_summary_csv": str(table_dir / "c3_ablation_summary.csv"),
        "policy_ablation_csv": str(table_dir / "c3_policy_ablation.csv"),
        "calibration_csv": str(table_dir / "c3_calibration_candidates.csv"),
        "markdown": str(markdown_path),
    }
    _write_csv(Path(outputs["robustness_summary_csv"]), robust_summary_rows)
    _write_csv(Path(outputs["robustness_scenarios_csv"]), robust_scenario_rows)
    _write_csv(Path(outputs["ablation_summary_csv"]), ablation_rows)
    _write_csv(Path(outputs["policy_ablation_csv"]), policy_rows)
    _write_csv(Path(outputs["calibration_csv"]), calibration_rows)

    figures = _write_figures(figure_dir, robust_scenario_rows, ablation_rows, policy_rows)
    outputs.update({name: str(path) for name, path in figures.items()})
    _write_markdown(markdown_path, robust_summary_rows, ablation_rows, policy_rows, calibration_rows, figures)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, metrics in sorted(summary.items()):
        rows.append(
            {
                "method": method,
                "clean_macro_f1": _round(metrics.get("clean_macro_f1")),
                "perturbed_macro_f1_mean": _round(metrics.get("perturbed_macro_f1_mean")),
                "macro_f1_drop": _round(metrics.get("macro_f1_drop")),
                "perturbed_false_action_rate_mean": _round(metrics.get("perturbed_no_gesture_false_action_rate_mean")),
            }
        )
    return rows


def _robustness_scenario_rows(robustness: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scenario, methods in robustness.items():
        for method, payload in methods.items():
            recognition = payload.get("recognition", {})
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "accuracy": _round(recognition.get("accuracy")),
                    "macro_f1": _round(recognition.get("macro_f1")),
                    "weighted_f1": _round(recognition.get("weighted_f1")),
                    "balanced_accuracy": _round(recognition.get("balanced_accuracy")),
                    "no_gesture_false_action_rate": _round(payload.get("no_gesture_false_action_rate")),
                }
            )
    return rows


def _policy_rows(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, variants in sorted(policy.items()):
        for mode, metrics in sorted(variants.items()):
            rows.append(
                {
                    "method": method,
                    "policy": mode,
                    "action_precision": _round(metrics.get("action_precision")),
                    "action_recall": _round(metrics.get("action_recall")),
                    "unintended_action_rate": _round(metrics.get("unintended_action_rate")),
                    "no_gesture_false_action_rate": _round(metrics.get("no_gesture_false_action_rate")),
                    "false_trigger_rate_per_minute": _round(metrics.get("false_trigger_rate_per_minute")),
                }
            )
    return rows


def _calibration_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        config = item.get("config", {})
        summary = item.get("summary", {})
        rows.append(
            {
                "rank": item.get("rank"),
                "score": _round(item.get("score")),
                "neural_weight": config.get("neural_weight"),
                "geometry_weight": config.get("geometry_weight"),
                "action_threshold": config.get("action_threshold"),
                "enable_safety_gate": config.get("enable_safety_gate"),
                "clean_macro_f1": _round(summary.get("clean_macro_f1")),
                "perturbed_macro_f1_mean": _round(summary.get("perturbed_macro_f1_mean")),
                "perturbed_false_action_rate_mean": _round(summary.get("perturbed_no_gesture_false_action_rate_mean")),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    figure_dir: Path,
    robust_scenario_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    figures: dict[str, Path] = {}
    if robust_scenario_rows:
        scenario_order = []
        for row in robust_scenario_rows:
            if row["scenario"] not in scenario_order:
                scenario_order.append(row["scenario"])
        _plot_grouped_bars(
            plt,
            figure_dir / "c3_robustness_macro_f1.png",
            scenario_order,
            robust_scenario_rows,
            "method",
            "macro_f1",
            "C3 robustness by perturbation",
            "Macro F1",
            ["c1t_direct", "c3_hybrid"],
        )
        figures["robustness_macro_f1_png"] = figure_dir / "c3_robustness_macro_f1.png"

    if ablation_rows:
        _plot_single_bars(
            plt,
            figure_dir / "c3_ablation_perturbed_macro_f1.png",
            ablation_rows,
            "method",
            "perturbed_macro_f1_mean",
            "Recognition ablation under perturbations",
            "Mean perturbed macro F1",
        )
        figures["ablation_macro_f1_png"] = figure_dir / "c3_ablation_perturbed_macro_f1.png"

    if policy_rows:
        _plot_single_bars(
            plt,
            figure_dir / "c3_policy_unintended_action_rate.png",
            policy_rows,
            lambda row: f"{row['method']} {row['policy']}",
            "unintended_action_rate",
            "Interaction-policy false action risk",
            "Unintended action rate",
        )
        figures["policy_unintended_png"] = figure_dir / "c3_policy_unintended_action_rate.png"
    return figures


def _plot_grouped_bars(plt, path: Path, labels: list[str], rows: list[dict[str, Any]], group_key: str, metric: str, title: str, ylabel: str, methods: list[str]) -> None:
    width = 0.38
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=150)
    for offset, method in zip([-width / 2, width / 2], methods):
        values = []
        for label in labels:
            match = next((row for row in rows if row["scenario"] == label and row[group_key] == method), None)
            values.append(float(match[metric]) if match else 0.0)
        ax.bar([pos + offset for pos in positions], values, width=width, label=method)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_single_bars(plt, path: Path, rows: list[dict[str, Any]], label_key, metric: str, title: str, ylabel: str) -> None:
    labels = [label_key(row) if callable(label_key) else row[label_key] for row in rows]
    values = [float(row[metric] or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_markdown(
    path: Path,
    robust_summary_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    figures: dict[str, Path],
) -> None:
    lines = [
        "# C3 Research Tables",
        "",
        "Generated from `artifacts/reports/c3_hybrid_robustness.json` and `artifacts/reports/c3_research_ablation.json`.",
        "",
        "## Robustness Summary",
        "",
        _markdown_table(robust_summary_rows),
        "",
        "## Recognition Ablation",
        "",
        _markdown_table(ablation_rows),
        "",
        "## Interaction Policy Ablation",
        "",
        _markdown_table(policy_rows),
        "",
        "## Calibration Candidates",
        "",
        _markdown_table(calibration_rows[:8]),
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
    if value is None:
        return ""
    if isinstance(value, (int, str, bool)):
        return value
    return round(float(value), 4)


if __name__ == "__main__":
    main()
