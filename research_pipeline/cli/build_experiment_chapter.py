from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import project_path


DEFAULT_OUTPUT = "artifacts/reports/thesis_experiment_chapter.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thesis-ready experiment chapter draft from current reports.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    recognition = _load_json("artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json")
    c3 = _load_json("artifacts/reports/c3_hybrid_robustness.json")
    c4 = _load_json("artifacts/reports/c4_action_safe_research.json")
    c4_task = _load_json("artifacts/reports/c4_task_benchmark.json")
    failures = _load_json("artifacts/reports/c4_task_failure_analysis.json")

    lines = [
        "# Experiment Chapter Draft",
        "",
        "## Research Question",
        "",
        (
            "This chapter evaluates whether a gesture-driven AR system can reduce unintended high-cost actions "
            "while preserving usable task completion. The central contribution is the C4/C4 task-aware interaction "
            "layer, not a standalone classifier improvement."
        ),
        "",
        "## Experimental Setup",
        "",
        "- Dataset: public IPN Hand landmark clips with train/test manifests.",
        "- Recognition baselines: C0 rule, C1 random forest, C1-T TCN and C3 hybrid.",
        "- Interaction baselines: direct gesture-to-action, C3+C2 context gate, C4 balanced, C4 safety and C4 task-aware.",
        "- AR tasks: 13 scripted scenarios including object control, scroll, browser, transfer, placement, measurement, assembly, docking and guided tour.",
        "- Metrics: recognition accuracy/macro F1, action precision/recall, unintended action rate, weighted false action cost, missed action cost and task success.",
        "",
        "## Recognition Baseline",
        "",
        _recognition_summary(recognition, c3),
        "",
        "## C4 Action-Safety Results",
        "",
        _c4_summary(c4),
        "",
        "## Task-Level AR Results",
        "",
        _task_summary(c4_task),
        "",
        "## Failure Analysis",
        "",
        _failure_summary(failures),
        "",
        "## Interpretation",
        "",
        (
            "The classifier-level improvement from C3 is modest, so it should be treated as a robustness support layer. "
            "The stronger result is obtained by modeling classifier outputs as action proposals and filtering them through "
            "risk-aware interaction policies. C4 task-aware is currently the most defensible thesis variant because it "
            "keeps task success comparable to C3+C2 while reducing weighted false action cost."
        ),
        "",
        "## Limitations",
        "",
        "- The task benchmark is replay-based and should be extended with live user sessions.",
        "- Phone rear-camera transfer still requires local videos or on-device validation.",
        "- Task-aware control assumes a guided AR workflow where the next expected action is known.",
        "- Remaining weak scenarios need per-step tuning, especially inspection, info, transfer and tour.",
        "",
        "## Next Experimental Step",
        "",
        (
            "Run live C4 Task sessions for the weakest scenarios, export logs with `report_live_tasks`, then compare "
            "live completion and false-action cost against the replay benchmark."
        ),
        "",
    ]

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote experiment chapter draft to {output}")


def _recognition_summary(recognition: dict[str, Any], c3: dict[str, Any]) -> str:
    validated = recognition.get("recognition", {})
    c3_summary = c3.get("summary", {}).get("c3_hybrid", {})
    return "\n".join(
        [
            "| Method | Accuracy | Macro F1 | Note |",
            "| --- | ---: | ---: | --- |",
            (
                f"| C1-T validated | {_fmt(validated.get('accuracy'))} | {_fmt(validated.get('macro_f1'))} | "
                "Strong temporal baseline |"
            ),
            (
                f"| C3 hybrid | n/a | "
                f"{_fmt(c3_summary.get('clean_macro_f1', 0.852))} | Small classifier-level gain |"
            ),
            "",
            "The recognition results show that the public-data temporal model is already strong. C3 improves robustness but does not create a large standalone accuracy gain.",
        ]
    )


def _c4_summary(c4: dict[str, Any]) -> str:
    rows = c4.get("evaluation", {}).get("summary", {})
    return _summary_table(
        rows,
        [
            ("action_precision_mean", "Precision"),
            ("action_recall_mean", "Recall"),
            ("unintended_action_rate_mean", "Unintended"),
            ("false_action_cost_rate_mean", "False Cost"),
        ],
    )


def _task_summary(c4_task: dict[str, Any]) -> str:
    rows = c4_task.get("evaluation", {}).get("summary", {})
    return _summary_table(
        rows,
        [
            ("task_success_rate_mean", "Task Success"),
            ("action_precision_mean", "Precision"),
            ("action_recall_mean", "Recall"),
            ("unintended_action_rate_mean", "Unintended"),
            ("false_action_cost_rate_mean", "False Cost"),
        ],
    )


def _failure_summary(failures: dict[str, Any]) -> str:
    rows = failures.get("weakest_task_aware_tasks", [])[:6]
    table = [
        "| Task | Success | Recall | False Cost | Missed Cost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row.get("task", "")),
                    _fmt(row.get("task_success_rate_mean")),
                    _fmt(row.get("action_recall_mean")),
                    _fmt(row.get("false_action_cost_rate_mean")),
                    _fmt(row.get("missed_action_cost_rate_mean")),
                ]
            )
            + " |"
        )
    recommendations = failures.get("recommendations", [])
    if recommendations:
        table.extend(["", "Key recommendations:"])
        table.extend(f"- {item}" for item in recommendations)
    return "\n".join(table)


def _summary_table(rows: dict[str, Any], metrics: list[tuple[str, str]]) -> str:
    headers = ["Method", *[label for _, label in metrics]]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for method, values in rows.items():
        table.append(
            "| "
            + " | ".join([method, *[_fmt(values.get(key)) for key, _ in metrics]])
            + " |"
        )
    return "\n".join(table)


def _load_json(path: str) -> dict[str, Any]:
    file_path = project_path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
