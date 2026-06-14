from __future__ import annotations

from statistics import median
from typing import Any


TASK_METRIC_KEYS = [
    "task_success_rate",
    "action_precision",
    "action_recall",
    "required_action_recall",
    "unintended_action_rate",
    "weighted_action_precision",
    "weighted_action_recall",
    "false_action_cost_rate",
    "missed_action_cost_rate",
    "false_action_cost_total",
    "missed_action_cost_total",
    "expected_action_cost_total",
    "latency_abs_median_ms",
    "latency_abs_p95_ms",
    "num_actual_actions",
    "num_expected_actions",
    "num_false_triggers",
    "num_missed_actions",
]


def task_report_to_metrics(report: dict[str, Any], action_costs: dict[str, float]) -> dict[str, float]:
    expected = report.get("expected_actions", [])
    matches = report.get("matches", [])
    false_triggers = report.get("false_triggers", [])
    missed = report.get("missed_expected", [])
    actual = report.get("actual_actions", [])

    expected_cost = sum(_cost(action_costs, str(item.get("action", ""))) for item in expected)
    matched_cost = sum(_cost(action_costs, str(item.get("action", ""))) for item in matches)
    false_cost = sum(_cost(action_costs, str(item.get("action", ""))) for item in false_triggers)
    missed_cost = sum(_cost(action_costs, str(item.get("action", ""))) for item in missed)
    latency = report.get("latency_abs_ms", {})

    return {
        "task_success_rate": 1.0 if report.get("task_success") else 0.0,
        "action_precision": float(report.get("action_precision", 0.0)),
        "action_recall": float(report.get("action_recall", 0.0)),
        "required_action_recall": float(report.get("required_action_recall", 0.0)),
        "unintended_action_rate": float(report.get("unintended_action_rate", 0.0)),
        "weighted_action_precision": matched_cost / (matched_cost + false_cost) if matched_cost + false_cost else 0.0,
        "weighted_action_recall": matched_cost / expected_cost if expected_cost else 0.0,
        "false_action_cost_rate": false_cost / expected_cost if expected_cost else 0.0,
        "missed_action_cost_rate": missed_cost / expected_cost if expected_cost else 0.0,
        "false_action_cost_total": false_cost,
        "missed_action_cost_total": missed_cost,
        "expected_action_cost_total": expected_cost,
        "latency_abs_median_ms": float(latency.get("median", 0.0)),
        "latency_abs_p95_ms": float(latency.get("p95", 0.0)),
        "num_actual_actions": float(len(actual)),
        "num_expected_actions": float(len(expected)),
        "num_false_triggers": float(len(false_triggers)),
        "num_missed_actions": float(len(missed)),
    }


def summarize_task_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in TASK_METRIC_KEYS:
        values = [float(row.get(key, 0.0)) for row in rows]
        summary[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
    return summary


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile_value / 100.0) * (len(ordered) - 1)))
    return ordered[min(len(ordered) - 1, max(0, index))]


def median_value(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _cost(action_costs: dict[str, float], action: str) -> float:
    return float(action_costs.get(action, 1.0 if action else 0.0))
