from __future__ import annotations

from research_pipeline.cli.benchmark_online_gesture import (
    COMPARISON_METHODS,
    _statistical_comparison,
)


def test_temporal_stabilized_arm_registered() -> None:
    # The anti-jitter stabilization arm sits in the smoothing stage of the ablation.
    assert "c6_temporal_stabilized" in COMPARISON_METHODS
    assert COMPARISON_METHODS.index("c6_temporal_stabilized") > COMPARISON_METHODS.index("direct_c6")


def _results_with_costs(costs_by_method: dict[str, list[float]]) -> dict[str, dict]:
    return {
        method: {"task_replay": {"tasks": [{"false_action_cost": c, "false_actions": c} for c in costs]}}
        for method, costs in costs_by_method.items()
    }


def test_statistical_comparison_reports_reduction_vs_baseline() -> None:
    results = _results_with_costs(
        {
            "direct_c6": [10.0, 12.0, 9.0, 11.0, 13.0, 10.0],
            "c6_validation_tarc": [1.0, 2.0, 0.0, 1.0, 2.0, 1.0],
        }
    )
    rows = _statistical_comparison(results, baseline_method="direct_c6")
    by_metric = {row["metric"]: row for row in rows if row["method"] == "c6_validation_tarc"}
    cost_row = by_metric["false_action_cost"]
    assert cost_row["delta"] < 0
    assert cost_row["delta_ci_high"] < 0  # whole CI below zero -> significant reduction
    assert cost_row["prob_improvement"] > 0.95
    assert cost_row["n"] == 6


def test_statistical_comparison_skips_mismatched_lengths() -> None:
    results = _results_with_costs({"direct_c6": [1.0, 2.0, 3.0], "broken": [1.0]})
    rows = _statistical_comparison(results, baseline_method="direct_c6")
    assert all(row["method"] != "broken" for row in rows)


def test_statistical_comparison_empty_without_baseline() -> None:
    results = _results_with_costs({"some_method": [1.0, 2.0]})
    assert _statistical_comparison(results, baseline_method="direct_c6") == []
