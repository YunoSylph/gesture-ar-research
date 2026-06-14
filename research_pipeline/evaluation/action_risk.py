from __future__ import annotations

from typing import Any

from research_pipeline.evaluation.interaction import ReplayFrame
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, InteractionEvent
from research_pipeline.models.common import Prediction


DEFAULT_ACTION_COSTS = {
    "idle": 0.0,
    "pointer_hover": 0.25,
    "navigate_previous": 1.0,
    "navigate_next": 1.0,
    "zoom_in": 1.25,
    "zoom_out": 1.25,
    "select_confirm": 2.0,
}


def normalize_action_costs(payload: dict[str, Any] | None = None) -> dict[str, float]:
    costs = dict(DEFAULT_ACTION_COSTS)
    raw = payload or {}
    if "action_costs" in raw and isinstance(raw["action_costs"], dict):
        raw = raw["action_costs"]
    for action, cost in raw.items():
        costs[str(action)] = max(0.0, float(cost))
    return costs


def direct_risk_metrics(rows: list[tuple[str, Prediction]], action_costs: dict[str, float] | None = None) -> dict[str, float]:
    costs = action_costs or DEFAULT_ACTION_COSTS
    matched_cost = 0.0
    expected_cost = 0.0
    false_cost = 0.0
    missed_cost = 0.0

    for true_label, prediction in rows:
        expected = ACTION_BY_LABEL.get(true_label, "")
        predicted = ACTION_BY_LABEL.get(prediction.label, "")
        if expected:
            expected_cost += _cost(costs, expected)
        if predicted and predicted == expected:
            matched_cost += _cost(costs, predicted)
            continue
        if predicted:
            false_cost += _cost(costs, predicted)
        if expected:
            missed_cost += _cost(costs, expected)

    return _risk_summary(matched_cost, expected_cost, false_cost, missed_cost)


def event_risk_metrics(
    frames: list[ReplayFrame],
    events: list[InteractionEvent],
    action_costs: dict[str, float] | None = None,
) -> dict[str, float]:
    costs = action_costs or DEFAULT_ACTION_COSTS
    expected_actions = [frame.expected_action for frame in frames if frame.expected_action]
    remaining = [event.action for event in events if event.action]
    matched_cost = 0.0
    expected_cost = sum(_cost(costs, action) for action in expected_actions)

    for expected in expected_actions:
        if expected not in remaining:
            continue
        remaining.remove(expected)
        matched_cost += _cost(costs, expected)

    false_cost = sum(_cost(costs, action) for action in remaining)
    missed_cost = max(0.0, expected_cost - matched_cost)
    return _risk_summary(matched_cost, expected_cost, false_cost, missed_cost)


def _risk_summary(matched_cost: float, expected_cost: float, false_cost: float, missed_cost: float) -> dict[str, float]:
    return {
        "weighted_action_precision": matched_cost / (matched_cost + false_cost) if matched_cost + false_cost else 0.0,
        "weighted_action_recall": matched_cost / expected_cost if expected_cost else 0.0,
        "false_action_cost_total": false_cost,
        "missed_action_cost_total": missed_cost,
        "expected_action_cost_total": expected_cost,
        "false_action_cost_rate": false_cost / expected_cost if expected_cost else 0.0,
        "missed_action_cost_rate": missed_cost / expected_cost if expected_cost else 0.0,
    }


def _cost(costs: dict[str, float], action: str) -> float:
    return float(costs.get(action, 1.0))
