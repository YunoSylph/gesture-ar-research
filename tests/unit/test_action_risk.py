from research_pipeline.evaluation.action_risk import direct_risk_metrics, event_risk_metrics
from research_pipeline.evaluation.interaction import ReplayFrame
from research_pipeline.interaction.fsm import InteractionEvent
from research_pipeline.models.common import prediction_from_scores


def test_direct_risk_metrics_weight_high_cost_false_action() -> None:
    rows = [
        ("no_gesture", prediction_from_scores({"click_2f": 1.0})),
        ("point_2f", prediction_from_scores({"point_2f": 1.0})),
    ]

    metrics = direct_risk_metrics(rows, {"select_confirm": 2.0, "pointer_hover": 0.25})

    assert metrics["false_action_cost_total"] == 2.0
    assert metrics["expected_action_cost_total"] == 0.25
    assert metrics["false_action_cost_rate"] == 8.0


def test_event_risk_metrics_matches_expected_actions_by_cost() -> None:
    frames = [
        ReplayFrame(timestamp_ms=0, label="point_2f", confidence=0.9, expected_action="pointer_hover"),
        ReplayFrame(timestamp_ms=100, label="click_2f", confidence=0.9, expected_action="select_confirm"),
    ]
    events = [
        InteractionEvent(timestamp_ms=0, gesture="point_2f", action="pointer_hover", confidence=0.9, state="cooldown"),
        InteractionEvent(timestamp_ms=100, gesture="zoom_in", action="zoom_in", confidence=0.9, state="cooldown"),
    ]

    metrics = event_risk_metrics(frames, events, {"pointer_hover": 0.25, "select_confirm": 2.0, "zoom_in": 1.25})

    assert metrics["weighted_action_recall"] == 0.25 / 2.25
    assert metrics["false_action_cost_total"] == 1.25
    assert metrics["missed_action_cost_total"] == 2.0
