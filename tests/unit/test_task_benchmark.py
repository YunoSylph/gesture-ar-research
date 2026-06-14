from research_pipeline.evaluation.live_sessions import evaluate_task_scenario
from research_pipeline.evaluation.task_benchmark import task_report_to_metrics


def test_task_report_to_metrics_weights_false_and_missed_actions():
    scenario = {
        "id": "unit_task",
        "expected_actions": [
            {"id": "select", "action": "select_confirm", "start_ms": 0, "target_ms": 100, "end_ms": 300},
            {"id": "zoom", "action": "zoom_in", "start_ms": 400, "target_ms": 600, "end_ms": 800},
        ],
    }
    records = [
        {"type": "prediction", "timestamp_ms": 100, "gesture": "click_2f", "action": "select_confirm", "confidence": 0.9},
        {"type": "prediction", "timestamp_ms": 500, "gesture": "swipe_left", "action": "navigate_previous", "confidence": 0.8},
    ]
    report = evaluate_task_scenario(records, scenario, tolerance_ms=100)

    metrics = task_report_to_metrics(report, {"select_confirm": 2.0, "zoom_in": 1.25, "navigate_previous": 1.0})

    assert metrics["task_success_rate"] == 0.0
    assert metrics["weighted_action_precision"] == 2.0 / 3.0
    assert metrics["weighted_action_recall"] == 2.0 / 3.25
    assert metrics["false_action_cost_rate"] == 1.0 / 3.25
    assert metrics["missed_action_cost_rate"] == 1.25 / 3.25
