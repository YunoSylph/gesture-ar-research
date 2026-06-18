from __future__ import annotations

from research_pipeline.evaluation.live_protocol import aggregate_session_reports
from research_pipeline.evaluation.live_sessions import summarize_task_records


def _session_report(success: bool, precision: float, recall: float) -> dict:
    return {
        "session": {
            "frames": 100,
            "fps": {"mean": 20.0},
            "processing_ms": {"p95": 40.0},
            "detection_rate_mean": 0.9,
            "confidence_mean": 0.8,
        },
        "tasks": {
            "object": {
                "ground_truth": {
                    "task_success": success,
                    "action_precision": precision,
                    "action_recall": recall,
                    "required_action_recall": recall,
                    "latency_abs_ms": {"median": 120.0},
                }
            }
        },
    }


def test_aggregate_averages_action_metrics_across_sessions() -> None:
    reports = [_session_report(True, 0.8, 0.9), _session_report(False, 0.6, 0.7)]
    agg = aggregate_session_reports(reports)
    assert agg["num_sessions"] == 2
    assert agg["tasks"]["object"]["sessions"] == 2
    assert agg["tasks"]["object"]["task_success_rate"] == 0.5
    assert agg["tasks"]["object"]["action_precision_mean"] == 0.7
    assert agg["overall"]["action_recall_mean"] == 0.8
    assert agg["quality"]["fps_mean"] == 20.0
    assert agg["quality"]["total_frames"] == 200


def test_aggregate_handles_sessions_without_ground_truth() -> None:
    reports = [
        {
            "session": {
                "frames": 50,
                "fps": {"mean": 15.0},
                "processing_ms": {"p95": 50.0},
                "detection_rate_mean": 0.5,
                "confidence_mean": 0.6,
            },
            "tasks": {"object": {}},
        }
    ]
    agg = aggregate_session_reports(reports)
    assert agg["num_sessions"] == 1
    assert agg["tasks"] == {}  # no ground truth -> no per-task action metrics
    assert agg["overall"]["scored_task_runs"] == 0
    assert agg["quality"]["fps_mean"] == 15.0


def test_end_to_end_summarize_then_aggregate() -> None:
    scenario = {
        "object": {
            "id": "object",
            "label": "Object",
            "expected_actions": [
                {"id": "s", "action": "select_confirm", "start_ms": 0, "end_ms": 500, "target_ms": 200, "required": True}
            ],
        }
    }
    records = [
        {"type": "prediction", "task": "object", "action": "select_confirm", "gesture": "click_2f",
         "timestamp_ms": 200, "confidence": 0.9, "fps": 20, "processing_ms": 40, "detection_rate": 0.9},
        {"type": "prediction", "task": "object", "action": "idle", "gesture": "no_gesture",
         "timestamp_ms": 300, "confidence": 0.5, "fps": 20, "processing_ms": 40, "detection_rate": 0.9},
    ]
    report = summarize_task_records(records, scenarios=scenario)
    agg = aggregate_session_reports([report])
    assert agg["tasks"]["object"]["task_success_rate"] == 1.0
    assert agg["overall"]["scored_task_runs"] == 1
