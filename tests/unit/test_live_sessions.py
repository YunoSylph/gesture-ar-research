from research_pipeline.evaluation.live_sessions import evaluate_task_scenario, summarize_live_records, summarize_task_records


def test_summarize_live_records_counts_and_latency() -> None:
    summary = summarize_live_records(
        [
            {
                "type": "prediction",
                "source": "webcam",
                "method": "onnx",
                "session_id": "session",
                "timestamp_ms": 10,
                "gesture": "point_2f",
                "action": "pointer_hover",
                "confidence": 0.8,
                "detection_rate": 1.0,
                "fps": 12.0,
                "processing_ms": 31.0,
            },
            {
                "type": "prediction",
                "source": "webcam",
                "method": "onnx",
                "session_id": "session",
                "timestamp_ms": 90,
                "gesture": "click_2f",
                "action": "select_confirm",
                "confidence": 0.6,
                "detection_rate": 0.5,
                "fps": 10.0,
                "processing_ms": 42.0,
            },
        ]
    )

    assert summary["frames"] == 2
    assert summary["source"] == "webcam"
    assert summary["method"] == "onnx"
    assert summary["duration_ms"] == 90
    assert summary["fps"]["mean"] == 11.0
    assert summary["processing_ms"]["p95"] == 42.0
    assert summary["gesture_counts"] == {"point_2f": 1, "click_2f": 1}
    assert summary["action_counts"] == {"pointer_hover": 1, "select_confirm": 1}


def test_summarize_task_records_reports_required_action_coverage() -> None:
    report = summarize_task_records(
        [
            {
                "type": "prediction",
                "task": "targets",
                "source": "webcam",
                "method": "onnx",
                "session_id": "session",
                "timestamp_ms": 0,
                "gesture": "point_2f",
                "action": "pointer_hover",
                "confidence": 0.9,
                "detection_rate": 1.0,
                "pointer": {"x": 0.4, "y": 0.5},
                "fps": 14.0,
                "processing_ms": 8.0,
            },
            {
                "type": "prediction",
                "task": "targets",
                "source": "webcam",
                "method": "onnx",
                "session_id": "session",
                "timestamp_ms": 1000,
                "gesture": "click_2f",
                "action": "select_confirm",
                "confidence": 0.8,
                "detection_rate": 1.0,
                "pointer": {"x": 0.4, "y": 0.5},
                "fps": 14.0,
                "processing_ms": 9.0,
            },
        ]
    )

    target_report = report["tasks"]["targets"]
    assert report["task_order"] == ["targets"]
    assert target_report["required_action_coverage"] == 1.0
    assert target_report["expected_action_coverage"] == 1.0
    assert target_report["pointer_coverage"] == 1.0
    assert target_report["active_action_rate_per_minute"] == 120.0


def test_evaluate_task_scenario_matches_expected_actions() -> None:
    report = evaluate_task_scenario(
        [
            {
                "type": "prediction",
                "timestamp_ms": 1000,
                "gesture": "point_2f",
                "action": "pointer_hover",
                "confidence": 0.9,
            },
            {
                "type": "prediction",
                "timestamp_ms": 2400,
                "gesture": "click_2f",
                "action": "select_confirm",
                "confidence": 0.8,
            },
            {
                "type": "prediction",
                "timestamp_ms": 3000,
                "gesture": "zoom_in",
                "action": "zoom_in",
                "confidence": 0.7,
            },
        ],
        {
            "id": "target_selection",
            "expected_actions": [
                {"id": "hover", "action": "pointer_hover", "start_ms": 0, "target_ms": 0, "end_ms": 500},
                {"id": "confirm", "action": "select_confirm", "start_ms": 1000, "target_ms": 1400, "end_ms": 1800},
            ],
        },
        tolerance_ms=150,
    )

    assert report is not None
    assert report["task_success"] is True
    assert report["action_precision"] == 0.6667
    assert report["action_recall"] == 1.0
    assert report["unintended_action_rate"] == 0.3333
    assert len(report["false_triggers"]) == 1


def test_summarize_task_records_includes_ground_truth() -> None:
    report = summarize_task_records(
        [
            {
                "type": "prediction",
                "task": "targets",
                "timestamp_ms": 0,
                "gesture": "point_2f",
                "action": "pointer_hover",
                "confidence": 0.9,
                "detection_rate": 1.0,
                "fps": 14.0,
                "processing_ms": 8.0,
            },
            {
                "type": "prediction",
                "task": "targets",
                "timestamp_ms": 1500,
                "gesture": "click_2f",
                "action": "select_confirm",
                "confidence": 0.8,
                "detection_rate": 1.0,
                "fps": 14.0,
                "processing_ms": 9.0,
            },
        ],
        scenarios={
            "targets": {
                "id": "target_selection",
                "expected_actions": [
                    {"id": "hover", "action": "pointer_hover", "start_ms": 0, "target_ms": 0, "end_ms": 250},
                    {"id": "confirm", "action": "select_confirm", "start_ms": 1200, "target_ms": 1500, "end_ms": 1800},
                ],
            }
        },
        tolerance_ms=100,
    )

    assert report["tasks"]["targets"]["ground_truth"]["task_success"] is True
    assert report["tasks"]["targets"]["ground_truth"]["required_action_recall"] == 1.0
