from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable

EXPECTED_ACTIONS_BY_TASK = {
    "object": {"pointer_hover", "select_confirm", "zoom_in", "zoom_out"},
    "carousel": {"navigate_previous", "navigate_next", "select_confirm"},
    "targets": {"pointer_hover", "select_confirm"},
}

REQUIRED_ACTIONS_BY_TASK = {
    "object": {"select_confirm"},
    "carousel": {"navigate_previous", "navigate_next"},
    "targets": {"pointer_hover", "select_confirm"},
}


def _numeric(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, int | float):
            output.append(float(value))
    return output


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarize_live_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [record for record in records if record.get("type") == "prediction"]
    fps_values = _numeric(record.get("fps") for record in predictions)
    processing_values = _numeric(record.get("processing_ms") for record in predictions)
    confidence_values = _numeric(record.get("confidence") for record in predictions)
    detection_values = _numeric(record.get("detection_rate") for record in predictions)

    gestures = Counter(str(record.get("gesture", "")) for record in predictions if record.get("gesture"))
    actions = Counter(str(record.get("action", "")) for record in predictions if record.get("action"))

    return {
        "frames": len(predictions),
        "source": predictions[0].get("source", "") if predictions else "",
        "method": predictions[0].get("method", "") if predictions else "",
        "session_id": predictions[0].get("session_id", "") if predictions else "",
        "duration_ms": int(predictions[-1].get("timestamp_ms", 0)) if predictions else 0,
        "fps": {
            "mean": round(_mean(fps_values), 3),
            "p50": round(_percentile(fps_values, 50), 3),
            "p95": round(_percentile(fps_values, 95), 3),
        },
        "processing_ms": {
            "mean": round(_mean(processing_values), 3),
            "p50": round(_percentile(processing_values, 50), 3),
            "p95": round(_percentile(processing_values, 95), 3),
        },
        "confidence_mean": round(_mean(confidence_values), 4),
        "detection_rate_mean": round(_mean(detection_values), 4),
        "gesture_counts": dict(gestures.most_common()),
        "action_counts": dict(actions.most_common()),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _rate_per_minute(count: int, duration_ms: int) -> float:
    if duration_ms <= 0:
        return 0.0
    return count / (duration_ms / 60000.0)


def _action_records(task_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not task_records:
        return []
    first_ms = int(task_records[0].get("timestamp_ms", 0))
    actions: list[dict[str, Any]] = []
    for index, record in enumerate(task_records):
        action = str(record.get("action") or "")
        if not action or action == "idle":
            continue
        actions.append(
            {
                "index": index,
                "timestamp_ms": int(record.get("timestamp_ms", 0)),
                "relative_ms": int(record.get("timestamp_ms", 0)) - first_ms,
                "action": action,
                "gesture": record.get("gesture", ""),
                "confidence": float(record.get("confidence", 0.0) or 0.0),
            }
        )
    return actions


def _normalize_expected_actions(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    expected = scenario.get("expected_actions", [])
    output: list[dict[str, Any]] = []
    if not isinstance(expected, list):
        return output
    for index, item in enumerate(expected):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if not action:
            continue
        start_ms = int(item.get("start_ms", item.get("earliest_ms", 0)))
        end_ms = int(item.get("end_ms", item.get("latest_ms", start_ms)))
        target_ms = int(item.get("target_ms", start_ms))
        output.append(
            {
                "id": str(item.get("id") or f"expected_{index + 1}"),
                "action": action,
                "start_ms": min(start_ms, end_ms),
                "end_ms": max(start_ms, end_ms),
                "target_ms": target_ms,
                "required": bool(item.get("required", True)),
            }
        )
    return output


def evaluate_task_scenario(
    task_records: list[dict[str, Any]],
    scenario: dict[str, Any] | None,
    *,
    tolerance_ms: int = 350,
) -> dict[str, Any] | None:
    if not scenario:
        return None

    expected_actions = _normalize_expected_actions(scenario)
    actual_actions = _action_records(task_records)
    unmatched_actual = actual_actions.copy()
    matches: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []

    for expected in expected_actions:
        window_start = expected["start_ms"] - tolerance_ms
        window_end = expected["end_ms"] + tolerance_ms
        candidates = [
            action
            for action in unmatched_actual
            if action["action"] == expected["action"] and window_start <= action["relative_ms"] <= window_end
        ]
        if not candidates:
            missed.append(expected)
            continue
        chosen = min(candidates, key=lambda item: abs(item["relative_ms"] - expected["target_ms"]))
        unmatched_actual.remove(chosen)
        matches.append(
            {
                "expected_id": expected["id"],
                "action": expected["action"],
                "expected_target_ms": expected["target_ms"],
                "actual_relative_ms": chosen["relative_ms"],
                "latency_ms": chosen["relative_ms"] - expected["target_ms"],
                "gesture": chosen["gesture"],
                "confidence": chosen["confidence"],
            }
        )

    required_expected = [item for item in expected_actions if item["required"]]
    matched_required_ids = {item["expected_id"] for item in matches}
    missed_required = [item for item in required_expected if item["id"] not in matched_required_ids]
    latencies = [abs(float(item["latency_ms"])) for item in matches]
    actual_count = len(actual_actions)
    matched_count = len(matches)
    expected_count = len(expected_actions)
    false_trigger_count = len(unmatched_actual)

    return {
        "scenario_id": scenario.get("id", ""),
        "scenario_label": scenario.get("label", ""),
        "tolerance_ms": tolerance_ms,
        "expected_actions": expected_actions,
        "actual_actions": actual_actions,
        "matches": matches,
        "missed_expected": missed,
        "false_triggers": unmatched_actual,
        "task_success": bool(required_expected) and not missed_required,
        "action_precision": round(_fraction(matched_count, actual_count), 4),
        "action_recall": round(_fraction(matched_count, expected_count), 4),
        "required_action_recall": round(_fraction(len(required_expected) - len(missed_required), len(required_expected)), 4)
        if required_expected
        else 0.0,
        "unintended_action_rate": round(_fraction(false_trigger_count, actual_count), 4),
        "latency_abs_ms": {
            "median": round(median(latencies), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 95), 3),
        },
    }


def _warnings_for_task(task: str, frames: int, metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if frames == 0:
        warnings.append("No prediction frames were recorded for this task.")
        return warnings
    if metrics["fps"]["mean"] < 10:
        warnings.append("Mean FPS is below 10; demonstration smoothness is likely weak.")
    if metrics["processing_ms"]["p95"] > 80:
        warnings.append("p95 processing latency is above 80 ms; lower preview/capture quality or optimize inference.")
    if metrics["detection_rate_mean"] < 0.35:
        warnings.append("Hand detection coverage is low; camera position, lighting, or gesture framing needs review.")
    if metrics["pointer_coverage"] < 0.35 and task in {"object", "targets"}:
        warnings.append("Pointer coverage is low for a pointer-driven AR task.")
    if metrics["required_action_coverage"] < 1.0:
        warnings.append("Not all required task actions were observed.")
    if metrics["idle_ratio"] > 0.9:
        warnings.append("Session is mostly idle; not enough active interaction was captured.")
    return warnings


def summarize_task_records(
    records: list[dict[str, Any]],
    *,
    scenarios: dict[str, Any] | None = None,
    tolerance_ms: int = 350,
) -> dict[str, Any]:
    predictions = [record for record in records if record.get("type") == "prediction"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in predictions:
        task = str(record.get("task") or "unknown")
        by_task.setdefault(task, []).append(record)

    task_reports: dict[str, Any] = {}
    for task, task_records in by_task.items():
        frames = len(task_records)
        first_ms = int(task_records[0].get("timestamp_ms", 0)) if task_records else 0
        last_ms = int(task_records[-1].get("timestamp_ms", 0)) if task_records else 0
        duration_ms = max(0, last_ms - first_ms)
        fps_values = _numeric(record.get("fps") for record in task_records)
        processing_values = _numeric(record.get("processing_ms") for record in task_records)
        confidence_values = _numeric(record.get("confidence") for record in task_records)
        detection_values = _numeric(record.get("detection_rate") for record in task_records)
        gestures = Counter(str(record.get("gesture", "")) for record in task_records if record.get("gesture"))
        actions = Counter(str(record.get("action", "")) for record in task_records if record.get("action"))
        non_idle_actions = sum(count for action, count in actions.items() if action != "idle")
        pointer_frames = sum(1 for record in task_records if record.get("pointer"))
        expected_actions = EXPECTED_ACTIONS_BY_TASK.get(task, set())
        required_actions = REQUIRED_ACTIONS_BY_TASK.get(task, set())
        observed_expected = {action for action in actions if action in expected_actions and action != "idle"}
        observed_required = {action for action in actions if action in required_actions}

        metrics = {
            "frames": frames,
            "duration_ms": duration_ms,
            "fps": {
                "mean": round(_mean(fps_values), 3),
                "p50": round(_percentile(fps_values, 50), 3),
                "p95": round(_percentile(fps_values, 95), 3),
            },
            "processing_ms": {
                "mean": round(_mean(processing_values), 3),
                "p50": round(_percentile(processing_values, 50), 3),
                "p95": round(_percentile(processing_values, 95), 3),
            },
            "confidence_mean": round(_mean(confidence_values), 4),
            "detection_rate_mean": round(_mean(detection_values), 4),
            "pointer_coverage": round(_fraction(pointer_frames, frames), 4),
            "idle_ratio": round(_fraction(actions.get("idle", 0), frames), 4),
            "active_action_rate_per_minute": round(_rate_per_minute(non_idle_actions, duration_ms), 4),
            "expected_action_coverage": round(_fraction(len(observed_expected), len(expected_actions)), 4)
            if expected_actions
            else 0.0,
            "required_action_coverage": round(_fraction(len(observed_required), len(required_actions)), 4)
            if required_actions
            else 0.0,
            "gesture_counts": dict(gestures.most_common()),
            "action_counts": dict(actions.most_common()),
        }
        metrics["warnings"] = _warnings_for_task(task, frames, metrics)
        scenario_report = evaluate_task_scenario(task_records, (scenarios or {}).get(task), tolerance_ms=tolerance_ms)
        if scenario_report is not None:
            metrics["ground_truth"] = scenario_report
            if not scenario_report["task_success"]:
                metrics["warnings"].append("Ground-truth scenario was not completed successfully.")
        task_reports[task] = metrics

    return {
        "session": summarize_live_records(predictions),
        "tasks": task_reports,
        "task_order": list(by_task.keys()),
        "notes": [
            "Proxy coverage is computed from live predictions for quick triage.",
            "Ground-truth scenario metrics are included when a task scenario file is provided.",
        ],
    }
