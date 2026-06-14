from __future__ import annotations

from typing import Any

import numpy as np

from research_pipeline.interaction.fsm import ACTION_BY_LABEL
from research_pipeline.labels import TARGET_LABELS, label_to_index


def analyze_recognition_risk(report: dict[str, Any]) -> dict[str, Any]:
    recognition = report.get("recognition", {})
    matrix = np.array(recognition.get("confusion_matrix", []), dtype=np.int64)
    if matrix.shape != (len(TARGET_LABELS), len(TARGET_LABELS)):
        return {"status": "invalid_confusion_matrix"}

    no_index = label_to_index("no_gesture")
    no_row = matrix[no_index]
    no_support = int(no_row.sum())
    false_actions = {
        label: int(no_row[index])
        for index, label in enumerate(TARGET_LABELS)
        if label != "no_gesture" and int(no_row[index]) > 0
    }
    false_action_total = int(sum(false_actions.values()))
    false_swipe_total = int(false_actions.get("swipe_left", 0) + false_actions.get("swipe_right", 0))
    swipe_left = label_to_index("swipe_left")
    swipe_right = label_to_index("swipe_right")
    zoom_in = label_to_index("zoom_in")
    zoom_out = label_to_index("zoom_out")

    return {
        "status": "ready",
        "no_gesture_false_action_total": false_action_total,
        "no_gesture_support": no_support,
        "no_gesture_false_action_rate": false_action_total / max(1, no_support),
        "no_gesture_false_swipe_total": false_swipe_total,
        "no_gesture_false_swipe_rate": false_swipe_total / max(1, no_support),
        "no_gesture_false_actions": false_actions,
        "directed_confusions": {
            "swipe_left_as_right": int(matrix[swipe_left, swipe_right]),
            "swipe_right_as_left": int(matrix[swipe_right, swipe_left]),
            "zoom_in_as_out": int(matrix[zoom_in, zoom_out]),
            "zoom_out_as_in": int(matrix[zoom_out, zoom_in]),
        },
        "action_labels": sorted(ACTION_BY_LABEL),
    }
