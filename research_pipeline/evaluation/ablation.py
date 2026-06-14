from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.evaluation.interaction import ReplayFrame, compute_interaction_metrics, replay_predictions
from research_pipeline.evaluation.robustness import PerturbationConfig, perturb_tensor
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, ContextPolicyConfig
from research_pipeline.models.common import Prediction


Recognizer = Callable[[LandmarkTensor], Prediction]


@dataclass(slots=True)
class CalibrationResult:
    rank: int
    score: float
    config: dict[str, Any]
    summary: dict[str, float]


def score_candidate(summary: dict[str, float], *, false_action_penalty: float = 0.25, drop_penalty: float = 0.1) -> float:
    return (
        float(summary.get("perturbed_macro_f1_mean", 0.0))
        - false_action_penalty * float(summary.get("perturbed_no_gesture_false_action_rate_mean", 0.0))
        - drop_penalty * float(summary.get("macro_f1_drop", 0.0))
    )


def benchmark_policy_manifest(
    manifest_path: str | Path,
    recognizers: dict[str, Recognizer],
    scenario: PerturbationConfig,
    *,
    seed: int = 42,
    max_records: int | None = None,
    c2_policy: ContextPolicyConfig | None = None,
    frames_per_clip: int = 3,
    frame_step_ms: int = 100,
    separator_ms: int = 220,
) -> dict[str, dict]:
    records = read_jsonl(manifest_path)
    if max_records is not None:
        records = records[:max_records]
    base_dir = Path(manifest_path).parent
    rng = np.random.default_rng(seed)
    predictions: dict[str, list[tuple[str, Prediction]]] = {name: [] for name in recognizers}

    for record in records:
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        perturbed = perturb_tensor(tensor, scenario, rng)
        for name, recognizer in recognizers.items():
            predictions[name].append((record.target_label, recognizer(perturbed)))

    report: dict[str, dict] = {}
    for name, rows in predictions.items():
        direct = _direct_policy_metrics(rows)
        c2_frames = _rows_to_replay_frames(
            rows,
            frames_per_clip=frames_per_clip,
            frame_step_ms=frame_step_ms,
            separator_ms=separator_ms,
        )
        c2_events = replay_predictions(c2_frames, c2_policy)
        report[name] = {
            "direct": direct,
            "c2": compute_interaction_metrics(c2_frames, c2_events),
        }
    return report


def _direct_policy_metrics(rows: list[tuple[str, Prediction]]) -> dict[str, float | int]:
    expected_actions = [ACTION_BY_LABEL.get(true, "") for true, _ in rows]
    predicted_actions = [ACTION_BY_LABEL.get(prediction.label, "") for _, prediction in rows]
    expected_count = sum(1 for action in expected_actions if action)
    predicted_count = sum(1 for action in predicted_actions if action)
    true_positive = sum(
        1
        for expected, predicted in zip(expected_actions, predicted_actions)
        if expected and predicted == expected
    )
    false_triggers = sum(
        1
        for expected, predicted in zip(expected_actions, predicted_actions)
        if predicted and predicted != expected
    )
    no_gesture_rows = sum(1 for expected in expected_actions if not expected)
    no_gesture_false = sum(
        1
        for expected, predicted in zip(expected_actions, predicted_actions)
        if not expected and predicted
    )
    return {
        "action_precision": true_positive / predicted_count if predicted_count else 0.0,
        "action_recall": true_positive / expected_count if expected_count else 0.0,
        "unintended_action_rate": false_triggers / max(1, predicted_count),
        "no_gesture_false_action_rate": no_gesture_false / max(1, no_gesture_rows),
        "num_expected_actions": expected_count,
        "num_events": predicted_count,
        "false_triggers": false_triggers,
    }


def _rows_to_replay_frames(
    rows: list[tuple[str, Prediction]],
    *,
    frames_per_clip: int,
    frame_step_ms: int,
    separator_ms: int,
) -> list[ReplayFrame]:
    frames: list[ReplayFrame] = []
    timestamp = 0
    for true_label, prediction in rows:
        expected_action = ACTION_BY_LABEL.get(true_label, "")
        for index in range(frames_per_clip):
            frames.append(
                ReplayFrame(
                    timestamp_ms=timestamp,
                    label=prediction.label,
                    confidence=prediction.confidence,
                    expected_action=expected_action if index == 0 else "",
                )
            )
            timestamp += frame_step_ms
        frames.append(ReplayFrame(timestamp_ms=timestamp, label="no_gesture", confidence=1.0))
        timestamp += separator_ms
    return frames
