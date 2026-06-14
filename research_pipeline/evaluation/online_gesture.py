from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from research_pipeline.interaction.fsm import ACTION_BY_LABEL
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.common import Prediction


EVENT_FIELDNAMES = [
    "sequence_id",
    "frame_index",
    "timestamp_ms",
    "ground_truth_label",
    "model_label",
    "model_confidence",
    "top2_margin",
    "proposal_label",
    "proposal_state",
    "controller_mode",
    "expected_label",
    "final_action",
    "action_accepted",
    "rejection_reason",
    "cooldown_remaining",
    "risk_cost",
    "task_id",
    "task_step",
]


DEFAULT_ACTION_COSTS = {
    "idle": 0.0,
    "pointer_hover": 0.25,
    "navigate_previous": 1.0,
    "navigate_next": 1.0,
    "zoom_in": 1.25,
    "zoom_out": 1.25,
    "select_confirm": 2.0,
}


@dataclass(slots=True)
class OnlineEvent:
    sequence_id: str
    frame_index: int
    timestamp_ms: int
    ground_truth_label: str = ""
    model_label: str = ""
    model_confidence: float = 0.0
    top2_margin: float = 0.0
    proposal_label: str = "no_gesture"
    proposal_state: str = "idle"
    controller_mode: str = "stability_threshold"
    expected_label: str = ""
    final_action: str = "idle"
    action_accepted: bool = False
    rejection_reason: str = ""
    cooldown_remaining: int = 0
    risk_cost: float = 0.0
    task_id: str = ""
    task_step: str = ""


@dataclass(slots=True)
class LabelSegment:
    sequence_id: str
    label: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class ProposalDecision:
    proposal_label: str
    proposal_state: str
    controller_mode: str
    final_action: str
    action_accepted: bool
    rejection_reason: str
    cooldown_remaining: int
    risk_cost: float


@dataclass(slots=True)
class ProposalControllerConfig:
    default_threshold: float = 0.55
    label_thresholds: dict[str, float] = field(default_factory=dict)
    default_stable_frames: int = 2
    label_stable_frames: dict[str, int] = field(default_factory=dict)
    cooldown_ms: int = 250
    expected_threshold_delta: float = -0.05
    unexpected_threshold_delta: float = 0.08
    action_costs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_COSTS))


class StabilityProposalController:
    """Small online validation layer used by the replay benchmark.

    It intentionally keeps the policy simple: raw model labels become proposals
    only after confidence and temporal stability checks. Accepted proposals are
    mapped to AR actions with a cooldown. TARC-specific task context can be
    emulated by passing an expected label, but no hidden ground truth is required.
    """

    def __init__(self, config: ProposalControllerConfig | None = None):
        self.config = config or ProposalControllerConfig()
        self.candidate_label = ""
        self.candidate_count = 0
        self.last_action_ms = -10**9

    def reset(self) -> None:
        self.candidate_label = ""
        self.candidate_count = 0
        self.last_action_ms = -10**9

    def update(self, prediction: Prediction, timestamp_ms: int, *, expected_label: str = "") -> ProposalDecision:
        label = prediction.label if prediction.label in TARGET_LABELS else "no_gesture"
        action = ACTION_BY_LABEL.get(label, "idle")
        remaining = max(0, self.last_action_ms + self.config.cooldown_ms - timestamp_ms)

        if label == "no_gesture":
            self.candidate_label = ""
            self.candidate_count = 0
            return ProposalDecision(
                proposal_label="no_gesture",
                proposal_state="idle",
                controller_mode="stability_threshold",
                final_action="idle",
                action_accepted=False,
                rejection_reason="idle",
                cooldown_remaining=remaining,
                risk_cost=0.0,
            )

        if prediction.confidence < self._threshold(label, expected_label):
            self.candidate_label = ""
            self.candidate_count = 0
            return ProposalDecision(
                proposal_label="no_gesture",
                proposal_state="rejected",
                controller_mode="stability_threshold",
                final_action="idle",
                action_accepted=False,
                rejection_reason="low_confidence",
                cooldown_remaining=remaining,
                risk_cost=_cost(self.config.action_costs, action),
            )

        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        if remaining > 0:
            return ProposalDecision(
                proposal_label=label,
                proposal_state="cooldown",
                controller_mode="stability_threshold",
                final_action="idle",
                action_accepted=False,
                rejection_reason="cooldown",
                cooldown_remaining=remaining,
                risk_cost=_cost(self.config.action_costs, action),
            )

        if self.candidate_count < self._stable_frames(label):
            return ProposalDecision(
                proposal_label=label,
                proposal_state="tracking",
                controller_mode="stability_threshold",
                final_action="idle",
                action_accepted=False,
                rejection_reason="unstable",
                cooldown_remaining=0,
                risk_cost=_cost(self.config.action_costs, action),
            )

        self.last_action_ms = timestamp_ms
        return ProposalDecision(
            proposal_label=label,
            proposal_state="accepted",
            controller_mode="stability_threshold",
            final_action=action,
            action_accepted=True,
            rejection_reason="",
            cooldown_remaining=0,
            risk_cost=_cost(self.config.action_costs, action),
        )

    def _threshold(self, label: str, expected_label: str) -> float:
        threshold = float(self.config.label_thresholds.get(label, self.config.default_threshold))
        if expected_label in TARGET_LABELS and expected_label != "no_gesture":
            threshold += self.config.expected_threshold_delta if label == expected_label else self.config.unexpected_threshold_delta
        return max(0.01, min(0.99, threshold))

    def _stable_frames(self, label: str) -> int:
        return max(1, int(self.config.label_stable_frames.get(label, self.config.default_stable_frames)))


def top2_margin(scores: dict[str, float]) -> float:
    ordered = sorted((float(scores.get(label, 0.0)) for label in TARGET_LABELS), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    return max(0.0, ordered[0] - ordered[1])


def event_to_dict(event: OnlineEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["model_confidence"] = round(float(payload["model_confidence"]), 6)
    payload["top2_margin"] = round(float(payload["top2_margin"]), 6)
    payload["risk_cost"] = round(float(payload["risk_cost"]), 6)
    payload["action_accepted"] = bool(payload["action_accepted"])
    return payload


def build_label_segments(
    events: Iterable[OnlineEvent],
    label_getter: Callable[[OnlineEvent], str],
    *,
    include_no_gesture: bool = False,
) -> list[LabelSegment]:
    segments: list[LabelSegment] = []
    by_sequence: dict[str, list[OnlineEvent]] = {}
    for event in events:
        by_sequence.setdefault(event.sequence_id, []).append(event)

    for sequence_id, sequence_events in sorted(by_sequence.items()):
        ordered = sorted(sequence_events, key=lambda item: item.frame_index)
        current_label = ""
        start_event: OnlineEvent | None = None
        previous_event: OnlineEvent | None = None

        def close_segment(end_event: OnlineEvent | None) -> None:
            if start_event is None or end_event is None:
                return
            if current_label == "no_gesture" and not include_no_gesture:
                return
            if not current_label:
                return
            segments.append(
                LabelSegment(
                    sequence_id=sequence_id,
                    label=current_label,
                    start_frame=start_event.frame_index,
                    end_frame=end_event.frame_index,
                    start_ms=start_event.timestamp_ms,
                    end_ms=end_event.timestamp_ms,
                )
            )

        for event in ordered:
            label = label_getter(event) or "no_gesture"
            if label not in TARGET_LABELS:
                label = "no_gesture"
            if start_event is None:
                current_label = label
                start_event = event
                previous_event = event
                continue
            if label != current_label:
                close_segment(previous_event)
                current_label = label
                start_event = event
            previous_event = event
        close_segment(previous_event)
    return segments


def compute_online_metrics(
    events: list[OnlineEvent],
    *,
    min_segment_iou: float = 0.1,
    latency_grace_ms: int = 1000,
) -> dict[str, Any]:
    unavailable: list[str] = []
    duration_minutes = _duration_minutes(events)
    labeled_events = [event for event in events if event.ground_truth_label in TARGET_LABELS]
    if not events:
        return {
            "frames": 0,
            "duration_minutes": 0.0,
            "metrics": {},
            "unavailable_metrics": ["No events were produced."],
        }

    metrics: dict[str, Any] = {
        "recognition_accuracy": None,
        "macro_f1": None,
        "macro_f1_model": None,
        "macro_f1_proposal": None,
        "frame_accuracy": None,
        "frame_accuracy_model": None,
        "frame_accuracy_proposal": None,
        "segment_precision": None,
        "segment_recall": None,
        "segment_f1": None,
        "onset_error_ms_mean": None,
        "onset_error_ms_median": None,
        "offset_error_ms_mean": None,
        "offset_error_ms_median": None,
        "decision_latency_ms_mean": None,
        "decision_latency_ms_median": None,
        "false_positives_per_minute": None,
        "false_negatives_per_gesture": None,
        "label_switch_rate_per_minute": _label_switch_rate(events, duration_minutes),
        "no_gesture_false_positive_rate": None,
        "accepted_action_rate_per_minute": None,
        "rejected_action_rate_per_minute": None,
        "accepted_action_count": 0,
        "rejected_action_count": 0,
    }

    if not labeled_events:
        unavailable.extend(
            [
                "Frame-level and segment-level metrics require ground_truth_label.",
                "no_gesture/action confusion requires ground_truth_label.",
            ]
        )
    else:
        model_correct = sum(1 for event in labeled_events if event.model_label == event.ground_truth_label)
        proposal_correct = sum(1 for event in labeled_events if event.proposal_label == event.ground_truth_label)
        metrics["frame_accuracy_model"] = model_correct / len(labeled_events)
        metrics["frame_accuracy_proposal"] = proposal_correct / len(labeled_events)
        metrics["frame_accuracy"] = metrics["frame_accuracy_proposal"]
        model_classification = _classification_metrics(
            [event.ground_truth_label for event in labeled_events],
            [event.model_label for event in labeled_events],
        )
        proposal_classification = _classification_metrics(
            [event.ground_truth_label for event in labeled_events],
            [event.proposal_label for event in labeled_events],
        )
        metrics["macro_f1_model"] = model_classification["macro_f1"]
        metrics["macro_f1_proposal"] = proposal_classification["macro_f1"]
        metrics["macro_f1"] = proposal_classification["macro_f1"]
        metrics["recognition_accuracy"] = metrics["frame_accuracy_proposal"]

        gt_no_gesture = [event for event in labeled_events if event.ground_truth_label == "no_gesture"]
        if gt_no_gesture:
            fp_no_gesture = sum(1 for event in gt_no_gesture if event.proposal_label != "no_gesture")
            metrics["no_gesture_false_positive_rate"] = fp_no_gesture / len(gt_no_gesture)
        else:
            unavailable.append("no_gesture_false_positive_rate requires no_gesture frames in the event log.")

        true_segments = build_label_segments(labeled_events, lambda event: event.ground_truth_label)
        pred_segments = build_label_segments(labeled_events, lambda event: event.proposal_label)
        matches = _match_segments(true_segments, pred_segments, min_segment_iou=min_segment_iou)
        matched_true = {match["true_index"] for match in matches}
        matched_pred = {match["pred_index"] for match in matches}

        precision = len(matches) / len(pred_segments) if pred_segments else 0.0
        recall = len(matches) / len(true_segments) if true_segments else 0.0
        metrics["segment_precision"] = precision
        metrics["segment_recall"] = recall
        metrics["segment_f1"] = _f1(precision, recall)

        onset_errors = [pred_segments[item["pred_index"]].start_ms - true_segments[item["true_index"]].start_ms for item in matches]
        offset_errors = [pred_segments[item["pred_index"]].end_ms - true_segments[item["true_index"]].end_ms for item in matches]
        latencies = _decision_latencies(labeled_events, true_segments, latency_grace_ms=latency_grace_ms)
        metrics["onset_error_ms_mean"] = _mean(onset_errors)
        metrics["onset_error_ms_median"] = _median(onset_errors)
        metrics["offset_error_ms_mean"] = _mean(offset_errors)
        metrics["offset_error_ms_median"] = _median(offset_errors)
        metrics["decision_latency_ms_mean"] = _mean(latencies)
        metrics["decision_latency_ms_median"] = _median(latencies)

        false_positive_segments = len(pred_segments) - len(matched_pred)
        false_negative_segments = len(true_segments) - len(matched_true)
        metrics["false_positives_per_minute"] = false_positive_segments / duration_minutes if duration_minutes else 0.0
        metrics["false_negatives_per_gesture"] = false_negative_segments / len(true_segments) if true_segments else 0.0
        metrics["true_segments"] = len(true_segments)
        metrics["predicted_segments"] = len(pred_segments)
        metrics["matched_segments"] = len(matches)

    action_candidates = [
        event
        for event in events
        if event.proposal_label in ACTION_BY_LABEL and ACTION_BY_LABEL.get(event.proposal_label, "idle") != "idle"
    ]
    if action_candidates:
        accepted = [event for event in action_candidates if event.action_accepted]
        rejected = [
            event
            for event in action_candidates
            if not event.action_accepted and event.rejection_reason not in {"", "idle"}
        ]
        metrics["accepted_action_count"] = len(accepted)
        metrics["rejected_action_count"] = len(rejected)
        metrics["accepted_action_rate_per_minute"] = len(accepted) / duration_minutes if duration_minutes else 0.0
        metrics["rejected_action_rate_per_minute"] = len(rejected) / duration_minutes if duration_minutes else 0.0
        metrics["accepted_action_ratio"] = len(accepted) / len(action_candidates)
        metrics["rejected_action_ratio"] = len(rejected) / len(action_candidates)
    else:
        unavailable.append("accepted/rejected action rates require non-idle action proposals.")

    return {
        "frames": len(events),
        "labeled_frames": len(labeled_events),
        "duration_minutes": duration_minutes,
        "metrics": metrics,
        "unavailable_metrics": unavailable,
    }


def write_events_csv(path: str | Path, events: Iterable[OnlineEvent]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        for event in events:
            writer.writerow(event_to_dict(event))


def write_events_jsonl(path: str | Path, events: Iterable[OnlineEvent]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event_to_dict(event), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_summary_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Online Gesture Evaluation Summary",
        "",
        f"Mode: `{summary.get('mode', 'unknown')}`",
        f"Data mode: `{summary.get('data_mode', 'unknown')}`",
        f"Manifest: `{summary.get('manifest', '')}`",
        f"Predictor: `{summary.get('predictor', 'unknown')}`",
        "",
        "## Data Availability",
        "",
    ]
    availability = summary.get("availability", {})
    for key, value in availability.items():
        lines.append(f"- `{key}`: {value}")
    limitations = summary.get("limitations", [])
    if limitations:
        lines.extend(["", "## Limitations", ""])
        for item in limitations:
            lines.append(f"- {item}")

    metrics = summary.get("evaluation", {}).get("metrics", {})
    lines.extend(["", "## Metrics", ""])
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6f}")
        else:
            lines.append(f"- `{key}`: {value}")

    unavailable = summary.get("evaluation", {}).get("unavailable_metrics", [])
    if unavailable:
        lines.extend(["", "## Graceful Fallback Notes", ""])
        for item in unavailable:
            lines.append(f"- {item}")

    outputs = summary.get("outputs", {})
    if outputs:
        lines.extend(["", "## Outputs", ""])
        for key, value in outputs.items():
            lines.append(f"- `{key}`: `{value}`")

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def write_summary_figure_svg(path: str | Path, metrics: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = [
        ("Frame acc.", metrics.get("frame_accuracy")),
        ("Seg. F1", metrics.get("segment_f1")),
        ("No-gesture FP", metrics.get("no_gesture_false_positive_rate")),
        ("Accepted/min", _scaled_rate(metrics.get("accepted_action_rate_per_minute"))),
        ("Rejected/min", _scaled_rate(metrics.get("rejected_action_rate_per_minute"))),
    ]
    width = 760
    height = 240
    bar_x = 150
    bar_width = 520
    row_h = 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Arial, sans-serif" font-size="20" fill="#172033">Online Gesture Metrics</text>',
    ]
    for index, (label, value) in enumerate(items):
        y = 62 + index * row_h
        numeric = None if value is None else max(0.0, min(1.0, float(value)))
        fill = "#5f6fdf" if index < 2 else "#d66b4d"
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial, sans-serif" font-size="14" fill="#3c465a">{label}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="20" rx="3" fill="#e8ecf4"/>')
        if numeric is not None:
            parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_width * numeric:.1f}" height="20" rx="3" fill="{fill}"/>')
            parts.append(
                f'<text x="{bar_x + bar_width + 12}" y="{y + 15}" font-family="Arial, sans-serif" font-size="13" fill="#172033">{numeric:.3f}</text>'
            )
        else:
            parts.append(
                f'<text x="{bar_x + 8}" y="{y + 15}" font-family="Arial, sans-serif" font-size="13" fill="#687386">n/a</text>'
            )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")


def _match_segments(true_segments: list[LabelSegment], pred_segments: list[LabelSegment], *, min_segment_iou: float) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int]] = []
    for true_index, true_segment in enumerate(true_segments):
        for pred_index, pred_segment in enumerate(pred_segments):
            if true_segment.sequence_id != pred_segment.sequence_id or true_segment.label != pred_segment.label:
                continue
            iou = _segment_iou(true_segment, pred_segment)
            if iou >= min_segment_iou:
                candidates.append((iou, true_index, pred_index))
    matches: list[dict[str, Any]] = []
    used_true: set[int] = set()
    used_pred: set[int] = set()
    for iou, true_index, pred_index in sorted(candidates, reverse=True):
        if true_index in used_true or pred_index in used_pred:
            continue
        used_true.add(true_index)
        used_pred.add(pred_index)
        matches.append({"iou": iou, "true_index": true_index, "pred_index": pred_index})
    return matches


def _segment_iou(left: LabelSegment, right: LabelSegment) -> float:
    overlap = max(0, min(left.end_frame, right.end_frame) - max(left.start_frame, right.start_frame) + 1)
    if overlap <= 0:
        return 0.0
    union = max(left.end_frame, right.end_frame) - min(left.start_frame, right.start_frame) + 1
    return overlap / union if union else 0.0


def _decision_latencies(events: list[OnlineEvent], true_segments: list[LabelSegment], *, latency_grace_ms: int) -> list[int]:
    by_sequence: dict[str, list[OnlineEvent]] = {}
    for event in events:
        by_sequence.setdefault(event.sequence_id, []).append(event)
    latencies: list[int] = []
    for segment in true_segments:
        candidates = [
            event
            for event in by_sequence.get(segment.sequence_id, [])
            if event.proposal_label == segment.label
            and event.timestamp_ms >= segment.start_ms
            and event.timestamp_ms <= segment.end_ms + latency_grace_ms
        ]
        if candidates:
            latencies.append(min(candidates, key=lambda item: item.timestamp_ms).timestamp_ms - segment.start_ms)
    return latencies


def _label_switch_rate(events: list[OnlineEvent], duration_minutes: float) -> float:
    if duration_minutes <= 0:
        return 0.0
    switches = 0
    by_sequence: dict[str, list[OnlineEvent]] = {}
    for event in events:
        by_sequence.setdefault(event.sequence_id, []).append(event)
    for sequence_events in by_sequence.values():
        previous = "no_gesture"
        for event in sorted(sequence_events, key=lambda item: item.frame_index):
            label = event.proposal_label or "no_gesture"
            if label != previous and label != "no_gesture" and previous != "no_gesture":
                switches += 1
            previous = label
    return switches / duration_minutes


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    f1_values = []
    for label in TARGET_LABELS:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(_f1(precision, recall))
    return {
        "accuracy": correct / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
    }


def _duration_minutes(events: list[OnlineEvent]) -> float:
    by_sequence: dict[str, list[int]] = {}
    for event in events:
        by_sequence.setdefault(event.sequence_id, []).append(event.timestamp_ms)
    total_ms = 0
    for timestamps in by_sequence.values():
        if not timestamps:
            continue
        ordered = sorted(timestamps)
        step = _median([right - left for left, right in zip(ordered, ordered[1:]) if right > left]) or 0
        total_ms += max(0, ordered[-1] - ordered[0] + step)
    return total_ms / 60000.0 if total_ms > 0 else 0.0


def _cost(costs: dict[str, float], action: str) -> float:
    return float(costs.get(action, 0.0 if action == "idle" else 1.0))


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[int | float]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _median(values: list[int | float]) -> float | None:
    return float(median(values)) if values else None


def _scaled_rate(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    # Rates can exceed one, but the compact SVG bar needs a stable 0..1 scale.
    return min(1.0, float(value) / 60.0)
