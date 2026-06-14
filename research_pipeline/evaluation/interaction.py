from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.interaction.fsm import ContextAwarePolicy, ContextPolicyConfig, InteractionEvent
from research_pipeline.models.common import Prediction


@dataclass(slots=True)
class ReplayFrame:
    timestamp_ms: int
    label: str
    confidence: float
    expected_action: str = ""


def replay_predictions(frames: list[ReplayFrame], config: ContextPolicyConfig | None = None) -> list[InteractionEvent]:
    policy = ContextAwarePolicy(config)
    events: list[InteractionEvent] = []
    for frame in frames:
        prediction = Prediction(frame.label, frame.confidence, {frame.label: frame.confidence})
        event = policy.update(prediction, frame.timestamp_ms)
        if event is not None:
            events.append(event)
    return events


def compute_interaction_metrics(frames: list[ReplayFrame], events: list[InteractionEvent]) -> dict:
    expected = [frame.expected_action for frame in frames if frame.expected_action]
    event_actions = [event.action for event in events]
    matched = 0
    remaining = event_actions.copy()
    for action in expected:
        if action in remaining:
            matched += 1
            remaining.remove(action)
    false_triggers = len(remaining)
    action_precision = matched / len(event_actions) if event_actions else 0.0
    action_recall = matched / len(expected) if expected else 0.0
    success = 1.0 if expected and matched == len(expected) else 0.0
    duration_minutes = 0.0
    if frames:
        duration_minutes = max(1e-9, (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 60000.0)
    return {
        "task_success_rate": success,
        "unintended_action_rate": false_triggers / max(1, len(event_actions)),
        "false_trigger_rate_per_minute": false_triggers / duration_minutes if duration_minutes else 0.0,
        "action_precision": action_precision,
        "action_recall": action_recall,
        "corrections_per_task": false_triggers,
        "num_events": len(events),
        "num_expected_actions": len(expected),
    }

