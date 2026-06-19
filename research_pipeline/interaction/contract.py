from __future__ import annotations

from typing import Any

from research_pipeline.interaction.fsm import ACTION_BY_LABEL, ContextAwarePolicy, ContextPolicyConfig
from research_pipeline.models.common import Prediction

# A single recognizer frame fed to the acceptance policy.
Frame = tuple[int, str, float]  # (timestamp_ms, label, confidence)


def validation_contract(config: ContextPolicyConfig | None = None) -> dict[str, Any]:
    """Exact spec of the on-device acceptance policy (ContextAwarePolicy / C2).

    This is the lightweight validation layer the Swift port must reproduce: a
    recognizer label is accepted as an AR action only after a confidence gate,
    temporal stability, and a cooldown; ``no_gesture`` / below-threshold frames
    are the release signal. Derived from the live config so it cannot drift.
    """

    config = config or ContextPolicyConfig()
    return {
        "policy": "ContextAwarePolicy",
        "config": {
            "activation_threshold": config.activation_threshold,
            "stable_frames": config.stable_frames,
            "cooldown_ms": config.cooldown_ms,
            "no_gesture_reset_frames": config.no_gesture_reset_frames,
        },
        "action_by_label": dict(ACTION_BY_LABEL),
        "states": ["idle", "tracking", "cooldown"],
        "event_fields": ["timestamp_ms", "gesture", "action", "confidence", "state"],
        "algorithm": [
            "Per frame (timestamp_ms, label, confidence), keep candidate, candidate_count, "
            "last_action_ms, no_gesture_count:",
            "1. If label == 'no_gesture' OR confidence < activation_threshold: no_gesture_count += 1; "
            "if no_gesture_count >= no_gesture_reset_frames, reset state to idle and clear candidate; emit nothing.",
            "2. Else no_gesture_count = 0.",
            "3. If label == candidate: candidate_count += 1; else candidate = label, candidate_count = 1, state = tracking.",
            "4. If candidate_count < stable_frames: emit nothing.",
            "5. If timestamp_ms - last_action_ms < cooldown_ms: emit nothing.",
            "6. action = action_by_label[label]; if absent (e.g. no_gesture), emit nothing.",
            "7. last_action_ms = timestamp_ms; state = cooldown; emit event(gesture, action, confidence, state).",
        ],
        "notes": (
            "TARC adds an optional task-expectation layer on top (accept only actions the current task "
            "expects); this contract covers the always-on confidence/stability/cooldown/release policy."
        ),
    }


def run_validation_trace(frames: list[Frame], config: ContextPolicyConfig | None = None) -> list[dict[str, Any]]:
    """Run the acceptance policy over a frame sequence and return emitted events."""

    policy = ContextAwarePolicy(config)
    events: list[dict[str, Any]] = []
    for timestamp_ms, label, confidence in frames:
        event = policy.update(Prediction(label, confidence, {label: confidence}), int(timestamp_ms))
        if event is not None:
            events.append(
                {
                    "timestamp_ms": event.timestamp_ms,
                    "gesture": event.gesture,
                    "action": event.action,
                    "confidence": event.confidence,
                    "state": event.state,
                }
            )
    return events


def default_validation_scenarios() -> list[tuple[str, list[Frame]]]:
    """Representative input sequences exercising each acceptance rule."""

    return [
        ("click_accept", [(0, "no_gesture", 0.95), (100, "click_2f", 0.80), (133, "click_2f", 0.85)]),
        ("single_frame_ignored", [(0, "swipe_left", 0.90)]),
        ("below_threshold_rejected", [(0, "click_2f", 0.50), (33, "click_2f", 0.55), (66, "click_2f", 0.50)]),
        (
            "cooldown_debounces_repeat",
            [(t, "zoom_in", 0.90) for t in (0, 33, 66, 99, 132)],
        ),
        (
            "no_gesture_resets_candidate",
            [
                (0, "swipe_right", 0.90),
                (33, "no_gesture", 0.90),
                (66, "no_gesture", 0.90),
                (99, "no_gesture", 0.90),
                (132, "swipe_right", 0.90),
                (165, "swipe_right", 0.90),
            ],
        ),
    ]


def golden_validation_traces(
    config: ContextPolicyConfig | None = None,
    scenarios: list[tuple[str, list[Frame]]] | None = None,
) -> list[dict[str, Any]]:
    """Golden input->events traces for verifying a Swift port of the policy."""

    config = config or ContextPolicyConfig()
    scenarios = scenarios or default_validation_scenarios()
    return [
        {
            "name": name,
            "frames": [[t, label, confidence] for (t, label, confidence) in frames],
            "events": run_validation_trace(frames, config),
        }
        for name, frames in scenarios
    ]
