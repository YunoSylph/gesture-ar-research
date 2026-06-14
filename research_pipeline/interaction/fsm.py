from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.labels import validate_target_label
from research_pipeline.models.common import Prediction


ACTION_BY_LABEL = {
    "point_2f": "pointer_hover",
    "click_2f": "select_confirm",
    "swipe_left": "navigate_previous",
    "swipe_right": "navigate_next",
    "zoom_in": "zoom_in",
    "zoom_out": "zoom_out",
}


@dataclass(slots=True)
class InteractionEvent:
    timestamp_ms: int
    gesture: str
    action: str
    confidence: float
    state: str


@dataclass(slots=True)
class ContextPolicyConfig:
    activation_threshold: float = 0.62
    stable_frames: int = 2
    cooldown_ms: int = 250
    no_gesture_reset_frames: int = 3


class ContextAwarePolicy:
    def __init__(self, config: ContextPolicyConfig | None = None):
        self.config = config or ContextPolicyConfig()
        self.state = "idle"
        self._candidate: str | None = None
        self._candidate_count = 0
        self._last_action_ms = -10**9
        self._no_gesture_count = 0

    def reset(self) -> None:
        self.state = "idle"
        self._candidate = None
        self._candidate_count = 0
        self._last_action_ms = -10**9
        self._no_gesture_count = 0

    def update(self, prediction: Prediction, timestamp_ms: int) -> InteractionEvent | None:
        validate_target_label(prediction.label)
        if prediction.label == "no_gesture" or prediction.confidence < self.config.activation_threshold:
            self._no_gesture_count += 1
            if self._no_gesture_count >= self.config.no_gesture_reset_frames:
                self.state = "idle"
                self._candidate = None
                self._candidate_count = 0
            return None

        self._no_gesture_count = 0
        if prediction.label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = prediction.label
            self._candidate_count = 1
            self.state = "tracking"

        if self._candidate_count < self.config.stable_frames:
            return None
        if timestamp_ms - self._last_action_ms < self.config.cooldown_ms:
            return None

        action = ACTION_BY_LABEL.get(prediction.label)
        if action is None:
            return None
        self._last_action_ms = timestamp_ms
        self.state = "cooldown"
        return InteractionEvent(
            timestamp_ms=timestamp_ms,
            gesture=prediction.label,
            action=action,
            confidence=prediction.confidence,
            state=self.state,
        )

