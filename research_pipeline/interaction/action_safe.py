from __future__ import annotations

from dataclasses import dataclass, field

from research_pipeline.interaction.fsm import ACTION_BY_LABEL, InteractionEvent
from research_pipeline.labels import TARGET_LABELS, validate_target_label
from research_pipeline.models.common import Prediction


@dataclass(slots=True)
class ActionSafePolicyConfig:
    default_threshold: float = 0.75
    label_thresholds: dict[str, float] = field(default_factory=dict)
    default_stable_frames: int = 2
    label_stable_frames: dict[str, int] = field(default_factory=dict)
    cooldown_ms: int = 200
    no_gesture_reset_frames: int = 3
    min_score_margin: float = 0.0


class ActionSafePolicy:
    """Risk-aware gesture-to-action controller for AR tasks.

    The policy treats classifier output as a proposal, not a command. It can
    abstain when confidence, score margin, temporal stability, or cooldown
    constraints are not satisfied.
    """

    def __init__(self, config: ActionSafePolicyConfig | None = None):
        self.config = config or ActionSafePolicyConfig()
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
        if prediction.label == "no_gesture" or not self._passes_confidence(prediction):
            self._register_abstention()
            return None

        self._no_gesture_count = 0
        if prediction.label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = prediction.label
            self._candidate_count = 1
            self.state = "tracking"

        if self._candidate_count < self._stable_frames(prediction.label):
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

    def _register_abstention(self) -> None:
        self._no_gesture_count += 1
        if self._no_gesture_count >= self.config.no_gesture_reset_frames:
            self.state = "idle"
            self._candidate = None
            self._candidate_count = 0

    def _passes_confidence(self, prediction: Prediction) -> bool:
        threshold = float(self.config.label_thresholds.get(prediction.label, self.config.default_threshold))
        if prediction.confidence < threshold:
            return False
        if self.config.min_score_margin <= 0.0:
            return True
        ordered_scores = sorted((float(prediction.scores.get(label, 0.0)) for label in TARGET_LABELS), reverse=True)
        runner_up = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
        return prediction.confidence - runner_up >= self.config.min_score_margin

    def _stable_frames(self, label: str) -> int:
        return max(1, int(self.config.label_stable_frames.get(label, self.config.default_stable_frames)))
