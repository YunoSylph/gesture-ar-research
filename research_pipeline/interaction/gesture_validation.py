from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from research_pipeline.interaction.fsm import ACTION_BY_LABEL
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.common import Prediction, prediction_from_scores


VALIDATION_STATES = {
    "idle",
    "background",
    "tracking",
    "candidate",
    "preparing",
    "ready",
    "locked",
    "cooldown",
    "release_required",
    "rejected",
}

DISCRETE_LABELS = {"click_2f", "swipe_left", "swipe_right", "zoom_in", "zoom_out"}


@dataclass(slots=True)
class GestureContractRule:
    role: str
    action: str
    risk_cost: float
    command: bool
    discrete: bool
    release_required: bool = False


DEFAULT_GESTURE_CONTRACT: dict[str, GestureContractRule] = {
    "no_gesture": GestureContractRule("background", "idle", 0.0, command=False, discrete=False),
    "point_2f": GestureContractRule("pointer_state", "pointer_hover", 0.25, command=False, discrete=False),
    "click_2f": GestureContractRule("command", "select_confirm", 2.0, command=True, discrete=True, release_required=True),
    "swipe_left": GestureContractRule("command", "navigate_previous", 1.0, command=True, discrete=True, release_required=True),
    "swipe_right": GestureContractRule("command", "navigate_next", 1.0, command=True, discrete=True, release_required=True),
    "zoom_in": GestureContractRule("transform_command", "zoom_in", 1.25, command=True, discrete=True, release_required=True),
    "zoom_out": GestureContractRule("transform_command", "zoom_out", 1.25, command=True, discrete=True, release_required=True),
}


@dataclass(slots=True)
class GestureValidationConfig:
    confidence_thresholds: dict[str, float] = field(default_factory=lambda: {
        "point_2f": 0.46,
        "click_2f": 0.58,
        "swipe_left": 0.56,
        "swipe_right": 0.56,
        "zoom_in": 0.56,
        "zoom_out": 0.56,
    })
    default_confidence_threshold: float = 0.55
    min_top2_margin: float = 0.0
    stable_frames: dict[str, int] = field(default_factory=lambda: {
        "point_2f": 1,
        "click_2f": 2,
        "swipe_left": 2,
        "swipe_right": 2,
        "zoom_in": 2,
        "zoom_out": 2,
    })
    default_stable_frames: int = 2
    cooldown_ms: int = 250
    lock_hold_ms: int = 120
    expected_confidence_delta: float = -0.05
    unexpected_confidence_delta: float = 0.08
    use_confidence: bool = True
    use_stability: bool = True
    use_cooldown: bool = True
    require_release: bool = True
    contract: dict[str, GestureContractRule] = field(default_factory=lambda: dict(DEFAULT_GESTURE_CONTRACT))


@dataclass(slots=True)
class GestureValidationInput:
    model_label: str
    model_confidence: float
    top2_margin: float
    timestamp_ms: int
    frame_index: int = 0
    expected_label: str = ""
    landmark_stats: dict[str, float] | None = None


@dataclass(slots=True)
class GestureValidationResult:
    proposal_label: str
    proposal_state: str
    proposal_confidence: float
    active: bool
    background: bool
    ready: bool
    accepted: bool
    rejected: bool
    rejection_reason: str
    lock_progress: float
    cooldown_remaining: int
    candidate_label: str = ""
    expected_label: str = ""
    final_action: str = "idle"
    risk_cost: float = 0.0
    last_accepted_action: str = "idle"
    stable_frames: int = 0
    required_frames: int = 0

    @property
    def is_ready_for_tarc(self) -> bool:
        return self.ready and self.proposal_state in {"ready", "locked"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prediction(self) -> Prediction:
        if self.is_ready_for_tarc:
            return prediction_from_scores({self.proposal_label: max(0.001, self.proposal_confidence)})
        return prediction_from_scores({"no_gesture": 1.0})


class GestureValidationLayer:
    """Online gesture proposal validator shared by replay and live AR paths.

    The layer treats recognizer output as a proposal. It validates that proposal
    with confidence, margin, temporal stability, cooldown, and release checks
    before TARC or a task replay is allowed to turn it into an action.
    """

    def __init__(self, config: GestureValidationConfig | None = None):
        self.config = config or GestureValidationConfig()
        self.candidate_label = ""
        self.candidate_count = 0
        self.last_action_ms = -10**9
        self.locked_label = ""
        self.lock_until_ms = -1
        self.release_label = ""
        self.last_accepted_action = "idle"

    def reset(self) -> None:
        self.candidate_label = ""
        self.candidate_count = 0
        self.last_action_ms = -10**9
        self.locked_label = ""
        self.lock_until_ms = -1
        self.release_label = ""
        self.last_accepted_action = "idle"

    def update(
        self,
        *,
        model_label: str,
        model_confidence: float,
        top2_margin: float,
        timestamp_ms: int,
        frame_index: int = 0,
        expected_label: str = "",
        landmark_stats: dict[str, float] | None = None,
    ) -> GestureValidationResult:
        return self.update_input(
            GestureValidationInput(
                model_label=model_label,
                model_confidence=model_confidence,
                top2_margin=top2_margin,
                timestamp_ms=timestamp_ms,
                frame_index=frame_index,
                expected_label=expected_label,
                landmark_stats=landmark_stats,
            )
        )

    def update_prediction(
        self,
        prediction: Prediction,
        *,
        timestamp_ms: int,
        frame_index: int = 0,
        expected_label: str = "",
        top2_margin_value: float | None = None,
        landmark_stats: dict[str, float] | None = None,
    ) -> GestureValidationResult:
        margin = _top2_margin(prediction.scores) if top2_margin_value is None else top2_margin_value
        return self.update(
            model_label=prediction.label,
            model_confidence=prediction.confidence,
            top2_margin=margin,
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            expected_label=expected_label,
            landmark_stats=landmark_stats,
        )

    def update_input(self, item: GestureValidationInput) -> GestureValidationResult:
        label = item.model_label if item.model_label in TARGET_LABELS else "no_gesture"
        expected = item.expected_label if item.expected_label in TARGET_LABELS and item.expected_label != "no_gesture" else ""
        confidence = max(0.0, min(1.0, float(item.model_confidence)))

        if self.locked_label and item.timestamp_ms <= self.lock_until_ms:
            rule = self._rule(self.locked_label)
            progress = 1.0
            return self._result(
                proposal_label=self.locked_label,
                state="locked",
                confidence=max(confidence, 0.92),
                expected_label=expected,
                ready=True,
                accepted=False,
                lock_progress=progress,
                stable_frames=self.candidate_count,
                required_frames=self._required_frames(self.locked_label),
                action="idle",
                risk_cost=rule.risk_cost,
            )
        if self.locked_label and item.timestamp_ms > self.lock_until_ms:
            self.locked_label = ""

        if label == "no_gesture":
            self.candidate_label = ""
            self.candidate_count = 0
            self.release_label = ""
            return self._result(
                proposal_label="no_gesture",
                state="background",
                confidence=confidence,
                expected_label=expected,
                ready=False,
                accepted=False,
                lock_progress=0.0,
                stable_frames=0,
                required_frames=0,
            )

        rule = self._rule(label)
        cooldown_remaining = self._cooldown_remaining(item.timestamp_ms)
        if self.config.require_release and self.release_label == label and rule.release_required:
            return self._result(
                proposal_label=label,
                state="release_required",
                confidence=confidence,
                expected_label=expected,
                ready=False,
                accepted=False,
                rejected=True,
                rejection_reason="release_required",
                lock_progress=0.0,
                cooldown_remaining=cooldown_remaining,
                stable_frames=self.candidate_count,
                required_frames=self._required_frames(label),
                action=rule.action,
                risk_cost=rule.risk_cost,
            )

        if self.config.use_confidence:
            threshold = self._confidence_threshold(label, expected)
            if confidence < threshold:
                self.candidate_label = ""
                self.candidate_count = 0
                return self._result(
                    proposal_label=label,
                    state="rejected",
                    confidence=confidence,
                    expected_label=expected,
                    ready=False,
                    accepted=False,
                    rejected=True,
                    rejection_reason="low_confidence",
                    lock_progress=0.0,
                    cooldown_remaining=cooldown_remaining,
                    stable_frames=0,
                    required_frames=self._required_frames(label),
                    action=rule.action,
                    risk_cost=rule.risk_cost,
                )
            if item.top2_margin < self.config.min_top2_margin:
                self.candidate_label = ""
                self.candidate_count = 0
                return self._result(
                    proposal_label=label,
                    state="rejected",
                    confidence=confidence,
                    expected_label=expected,
                    ready=False,
                    accepted=False,
                    rejected=True,
                    rejection_reason="low_margin",
                    lock_progress=0.0,
                    cooldown_remaining=cooldown_remaining,
                    stable_frames=0,
                    required_frames=self._required_frames(label),
                    action=rule.action,
                    risk_cost=rule.risk_cost,
                )

        if self.config.use_cooldown and cooldown_remaining > 0 and rule.command:
            return self._result(
                proposal_label=label,
                state="cooldown",
                confidence=confidence,
                expected_label=expected,
                ready=False,
                accepted=False,
                rejected=True,
                rejection_reason="cooldown",
                lock_progress=0.0,
                cooldown_remaining=cooldown_remaining,
                stable_frames=self.candidate_count,
                required_frames=self._required_frames(label),
                action=rule.action,
                risk_cost=rule.risk_cost,
            )

        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        required = self._required_frames(label) if self.config.use_stability else 1
        progress = min(1.0, self.candidate_count / max(1, required))
        if self.candidate_count < required:
            state = "candidate" if self.candidate_count == 1 else "preparing"
            return self._result(
                proposal_label=label,
                state=state,
                confidence=confidence,
                expected_label=expected,
                ready=False,
                accepted=False,
                lock_progress=progress,
                stable_frames=self.candidate_count,
                required_frames=required,
                action=rule.action,
                risk_cost=rule.risk_cost,
            )

        self.last_action_ms = item.timestamp_ms if rule.command else self.last_action_ms
        self.locked_label = label if rule.command else ""
        self.lock_until_ms = item.timestamp_ms + max(0, self.config.lock_hold_ms) if rule.command else -1
        self.release_label = label if rule.release_required else ""
        self.last_accepted_action = rule.action if rule.action else "idle"
        state = "ready" if not rule.command else "locked"
        return self._result(
            proposal_label=label,
            state=state,
            confidence=confidence,
            expected_label=expected,
            ready=True,
            accepted=True,
            lock_progress=1.0,
            stable_frames=self.candidate_count,
            required_frames=required,
            action=rule.action,
            risk_cost=rule.risk_cost,
        )

    def _rule(self, label: str) -> GestureContractRule:
        return self.config.contract.get(label, DEFAULT_GESTURE_CONTRACT[label])

    def _required_frames(self, label: str) -> int:
        return max(1, int(self.config.stable_frames.get(label, self.config.default_stable_frames)))

    def _confidence_threshold(self, label: str, expected_label: str) -> float:
        threshold = float(self.config.confidence_thresholds.get(label, self.config.default_confidence_threshold))
        if expected_label:
            threshold += self.config.expected_confidence_delta if label == expected_label else self.config.unexpected_confidence_delta
        return max(0.01, min(0.99, threshold))

    def _cooldown_remaining(self, timestamp_ms: int) -> int:
        if not self.config.use_cooldown:
            return 0
        return max(0, int(self.last_action_ms + self.config.cooldown_ms - timestamp_ms))

    def _result(
        self,
        *,
        proposal_label: str,
        state: str,
        confidence: float,
        expected_label: str,
        ready: bool,
        accepted: bool,
        lock_progress: float,
        rejected: bool = False,
        rejection_reason: str = "",
        cooldown_remaining: int = 0,
        stable_frames: int = 0,
        required_frames: int = 0,
        action: str = "idle",
        risk_cost: float = 0.0,
    ) -> GestureValidationResult:
        if state not in VALIDATION_STATES:
            state = "rejected"
            rejected = True
            rejection_reason = rejection_reason or "invalid_state"
        active = proposal_label != "no_gesture" and state not in {"idle", "background", "rejected", "cooldown", "release_required"}
        background = proposal_label == "no_gesture" or state in {"idle", "background"}
        return GestureValidationResult(
            proposal_label=proposal_label,
            proposal_state=state,
            proposal_confidence=round(float(confidence), 6),
            active=active,
            background=background,
            ready=ready,
            accepted=accepted,
            rejected=rejected,
            rejection_reason=rejection_reason,
            lock_progress=round(max(0.0, min(1.0, float(lock_progress))), 6),
            cooldown_remaining=int(cooldown_remaining),
            candidate_label=self.candidate_label or proposal_label,
            expected_label=expected_label,
            final_action=action if accepted else "idle",
            risk_cost=float(risk_cost),
            last_accepted_action=self.last_accepted_action,
            stable_frames=int(stable_frames),
            required_frames=int(required_frames),
        )


def config_from_mapping(payload: dict[str, Any] | None) -> GestureValidationConfig:
    raw = payload or {}
    return GestureValidationConfig(
        confidence_thresholds={str(key): float(value) for key, value in raw.get("confidence_thresholds", {}).items()}
        or GestureValidationConfig().confidence_thresholds,
        default_confidence_threshold=float(raw.get("default_confidence_threshold", 0.55)),
        min_top2_margin=float(raw.get("min_top2_margin", 0.0)),
        stable_frames={str(key): int(value) for key, value in raw.get("stable_frames", {}).items()}
        or GestureValidationConfig().stable_frames,
        default_stable_frames=int(raw.get("default_stable_frames", 2)),
        cooldown_ms=int(raw.get("cooldown_ms", 250)),
        lock_hold_ms=int(raw.get("lock_hold_ms", 120)),
        expected_confidence_delta=float(raw.get("expected_confidence_delta", -0.05)),
        unexpected_confidence_delta=float(raw.get("unexpected_confidence_delta", 0.08)),
        use_confidence=bool(raw.get("use_confidence", True)),
        use_stability=bool(raw.get("use_stability", True)),
        use_cooldown=bool(raw.get("use_cooldown", True)),
        require_release=bool(raw.get("require_release", True)),
    )


def _top2_margin(scores: dict[str, float]) -> float:
    ordered = sorted((float(scores.get(label, 0.0)) for label in TARGET_LABELS), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    return max(0.0, ordered[0] - ordered[1])
