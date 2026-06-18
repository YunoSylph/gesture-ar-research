from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.common import Prediction, prediction_from_scores


@dataclass(slots=True)
class TemporalStabilizerConfig:
    """Configuration for the W-frame temporal label stabilizer.

    ``window`` is the look-back ``W`` (OO-dMVMT, Sec. 3.4, post-processing step:
    "assign the class label that is the most frequent in the last ``W``
    preliminary classifications"). ``enter_fraction`` requires a gesture label to
    occupy at least this share of the window before it is emitted, which
    suppresses brief spurious activations. Frames below ``min_confidence`` vote as
    background (``no_gesture``), so low-confidence jitter pulls toward idle.
    ``sticky`` adds hysteresis: the current stable gesture is kept unless a
    challenger is *strictly* more frequent, avoiding flicker on ties.
    """

    window: int = 7
    enter_fraction: float = 0.5
    min_confidence: float = 0.0
    sticky: bool = True


@dataclass(slots=True)
class StabilizedResult:
    label: str
    confidence: float
    support: int
    window: int
    switched: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalLabelStabilizer:
    """Majority-vote temporal smoothing over the last ``W`` frame labels.

    This is a label-level stabilizer, distinct from score smoothing
    (``benchmark_online_gesture._smooth_prediction``) and from the
    consecutive-frame stability check in ``GestureValidationLayer`` (which
    requires ``N`` identical frames in a row and so adds latency). Majority voting
    tolerates isolated single-frame flips without demanding an unbroken run, which
    is the failure mode observed live where the raw recognizer jumps between
    ``no_gesture``, ``click_2f``, ``swipe_left`` and ``zoom_in`` on adjacent
    frames. It runs on the landmark prediction stream, so it is identical on
    webcam and on a rear phone camera.
    """

    def __init__(self, config: TemporalStabilizerConfig | None = None):
        self.config = config or TemporalStabilizerConfig()
        self.history: deque[str] = deque(maxlen=max(1, self.config.window))
        self.confidence_history: deque[float] = deque(maxlen=max(1, self.config.window))
        self.stable_label = "no_gesture"

    def reset(self) -> None:
        self.history.clear()
        self.confidence_history.clear()
        self.stable_label = "no_gesture"

    def update(self, label: str, confidence: float = 1.0) -> StabilizedResult:
        vote = label if label in TARGET_LABELS else "no_gesture"
        conf = max(0.0, min(1.0, float(confidence)))
        if vote != "no_gesture" and conf < self.config.min_confidence:
            vote = "no_gesture"
        self.history.append(vote)
        self.confidence_history.append(conf)

        counts: dict[str, int] = {}
        for item in self.history:
            counts[item] = counts.get(item, 0) + 1
        window = len(self.history)

        winner = max(counts, key=lambda key: counts[key])
        winner_count = counts[winner]
        previous = self.stable_label

        if winner == "no_gesture":
            resolved = "no_gesture"
        elif winner_count < max(1, round(self.config.enter_fraction * window)):
            # No label is dominant enough; hold the current gesture if it still
            # has any support in the window, otherwise fall back to background.
            resolved = previous if self.config.sticky and counts.get(previous, 0) > 0 else "no_gesture"
        elif self.config.sticky and previous not in {"no_gesture", winner} and counts.get(previous, 0) == winner_count:
            resolved = previous
        else:
            resolved = winner

        self.stable_label = resolved
        support = counts.get(resolved, 0)
        resolved_conf = self._mean_confidence(resolved)
        return StabilizedResult(
            label=resolved,
            confidence=resolved_conf,
            support=support,
            window=window,
            switched=resolved != previous,
        )

    def update_prediction(self, prediction: Prediction) -> Prediction:
        result = self.update(prediction.label, prediction.confidence)
        if result.label == "no_gesture":
            return prediction_from_scores({"no_gesture": 1.0})
        return prediction_from_scores({result.label: max(1e-3, result.confidence)})

    def _mean_confidence(self, label: str) -> float:
        values = [conf for item, conf in zip(self.history, self.confidence_history) if item == label]
        if not values:
            return 0.0
        return float(sum(values) / len(values))
