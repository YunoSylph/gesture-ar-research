from __future__ import annotations

from research_pipeline.interaction.stabilizer import (
    TemporalLabelStabilizer,
    TemporalStabilizerConfig,
)


def _run(stabilizer: TemporalLabelStabilizer, labels: list[str], confidence: float = 1.0) -> list[str]:
    return [stabilizer.update(label, confidence).label for label in labels]


def test_stable_stream_passes_through() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=5))
    out = _run(stabilizer, ["swipe_left"] * 5)
    assert out[-1] == "swipe_left"


def test_single_frame_flip_is_suppressed() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=5, enter_fraction=0.5))
    # Stream is mostly point_2f with one stray click_2f frame.
    out = _run(stabilizer, ["point_2f", "point_2f", "point_2f", "click_2f", "point_2f"])
    assert out[-1] == "point_2f"
    assert "click_2f" not in out


def test_sustained_switch_is_followed() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=5, enter_fraction=0.5))
    _run(stabilizer, ["point_2f"] * 5)
    out = _run(stabilizer, ["swipe_right"] * 5)
    assert out[-1] == "swipe_right"


def test_low_confidence_votes_as_background() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=4, min_confidence=0.6))
    out = _run(stabilizer, ["zoom_in"] * 4, confidence=0.3)
    assert out[-1] == "no_gesture"


def test_enter_fraction_blocks_weak_activation() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=6, enter_fraction=0.6))
    # Only 2 of 6 frames are the gesture -> below the 0.6 entry share.
    out = _run(stabilizer, ["no_gesture", "no_gesture", "click_2f", "no_gesture", "click_2f", "no_gesture"])
    assert out[-1] == "no_gesture"


def test_update_prediction_returns_background_for_idle() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=3))
    from research_pipeline.models.common import prediction_from_scores

    result = stabilizer.update_prediction(prediction_from_scores({"no_gesture": 1.0}))
    assert result.label == "no_gesture"


def test_unknown_label_treated_as_background() -> None:
    stabilizer = TemporalLabelStabilizer(TemporalStabilizerConfig(window=3))
    out = _run(stabilizer, ["garbage", "garbage", "garbage"])
    assert out[-1] == "no_gesture"
