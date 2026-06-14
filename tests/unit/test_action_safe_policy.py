from research_pipeline.interaction.action_safe import ActionSafePolicy, ActionSafePolicyConfig
from research_pipeline.models.common import prediction_from_scores


def test_action_safe_policy_abstains_below_label_threshold() -> None:
    policy = ActionSafePolicy(
        ActionSafePolicyConfig(
            default_threshold=0.5,
            label_thresholds={"click_2f": 0.8},
            default_stable_frames=1,
        )
    )
    prediction = prediction_from_scores({"click_2f": 0.7, "no_gesture": 0.3})
    assert policy.update(prediction, 0) is None


def test_action_safe_policy_emits_after_stable_frames() -> None:
    policy = ActionSafePolicy(ActionSafePolicyConfig(default_threshold=0.5, default_stable_frames=2))
    prediction = prediction_from_scores({"swipe_right": 0.8, "no_gesture": 0.2})
    assert policy.update(prediction, 0) is None
    event = policy.update(prediction, 100)
    assert event is not None
    assert event.action == "navigate_next"


def test_action_safe_policy_uses_score_margin() -> None:
    policy = ActionSafePolicy(
        ActionSafePolicyConfig(
            default_threshold=0.5,
            default_stable_frames=1,
            min_score_margin=0.2,
        )
    )
    ambiguous = prediction_from_scores({"zoom_in": 0.52, "zoom_out": 0.48})
    assert policy.update(ambiguous, 0) is None
