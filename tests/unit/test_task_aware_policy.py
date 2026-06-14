from research_pipeline.interaction.action_safe import ActionSafePolicyConfig
from research_pipeline.interaction.task_aware import TaskAwareActionSafePolicy, TaskAwarePolicyConfig
from research_pipeline.models.common import prediction_from_scores


def test_task_aware_policy_lowers_threshold_for_expected_action() -> None:
    scenario = {
        "id": "unit",
        "expected_actions": [{"id": "confirm", "action": "select_confirm"}],
    }
    policy = TaskAwareActionSafePolicy(
        scenario,
        TaskAwarePolicyConfig(
            base=ActionSafePolicyConfig(
                default_threshold=0.8,
                label_thresholds={"click_2f": 0.8, "swipe_right": 0.8},
                default_stable_frames=1,
                cooldown_ms=0,
            ),
            expected_threshold_delta=-0.2,
            unexpected_threshold_delta=0.1,
        ),
    )

    event = policy.update(prediction_from_scores({"click_2f": 0.7, "no_gesture": 0.3}), 0)

    assert event is not None
    assert event.action == "select_confirm"
    assert policy.context()["step_index"] == 1


def test_task_aware_policy_raises_threshold_for_unexpected_action() -> None:
    scenario = {
        "id": "unit",
        "expected_actions": [{"id": "confirm", "action": "select_confirm"}],
    }
    policy = TaskAwareActionSafePolicy(
        scenario,
        TaskAwarePolicyConfig(
            base=ActionSafePolicyConfig(default_threshold=0.8, default_stable_frames=1),
            expected_threshold_delta=-0.2,
            unexpected_threshold_delta=0.1,
        ),
    )

    event = policy.update(prediction_from_scores({"swipe_right": 0.85, "no_gesture": 0.15}), 0)

    assert event is None
    assert policy.context()["expected_action"] == "select_confirm"
