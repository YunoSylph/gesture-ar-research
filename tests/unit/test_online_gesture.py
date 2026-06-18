from pathlib import Path

from research_pipeline.cli.benchmark_online_gesture import run_online_benchmark
from research_pipeline.evaluation.online_gesture import (
    EVENT_FIELDNAMES,
    OnlineEvent,
    compute_online_metrics,
    event_to_dict,
)
from research_pipeline.interaction.gesture_validation import GestureValidationLayer
from research_pipeline.evaluation.task_replay import default_task_scenarios, evaluate_task_replay
from research_pipeline.models.common import prediction_from_scores


def test_online_metrics_basic() -> None:
    events = [
        OnlineEvent("s1", 0, 0, ground_truth_label="no_gesture", proposal_label="no_gesture", model_label="no_gesture"),
        OnlineEvent("s1", 1, 100, ground_truth_label="swipe_right", proposal_label="no_gesture", model_label="swipe_right"),
        OnlineEvent("s1", 2, 200, ground_truth_label="swipe_right", proposal_label="swipe_right", model_label="swipe_right"),
        OnlineEvent("s1", 3, 300, ground_truth_label="swipe_right", proposal_label="swipe_right", model_label="swipe_right"),
    ]

    metrics = compute_online_metrics(events)["metrics"]

    assert metrics["frame_accuracy_proposal"] == 3 / 4
    assert metrics["segment_precision"] == 1.0
    assert metrics["segment_recall"] == 1.0
    assert metrics["decision_latency_ms_mean"] == 100.0


def test_false_positive_rate() -> None:
    events = [
        OnlineEvent("s1", 0, 0, ground_truth_label="no_gesture", proposal_label="no_gesture"),
        OnlineEvent("s1", 1, 100, ground_truth_label="no_gesture", proposal_label="click_2f"),
        OnlineEvent("s1", 2, 200, ground_truth_label="no_gesture", proposal_label="swipe_left"),
        OnlineEvent("s1", 3, 300, ground_truth_label="point_2f", proposal_label="point_2f"),
    ]

    metrics = compute_online_metrics(events)["metrics"]

    assert metrics["no_gesture_false_positive_rate"] == 2 / 3
    assert metrics["false_positives_per_minute"] > 0


def test_label_switch_rate() -> None:
    events = [
        OnlineEvent("s1", 0, 0, ground_truth_label="swipe_left", proposal_label="swipe_left"),
        OnlineEvent("s1", 1, 100, ground_truth_label="swipe_left", proposal_label="swipe_right"),
        OnlineEvent("s1", 2, 200, ground_truth_label="swipe_left", proposal_label="zoom_in"),
    ]

    metrics = compute_online_metrics(events)["metrics"]

    assert metrics["label_switch_rate_per_minute"] > 0


def test_event_log_schema() -> None:
    event = OnlineEvent("s1", 0, 0, ground_truth_label="no_gesture")
    payload = event_to_dict(event)

    assert list(payload.keys()) == EVENT_FIELDNAMES


def test_task_replay_success() -> None:
    scenario = default_task_scenarios()["object_control"]
    events = [
        OnlineEvent("s1", 0, 0, task_id=scenario.id, final_action="pointer_hover", action_accepted=True),
        OnlineEvent("s1", 1, 100, task_id=scenario.id, final_action="select_confirm", action_accepted=True),
        OnlineEvent("s1", 2, 200, task_id=scenario.id, final_action="zoom_in", action_accepted=True),
        OnlineEvent("s1", 3, 300, task_id=scenario.id, final_action="zoom_out", action_accepted=True),
    ]

    result = evaluate_task_replay(events, scenario)

    assert result.task_success is True
    assert result.false_action_cost == 0.0
    assert result.missed_action_cost == 0.0


def test_task_replay_false_action_cost() -> None:
    scenario = default_task_scenarios()["scroll_open"]
    events = [
        OnlineEvent("s1", 0, 0, task_id=scenario.id, final_action="navigate_next", action_accepted=True),
        OnlineEvent("s1", 1, 100, task_id=scenario.id, final_action="zoom_in", action_accepted=True),
    ]

    result = evaluate_task_replay(events, scenario)

    assert result.task_success is False
    assert result.false_action_cost > 0
    assert result.missed_action_cost > 0


def test_direct_vs_tarc_comparison_runs(tmp_path) -> None:
    summary = run_online_benchmark(Path("configs/eval/online_gesture.yaml"), tmp_path)
    rows = {row["method"]: row for row in summary["method_comparison"]}

    assert "direct_c6" in rows
    assert "c6_validation_confidence_only" in rows
    assert "c6_validation_confidence_stability_cooldown" in rows
    assert "c6_validation_tarc" in rows
    assert (tmp_path / "method_comparison.csv").exists()
    assert (tmp_path / "method_comparison.md").exists()


def test_stability_controller_accepts_after_required_frames() -> None:
    controller = GestureValidationLayer()
    prediction = prediction_from_scores({"click_2f": 1.0})

    first = controller.update_prediction(prediction, timestamp_ms=0)
    second = controller.update_prediction(prediction, timestamp_ms=100)
    third = controller.update_prediction(prediction, timestamp_ms=150)

    assert first.accepted is False
    assert first.proposal_state == "candidate"
    assert second.accepted is True
    assert second.final_action == "select_confirm"
    assert third.accepted is False
    assert third.proposal_state == "locked"
    assert third.final_action == "idle"


def test_tarc_only_receives_ready_or_locked_proposal() -> None:
    controller = GestureValidationLayer()
    prediction = prediction_from_scores({"swipe_right": 1.0})

    first = controller.update_prediction(prediction, timestamp_ms=0, expected_label="swipe_right")
    second = controller.update_prediction(prediction, timestamp_ms=100, expected_label="swipe_right")

    assert first.is_ready_for_tarc is False
    assert second.is_ready_for_tarc is True


def test_global_release_blocks_new_command_until_release() -> None:
    from research_pipeline.interaction.gesture_validation import GestureValidationConfig

    controller = GestureValidationLayer(GestureValidationConfig(require_global_release=True))
    click = prediction_from_scores({"click_2f": 1.0})
    controller.update_prediction(click, timestamp_ms=0)
    accepted = controller.update_prediction(click, timestamp_ms=100)
    assert accepted.accepted is True

    # A different command after the lock/cooldown window is still blocked until release.
    swipe = prediction_from_scores({"swipe_right": 1.0})
    blocked = controller.update_prediction(swipe, timestamp_ms=400)
    blocked_again = controller.update_prediction(swipe, timestamp_ms=500)
    assert blocked.accepted is False and blocked.rejection_reason == "awaiting_release"
    assert blocked_again.accepted is False

    # A no_gesture release re-arms the layer; the next command can then be accepted.
    controller.update_prediction(prediction_from_scores({"no_gesture": 1.0}), timestamp_ms=600)
    controller.update_prediction(swipe, timestamp_ms=700)
    rearmed = controller.update_prediction(swipe, timestamp_ms=800)
    assert rearmed.accepted is True and rearmed.final_action == "navigate_next"
