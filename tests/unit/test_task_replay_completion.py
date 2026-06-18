from __future__ import annotations

import pytest

from research_pipeline.evaluation.online_gesture import OnlineEvent
from research_pipeline.evaluation.task_replay import TaskScenario, TaskStep, evaluate_task_replay

ACTION_COSTS = {"pointer_hover": 1.0, "select_confirm": 2.0}


def _scenario() -> TaskScenario:
    return TaskScenario(
        id="unit_task",
        label="Unit task",
        expected_steps=(
            TaskStep("point", "pointer_hover"),
            TaskStep("select", "select_confirm"),
        ),
        allowed_gestures=("point_2f", "click_2f", "no_gesture"),
        success_condition="both steps accepted in order with no false action",
        penalty_condition="extra accepted actions add cost",
        failure_condition="missed or false action fails the task",
        max_false_action_cost=0.0,
        action_costs=ACTION_COSTS,
    )


def _accept(action: str, index: int) -> OnlineEvent:
    return OnlineEvent(
        sequence_id="seq",
        frame_index=index,
        timestamp_ms=index * 100,
        final_action=action,
        action_accepted=True,
        task_id="unit_task",
    )


def test_perfect_run_scores_one_and_succeeds() -> None:
    events = [_accept("pointer_hover", 0), _accept("select_confirm", 1)]
    result = evaluate_task_replay(events, _scenario())
    assert result.task_success is True
    assert result.task_completion_score == 1.0


def test_clean_but_imperfect_run_is_graded_not_zero() -> None:
    # All steps done in order, plus one extra (false) confirm -> binary success fails,
    # but the graded completion stays high instead of collapsing to the floor.
    events = [
        _accept("pointer_hover", 0),
        _accept("select_confirm", 1),
        _accept("select_confirm", 2),
    ]
    result = evaluate_task_replay(events, _scenario())
    assert result.task_success is False
    # precision = 3/(3+2) = 0.6, recall = 1.0 -> F1 = 0.75
    assert result.task_completion_score == pytest.approx(0.75)


def test_partial_progress_scores_between_zero_and_one() -> None:
    events = [_accept("pointer_hover", 0)]
    result = evaluate_task_replay(events, _scenario())
    assert result.task_success is False
    # matched cost 1, expected 3 -> recall 1/3, precision 1.0 -> F1 = 0.5
    assert result.task_completion_score == 0.5


def test_confident_completion_uses_threshold() -> None:
    # Clean-but-imperfect run (one extra confirm): completion 0.75, strict success fails.
    events = [
        _accept("pointer_hover", 0),
        _accept("select_confirm", 1),
        _accept("select_confirm", 2),
    ]
    assert evaluate_task_replay(events, _scenario(), completion_threshold=0.5).confident_completion is True
    assert evaluate_task_replay(events, _scenario(), completion_threshold=0.8).confident_completion is False
    # Binary success stays False either way; confident completion is the graded headline.
    assert evaluate_task_replay(events, _scenario(), completion_threshold=0.5).task_success is False


def test_confident_completion_rate_in_summary() -> None:
    from research_pipeline.evaluation.task_replay import evaluate_task_set

    events = [_accept("pointer_hover", 0), _accept("select_confirm", 1)]  # perfect -> completion 1.0
    report = evaluate_task_set(events, {"unit_task": _scenario()}, completion_threshold=0.5)
    summary = report["summary"]
    assert summary["completion_threshold"] == 0.5
    assert summary["confident_completion_rate"] == 1.0
    assert summary["task_success_rate"] == 1.0
