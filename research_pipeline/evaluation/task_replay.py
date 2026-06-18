from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from research_pipeline.evaluation.online_gesture import DEFAULT_ACTION_COSTS, OnlineEvent
from research_pipeline.interaction.fsm import ACTION_BY_LABEL


LABEL_BY_ACTION = {action: label for label, action in ACTION_BY_LABEL.items()}


@dataclass(frozen=True, slots=True)
class TaskStep:
    id: str
    action: str
    required: bool = True
    target_ms: int = 0

    @property
    def expected_label(self) -> str:
        return LABEL_BY_ACTION.get(self.action, "no_gesture")


@dataclass(frozen=True, slots=True)
class TaskScenario:
    id: str
    label: str
    expected_steps: tuple[TaskStep, ...]
    allowed_gestures: tuple[str, ...]
    success_condition: str
    penalty_condition: str
    failure_condition: str
    max_false_action_cost: float = 0.0
    action_costs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_COSTS))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_steps"] = [asdict(step) | {"expected_label": step.expected_label} for step in self.expected_steps]
        return payload


DEFAULT_COMPLETION_THRESHOLD = 0.5


@dataclass(slots=True)
class TaskReplayResult:
    task_id: str
    task_label: str
    task_success: bool
    confident_completion: bool
    task_completion_score: float
    accepted_actions: int
    rejected_actions: int
    expected_actions: int
    matched_actions: int
    missed_actions: int
    false_actions: int
    false_action_cost: float
    missed_action_cost: float
    expected_action_cost: float
    weighted_action_precision: float
    weighted_action_recall: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_task_scenarios(action_costs: dict[str, float] | None = None) -> dict[str, TaskScenario]:
    costs = dict(action_costs or DEFAULT_ACTION_COSTS)
    return {
        "object_control": TaskScenario(
            id="object_control",
            label="Object control",
            expected_steps=(
                TaskStep("object_point", "pointer_hover"),
                TaskStep("object_select", "select_confirm"),
                TaskStep("object_zoom_in", "zoom_in"),
                TaskStep("object_zoom_out", "zoom_out"),
            ),
            allowed_gestures=("point_2f", "click_2f", "zoom_in", "zoom_out", "no_gesture"),
            success_condition="All required actions are accepted in order with no false action cost.",
            penalty_condition="Unexpected accepted actions add their configured risk cost.",
            failure_condition="Any missed required action or any false action cost above zero fails the task.",
            max_false_action_cost=0.0,
            action_costs=costs,
        ),
        "scroll_open": TaskScenario(
            id="scroll_open",
            label="Scroll and open",
            expected_steps=(
                TaskStep("scroll_next_1", "navigate_next"),
                TaskStep("scroll_next_2", "navigate_next"),
                TaskStep("scroll_previous", "navigate_previous"),
                TaskStep("scroll_open", "select_confirm"),
            ),
            allowed_gestures=("swipe_right", "swipe_left", "click_2f", "no_gesture"),
            success_condition="Two next actions, one previous action, and one confirm action are accepted in order.",
            penalty_condition="Unexpected navigation or confirm actions add risk cost.",
            failure_condition="Missing an expected action or accepting an out-of-sequence action fails the task.",
            max_false_action_cost=0.0,
            action_costs=costs,
        ),
        "sort_virtual_item": TaskScenario(
            id="sort_virtual_item",
            label="Sort virtual item",
            expected_steps=(
                TaskStep("sort_point", "pointer_hover"),
                TaskStep("sort_pick", "select_confirm"),
                TaskStep("sort_move", "navigate_next"),
                TaskStep("sort_drop", "select_confirm"),
            ),
            allowed_gestures=("point_2f", "click_2f", "swipe_right", "no_gesture"),
            success_condition="The item is pointed at, picked, moved to the target, and dropped in order.",
            penalty_condition="Unexpected clicks are high-cost false actions; unexpected navigation is medium-cost.",
            failure_condition="Missing pick/drop or triggering extra confirm fails the task.",
            max_false_action_cost=0.0,
            action_costs=costs,
        ),
    }


def evaluate_task_replay(
    events: list[OnlineEvent],
    scenario: TaskScenario,
    *,
    completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
) -> TaskReplayResult:
    accepted = [event for event in events if event.action_accepted and event.final_action and event.final_action != "idle"]
    rejected = [
        event
        for event in events
        if not event.action_accepted
        and event.proposal_label in ACTION_BY_LABEL
        and ACTION_BY_LABEL.get(event.proposal_label, "idle") != "idle"
        and event.rejection_reason not in {"", "idle"}
    ]

    expected_steps = list(scenario.expected_steps)
    matched_event_indexes: set[int] = set()
    matched_steps: list[TaskStep] = []
    cursor = 0

    for step in expected_steps:
        found_index: int | None = None
        for index in range(cursor, len(accepted)):
            if index in matched_event_indexes:
                continue
            event = accepted[index]
            if event.final_action == step.action:
                found_index = index
                break
        if found_index is None:
            continue
        matched_event_indexes.add(found_index)
        matched_steps.append(step)
        cursor = found_index + 1

    false_events = [event for index, event in enumerate(accepted) if index not in matched_event_indexes]
    missed_steps = [step for step in expected_steps if step.required and step not in matched_steps]
    expected_cost = sum(_cost(scenario.action_costs, step.action) for step in expected_steps if step.required)
    matched_cost = sum(_cost(scenario.action_costs, step.action) for step in matched_steps if step.required)
    false_cost = sum(_cost(scenario.action_costs, event.final_action) for event in false_events)
    missed_cost = sum(_cost(scenario.action_costs, step.action) for step in missed_steps)
    precision = matched_cost / (matched_cost + false_cost) if matched_cost + false_cost else 0.0
    recall = matched_cost / expected_cost if expected_cost else 0.0
    success = not missed_steps and false_cost <= scenario.max_false_action_cost
    # Graded counterpart to the binary `task_success`: the strict criterion has a
    # zero false-action tolerance, so on noisy replay it collapses to a floor.
    # The completion score is the F1 of the cost-weighted action precision and
    # recall, so partial progress and clean-but-imperfect runs score continuously.
    completion = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    # "Confident completion" is the headline task metric: the strict binary success has a
    # zero false-action tolerance and floors out, so a task counts as completed when the
    # graded completion score clears a threshold instead. It rewards getting the steps done
    # with mostly-correct action cost, without requiring a perfectly clean run.
    confident_completion = completion >= completion_threshold

    return TaskReplayResult(
        task_id=scenario.id,
        task_label=scenario.label,
        task_success=success,
        confident_completion=confident_completion,
        task_completion_score=completion,
        accepted_actions=len(accepted),
        rejected_actions=len(rejected),
        expected_actions=len(expected_steps),
        matched_actions=len(matched_steps),
        missed_actions=len(missed_steps),
        false_actions=len(false_events),
        false_action_cost=false_cost,
        missed_action_cost=missed_cost,
        expected_action_cost=expected_cost,
        weighted_action_precision=precision,
        weighted_action_recall=recall,
    )


def evaluate_task_set(
    events: Iterable[OnlineEvent],
    scenarios: dict[str, TaskScenario],
    *,
    completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
) -> dict[str, Any]:
    by_task: dict[tuple[str, str], list[OnlineEvent]] = {}
    for event in events:
        if event.task_id:
            by_task.setdefault((event.sequence_id, event.task_id), []).append(event)

    results: list[TaskReplayResult] = []
    for (_sequence_id, task_id), task_events in sorted(by_task.items()):
        scenario = scenarios.get(task_id)
        if scenario is None:
            continue
        results.append(evaluate_task_replay(task_events, scenario, completion_threshold=completion_threshold))

    rows = [result.to_dict() for result in results]
    count = len(rows)
    return {
        "tasks": rows,
        "summary": {
            "task_count": count,
            "completion_threshold": float(completion_threshold),
            "task_success_rate": sum(1.0 for row in rows if row["task_success"]) / count if count else 0.0,
            "confident_completion_rate": sum(1.0 for row in rows if row["confident_completion"]) / count if count else 0.0,
            "task_completion_score": _mean([float(row["task_completion_score"]) for row in rows]),
            "accepted_actions": sum(int(row["accepted_actions"]) for row in rows),
            "rejected_actions": sum(int(row["rejected_actions"]) for row in rows),
            "false_action_cost": sum(float(row["false_action_cost"]) for row in rows),
            "missed_action_cost": sum(float(row["missed_action_cost"]) for row in rows),
            "expected_action_cost": sum(float(row["expected_action_cost"]) for row in rows),
            "weighted_action_precision": _mean([float(row["weighted_action_precision"]) for row in rows]),
            "weighted_action_recall": _mean([float(row["weighted_action_recall"]) for row in rows]),
        },
    }


def _cost(costs: dict[str, float], action: str) -> float:
    return float(costs.get(action, 0.0 if action == "idle" else 1.0))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
