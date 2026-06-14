from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from research_pipeline.interaction.action_safe import ActionSafePolicy, ActionSafePolicyConfig
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, InteractionEvent
from research_pipeline.models.common import Prediction


LABEL_BY_ACTION = {action: label for label, action in ACTION_BY_LABEL.items()}


@dataclass(slots=True)
class TaskAwarePolicyConfig:
    base: ActionSafePolicyConfig = field(default_factory=ActionSafePolicyConfig)
    expected_threshold_delta: float = -0.12
    unexpected_threshold_delta: float = 0.08
    idle_threshold_delta: float = 0.12
    expected_stable_frames: int = 1


class TaskAwareActionSafePolicy:
    """Scenario-aware C4 policy for guided AR tasks.

    The policy keeps the calibrated action-safe controller, but temporarily
    lowers the threshold for the current expected task action and raises it for
    unexpected actions. This models guided AR workflows where the interface
    knows the next meaningful command.
    """

    def __init__(self, scenario: dict[str, Any] | None, config: TaskAwarePolicyConfig | None = None):
        self.scenario = scenario or {}
        self.config = config or TaskAwarePolicyConfig()
        self.expected_actions = _normalize_expected_actions(self.scenario)
        self.step_index = 0
        self.false_events = 0
        self.policy = ActionSafePolicy(copy.deepcopy(self.config.base))

    def reset(self) -> None:
        self.step_index = 0
        self.false_events = 0
        self.policy = ActionSafePolicy(copy.deepcopy(self.config.base))

    def update(self, prediction: Prediction, timestamp_ms: int) -> InteractionEvent | None:
        self.policy.config = self._config_for_current_step()
        event = self.policy.update(prediction, timestamp_ms)
        if event is None:
            return None
        expected = self.current_expected_action()
        if expected and event.action == expected["action"]:
            self.step_index = min(self.step_index + 1, len(self.expected_actions))
        else:
            self.false_events += 1
        return event

    def current_expected_action(self) -> dict[str, Any] | None:
        if self.step_index >= len(self.expected_actions):
            return None
        return self.expected_actions[self.step_index]

    def context(self) -> dict[str, Any]:
        expected = self.current_expected_action()
        expected_action = str(expected.get("action", "")) if expected else ""
        return {
            "mode": "c4_task_aware",
            "task_id": self.scenario.get("id", ""),
            "task_label": self.scenario.get("label", ""),
            "step_index": self.step_index,
            "step_count": len(self.expected_actions),
            "expected_action": expected_action,
            "expected_label": LABEL_BY_ACTION.get(expected_action, ""),
            "expected_id": str(expected.get("id", "")) if expected else "",
            "false_events": self.false_events,
        }

    def _config_for_current_step(self) -> ActionSafePolicyConfig:
        config = copy.deepcopy(self.config.base)
        thresholds = dict(config.label_thresholds)
        stable_frames = dict(config.label_stable_frames)
        expected = self.current_expected_action()
        expected_label = LABEL_BY_ACTION.get(str(expected.get("action", ""))) if expected else ""

        if expected_label:
            for label in ACTION_BY_LABEL:
                base_threshold = thresholds.get(label, config.default_threshold)
                delta = self.config.expected_threshold_delta if label == expected_label else self.config.unexpected_threshold_delta
                thresholds[label] = _clamp_threshold(base_threshold + delta)
            stable_frames[expected_label] = max(1, int(self.config.expected_stable_frames))
        else:
            for label in ACTION_BY_LABEL:
                base_threshold = thresholds.get(label, config.default_threshold)
                thresholds[label] = _clamp_threshold(base_threshold + self.config.idle_threshold_delta)

        config.label_thresholds = thresholds
        config.label_stable_frames = stable_frames
        return config


def task_aware_config_from_mapping(payload: dict[str, Any] | None) -> TaskAwarePolicyConfig:
    raw = payload or {}
    base_payload = raw.get("base", {})
    return TaskAwarePolicyConfig(
        base=ActionSafePolicyConfig(**base_payload),
        expected_threshold_delta=float(raw.get("expected_threshold_delta", -0.12)),
        unexpected_threshold_delta=float(raw.get("unexpected_threshold_delta", 0.08)),
        idle_threshold_delta=float(raw.get("idle_threshold_delta", 0.12)),
        expected_stable_frames=int(raw.get("expected_stable_frames", 1)),
    )


def load_task_scenarios(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    tasks = payload.get("tasks", {})
    return tasks if isinstance(tasks, dict) else {}


def _normalize_expected_actions(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    expected = scenario.get("expected_actions", [])
    if not isinstance(expected, list):
        return []
    output = []
    for index, item in enumerate(expected):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if action not in LABEL_BY_ACTION:
            continue
        output.append(
            {
                "id": str(item.get("id") or f"expected_{index + 1}"),
                "action": action,
                "target_ms": int(item.get("target_ms", 0)),
            }
        )
    return output


def _clamp_threshold(value: float) -> float:
    return max(0.05, min(0.99, float(value)))
