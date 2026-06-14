from __future__ import annotations

import argparse
import copy
import hashlib
import time
from dataclasses import asdict
from typing import Any, Callable

import numpy as np

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import ManifestRecord, resolve_path
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.evaluation.action_risk import normalize_action_costs
from research_pipeline.evaluation.live_sessions import evaluate_task_scenario
from research_pipeline.evaluation.robustness import PerturbationConfig, perturb_tensor
from research_pipeline.evaluation.task_benchmark import summarize_task_metric_rows, task_report_to_metrics
from research_pipeline.interaction.action_safe import ActionSafePolicy, ActionSafePolicyConfig
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, ContextAwarePolicy, ContextPolicyConfig
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.c6_ensemble import C6EnsembleRecognizer, c6_config_from_mapping
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.hybrid import CachedArtifactPredictor, HybridConfig, HybridRecognizer


Recognizer = Callable[[LandmarkTensor], Prediction]
ACTION_LABELS = {action: label for label, action in ACTION_BY_LABEL.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-level C4 AR interaction benchmark.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    start = time.perf_counter()
    task_payload = load_yaml(project_path(config["task_scenarios_path"]))
    tasks = task_payload.get("tasks", {})
    tolerance_ms = int(task_payload.get("tolerance_ms", 450))
    scenarios = [_scenario_from_dict(item) for item in config.get("evaluation_scenarios", [{"name": "clean", "kind": "clean"}])]
    action_costs = _load_action_costs(config)
    manifest_path = project_path(config["evaluation_manifest"])
    records_by_label = _records_by_label(read_jsonl(manifest_path))

    artifact = load_artifact(project_path(config["model_path"]))
    direct = CachedArtifactPredictor(artifact)
    c3 = HybridRecognizer(artifact, HybridConfig(**config.get("hybrid", {})))
    c6 = _build_c6_recognizer(config)
    prediction_cache: dict[tuple[str, str, str], Prediction] = {}
    recognizers: dict[str, Recognizer] = {"c1t": direct.predict, "c3": c3.predict}
    if c6 is not None:
        recognizers["c6"] = c6.predict
    method_runners = _method_runners(config)

    trials = []
    seed = int(config.get("seed", 42))
    num_trials = int(config.get("num_trials_per_task", 6))
    for scenario_index, scenario in enumerate(scenarios):
        for task_key, scenario_task in sorted(tasks.items()):
            for trial_index in range(num_trials):
                rng = np.random.default_rng(seed + scenario_index * 100_003 + trial_index * 997 + _stable_int(task_key))
                items = _build_trial_items(
                    scenario_task,
                    task_key=task_key,
                    trial_index=trial_index,
                    scenario=scenario,
                    records_by_label=records_by_label,
                    manifest_base_dir=manifest_path.parent,
                    recognizers=recognizers,
                    prediction_cache=prediction_cache,
                    rng=rng,
                    seed=seed,
                    idle_distractors_per_gap=int(config.get("idle_distractors_per_gap", 1)),
                )
                for method, runner in method_runners.items():
                    task_records = runner(items)
                    task_report = evaluate_task_scenario(task_records, scenario_task, tolerance_ms=tolerance_ms)
                    if task_report is None:
                        continue
                    metrics = task_report_to_metrics(task_report, action_costs)
                    trials.append(
                        {
                            "scenario": scenario.name,
                            "task": task_key,
                            "task_id": scenario_task.get("id", task_key),
                            "task_label": scenario_task.get("label", task_key),
                            "trial": trial_index + 1,
                            "method": method,
                            **metrics,
                        }
                    )

    report = {
        "method": {
            "name": "C4 Task-Level AR Benchmark",
            "description": (
                "Task-level replay benchmark for gesture-driven AR scenarios. "
                "It measures whether full interface tasks complete, not only whether isolated gestures are classified."
            ),
        },
        "config": config,
        "task_scenarios": {
            "path": config["task_scenarios_path"],
            "tolerance_ms": tolerance_ms,
            "tasks": tasks,
        },
        "evaluation": {
            "manifest": config["evaluation_manifest"],
            "scenarios": [asdict(item) for item in scenarios],
            "num_trials": len(trials),
            "trials": trials,
            "summary": _summary_by(trials, ["method"]),
            "by_task": _summary_by(trials, ["task", "method"]),
            "by_scenario": _summary_by(trials, ["scenario", "method"]),
            "bootstrap_ci": _bootstrap_ci(
                trials,
                metrics=[
                    "task_success_rate",
                    "action_precision",
                    "action_recall",
                    "unintended_action_rate",
                    "false_action_cost_rate",
                    "missed_action_cost_rate",
                ],
                seed=seed + 9091,
                iterations=int(config.get("bootstrap_iterations", 1000)),
            ),
        },
        "action_costs": action_costs,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_report(config.get("output_report", "artifacts/reports/c4_task_benchmark.json"), report)
    if config.get("method_set") == "official_compact":
        summary = report["evaluation"]["summary"]
        baseline = summary.get("baseline_direct", {})
        robust = summary.get("robust_recognizer_direct", {})
        proposed = summary.get("proposed_tarc", {})
        print(
            "baseline_success="
            f"{baseline.get('task_success_rate_mean', 0.0):.4f} "
            f"robust_success={robust.get('task_success_rate_mean', 0.0):.4f} "
            f"proposed_success={proposed.get('task_success_rate_mean', 0.0):.4f} "
            f"proposed_false_cost={proposed.get('false_action_cost_rate_mean', 0.0):.4f} "
            f"proposed_unintended={proposed.get('unintended_action_rate_mean', 0.0):.4f}"
        )
        return
    c4 = report["evaluation"]["summary"].get("c4_safety", {})
    c2 = report["evaluation"]["summary"].get("c3_c2_default", {})
    task_aware = report["evaluation"]["summary"].get("c4_task_aware", {})
    print(
        "c4_task_success="
        f"{c4.get('task_success_rate_mean', 0.0):.4f} "
        f"c2_task_success={c2.get('task_success_rate_mean', 0.0):.4f} "
        f"c4_false_cost={c4.get('false_action_cost_rate_mean', 0.0):.4f} "
        f"c4_unintended={c4.get('unintended_action_rate_mean', 0.0):.4f} "
        f"task_aware_success={task_aware.get('task_success_rate_mean', 0.0):.4f} "
        f"task_aware_false_cost={task_aware.get('false_action_cost_rate_mean', 0.0):.4f}"
    )


def _build_c6_recognizer(config: dict[str, Any]) -> C6EnsembleRecognizer | None:
    payload = copy.deepcopy(config.get("c6", {}))
    if not payload:
        return None
    payload["model_paths"] = [str(project_path(path)) for path in payload.get("model_paths", [])]
    return C6EnsembleRecognizer(c6_config_from_mapping(payload))


def _method_runners(config: dict[str, Any]) -> dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]:
    if config.get("method_set") == "official_compact":
        return {
            "baseline_direct": lambda items: _direct_records(items, "c1t", method_name="baseline_direct"),
            "robust_recognizer_direct": lambda items: _direct_records(
                items,
                "c6" if config.get("c6") else "c3",
                method_name="robust_recognizer_direct",
            ),
            "proposed_tarc": lambda items: _policy_records(
                items,
                _TaskAwareActionSafePolicyAdapter(config.get("c4_task_aware_policy", {}), name="proposed_tarc"),
                config,
                recognizer_name="c6" if config.get("c6") else "c3",
            ),
        }
    return {
        "c1t_direct": lambda items: _direct_records(items, "c1t"),
        "c3_direct": lambda items: _direct_records(items, "c3"),
        "c3_c2_default": lambda items: _policy_records(
            items,
            _ContextPolicyAdapter(ContextPolicyConfig(**config.get("c2_policy", {}))),
            config,
        ),
        "c4_balanced": lambda items: _policy_records(
            items,
            _ActionSafePolicyAdapter(ActionSafePolicyConfig(**config.get("c4_balanced_policy", {}))),
            config,
        ),
        "c4_safety": lambda items: _policy_records(
            items,
            _ActionSafePolicyAdapter(ActionSafePolicyConfig(**config.get("c4_safety_policy", {}))),
            config,
        ),
        "c4_task_aware": lambda items: _policy_records(
            items,
            _TaskAwareActionSafePolicyAdapter(config.get("c4_task_aware_policy", {})),
            config,
        ),
    }


def _build_trial_items(
    task: dict[str, Any],
    *,
    task_key: str,
    trial_index: int,
    scenario: PerturbationConfig,
    records_by_label: dict[str, list[ManifestRecord]],
    manifest_base_dir,
    recognizers: dict[str, Recognizer],
    prediction_cache: dict[tuple[str, str, str], Prediction],
    rng: np.random.Generator,
    seed: int,
    idle_distractors_per_gap: int,
) -> list[dict[str, Any]]:
    expected = task.get("expected_actions", [])
    items = []
    for step_index, step in enumerate(expected):
        action = str(step["action"])
        label = ACTION_LABELS[action]
        record = _choose_record(records_by_label, label, rng)
        predictions = _predictions_for_record(
            record,
            scenario,
            manifest_base_dir,
            recognizers,
            prediction_cache,
            seed=seed + _stable_int(f"{task_key}-{trial_index}-{step_index}"),
        )
        items.append(
            {
                "kind": "expected",
                "action": action,
                "expected_label": label,
                "timestamp_ms": int(step.get("target_ms", step.get("start_ms", 0))),
                "start_ms": int(step.get("start_ms", 0)),
                "end_ms": int(step.get("end_ms", step.get("target_ms", 0))),
                "predictions": predictions,
            }
        )

    for timestamp in _idle_timestamps(expected, idle_distractors_per_gap):
        record = _choose_record(records_by_label, "no_gesture", rng)
        predictions = _predictions_for_record(
            record,
            scenario,
            manifest_base_dir,
            recognizers,
            prediction_cache,
            seed=seed + _stable_int(f"{task_key}-{trial_index}-idle-{timestamp}"),
        )
        items.append(
            {
                "kind": "idle",
                "action": "",
                "expected_label": "no_gesture",
                "timestamp_ms": timestamp,
                "start_ms": timestamp,
                "end_ms": timestamp,
                "predictions": predictions,
            }
        )
    return sorted(items, key=lambda item: (int(item["timestamp_ms"]), 0 if item["kind"] == "expected" else 1))


def _direct_records(items: list[dict[str, Any]], recognizer_name: str, *, method_name: str | None = None) -> list[dict[str, Any]]:
    method = method_name or f"{recognizer_name}_direct"
    records = [_idle_anchor_record(method=method)]
    for item in items:
        prediction = item["predictions"][recognizer_name]
        action = ACTION_BY_LABEL.get(prediction.label, "idle")
        records.append(_task_record(item, prediction, action, method=method))
    return records


def _policy_records(
    items: list[dict[str, Any]],
    adapter: "_PolicyAdapter",
    config: dict[str, Any],
    *,
    recognizer_name: str = "c3",
) -> list[dict[str, Any]]:
    adapter.reset()
    records = [_idle_anchor_record(method=adapter.name)]
    frames_per_clip = int(config.get("policy_frames_per_clip", 3))
    frame_step_ms = int(config.get("policy_frame_step_ms", 100))
    reset_frames = int(config.get("policy_reset_frames", 3))
    reset_step_ms = int(config.get("policy_reset_step_ms", 90))
    no_gesture = prediction_from_scores({"no_gesture": 1.0})

    for item in items:
        prediction = item["predictions"][recognizer_name]
        target_ms = int(item["timestamp_ms"])
        start_ms = target_ms - frame_step_ms * max(0, frames_per_clip - 1)
        for frame_index in range(frames_per_clip):
            timestamp = start_ms + frame_index * frame_step_ms
            event = adapter.update(prediction, timestamp, item)
            action = event.action if event else "idle"
            records.append(_task_record(item, prediction, action, timestamp_ms=timestamp, event=event, method=adapter.name))
        for reset_index in range(reset_frames):
            timestamp = int(item["end_ms"]) + (reset_index + 1) * reset_step_ms
            adapter.update(no_gesture, timestamp, item)
            records.append(_task_record(item, no_gesture, "idle", timestamp_ms=timestamp, method=adapter.name))
    return records


def _idle_anchor_record(*, method: str) -> dict[str, Any]:
    return {
        "type": "prediction",
        "timestamp_ms": 0,
        "task": "",
        "gesture": "no_gesture",
        "action": "idle",
        "confidence": 1.0,
        "method": method,
    }


def _task_record(
    item: dict[str, Any],
    prediction: Prediction,
    action: str,
    *,
    timestamp_ms: int | None = None,
    event: Any | None = None,
    method: str,
) -> dict[str, Any]:
    return {
        "type": "prediction",
        "timestamp_ms": int(item["timestamp_ms"] if timestamp_ms is None else timestamp_ms),
        "task": item.get("task", ""),
        "gesture": event.gesture if event else prediction.label,
        "action": action,
        "confidence": float(event.confidence if event else prediction.confidence),
        "method": method,
    }


def _predictions_for_record(
    record: ManifestRecord,
    scenario: PerturbationConfig,
    manifest_base_dir,
    recognizers: dict[str, Recognizer],
    prediction_cache: dict[tuple[str, str, str], Prediction],
    *,
    seed: int,
) -> dict[str, Prediction]:
    output = {}
    for name, recognizer in recognizers.items():
        key = (record.sample_id, scenario.name, name)
        if key not in prediction_cache:
            rng = np.random.default_rng(seed + _stable_int(f"{record.sample_id}-{scenario.name}-{name}"))
            tensor = load_landmark_npz(resolve_path(record.tensor_path, manifest_base_dir))
            prediction_cache[key] = recognizer(perturb_tensor(tensor, scenario, rng))
        output[name] = prediction_cache[key]
    return output


def _idle_timestamps(expected: list[dict[str, Any]], distractors_per_gap: int) -> list[int]:
    if distractors_per_gap <= 0 or not expected:
        return []
    ordered = sorted(expected, key=lambda item: int(item.get("target_ms", item.get("start_ms", 0))))
    timestamps = []
    first_start = int(ordered[0].get("start_ms", 0))
    if first_start > 400:
        timestamps.append(max(0, first_start // 2))
    for left, right in zip(ordered, ordered[1:]):
        left_end = int(left.get("end_ms", left.get("target_ms", 0)))
        right_start = int(right.get("start_ms", right.get("target_ms", 0)))
        if right_start <= left_end:
            continue
        for index in range(distractors_per_gap):
            fraction = (index + 1) / (distractors_per_gap + 1)
            timestamps.append(int(left_end + (right_start - left_end) * fraction))
    last = ordered[-1]
    timestamps.append(int(last.get("end_ms", last.get("target_ms", 0))) + 700)
    return timestamps


def _choose_record(records_by_label: dict[str, list[ManifestRecord]], label: str, rng: np.random.Generator) -> ManifestRecord:
    candidates = records_by_label.get(label)
    if not candidates:
        raise ValueError(f"No manifest records found for label '{label}'.")
    return candidates[int(rng.integers(0, len(candidates)))]


def _records_by_label(records: list[ManifestRecord]) -> dict[str, list[ManifestRecord]]:
    output = {label: [] for label in TARGET_LABELS}
    for record in records:
        output.setdefault(record.target_label, []).append(record)
    return output


def _summary_by(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = " / ".join(str(row[key]) for key in keys)
        grouped.setdefault(group_key, []).append(row)
    return {key: summarize_task_metric_rows(values) for key, values in sorted(grouped.items())}


def _bootstrap_ci(
    rows: list[dict[str, Any]],
    *,
    metrics: list[str],
    seed: int,
    iterations: int,
) -> dict[str, dict[str, dict[str, float]]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for method, method_rows in sorted(by_method.items()):
        output[method] = {}
        if not method_rows:
            continue
        for metric in metrics:
            values = []
            for _ in range(max(1, iterations)):
                indexes = rng.integers(0, len(method_rows), size=len(method_rows))
                sample = [float(method_rows[int(index)].get(metric, 0.0)) for index in indexes]
                values.append(sum(sample) / len(sample) if sample else 0.0)
            ordered = sorted(values)
            output[method][metric] = {
                "p2_5": _percentile_sorted(ordered, 2.5),
                "mean": sum(values) / len(values),
                "p97_5": _percentile_sorted(ordered, 97.5),
            }
    return output


def _scenario_from_dict(payload: dict[str, Any]) -> PerturbationConfig:
    return PerturbationConfig(
        name=str(payload["name"]),
        kind=str(payload.get("kind", "clean")),
        sigma=float(payload.get("sigma", 0.0)),
        drop_rate=float(payload.get("drop_rate", 0.0)),
        mask_rate=float(payload.get("mask_rate", 0.0)),
        jitter=int(payload.get("jitter", 0)),
        translation=float(payload.get("translation", 0.0)),
        scale=float(payload.get("scale", 0.0)),
    )


def _load_action_costs(config: dict[str, Any]) -> dict[str, float]:
    if config.get("risk_costs_path"):
        return normalize_action_costs(load_yaml(project_path(config["risk_costs_path"])))
    return normalize_action_costs(config.get("action_costs"))


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _percentile_sorted(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    index = int(round((percentile_value / 100.0) * (len(values) - 1)))
    return values[min(len(values) - 1, max(0, index))]


class _PolicyAdapter:
    name: str

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, prediction: Prediction, timestamp_ms: int, item: dict[str, Any] | None = None):
        raise NotImplementedError


class _ContextPolicyAdapter(_PolicyAdapter):
    name = "c3_c2_default"

    def __init__(self, config: ContextPolicyConfig):
        self.policy = ContextAwarePolicy(config)

    def reset(self) -> None:
        self.policy.reset()

    def update(self, prediction: Prediction, timestamp_ms: int, item: dict[str, Any] | None = None):
        return self.policy.update(prediction, timestamp_ms)


class _ActionSafePolicyAdapter(_PolicyAdapter):
    def __init__(self, config: ActionSafePolicyConfig):
        self.policy = ActionSafePolicy(config)
        self.name = "c4_safety" if config.default_threshold >= 0.7 else "c4_balanced"

    def reset(self) -> None:
        self.policy.reset()

    def update(self, prediction: Prediction, timestamp_ms: int, item: dict[str, Any] | None = None):
        return self.policy.update(prediction, timestamp_ms)


class _TaskAwareActionSafePolicyAdapter(_PolicyAdapter):
    name = "c4_task_aware"

    def __init__(self, config: dict[str, Any], *, name: str = "c4_task_aware"):
        self.name = name
        self.raw_config = config
        self.base_config = ActionSafePolicyConfig(**config.get("base", {}))
        self.policy = ActionSafePolicy(copy.deepcopy(self.base_config))
        self.expected_threshold_delta = float(config.get("expected_threshold_delta", -0.1))
        self.unexpected_threshold_delta = float(config.get("unexpected_threshold_delta", 0.08))
        self.idle_threshold_delta = float(config.get("idle_threshold_delta", 0.12))
        self.expected_stable_frames = int(config.get("expected_stable_frames", 1))

    def reset(self) -> None:
        self.policy.reset()
        self.policy.config = copy.deepcopy(self.base_config)

    def update(self, prediction: Prediction, timestamp_ms: int, item: dict[str, Any] | None = None):
        self.policy.config = self._config_for_item(item)
        return self.policy.update(prediction, timestamp_ms)

    def _config_for_item(self, item: dict[str, Any] | None) -> ActionSafePolicyConfig:
        config = copy.deepcopy(self.base_config)
        thresholds = dict(config.label_thresholds)
        stable_frames = dict(config.label_stable_frames)
        expected_label = str((item or {}).get("expected_label", ""))

        if expected_label in ACTION_BY_LABEL:
            for label in ACTION_BY_LABEL:
                base_threshold = thresholds.get(label, config.default_threshold)
                delta = self.expected_threshold_delta if label == expected_label else self.unexpected_threshold_delta
                thresholds[label] = _clamp_threshold(base_threshold + delta)
            stable_frames[expected_label] = max(1, self.expected_stable_frames)
        else:
            for label in ACTION_BY_LABEL:
                base_threshold = thresholds.get(label, config.default_threshold)
                thresholds[label] = _clamp_threshold(base_threshold + self.idle_threshold_delta)

        config.label_thresholds = thresholds
        config.label_stable_frames = stable_frames
        return config


def _clamp_threshold(value: float) -> float:
    return max(0.05, min(0.99, float(value)))


if __name__ == "__main__":
    main()
