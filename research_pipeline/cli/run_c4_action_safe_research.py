from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.evaluation.ablation import _direct_policy_metrics, _rows_to_replay_frames
from research_pipeline.evaluation.action_risk import direct_risk_metrics, event_risk_metrics, normalize_action_costs
from research_pipeline.evaluation.interaction import ReplayFrame, compute_interaction_metrics, replay_predictions
from research_pipeline.evaluation.robustness import PerturbationConfig, perturb_tensor
from research_pipeline.interaction.action_safe import ActionSafePolicy, ActionSafePolicyConfig
from research_pipeline.interaction.fsm import ContextPolicyConfig
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.hybrid import CachedArtifactPredictor, HybridConfig, HybridRecognizer


Recognizer = Callable[[LandmarkTensor], Prediction]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C4 action-safe AR interaction research benchmark.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    start = time.perf_counter()
    action_costs = _load_action_costs(config)
    artifact = load_artifact(project_path(config["model_path"]))
    hybrid_config = HybridConfig(**config.get("hybrid", {}))
    direct = CachedArtifactPredictor(artifact)
    c3 = HybridRecognizer(artifact, hybrid_config)
    calibration_scenarios = [_scenario_from_dict(item) for item in config.get("calibration_scenarios", [])]
    evaluation_scenarios = [_scenario_from_dict(item) for item in config.get("evaluation_scenarios", [])]
    if not calibration_scenarios:
        calibration_scenarios = [_scenario_from_dict({"name": "clean", "kind": "clean"})]
    if not evaluation_scenarios:
        evaluation_scenarios = calibration_scenarios

    calibration = _calibrate_policy(c3.predict, config, calibration_scenarios, action_costs)
    balanced_policy = ActionSafePolicyConfig(**calibration["best_balanced"]["config"])
    safety_policy = ActionSafePolicyConfig(**calibration["best_safety"]["config"])
    c2_policy = ContextPolicyConfig(**config.get("c2_policy", {}))
    evaluation = _evaluate_methods(
        {"c1t": direct.predict, "c3": c3.predict},
        config,
        evaluation_scenarios,
        action_costs=action_costs,
        c2_policy=c2_policy,
        c4_balanced_policy=balanced_policy,
        c4_safety_policy=safety_policy,
    )
    report = {
        "method": {
            "name": "C4 Action-Safe Interaction",
            "description": (
                "A risk-aware AR interaction layer that combines C3 recognition, "
                "utility-calibrated confidence thresholds, temporal stability, cooldown, and abstention."
            ),
        },
        "calibration": calibration,
        "evaluation": evaluation,
        "action_costs": action_costs,
        "config": config,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_report(config.get("output_report", "artifacts/reports/c4_action_safe_research.json"), report)
    summary = evaluation["summary"]
    c4 = summary.get("c4_safety", {})
    baseline = summary.get("c3_c2_default", {})
    print(
        "c4_unintended="
        f"{c4.get('unintended_action_rate_mean', 0.0):.4f} "
        f"c3_c2_unintended={baseline.get('unintended_action_rate_mean', 0.0):.4f} "
        f"c4_precision={c4.get('action_precision_mean', 0.0):.4f} "
        f"c4_recall={c4.get('action_recall_mean', 0.0):.4f}"
    )


def _calibrate_policy(
    recognizer: Recognizer,
    config: dict[str, Any],
    scenarios: list[PerturbationConfig],
    action_costs: dict[str, float],
) -> dict[str, Any]:
    candidates = _candidate_policy_configs(config)
    objective = config.get("objective", {})
    scenario_rows = []
    for index, scenario in enumerate(scenarios):
        scenario_rows.append(
            (
                scenario,
                _predict_manifest(
                    project_path(config["calibration_manifest"]),
                    recognizer,
                    scenario,
                    seed=int(config.get("seed", 42)) + 1009 * (index + 1),
                    max_records=int(config["max_calibration_records"]) if config.get("max_calibration_records") else None,
                    sample_strategy=str(config.get("calibration_sample_strategy", "first")),
                ),
            )
        )

    ranked = []
    for candidate in candidates:
        metrics_by_scenario = []
        policy_config = ActionSafePolicyConfig(**candidate)
        for scenario, rows in scenario_rows:
            metrics_by_scenario.append(
                {
                    "scenario": scenario.name,
                    **_action_safe_metrics(rows, policy_config, config, action_costs),
                }
            )
        summary = _summarize_policy_metrics(metrics_by_scenario)
        score = _score_policy_candidate(
            summary,
            precision_weight=float(objective.get("precision_weight", 0.45)),
            recall_weight=float(objective.get("recall_weight", 0.45)),
            unintended_penalty=float(objective.get("unintended_penalty", 0.55)),
            false_rate_penalty=float(objective.get("false_rate_penalty", 0.01)),
            min_recall=float(objective.get("min_recall", 0.88)),
            recall_floor_penalty=float(objective.get("recall_floor_penalty", 2.0)),
        )
        ranked.append({"rank": 0, "score": score, "config": candidate, "summary": summary})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    safety_ranked = sorted(
        ranked,
        key=lambda item: (
            _score_safety_candidate(
                item["summary"],
                min_recall=float(config.get("objective", {}).get("safety_min_recall", 0.88)),
            ),
            float(item["config"].get("default_threshold", 0.0)),
            _mean_label_threshold(item["config"]),
        ),
        reverse=True,
    )
    return {
        "manifest": config["calibration_manifest"],
        "scenarios": [asdict(item) for item in scenarios],
        "num_candidates": len(ranked),
        "best": ranked[0],
        "best_balanced": ranked[0],
        "best_safety": safety_ranked[0],
        "top_candidates": ranked[: min(10, len(ranked))],
        "top_safety_candidates": safety_ranked[: min(10, len(safety_ranked))],
    }


def _evaluate_methods(
    recognizers: dict[str, Recognizer],
    config: dict[str, Any],
    scenarios: list[PerturbationConfig],
    *,
    action_costs: dict[str, float],
    c2_policy: ContextPolicyConfig,
    c4_balanced_policy: ActionSafePolicyConfig,
    c4_safety_policy: ActionSafePolicyConfig,
) -> dict[str, Any]:
    by_scenario = {}
    for index, scenario in enumerate(scenarios):
        c1t_rows = _predict_manifest(
            project_path(config["evaluation_manifest"]),
            recognizers["c1t"],
            scenario,
            seed=int(config.get("seed", 42)) + 2003 * (index + 1),
            max_records=int(config["max_evaluation_records"]) if config.get("max_evaluation_records") else None,
            sample_strategy=str(config.get("evaluation_sample_strategy", "first")),
        )
        c3_rows = _predict_manifest(
            project_path(config["evaluation_manifest"]),
            recognizers["c3"],
            scenario,
            seed=int(config.get("seed", 42)) + 2003 * (index + 1),
            max_records=int(config["max_evaluation_records"]) if config.get("max_evaluation_records") else None,
            sample_strategy=str(config.get("evaluation_sample_strategy", "first")),
        )
        by_scenario[scenario.name] = {
            "c1t_direct": _direct_metrics(c1t_rows, action_costs),
            "c3_direct": _direct_metrics(c3_rows, action_costs),
            "c3_c2_default": _context_metrics(c3_rows, c2_policy, config, action_costs),
            "c4_balanced": _action_safe_metrics(c3_rows, c4_balanced_policy, config, action_costs),
            "c4_safety": _action_safe_metrics(c3_rows, c4_safety_policy, config, action_costs),
        }
    summary = _summarize_by_method(by_scenario)
    return {
        "manifest": config["evaluation_manifest"],
        "scenarios": [asdict(item) for item in scenarios],
        "by_scenario": by_scenario,
        "summary": summary,
        "bootstrap_ci": _bootstrap_summary_ci(
            by_scenario,
            metrics=[
                "action_precision",
                "action_recall",
                "unintended_action_rate",
                "weighted_action_precision",
                "weighted_action_recall",
                "false_action_cost_rate",
                "missed_action_cost_rate",
            ],
            seed=int(config.get("seed", 42)) + 9091,
            iterations=int(config.get("bootstrap_iterations", 1000)),
        ),
    }


def _predict_manifest(
    manifest_path: Path,
    recognizer: Recognizer,
    scenario: PerturbationConfig,
    *,
    seed: int,
    max_records: int | None,
    sample_strategy: str = "first",
) -> list[tuple[str, Prediction]]:
    records = read_jsonl(manifest_path)
    records = _sample_records(records, max_records=max_records, strategy=sample_strategy, seed=seed)
    base_dir = manifest_path.parent
    rng = np.random.default_rng(seed)
    output = []
    for record in records:
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        perturbed = perturb_tensor(tensor, scenario, rng)
        output.append((record.target_label, recognizer(perturbed)))
    return output


def _sample_records(records: list[Any], *, max_records: int | None, strategy: str, seed: int) -> list[Any]:
    if max_records is None or max_records >= len(records):
        return records
    if strategy == "first":
        return records[:max_records]
    rng = np.random.default_rng(seed)
    if strategy == "random":
        indexes = rng.choice(len(records), size=max_records, replace=False)
        return [records[int(index)] for index in sorted(indexes)]
    if strategy != "stratified":
        raise ValueError(f"Unknown sample strategy '{strategy}'.")

    by_label: dict[str, list[Any]] = {}
    for record in records:
        by_label.setdefault(record.target_label, []).append(record)
    labels = sorted(by_label)
    quotas = {label: int(max_records * len(items) / len(records)) for label, items in by_label.items()}
    for label in labels:
        quotas[label] = max(1, min(len(by_label[label]), quotas[label]))
    while sum(quotas.values()) < max_records:
        label = max(labels, key=lambda item: len(by_label[item]) - quotas[item])
        if quotas[label] >= len(by_label[label]):
            break
        quotas[label] += 1
    while sum(quotas.values()) > max_records:
        label = max(labels, key=lambda item: quotas[item])
        if quotas[label] <= 1:
            break
        quotas[label] -= 1

    selected = []
    for label in labels:
        items = by_label[label]
        indexes = rng.choice(len(items), size=quotas[label], replace=False)
        selected.extend(items[int(index)] for index in indexes)
    return sorted(selected, key=lambda record: record.sample_id)


def _direct_metrics(rows: list[tuple[str, Prediction]], action_costs: dict[str, float]) -> dict[str, float | int]:
    return {**_direct_policy_metrics(rows), **direct_risk_metrics(rows, action_costs)}


def _context_metrics(
    rows: list[tuple[str, Prediction]],
    policy: ContextPolicyConfig,
    config: dict[str, Any],
    action_costs: dict[str, float],
) -> dict[str, float | int]:
    frames = _rows_to_replay_frames(
        rows,
        frames_per_clip=int(config.get("policy_frames_per_clip", 3)),
        frame_step_ms=int(config.get("policy_frame_step_ms", 100)),
        separator_ms=int(config.get("policy_separator_ms", 220)),
    )
    events = replay_predictions(frames, policy)
    return {**compute_interaction_metrics(frames, events), **event_risk_metrics(frames, events, action_costs)}


def _action_safe_metrics(
    rows: list[tuple[str, Prediction]],
    policy: ActionSafePolicyConfig,
    config: dict[str, Any],
    action_costs: dict[str, float],
) -> dict[str, float | int]:
    frames, events = _action_safe_replay(
        rows,
        policy,
        frames_per_clip=int(config.get("policy_frames_per_clip", 3)),
        frame_step_ms=int(config.get("policy_frame_step_ms", 100)),
        separator_ms=int(config.get("policy_separator_ms", 220)),
    )
    return {**compute_interaction_metrics(frames, events), **event_risk_metrics(frames, events, action_costs)}


def _action_safe_replay(
    rows: list[tuple[str, Prediction]],
    policy_config: ActionSafePolicyConfig,
    *,
    frames_per_clip: int,
    frame_step_ms: int,
    separator_ms: int,
):
    from research_pipeline.interaction.fsm import ACTION_BY_LABEL

    policy = ActionSafePolicy(policy_config)
    frames: list[ReplayFrame] = []
    events = []
    timestamp = 0
    no_gesture_prediction = prediction_from_scores({"no_gesture": 1.0})
    for true_label, prediction in rows:
        expected_action = ACTION_BY_LABEL.get(true_label, "")
        for index in range(frames_per_clip):
            frames.append(
                ReplayFrame(
                    timestamp_ms=timestamp,
                    label=prediction.label,
                    confidence=prediction.confidence,
                    expected_action=expected_action if index == 0 else "",
                )
            )
            event = policy.update(prediction, timestamp)
            if event is not None:
                events.append(event)
            timestamp += frame_step_ms
        frames.append(ReplayFrame(timestamp_ms=timestamp, label="no_gesture", confidence=1.0))
        policy.update(no_gesture_prediction, timestamp)
        timestamp += separator_ms
    return frames, events


def _candidate_policy_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("candidates"):
        return [_normalize_policy_config(item) for item in config["candidates"]]
    grid = config.get("policy_grid", {})
    keys = list(grid)
    values = [grid[key] for key in keys]
    candidates = []
    for combo in itertools.product(*values):
        raw = {key: value for key, value in zip(keys, combo)}
        candidates.append(_normalize_policy_config(raw))
    return candidates or [asdict(ActionSafePolicyConfig())]


def _normalize_policy_config(raw: dict[str, Any]) -> dict[str, Any]:
    default_threshold = float(raw.get("default_threshold", 0.75))
    high_risk_delta = float(raw.get("high_risk_delta", 0.0))
    pointer_delta = float(raw.get("pointer_delta", 0.0))
    default_stable = int(raw.get("default_stable_frames", 2))
    high_risk_stable = int(raw.get("high_risk_stable_frames", default_stable))
    label_thresholds = {
        label: float(value)
        for label, value in raw.get("label_thresholds", {}).items()
        if label in TARGET_LABELS
    }
    if pointer_delta:
        label_thresholds["point_2f"] = max(0.0, min(0.99, default_threshold + pointer_delta))
    if high_risk_delta:
        for label in ("click_2f", "swipe_left", "swipe_right", "zoom_in", "zoom_out"):
            label_thresholds[label] = max(0.0, min(0.99, default_threshold + high_risk_delta))
    label_stable = {
        label: int(value)
        for label, value in raw.get("label_stable_frames", {}).items()
        if label in TARGET_LABELS
    }
    if high_risk_stable != default_stable:
        for label in ("click_2f", "swipe_left", "swipe_right", "zoom_in", "zoom_out"):
            label_stable[label] = high_risk_stable
    return {
        "default_threshold": default_threshold,
        "label_thresholds": label_thresholds,
        "default_stable_frames": default_stable,
        "label_stable_frames": label_stable,
        "cooldown_ms": int(raw.get("cooldown_ms", 200)),
        "no_gesture_reset_frames": int(raw.get("no_gesture_reset_frames", 3)),
        "min_score_margin": float(raw.get("min_score_margin", 0.0)),
    }


def _summarize_by_method(by_scenario: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    methods = sorted({method for scenario in by_scenario.values() for method in scenario})
    for method in methods:
        metrics = [scenario[method] for scenario in by_scenario.values() if method in scenario]
        output[method] = _summarize_policy_metrics(metrics)
    return output


def _summarize_policy_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "action_precision",
        "action_recall",
        "unintended_action_rate",
        "false_trigger_rate_per_minute",
        "num_events",
        "num_expected_actions",
        "weighted_action_precision",
        "weighted_action_recall",
        "false_action_cost_rate",
        "missed_action_cost_rate",
        "false_action_cost_total",
        "missed_action_cost_total",
        "expected_action_cost_total",
    ]
    summary = {}
    for key in keys:
        values = [float(item.get(key, 0.0)) for item in metrics]
        summary[f"{key}_mean"] = sum(values) / len(values) if values else 0.0
    return summary


def _score_policy_candidate(
    summary: dict[str, float],
    *,
    precision_weight: float,
    recall_weight: float,
    unintended_penalty: float,
    false_rate_penalty: float,
    min_recall: float,
    recall_floor_penalty: float,
) -> float:
    recall = summary.get("action_recall_mean", 0.0)
    recall_gap = max(0.0, min_recall - recall)
    return (
        precision_weight * summary.get("action_precision_mean", 0.0)
        + recall_weight * recall
        - unintended_penalty * summary.get("unintended_action_rate_mean", 0.0)
        - false_rate_penalty * summary.get("false_trigger_rate_per_minute_mean", 0.0)
        - float(summary.get("false_action_cost_rate_mean", 0.0)) * 0.05
        - recall_floor_penalty * recall_gap
    )


def _score_safety_candidate(summary: dict[str, float], *, min_recall: float) -> float:
    recall = summary.get("action_recall_mean", 0.0)
    if recall < min_recall:
        return -100.0 + recall
    return (
        1.2 * summary.get("action_precision_mean", 0.0)
        + 0.25 * recall
        - 2.0 * summary.get("unintended_action_rate_mean", 0.0)
        - 0.02 * summary.get("false_trigger_rate_per_minute_mean", 0.0)
        - 0.2 * summary.get("false_action_cost_rate_mean", 0.0)
    )


def _mean_label_threshold(config: dict[str, Any]) -> float:
    thresholds = [float(config.get("default_threshold", 0.0))]
    thresholds.extend(float(value) for value in config.get("label_thresholds", {}).values())
    return sum(thresholds) / len(thresholds)


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


def _bootstrap_summary_ci(
    by_scenario: dict[str, dict[str, dict]],
    *,
    metrics: list[str],
    seed: int,
    iterations: int,
) -> dict[str, dict[str, dict[str, float]]]:
    scenario_names = list(by_scenario)
    methods = sorted({method for scenario in by_scenario.values() for method in scenario})
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, dict[str, float]]] = {method: {} for method in methods}
    if not scenario_names:
        return output

    for method in methods:
        for metric in metrics:
            values = []
            for _ in range(max(1, iterations)):
                sample = rng.choice(scenario_names, size=len(scenario_names), replace=True)
                sampled_values = [
                    float(by_scenario[scenario][method].get(metric, 0.0))
                    for scenario in sample
                    if method in by_scenario[scenario]
                ]
                values.append(sum(sampled_values) / len(sampled_values) if sampled_values else 0.0)
            ordered = sorted(values)
            output[method][metric] = {
                "p2_5": _percentile_sorted(ordered, 2.5),
                "mean": sum(values) / len(values),
                "p97_5": _percentile_sorted(ordered, 97.5),
            }
    return output


def _percentile_sorted(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = int(round((percentile / 100.0) * (len(values) - 1)))
    return values[min(len(values) - 1, max(0, index))]


if __name__ == "__main__":
    main()
