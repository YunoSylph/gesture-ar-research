from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np

from research_pipeline.cli.common import load_yaml, project_path
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import ManifestRecord, resolve_path
from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.features.preprocessing import resample_landmarks
from research_pipeline.evaluation.action_risk import normalize_action_costs
from research_pipeline.evaluation.online_gesture import (
    EVENT_FIELDNAMES,
    OnlineEvent,
    ProposalDecision,
    compute_online_metrics,
    top2_margin,
    write_events_csv,
    write_events_jsonl,
    write_summary_figure_svg,
    write_summary_markdown,
)
from research_pipeline.interaction.gesture_validation import (
    GestureValidationConfig,
    GestureValidationLayer,
    GestureValidationResult,
    config_from_mapping as validation_config_from_mapping,
)
from research_pipeline.evaluation.task_replay import (
    LABEL_BY_ACTION,
    TaskScenario,
    default_task_scenarios,
    evaluate_task_set,
)
from research_pipeline.interaction.fsm import ACTION_BY_LABEL
from research_pipeline.interaction.stabilizer import TemporalLabelStabilizer, TemporalStabilizerConfig
from research_pipeline.evaluation.statistics import PairedComparison, paired_comparison
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.rule_based import RuleBasedRecognizer


COMPARISON_METHODS = [
    "direct_c6",
    "c6_smoothing",
    "c6_temporal_stabilized",
    "c6_validation_confidence_only",
    "c6_validation_confidence_stability",
    "c6_validation_confidence_stability_cooldown",
    "c6_validation_tarc",
    "c6_validation_tarc_release",
    "landmark_controller",
    "landmark_controller_tarc",
]


class Predictor(Protocol):
    name: str

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        ...


@dataclass(slots=True)
class BuiltSequence:
    sequence_id: str
    tensor: LandmarkTensor
    labels: list[str]
    expected_labels: list[str]
    task_steps: list[str]
    segment_sources: list[dict[str, Any]]
    data_mode: str
    task_id: str = ""


class RuleBasedPredictor:
    name = "rule_based"

    def __init__(self) -> None:
        self.recognizer = RuleBasedRecognizer()

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        return self.recognizer.predict(tensor)


class ArtifactPredictor:
    def __init__(self, path: Path) -> None:
        from research_pipeline.models.artifacts import load_artifact
        from research_pipeline.models.hybrid import CachedArtifactPredictor

        self.path = path
        self.name = f"artifact:{path.as_posix()}"
        self.predictor = CachedArtifactPredictor(load_artifact(path))

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        return self.predictor.predict(tensor)


class C6Predictor:
    def __init__(self, payload: dict[str, Any]) -> None:
        from research_pipeline.models.c6_ensemble import C6EnsembleRecognizer, c6_config_from_mapping

        resolved = dict(payload)
        resolved["model_paths"] = [str(project_path(path)) for path in payload.get("model_paths", [])]
        self.name = "c6_ensemble"
        self.predictor = C6EnsembleRecognizer(c6_config_from_mapping(resolved))

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        return self.predictor.predict(tensor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark online/pseudo-continuous gesture-to-action methods.")
    parser.add_argument("--config", required=True, help="Path to configs/eval/online_gesture.yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory for events.csv, summary.json, comparison tables, figures/.")
    args = parser.parse_args()

    run_online_benchmark(project_path(args.config), project_path(args.output_dir))


def run_online_benchmark(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    manifest_path = project_path(config["manifest"])
    records = read_jsonl(manifest_path)
    action_costs = _load_action_costs(config)
    scenarios = _scenario_subset(default_task_scenarios(action_costs), config)
    availability = _analyze_availability(records, manifest_path, config)
    limitations: list[str] = []

    if not availability["continuous_timeline_available"]:
        limitations.append(str(availability["continuous_timeline_reason"]))
    limitations.append(
        "OO-dMVMT is used as methodological direction for online classification/segmentation metrics; "
        "this benchmark does not compare against OO-dMVMT numeric results."
    )

    predictor, predictor_limitations = _build_predictor(config)
    limitations.extend(predictor_limitations)
    if predictor.name != "c6_ensemble":
        limitations.append(
            f"C6 artifacts are unavailable in this run; C6-named baselines use effective predictor '{predictor.name}'."
        )

    sequences, sequence_limitations = _build_sequences(records, manifest_path, config, scenarios)
    limitations.extend(sequence_limitations)
    if not sequences:
        raise SystemExit("No pseudo-continuous sequences could be built from the manifest.")

    data_mode = _combine_data_modes(sequence.data_mode for sequence in sequences)
    if data_mode == "real_landmark_tensors":
        limitations.append(
            "Gesture and idle segments use real extracted MediaPipe landmarks from IPN Hand clips; "
            "the only synthetic element is the concatenation order (pseudo-continuous replay)."
        )
    elif "real" in data_mode:
        limitations.append(
            "Gesture clips use real extracted landmarks, but some idle gaps or clips fell back to synthetic "
            "landmarks; treat the affected segments accordingly."
        )
    else:
        limitations.append(
            "This run used synthetic fallback landmarks because processed tensors were not available; "
            "do not interpret recognition metrics as public benchmark results."
        )

    validation_config = _validation_config(config, action_costs)
    raw_predictions = _precompute_predictions(sequences, predictor, config)
    method_results = _run_method_comparison(sequences, raw_predictions, validation_config, config, scenarios, limitations)
    all_events = [event for result in method_results.values() for event in result["events"]]
    comparison_rows = [
        _comparison_row(method, predictor.name, method_results[method]["evaluation"], method_results[method]["task_replay"])
        for method in COMPARISON_METHODS
        if method in method_results
    ]
    statistical_comparison = _statistical_comparison(method_results)
    paired_units = len(method_results.get("direct_c6", {}).get("task_replay", {}).get("tasks", []))
    if 0 < paired_units < 5:
        limitations.append(
            f"Statistical comparison has only {paired_units} paired sequence(s); raise task_replay.trials_per_task "
            "so the confidence intervals and McNemar p-values become meaningful."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    events_csv = output_dir / "events.csv"
    events_jsonl = output_dir / "events.jsonl"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    comparison_csv = output_dir / "method_comparison.csv"
    comparison_md = output_dir / "method_comparison.md"
    figure_svg = figures_dir / "summary_metrics.svg"

    proposed = method_results.get("c6_validation_tarc", next(iter(method_results.values())))
    summary = {
        "mode": str(config.get("mode", "pseudo_continuous")),
        "data_mode": data_mode,
        "config": str(config_path),
        "manifest": str(manifest_path),
        "predictor": predictor.name,
        "availability": availability,
        "task_scenarios": {key: scenario.to_dict() for key, scenario in scenarios.items()},
        "sequence_count": len(sequences),
        "sequence_ids": [sequence.sequence_id for sequence in sequences],
        "event_log_schema": EVENT_FIELDNAMES,
        "label_counts": dict(Counter(event.ground_truth_label for event in all_events)),
        "evaluation": proposed["evaluation"],
        "method_comparison": comparison_rows,
        "statistical_comparison": statistical_comparison,
        "method_details": {
            method: {
                "evaluation": result["evaluation"],
                "task_replay": result["task_replay"],
            }
            for method, result in method_results.items()
        },
        "limitations": _dedupe(limitations),
        "outputs": {
            "events_csv": str(events_csv),
            "events_jsonl": str(events_jsonl),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "method_comparison_csv": str(comparison_csv),
            "method_comparison_md": str(comparison_md),
            "figures_dir": str(figures_dir),
            "summary_figure": str(figure_svg),
        },
    }

    write_events_csv(events_csv, all_events)
    write_events_jsonl(events_jsonl, all_events)
    _write_comparison_csv(comparison_csv, comparison_rows)
    _write_comparison_markdown(comparison_md, comparison_rows, summary["limitations"], statistical_comparison)
    with summary_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_summary_markdown(summary_md, summary)
    write_summary_figure_svg(figure_svg, proposed["evaluation"].get("metrics", {}))
    (figures_dir / "README.md").write_text(
        "Generated by benchmark_online_gesture.py. The SVG is dependency-free and intended as a quick metric preview.\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "online_gesture_methods={methods} frames={frames} sequences={seq} data_mode={mode} comparison={comparison}".format(
            methods=len(method_results),
            frames=len(all_events),
            seq=len(sequences),
            mode=data_mode,
            comparison=comparison_csv,
        )
    )
    return summary


def _load_action_costs(config: dict[str, Any]) -> dict[str, float]:
    path = config.get("risk_costs_path")
    if path:
        return normalize_action_costs(load_yaml(project_path(path)))
    return normalize_action_costs(config.get("action_costs"))


def _scenario_subset(scenarios: dict[str, TaskScenario], config: dict[str, Any]) -> dict[str, TaskScenario]:
    requested = config.get("task_replay", {}).get("tasks")
    if not requested:
        return scenarios
    return {task_id: scenarios[task_id] for task_id in requested if task_id in scenarios}


def _analyze_availability(records: list[ManifestRecord], manifest_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    session_ids = {record.session_id for record in records}
    labels = Counter(record.target_label for record in records)
    timestamped = [record for record in records if record.clip_end_ms >= record.clip_start_ms]
    raw_videos_found = sum(1 for record in records if record.raw_video_path and project_path(record.raw_video_path).exists())
    tensor_paths = [resolve_path(record.tensor_path, manifest_path.parent) for record in records if record.tensor_path]
    tensor_files_found = sum(1 for path in tensor_paths if path.exists())
    predictor_config = config.get("predictor", {})
    model_paths = []
    if predictor_config.get("kind") == "c6_ensemble":
        model_paths.extend(project_path(path) for path in predictor_config.get("c6", {}).get("model_paths", []))
    elif predictor_config.get("model_path"):
        model_paths.append(project_path(predictor_config["model_path"]))
    model_artifacts_found = sum(1 for path in model_paths if path.exists())

    has_annotation_order = len(timestamped) == len(records) and bool(records)
    tensors_complete = bool(tensor_paths) and tensor_files_found == len(tensor_paths)
    if tensors_complete:
        reason = (
            "Real extracted landmark tensors are available for every manifest clip, but the dataset stores "
            "segmented clips rather than the original uncut IPN recordings. The evaluator therefore builds an "
            "explicitly marked pseudo-continuous stream by concatenating real gesture clips with real no_gesture "
            "idle gaps; this is real-clip replay, not the original continuous IPN timeline."
        )
    else:
        reason = (
            "Full continuous IPN timeline is not available: the manifest stores selected/remapped clips with "
            "timestamps, and some processed tensors are missing, so the evaluator runs in explicitly marked "
            "pseudo-continuous mode with synthetic fallback for the missing clips."
        )
    return {
        "manifest_records": len(records),
        "manifest_path": str(manifest_path),
        "sessions": len(session_ids),
        "label_counts": dict(labels),
        "manifest_has_clip_timestamps": has_annotation_order,
        "ipn_annotation_order_recoverable": has_annotation_order,
        "continuous_timeline_available": False,
        "continuous_timeline_reason": reason,
        "raw_videos_found": raw_videos_found,
        "tensor_paths_declared": len(tensor_paths),
        "tensor_files_found": tensor_files_found,
        "model_paths_declared": len(model_paths),
        "model_artifacts_found": model_artifacts_found,
    }


def _build_predictor(config: dict[str, Any]) -> tuple[Predictor, list[str]]:
    predictor_config = config.get("predictor", {})
    kind = str(predictor_config.get("kind", "artifact"))
    fallback = str(predictor_config.get("fallback", "rule_based"))
    limitations: list[str] = []

    try:
        if kind == "artifact":
            path = project_path(predictor_config.get("model_path", ""))
            if path.exists():
                return ArtifactPredictor(path), limitations
            limitations.append(f"Configured model artifact was not found: {path}.")
        elif kind == "c6_ensemble":
            c6_payload = predictor_config.get("c6", {})
            paths = [project_path(path) for path in c6_payload.get("model_paths", [])]
            missing = [path for path in paths if not path.exists()]
            if paths and not missing:
                return C6Predictor(c6_payload), limitations
            limitations.append(f"Configured C6 model artifacts were not found: {[str(path) for path in missing]}.")
        elif kind == "rule_based":
            return RuleBasedPredictor(), limitations
        else:
            limitations.append(f"Unsupported predictor kind '{kind}', falling back to {fallback}.")
    except Exception as exc:  # pragma: no cover - defensive fallback for optional ML dependencies
        limitations.append(f"Failed to initialize predictor '{kind}': {exc}.")

    if fallback == "rule_based":
        limitations.append("Using rule_based predictor fallback for evaluator smoke run.")
        return RuleBasedPredictor(), limitations
    raise SystemExit(f"Cannot initialize predictor kind '{kind}' and fallback '{fallback}' is not supported.")


def _validation_config(config: dict[str, Any], action_costs: dict[str, float]) -> GestureValidationConfig:
    raw = dict(config.get("gesture_validation", config.get("proposal_controller", {})))
    if "label_thresholds" in raw and "confidence_thresholds" not in raw:
        raw["confidence_thresholds"] = raw["label_thresholds"]
    if "expected_threshold_delta" in raw and "expected_confidence_delta" not in raw:
        raw["expected_confidence_delta"] = raw["expected_threshold_delta"]
    if "unexpected_threshold_delta" in raw and "unexpected_confidence_delta" not in raw:
        raw["unexpected_confidence_delta"] = raw["unexpected_threshold_delta"]
    if "default_threshold" in raw and "default_confidence_threshold" not in raw:
        raw["default_confidence_threshold"] = raw["default_threshold"]
    if "label_stable_frames" in raw and "stable_frames" not in raw:
        raw["stable_frames"] = raw["label_stable_frames"]
    config_obj = validation_config_from_mapping(raw)
    for label, rule in config_obj.contract.items():
        if rule.action in action_costs:
            rule.risk_cost = float(action_costs[rule.action])
    return config_obj


def _build_sequences(
    records: list[ManifestRecord],
    manifest_path: Path,
    config: dict[str, Any],
    scenarios: dict[str, TaskScenario],
) -> tuple[list[BuiltSequence], list[str]]:
    if bool(config.get("task_replay", {}).get("enabled", True)):
        return _build_task_sequences(records, manifest_path, config, scenarios)
    return _build_manifest_order_sequences(records, manifest_path, config)


def _build_task_sequences(
    records: list[ManifestRecord],
    manifest_path: Path,
    config: dict[str, Any],
    scenarios: dict[str, TaskScenario],
) -> tuple[list[BuiltSequence], list[str]]:
    replay = config.get("pseudo_continuous", {})
    task_config = config.get("task_replay", {})
    idle_gap_ms = int(replay.get("idle_gap_ms", 600))
    frame_step_ms = int(replay.get("frame_step_ms", 33))
    max_frames_per_clip = int(replay.get("max_frames_per_clip", 48))
    synthetic_fallback = bool(replay.get("synthetic_fallback", True))
    trials_per_task = int(task_config.get("trials_per_task", 1))
    records_by_label: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        records_by_label[record.target_label].append(record)

    limitations: list[str] = []
    sequences: list[BuiltSequence] = []
    missing_tensor_count = 0
    counters: Counter[str] = Counter()
    for task_id, scenario in scenarios.items():
        for trial in range(trials_per_task):
            tensors: list[LandmarkTensor] = []
            labels: list[str] = []
            expected_labels: list[str] = []
            task_steps: list[str] = []
            segment_sources: list[dict[str, Any]] = []
            data_modes: set[str] = set()

            for step_index, step in enumerate(scenario.expected_steps):
                if tensors:
                    idle_frames = max(2, int(round(idle_gap_ms / max(1, frame_step_ms))))
                    idle_tensor, idle_mode = _idle_tensor(
                        records_by_label.get("no_gesture", []),
                        manifest_path,
                        counters=counters,
                        idle_frames=idle_frames,
                        max_frames_per_clip=max_frames_per_clip,
                        seed=200_000 + trial * 1000 + step_index,
                    )
                    tensors.append(idle_tensor)
                    labels.extend(["no_gesture"] * idle_frames)
                    expected_labels.extend([""] * idle_frames)
                    task_steps.extend([""] * idle_frames)
                    segment_sources.append({"kind": "inserted_idle", "label": "no_gesture", "frames": idle_frames, "data_mode": idle_mode})
                    data_modes.add(idle_mode)

                label = step.expected_label
                candidates = records_by_label.get(label, [])
                if candidates:
                    record = candidates[counters[label] % len(candidates)]
                    counters[label] += 1
                    tensor, data_mode, missing = _tensor_for_record(
                        record,
                        manifest_path,
                        synthetic_fallback=synthetic_fallback,
                        max_frames_per_clip=max_frames_per_clip,
                    )
                    missing_tensor_count += int(missing)
                    source = {
                        "kind": "manifest_clip",
                        "label": label,
                        "sample_id": record.sample_id,
                        "session_id": record.session_id,
                        "clip_start_ms": record.clip_start_ms,
                        "clip_end_ms": record.clip_end_ms,
                        "frames": int(tensor.landmarks.shape[0]),
                        "data_mode": data_mode,
                        "task_step": step.id,
                    }
                else:
                    frames = max(8, min(max_frames_per_clip, 32))
                    tensor = synthetic_landmarks(label, length=frames, seed=300_000 + trial * 1000 + step_index, sample_id=f"{task_id}:{step.id}")
                    data_mode = "synthetic_fallback_from_task_label"
                    source = {"kind": "synthetic_task_clip", "label": label, "frames": frames, "task_step": step.id}
                    data_modes.add(data_mode)
                tensors.append(tensor)
                labels.extend([label] * tensor.landmarks.shape[0])
                expected_labels.extend([label] * tensor.landmarks.shape[0])
                task_steps.extend([step.id] * tensor.landmarks.shape[0])
                segment_sources.append(source)
                data_modes.add(data_mode)

            sequence_id = f"task_{task_id}_{trial + 1:02d}"
            sequences.append(
                BuiltSequence(
                    sequence_id=sequence_id,
                    tensor=_concat_tensors(tensors),
                    labels=labels,
                    expected_labels=expected_labels,
                    task_steps=task_steps,
                    segment_sources=segment_sources,
                    data_mode=_combine_data_modes(data_modes),
                    task_id=task_id,
                )
            )

    if missing_tensor_count:
        limitations.append(
            f"{missing_tensor_count} task replay clips referenced missing tensor files and were replaced by synthetic landmarks."
        )
    return sequences, limitations


def _build_manifest_order_sequences(
    records: list[ManifestRecord],
    manifest_path: Path,
    config: dict[str, Any],
) -> tuple[list[BuiltSequence], list[str]]:
    replay = config.get("pseudo_continuous", {})
    max_sequences = int(replay.get("max_sequences", 4))
    max_clips_per_sequence = int(replay.get("max_clips_per_sequence", 12))
    idle_gap_ms = int(replay.get("idle_gap_ms", 600))
    frame_step_ms = int(replay.get("frame_step_ms", 33))
    max_frames_per_clip = int(replay.get("max_frames_per_clip", 48))
    synthetic_fallback = bool(replay.get("synthetic_fallback", True))
    insert_idle = bool(replay.get("insert_idle_between_clips", True))
    no_gesture_pool = [record for record in records if record.target_label == "no_gesture"]
    idle_counters: Counter[str] = Counter()

    by_session: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_session[record.session_id].append(record)

    limitations: list[str] = []
    sequences: list[BuiltSequence] = []
    missing_tensor_count = 0
    for session_index, (session_id, session_records) in enumerate(sorted(by_session.items())):
        if len(sequences) >= max_sequences:
            break
        ordered = sorted(session_records, key=lambda item: (item.clip_start_ms, item.clip_end_ms, item.sample_id))
        selected = ordered[:max_clips_per_sequence]
        tensors: list[LandmarkTensor] = []
        labels: list[str] = []
        expected_labels: list[str] = []
        task_steps: list[str] = []
        segment_sources: list[dict[str, Any]] = []
        data_modes: set[str] = set()
        for clip_index, record in enumerate(selected):
            if insert_idle and tensors:
                idle_frames = max(2, int(round(idle_gap_ms / max(1, frame_step_ms))))
                idle_tensor, idle_mode = _idle_tensor(
                    no_gesture_pool,
                    manifest_path,
                    counters=idle_counters,
                    idle_frames=idle_frames,
                    max_frames_per_clip=max_frames_per_clip,
                    seed=100_000 + session_index * 1000 + clip_index,
                )
                tensors.append(idle_tensor)
                labels.extend(["no_gesture"] * idle_frames)
                expected_labels.extend([""] * idle_frames)
                task_steps.extend([""] * idle_frames)
                segment_sources.append({"kind": "inserted_idle", "label": "no_gesture", "frames": idle_frames, "data_mode": idle_mode})
                data_modes.add(idle_mode)
            tensor, data_mode, missing = _tensor_for_record(
                record,
                manifest_path,
                synthetic_fallback=synthetic_fallback,
                max_frames_per_clip=max_frames_per_clip,
            )
            missing_tensor_count += int(missing)
            tensors.append(tensor)
            labels.extend([record.target_label] * tensor.landmarks.shape[0])
            expected_labels.extend([""] * tensor.landmarks.shape[0])
            task_steps.extend([""] * tensor.landmarks.shape[0])
            segment_sources.append({"kind": "manifest_clip", "label": record.target_label, "sample_id": record.sample_id})
            data_modes.add(data_mode)
        if tensors:
            sequences.append(
                BuiltSequence(
                    sequence_id=f"pseudo_{session_index + 1:03d}_{session_id}",
                    tensor=_concat_tensors(tensors),
                    labels=labels,
                    expected_labels=expected_labels,
                    task_steps=task_steps,
                    segment_sources=segment_sources,
                    data_mode=_combine_data_modes(data_modes),
                )
            )
    if missing_tensor_count:
        limitations.append(
            f"{missing_tensor_count} manifest clips referenced missing tensor files and were replaced by synthetic landmarks."
        )
    return sequences, limitations


def _idle_tensor(
    no_gesture_records: list[ManifestRecord],
    manifest_path: Path,
    *,
    counters: Counter[str],
    idle_frames: int,
    max_frames_per_clip: int,
    seed: int,
) -> tuple[LandmarkTensor, str]:
    """Idle gap between gesture clips.

    Prefer a real ``no_gesture`` clip (the manifest has hundreds), resampled to
    the gap length, so the replay stream is real end to end. Only fall back to
    synthetic landmarks when no real ``no_gesture`` tensor is available.
    """

    if no_gesture_records:
        record = no_gesture_records[counters["__idle__"] % len(no_gesture_records)]
        counters["__idle__"] += 1
        tensor, data_mode, _ = _tensor_for_record(
            record,
            manifest_path,
            synthetic_fallback=True,
            max_frames_per_clip=max_frames_per_clip,
        )
        if data_mode == "real_landmark_tensors":
            return resample_landmarks(tensor, target_length=idle_frames), "real_landmark_tensors"
    return synthetic_landmarks("no_gesture", length=idle_frames, seed=seed), "synthetic_inserted_idle"


def _tensor_for_record(
    record: ManifestRecord,
    manifest_path: Path,
    *,
    synthetic_fallback: bool,
    max_frames_per_clip: int,
) -> tuple[LandmarkTensor, str, bool]:
    if record.tensor_path:
        tensor_path = resolve_path(record.tensor_path, manifest_path.parent)
        if tensor_path.exists():
            return load_landmark_npz(tensor_path), "real_landmark_tensors", False
    if not synthetic_fallback:
        raise FileNotFoundError(f"Tensor is missing for record '{record.sample_id}'.")
    fps = record.fps or 30.0
    duration_ms = max(1, int(record.clip_end_ms) - int(record.clip_start_ms))
    frames = max(8, min(max_frames_per_clip, int(round(duration_ms * fps / 1000.0))))
    return (
        synthetic_landmarks(record.target_label, length=frames, seed=17, sample_id=record.sample_id),
        "synthetic_fallback_from_manifest_label",
        bool(record.tensor_path),
    )


def _concat_tensors(tensors: list[LandmarkTensor]) -> LandmarkTensor:
    landmarks = np.concatenate([tensor.landmarks for tensor in tensors], axis=0).astype(np.float32)
    sequence_mask = np.concatenate([tensor.sequence_mask for tensor in tensors], axis=0).astype(bool)
    confidence = np.concatenate([tensor.frame_confidence for tensor in tensors], axis=0).astype(np.float32)
    handedness = np.concatenate(
        [
            tensor.handedness_score
            if tensor.handedness_score.shape == (tensor.landmarks.shape[0],)
            else np.repeat(tensor.handedness_score.astype(np.float32), tensor.landmarks.shape[0])
            for tensor in tensors
        ],
        axis=0,
    ).astype(np.float32)
    world = None
    if all(tensor.world_landmarks is not None for tensor in tensors):
        world = np.concatenate([tensor.world_landmarks for tensor in tensors if tensor.world_landmarks is not None], axis=0).astype(np.float32)
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=sequence_mask,
        frame_confidence=confidence,
        handedness_score=handedness,
        coord_space=tensors[0].coord_space if tensors else "image_normalized_xyz",
        world_landmarks=world,
    )


def _precompute_predictions(
    sequences: list[BuiltSequence],
    predictor: Predictor,
    config: dict[str, Any],
) -> dict[tuple[str, int], Prediction]:
    replay = config.get("pseudo_continuous", {})
    window_size = int(replay.get("window_size", 32))
    output: dict[tuple[str, int], Prediction] = {}
    for sequence in sequences:
        for frame_index in range(len(sequence.labels)):
            output[(sequence.sequence_id, frame_index)] = predictor.predict(_window_tensor(sequence.tensor, frame_index, window_size))
    return output


def _run_method_comparison(
    sequences: list[BuiltSequence],
    raw_predictions: dict[tuple[str, int], Prediction],
    validation_config: GestureValidationConfig,
    config: dict[str, Any],
    scenarios: dict[str, TaskScenario],
    limitations: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for method in COMPARISON_METHODS:
        events = _run_method(method, sequences, raw_predictions, validation_config, config, limitations)
        results[method] = {
            "events": events,
            "evaluation": compute_online_metrics(
                events,
                min_segment_iou=float(config.get("metrics", {}).get("min_segment_iou", 0.1)),
                latency_grace_ms=int(config.get("metrics", {}).get("latency_grace_ms", 1000)),
            ),
            "task_replay": evaluate_task_set(
                events,
                scenarios,
                completion_threshold=float(config.get("task_replay", {}).get("completion_threshold", 0.5)),
            ),
        }
    return results


def _run_method(
    method: str,
    sequences: list[BuiltSequence],
    raw_predictions: dict[tuple[str, int], Prediction],
    validation_config: GestureValidationConfig,
    config: dict[str, Any],
    limitations: list[str],
) -> list[OnlineEvent]:
    replay = config.get("pseudo_continuous", {})
    frame_step_ms = int(replay.get("frame_step_ms", 33))
    window_size = int(replay.get("window_size", 32))
    smoothing_window = int(config.get("smoothing", {}).get("window", 5))
    stabilizer_cfg = config.get("stabilizer", {})
    events: list[OnlineEvent] = []
    live_controller_available = True
    live_controller_error = ""
    if method.startswith("landmark_controller"):
        live_controller_available, live_controller_error = _live_controller_available()
        if not live_controller_available:
            limitations.append(f"LiveLandmarkGestureController unavailable for {method}: {live_controller_error}.")

    for sequence in sequences:
        validation_layer = GestureValidationLayer(_config_for_method(validation_config, method))
        landmark_validation_layer = GestureValidationLayer(_config_for_method(validation_config, method))
        smoothing_history: deque[Prediction] = deque(maxlen=max(1, smoothing_window))
        stabilizer = TemporalLabelStabilizer(
            TemporalStabilizerConfig(
                window=int(stabilizer_cfg.get("window", 7)),
                enter_fraction=float(stabilizer_cfg.get("enter_fraction", 0.5)),
                min_confidence=float(stabilizer_cfg.get("min_confidence", 0.0)),
                sticky=bool(stabilizer_cfg.get("sticky", True)),
            )
        )
        live_controller = _new_live_controller() if live_controller_available and method.startswith("landmark_controller") else None
        for frame_index, ground_truth_label in enumerate(sequence.labels):
            timestamp_ms = frame_index * frame_step_ms
            raw = raw_predictions[(sequence.sequence_id, frame_index)]
            expected_label = sequence.expected_labels[frame_index] if "tarc" in method else ""
            prediction_for_decision = raw

            if method == "direct_c6":
                decision = _direct_decision(raw, controller_mode="direct_c6", action_costs=_action_costs(validation_config))
            elif method == "c6_smoothing":
                smoothing_history.append(raw)
                smoothed = _smooth_prediction(smoothing_history)
                prediction_for_decision = smoothed
                decision = _direct_decision(smoothed, controller_mode="c6_smoothing", action_costs=_action_costs(validation_config))
            elif method == "c6_temporal_stabilized":
                stabilized = stabilizer.update_prediction(raw)
                prediction_for_decision = stabilized
                decision = _direct_decision(stabilized, controller_mode="c6_temporal_stabilized", action_costs=_action_costs(validation_config))
            elif method in {
                "c6_validation_confidence_only",
                "c6_validation_confidence_stability",
                "c6_validation_confidence_stability_cooldown",
            }:
                validation = validation_layer.update_prediction(
                    raw,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    top2_margin_value=top2_margin(raw.scores),
                )
                decision = _decision_from_validation(validation, controller_mode=method, tarc=False)
            elif method in {"c6_validation_tarc", "c6_validation_tarc_release"}:
                validation = validation_layer.update_prediction(
                    raw,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    expected_label=expected_label,
                    top2_margin_value=top2_margin(raw.scores),
                )
                decision = _decision_from_validation(validation, controller_mode=method, tarc=True)
            elif method == "landmark_controller":
                landmark_prediction = _landmark_prediction(live_controller, raw, _window_tensor(sequence.tensor, frame_index, window_size), "")
                prediction_for_decision = landmark_prediction
                mode = "live_landmark_controller" if live_controller is not None else "landmark_unavailable_raw_fallback"
                decision = _direct_decision(landmark_prediction, controller_mode=mode, action_costs=_action_costs(validation_config))
            elif method == "landmark_controller_tarc":
                landmark_prediction = _landmark_prediction(
                    live_controller,
                    raw,
                    _window_tensor(sequence.tensor, frame_index, window_size),
                    expected_label,
                )
                prediction_for_decision = landmark_prediction
                validation = landmark_validation_layer.update_prediction(
                    landmark_prediction,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    expected_label=expected_label,
                    top2_margin_value=top2_margin(landmark_prediction.scores),
                )
                decision = _decision_from_validation(validation, controller_mode="landmark_controller_tarc", tarc=True)
                if live_controller is None:
                    decision.controller_mode = "landmark_unavailable_raw_fallback_tarc"
            else:  # pragma: no cover - COMPARISON_METHODS guards this path
                raise ValueError(f"Unknown method '{method}'.")

            events.append(
                OnlineEvent(
                    sequence_id=f"{method}:{sequence.sequence_id}",
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    ground_truth_label=ground_truth_label,
                    model_label=raw.label,
                    model_confidence=float(raw.confidence),
                    top2_margin=top2_margin(raw.scores),
                    proposal_label=prediction_for_decision.label if decision.proposal_label == "no_gesture" and method in {"direct_c6", "c6_smoothing", "c6_temporal_stabilized"} else decision.proposal_label,
                    proposal_state=decision.proposal_state,
                    controller_mode=decision.controller_mode,
                    expected_label=expected_label,
                    final_action=decision.final_action,
                    action_accepted=decision.action_accepted,
                    rejection_reason=decision.rejection_reason,
                    cooldown_remaining=decision.cooldown_remaining,
                    risk_cost=decision.risk_cost,
                    task_id=sequence.task_id,
                    task_step=sequence.task_steps[frame_index],
                )
            )
    return events


def _direct_decision(prediction: Prediction, *, controller_mode: str, action_costs: dict[str, float]) -> ProposalDecision:
    label = prediction.label if prediction.label in TARGET_LABELS else "no_gesture"
    action = ACTION_BY_LABEL.get(label, "idle")
    accepted = action != "idle"
    return ProposalDecision(
        proposal_label=label,
        proposal_state="accepted" if accepted else "idle",
        controller_mode=controller_mode,
        final_action=action if accepted else "idle",
        action_accepted=accepted,
        rejection_reason="" if accepted else "idle",
        cooldown_remaining=0,
        risk_cost=float(action_costs.get(action, 0.0)),
    )


def _decision_from_validation(validation: GestureValidationResult, *, controller_mode: str, tarc: bool) -> ProposalDecision:
    if not validation.is_ready_for_tarc:
        return ProposalDecision(
            proposal_label=validation.proposal_label,
            proposal_state=validation.proposal_state,
            controller_mode=controller_mode,
            final_action="idle",
            action_accepted=False,
            rejection_reason=validation.rejection_reason or "not_ready",
            cooldown_remaining=validation.cooldown_remaining,
            risk_cost=validation.risk_cost,
        )
    if tarc and validation.expected_label and validation.proposal_label != validation.expected_label:
        return ProposalDecision(
            proposal_label=validation.proposal_label,
            proposal_state="rejected",
            controller_mode=controller_mode,
            final_action="idle",
            action_accepted=False,
            rejection_reason="unexpected_for_tarc",
            cooldown_remaining=validation.cooldown_remaining,
            risk_cost=validation.risk_cost,
        )
    return ProposalDecision(
        proposal_label=validation.proposal_label,
        proposal_state=validation.proposal_state,
        controller_mode=controller_mode,
        final_action=validation.final_action,
        action_accepted=validation.final_action != "idle",
        rejection_reason="",
        cooldown_remaining=validation.cooldown_remaining,
        risk_cost=validation.risk_cost,
    )


def _config_for_method(base: GestureValidationConfig, method: str) -> GestureValidationConfig:
    config = GestureValidationConfig(
        confidence_thresholds=dict(base.confidence_thresholds),
        default_confidence_threshold=base.default_confidence_threshold,
        min_top2_margin=base.min_top2_margin,
        stable_frames=dict(base.stable_frames),
        default_stable_frames=base.default_stable_frames,
        cooldown_ms=base.cooldown_ms,
        lock_hold_ms=base.lock_hold_ms,
        expected_confidence_delta=base.expected_confidence_delta,
        unexpected_confidence_delta=base.unexpected_confidence_delta,
        use_confidence=base.use_confidence,
        use_stability=base.use_stability,
        use_cooldown=base.use_cooldown,
        require_release=base.require_release,
        require_global_release=base.require_global_release,
        contract=dict(base.contract),
    )
    if method == "c6_validation_confidence_only":
        config.use_stability = False
        config.use_cooldown = False
        config.require_release = False
        config.require_global_release = False
    elif method == "c6_validation_confidence_stability":
        config.use_cooldown = False
        config.require_release = False
        config.require_global_release = False
    elif method == "c6_validation_confidence_stability_cooldown":
        config.require_release = True
        config.require_global_release = False
    elif method == "c6_validation_tarc":
        config.require_global_release = False
    elif method == "c6_validation_tarc_release":
        config.require_release = True
        config.require_global_release = True
    return config


def _action_costs(config: GestureValidationConfig) -> dict[str, float]:
    return {rule.action: rule.risk_cost for rule in config.contract.values()}


def _smooth_prediction(history: Iterable[Prediction]) -> Prediction:
    scores = {label: 0.0 for label in TARGET_LABELS}
    total_weight = 0.0
    for index, prediction in enumerate(history, start=1):
        weight = index * max(0.05, float(prediction.confidence))
        total_weight += weight
        for label in TARGET_LABELS:
            scores[label] += float(prediction.scores.get(label, 0.0)) * weight
    if total_weight > 0:
        scores = {label: value / total_weight for label, value in scores.items()}
    return prediction_from_scores(scores)


def _live_controller_available() -> tuple[bool, str]:
    try:
        from research_pipeline.serve.live_backend import LiveLandmarkGestureController  # noqa: F401

        return True, ""
    except Exception as exc:
        return False, str(exc)


def _new_live_controller():
    from research_pipeline.serve.live_backend import LiveLandmarkGestureController

    return LiveLandmarkGestureController()


def _landmark_prediction(controller: Any, raw: Prediction, tensor: LandmarkTensor, expected_label: str) -> Prediction:
    if controller is None:
        return raw
    return controller.update(raw, tensor, expected_label=expected_label)


def _window_tensor(tensor: LandmarkTensor, frame_index: int, window_size: int) -> LandmarkTensor:
    start = max(0, frame_index - max(1, window_size) + 1)
    end = frame_index + 1
    world = tensor.world_landmarks[start:end] if tensor.world_landmarks is not None else None
    return LandmarkTensor(
        landmarks=tensor.landmarks[start:end],
        sequence_mask=tensor.sequence_mask[start:end],
        frame_confidence=tensor.frame_confidence[start:end],
        handedness_score=tensor.handedness_score[start:end],
        coord_space=tensor.coord_space,
        world_landmarks=world,
    )


def _comparison_row(
    method: str,
    predictor_name: str,
    evaluation: dict[str, Any],
    task_replay: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluation.get("metrics", {})
    task = task_replay.get("summary", {})
    return {
        "method": method,
        "effective_predictor": predictor_name,
        "recognition_accuracy": _round(metrics.get("recognition_accuracy")),
        "macro_f1": _round(metrics.get("macro_f1")),
        "segment_f1": _round(metrics.get("segment_f1")),
        "false_positives_per_minute": _round(metrics.get("false_positives_per_minute")),
        "label_switch_rate": _round(metrics.get("label_switch_rate_per_minute")),
        "decision_latency_ms": _round(metrics.get("decision_latency_ms_mean")),
        "accepted_actions": int(task.get("accepted_actions", metrics.get("accepted_action_count", 0)) or 0),
        "rejected_actions": int(task.get("rejected_actions", metrics.get("rejected_action_count", 0)) or 0),
        "false_action_cost": _round(task.get("false_action_cost")),
        "missed_action_cost": _round(task.get("missed_action_cost")),
        "action_precision": _round(task.get("weighted_action_precision")),
        "action_recall": _round(task.get("weighted_action_recall")),
        "task_completion": _round(task.get("task_completion_score")),
        "confident_completion": _round(task.get("confident_completion_rate")),
        "task_success": _round(task.get("task_success_rate")),
    }


def _statistical_comparison(
    method_results: dict[str, dict[str, Any]],
    *,
    baseline_method: str = "direct_c6",
    metrics: tuple[tuple[str, bool], ...] = (
        ("false_action_cost", True),
        ("false_actions", True),
        ("task_completion_score", False),
    ),
) -> list[dict[str, Any]]:
    """Paired comparison of each method against the direct baseline.

    Units are the per-(sequence, task) rows produced by ``evaluate_task_set``;
    because every method replays the identical sequences in the same order, the
    rows align by index and the difference is paired. Each metric carries its own
    direction: cost-like metrics use ``lower_is_better`` (a negative ``delta`` with
    a CI fully below zero is the "pipeline reduces false AR actions" result), while
    ``task_completion_score`` uses higher-is-better (a positive ``delta`` with a CI
    fully above zero is the "pipeline completes tasks more confidently" result).
    """

    if baseline_method not in method_results:
        return []
    baseline_rows = method_results[baseline_method]["task_replay"].get("tasks", [])
    if not baseline_rows:
        return []
    rows: list[dict[str, Any]] = []
    for method, result in method_results.items():
        if method == baseline_method:
            continue
        method_rows = result["task_replay"].get("tasks", [])
        if len(method_rows) != len(baseline_rows):
            continue
        for metric, lower_is_better in metrics:
            base_vec = [float(row.get(metric, 0.0)) for row in baseline_rows]
            method_vec = [float(row.get(metric, 0.0)) for row in method_rows]
            comparison: PairedComparison = paired_comparison(base_vec, method_vec, lower_is_better=lower_is_better)
            rows.append({"method": method, "baseline": baseline_method, "metric": metric, **comparison.to_dict()})
    return rows


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "effective_predictor",
        "recognition_accuracy",
        "macro_f1",
        "segment_f1",
        "false_positives_per_minute",
        "label_switch_rate",
        "decision_latency_ms",
        "accepted_actions",
        "rejected_actions",
        "false_action_cost",
        "missed_action_cost",
        "action_precision",
        "action_recall",
        "task_completion",
        "confident_completion",
        "task_success",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    limitations: list[str],
    statistical_comparison: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "effective_predictor",
        "recognition_accuracy",
        "macro_f1",
        "segment_f1",
        "false_positives_per_minute",
        "label_switch_rate",
        "decision_latency_ms",
        "accepted_actions",
        "rejected_actions",
        "false_action_cost",
        "missed_action_cost",
        "action_precision",
        "action_recall",
        "task_completion",
        "confident_completion",
        "task_success",
    ]
    lines = ["# Online Gesture Method Comparison", ""]
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    if statistical_comparison:
        stat_fields = ["method", "metric", "baseline_mean", "method_mean", "delta", "delta_ci_low", "delta_ci_high", "prob_improvement", "p_value", "n"]
        lines.extend(
            [
                "",
                "## Paired Comparison vs direct_c6 (lower is better)",
                "",
                "Per-(sequence, task) paired bootstrap. `delta` = method - baseline; a `delta_ci_high` below 0 means the reduction is significant at the chosen level. `p_value` is the exact McNemar test.",
                "",
                "| " + " | ".join(stat_fields) + " |",
                "| " + " | ".join("---" for _ in stat_fields) + " |",
            ]
        )
        for row in statistical_comparison:
            lines.append("| " + " | ".join(_format_stat(row.get(field)) for field in stat_fields) + " |")
    if limitations:
        lines.extend(["", "## Limitations", ""])
        for item in _dedupe(limitations):
            lines.append(f"- {item}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def _combine_data_modes(values: Any) -> str:
    modes = set(values)
    if modes == {"real_landmark_tensors"}:
        return "real_landmark_tensors"
    if "real_landmark_tensors" in modes:
        return "mixed_real_and_synthetic_fallback"
    if "synthetic_inserted_idle" in modes and len(modes) == 1:
        return "synthetic_inserted_idle"
    return "synthetic_fallback_pseudo_continuous"


def _round(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _format_stat(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


if __name__ == "__main__":
    main()
