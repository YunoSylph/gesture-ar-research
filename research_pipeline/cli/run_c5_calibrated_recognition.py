from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import ManifestRecord, resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.evaluation.calibration import compute_calibration_report
from research_pipeline.evaluation.error_analysis import analyze_recognition_risk
from research_pipeline.evaluation.metrics import RecognitionMetrics, compute_recognition_metrics
from research_pipeline.evaluation.robustness import PerturbationConfig, perturb_tensor, summarize_robustness
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.calibrated import CalibratedFusionConfig, calibrated_fusion_matrix
from research_pipeline.models.common import prediction_from_scores
from research_pipeline.models.hybrid import (
    CachedArtifactPredictor,
    HybridConfig,
    fuse_hybrid_predictions,
    geometry_prior_prediction,
)


@dataclass(slots=True)
class ScorePack:
    y_true: list[str]
    c1_scores: np.ndarray
    c3_scores: np.ndarray
    latencies_ms: list[float]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C5 calibrated score fusion for stronger gesture recognition evaluation."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    start = time.perf_counter()
    rng = np.random.default_rng(int(config.get("seed", 42)))
    raw_model_paths = config.get("model_paths") or [config["model_path"]]
    model_paths = [project_path(path) for path in raw_model_paths]
    direct_predictors = [CachedArtifactPredictor(load_artifact(path)) for path in model_paths]
    hybrid_config = HybridConfig(**config.get("hybrid", {}))

    calibration_manifest = project_path(config["calibration_manifest"])
    evaluation_manifest = project_path(config["evaluation_manifest"])
    calibration_records = _select_records(
        read_jsonl(calibration_manifest),
        max_records=_optional_int(config.get("max_calibration_records")),
        rng=rng,
    )
    calibration_scenarios = [_scenario_from_dict(item) for item in config.get("calibration_scenarios", [])]
    if not calibration_scenarios:
        calibration_scenarios = [_scenario_from_dict({"name": "clean", "kind": "clean"})]

    print(
        "c5: calibration records="
        f"{len(calibration_records)} scenarios={','.join(item.name for item in calibration_scenarios)}"
    )
    calibration_pack = _predict_scenarios(
        calibration_manifest,
        calibration_records,
        direct_predictors,
        hybrid_config,
        calibration_scenarios,
        seed=int(config.get("seed", 42)) + 1000,
    )

    candidates = _candidate_configs(config)
    ranked = _rank_candidates(
        calibration_pack,
        candidates,
        config.get("objective", {}),
        num_bins=int(config.get("calibration_bins", 15)),
    )
    selected = _select_configs(ranked)

    print(
        "c5: candidates="
        f"{len(candidates)} best_macro={ranked[0]['macro_f1']:.4f} best_safety={selected['safety']['safety_score']:.4f}"
    )
    evaluation_records = _select_records(
        read_jsonl(evaluation_manifest),
        max_records=_optional_int(config.get("max_evaluation_records")),
        rng=rng,
    )
    evaluation_scenarios = [_scenario_from_dict(item) for item in config.get("evaluation_scenarios", [])]
    if not evaluation_scenarios:
        evaluation_scenarios = [_scenario_from_dict({"name": "clean", "kind": "clean"})]

    robustness = _evaluate_selected_configs(
        evaluation_manifest,
        evaluation_records,
        direct_predictors,
        hybrid_config,
        evaluation_scenarios,
        selected,
        seed=int(config.get("seed", 42)) + 2000,
        calibration_bins=int(config.get("calibration_bins", 15)),
    )
    summary = summarize_robustness(robustness)
    clean = robustness.get("clean", {})
    improvements = _improvement_summary(clean, summary)

    report = {
        "method": {
            "name": "C5 calibrated recognition fusion",
            "description": (
                "Train-split calibrated score fusion over C1-TCN and C3 geometry-aware scores with "
                "class-bias tuning, temperature calibration, and optional action abstention."
            ),
            "model_paths": [str(path) for path in model_paths],
        },
        "calibration": {
            "manifest": str(calibration_manifest),
            "num_records": len(calibration_records),
            "num_augmented_records": len(calibration_pack.y_true),
            "scenarios": [asdict(item) for item in calibration_scenarios],
            "num_candidates": len(candidates),
            "best_macro": selected["macro"],
            "best_safety": selected["safety"],
            "top_candidates": ranked[: min(12, len(ranked))],
        },
        "evaluation": {
            "manifest": str(evaluation_manifest),
            "num_records": len(evaluation_records),
            "scenarios": [asdict(item) for item in evaluation_scenarios],
            "robustness": robustness,
            "summary": summary,
            "improvements": improvements,
        },
        "latency": {
            "median_ms": _percentile(_all_latencies(robustness), 50),
            "p95_ms": _percentile(_all_latencies(robustness), 95),
            "note": "Measured as cached TCN pass(es) plus lightweight C3 geometry and C5/C6 calibration.",
        },
        "config": config,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_report(config.get("output_report", "artifacts/reports/c5_calibrated_recognition.json"), report)

    c1 = summary.get("c1t_direct", {}).get("perturbed_macro_f1_mean", 0.0)
    c5 = summary.get("c5_safety", {}).get("perturbed_macro_f1_mean", 0.0)
    c1_ece = summary.get("c1t_direct", {}).get("clean_ece", 0.0)
    c5_ece = summary.get("c5_safety", {}).get("clean_ece", 0.0)
    print(f"c5_perturbed_macro_f1={c5:.4f} c1t_direct={c1:.4f} delta={c5 - c1:+.4f}")
    print(f"clean_ece c5_safety={c5_ece:.4f} c1t_direct={c1_ece:.4f} delta={c5_ece - c1_ece:+.4f}")


def _predict_scenarios(
    manifest_path: Path,
    records: list[ManifestRecord],
    direct_predictors: list[CachedArtifactPredictor],
    hybrid_config: HybridConfig,
    scenarios: list[PerturbationConfig],
    *,
    seed: int,
) -> ScorePack:
    packs = []
    for scenario_index, scenario in enumerate(scenarios):
        packs.append(
            _predict_records(
                manifest_path,
                records,
                direct_predictors,
                hybrid_config,
                scenario,
                seed=seed + scenario_index * 997,
            )
        )
    return ScorePack(
        y_true=[label for pack in packs for label in pack.y_true],
        c1_scores=np.concatenate([pack.c1_scores for pack in packs], axis=0),
        c3_scores=np.concatenate([pack.c3_scores for pack in packs], axis=0),
        latencies_ms=[latency for pack in packs for latency in pack.latencies_ms],
    )


def _predict_records(
    manifest_path: Path,
    records: list[ManifestRecord],
    direct_predictors: list[CachedArtifactPredictor],
    hybrid_config: HybridConfig,
    scenario: PerturbationConfig,
    *,
    seed: int,
) -> ScorePack:
    base_dir = manifest_path.parent
    rng = np.random.default_rng(seed)
    y_true: list[str] = []
    c1_scores = np.zeros((len(records), len(TARGET_LABELS)), dtype=np.float64)
    c3_scores = np.zeros_like(c1_scores)
    latencies_ms: list[float] = []

    for index, record in enumerate(records):
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        tensor = perturb_tensor(tensor, scenario, rng)
        start = time.perf_counter()
        c1 = _ensemble_predict(direct_predictors, tensor)
        geometry = geometry_prior_prediction(tensor, hybrid_config)
        c3 = fuse_hybrid_predictions(c1, geometry, tensor, hybrid_config)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        y_true.append(record.target_label)
        c1_scores[index] = [float(c1.scores.get(label, 0.0)) for label in TARGET_LABELS]
        c3_scores[index] = [float(c3.scores.get(label, 0.0)) for label in TARGET_LABELS]

    return ScorePack(y_true=y_true, c1_scores=c1_scores, c3_scores=c3_scores, latencies_ms=latencies_ms)


def _rank_candidates(
    pack: ScorePack,
    candidates: list[CalibratedFusionConfig],
    objective: dict[str, Any],
    *,
    num_bins: int = 15,
) -> list[dict]:
    ranked: list[dict] = []
    weak_labels = list(objective.get("weak_labels", ["click_2f", "swipe_left", "zoom_out"]))
    weak_weight = float(objective.get("weak_f1_weight", 0.04))
    false_action_penalty = float(objective.get("false_action_penalty", 0.18))
    # Calibration is now a first-class selection criterion: an ece_penalty > 0 makes the
    # safety objective prefer configs that keep the expected calibration error low, instead
    # of only maximising macro F1 and the no_gesture safety margin (which previously let the
    # bias/temperature search degrade calibration as a side effect).
    ece_penalty = float(objective.get("ece_penalty", 0.0))

    for candidate in candidates:
        probabilities = calibrated_fusion_matrix(pack.c1_scores, pack.c3_scores, candidate)
        y_pred = [TARGET_LABELS[int(index)] for index in probabilities.argmax(axis=1)]
        metrics, risk = _metrics_and_risk(pack.y_true, y_pred)
        calibration = compute_calibration_report(probabilities, pack.y_true, num_bins=num_bins)
        ece = calibration.expected_calibration_error
        weak_mean = _weak_mean_f1(metrics, weak_labels)
        false_action = float(risk.get("no_gesture_false_action_rate", 0.0))
        macro_score = metrics.macro_f1 + weak_weight * weak_mean
        safety_score = macro_score - false_action_penalty * false_action - ece_penalty * ece
        ranked.append(
            {
                "rank": 0,
                "macro_score": macro_score,
                "safety_score": safety_score,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "balanced_accuracy": metrics.balanced_accuracy,
                "weak_mean_f1": weak_mean,
                "no_gesture_false_action_rate": false_action,
                "expected_calibration_error": ece,
                "config": _config_to_dict(candidate),
            }
        )
    ranked.sort(key=lambda item: (item["macro_score"], item["safety_score"]), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def _select_configs(ranked: list[dict]) -> dict[str, dict]:
    macro = ranked[0]
    safety = max(ranked, key=lambda item: (item["safety_score"], item["macro_score"]))
    return {"macro": macro, "safety": safety}


def _evaluate_selected_configs(
    manifest_path: Path,
    records: list[ManifestRecord],
    direct_predictors: list[CachedArtifactPredictor],
    hybrid_config: HybridConfig,
    scenarios: list[PerturbationConfig],
    selected: dict[str, dict],
    *,
    seed: int,
    calibration_bins: int = 15,
) -> dict[str, dict]:
    macro_config = _config_from_dict(selected["macro"]["config"])
    safety_config = _config_from_dict(selected["safety"]["config"])
    report: dict[str, dict] = {}

    for scenario_index, scenario in enumerate(scenarios):
        pack = _predict_records(
            manifest_path,
            records,
            direct_predictors,
            hybrid_config,
            scenario,
            seed=seed + scenario_index * 997,
        )
        method_probabilities = {
            "c1t_direct": pack.c1_scores,
            "c3_hybrid": pack.c3_scores,
            "c5_macro": calibrated_fusion_matrix(pack.c1_scores, pack.c3_scores, macro_config),
            "c5_safety": calibrated_fusion_matrix(pack.c1_scores, pack.c3_scores, safety_config),
        }
        scenario_report: dict[str, dict] = {}
        for name, probabilities in method_probabilities.items():
            y_pred = [TARGET_LABELS[int(index)] for index in probabilities.argmax(axis=1)]
            metrics, risk = _metrics_and_risk(pack.y_true, y_pred)
            calibration = compute_calibration_report(probabilities, pack.y_true, num_bins=calibration_bins)
            scenario_report[name] = {
                "recognition": metrics.to_dict(),
                "calibration": calibration.to_dict(),
                "risk": risk,
                "latency": {
                    "median_ms": _percentile(pack.latencies_ms, 50),
                    "p95_ms": _percentile(pack.latencies_ms, 95),
                },
            }
        report[scenario.name] = scenario_report
        print(f"c5: evaluated scenario={scenario.name} records={len(records)}")
    return report


def _ensemble_predict(direct_predictors, tensor):
    if len(direct_predictors) == 1:
        return direct_predictors[0].predict(tensor)
    scores = {label: 0.0 for label in TARGET_LABELS}
    for predictor in direct_predictors:
        prediction = predictor.predict(tensor)
        for label in TARGET_LABELS:
            scores[label] += float(prediction.scores.get(label, 0.0))
    scale = 1.0 / len(direct_predictors)
    return prediction_from_scores({label: value * scale for label, value in scores.items()})


def _metrics_and_risk(y_true: list[str], y_pred: list[str]) -> tuple[RecognitionMetrics, dict[str, Any]]:
    metrics = compute_recognition_metrics(y_true, y_pred)
    return metrics, analyze_recognition_risk({"recognition": metrics.to_dict()})


def _candidate_configs(config: dict[str, Any]) -> list[CalibratedFusionConfig]:
    grid = config.get("grid", {})
    c3_weights = grid.get("c3_weight", [0.0])
    temperatures = grid.get("temperature", [1.0])
    min_action_confidences = grid.get("min_action_confidence", [0.0])
    min_action_margins = grid.get("min_action_margin", [0.0])
    label_bias_grid = grid.get("label_biases", {})
    labels = list(label_bias_grid)
    bias_values = [label_bias_grid[label] for label in labels]
    base_biases = dict(config.get("base_label_biases", {}))

    candidates: list[CalibratedFusionConfig] = []
    for c3_weight, temperature, min_confidence, min_margin, *biases in itertools.product(
        c3_weights,
        temperatures,
        min_action_confidences,
        min_action_margins,
        *bias_values,
    ):
        label_biases = dict(base_biases)
        label_biases.update({label: float(value) for label, value in zip(labels, biases)})
        candidates.append(
            CalibratedFusionConfig(
                c3_weight=float(c3_weight),
                temperature=float(temperature),
                min_action_confidence=float(min_confidence),
                min_action_margin=float(min_margin),
                label_biases=label_biases,
            )
        )
    return candidates or [CalibratedFusionConfig()]


def _select_records(
    records: list[ManifestRecord],
    *,
    max_records: int | None,
    rng: np.random.Generator,
) -> list[ManifestRecord]:
    if max_records is None or max_records >= len(records):
        return records
    by_label: dict[str, list[ManifestRecord]] = {label: [] for label in TARGET_LABELS}
    for record in records:
        by_label.setdefault(record.target_label, []).append(record)

    selected: list[ManifestRecord] = []
    quota = max(1, max_records // len(TARGET_LABELS))
    for label in TARGET_LABELS:
        label_records = list(by_label.get(label, []))
        rng.shuffle(label_records)
        selected.extend(label_records[: min(quota, len(label_records))])

    if len(selected) < max_records:
        selected_ids = {record.sample_id for record in selected}
        remainder = [record for record in records if record.sample_id not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: max_records - len(selected)])
    rng.shuffle(selected)
    return selected[:max_records]


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


def _improvement_summary(clean: dict[str, dict], summary: dict[str, dict]) -> dict[str, Any]:
    base_clean = clean.get("c1t_direct", {}).get("recognition", {})
    c3_clean = clean.get("c3_hybrid", {}).get("recognition", {})
    c5_clean = clean.get("c5_safety", {}).get("recognition", {})
    base_calib = clean.get("c1t_direct", {}).get("calibration", {})
    c5_calib = clean.get("c5_safety", {}).get("calibration", {})
    base_robust = summary.get("c1t_direct", {})
    c3_robust = summary.get("c3_hybrid", {})
    c5_robust = summary.get("c5_safety", {})
    return {
        "clean_macro_f1_delta_vs_c1t": float(c5_clean.get("macro_f1", 0.0)) - float(base_clean.get("macro_f1", 0.0)),
        "clean_macro_f1_delta_vs_c3": float(c5_clean.get("macro_f1", 0.0)) - float(c3_clean.get("macro_f1", 0.0)),
        "perturbed_macro_f1_delta_vs_c1t": float(c5_robust.get("perturbed_macro_f1_mean", 0.0))
        - float(base_robust.get("perturbed_macro_f1_mean", 0.0)),
        "perturbed_macro_f1_delta_vs_c3": float(c5_robust.get("perturbed_macro_f1_mean", 0.0))
        - float(c3_robust.get("perturbed_macro_f1_mean", 0.0)),
        "perturbed_false_action_delta_vs_c1t": float(
            c5_robust.get("perturbed_no_gesture_false_action_rate_mean", 0.0)
        )
        - float(base_robust.get("perturbed_no_gesture_false_action_rate_mean", 0.0)),
        "clean_ece_delta_vs_c1t": float(c5_calib.get("expected_calibration_error", 0.0))
        - float(base_calib.get("expected_calibration_error", 0.0)),
        "clean_brier_delta_vs_c1t": float(c5_calib.get("brier_score", 0.0))
        - float(base_calib.get("brier_score", 0.0)),
    }


def _weak_mean_f1(metrics: RecognitionMetrics, weak_labels: list[str]) -> float:
    values = [
        float(metrics.per_class[label]["f1"])
        for label in weak_labels
        if label in metrics.per_class and metrics.per_class[label]["support"] > 0
    ]
    return float(np.mean(values)) if values else 0.0


def _config_to_dict(config: CalibratedFusionConfig) -> dict[str, Any]:
    return {
        "c3_weight": config.c3_weight,
        "temperature": config.temperature,
        "label_biases": dict(config.label_biases),
        "min_action_confidence": config.min_action_confidence,
        "min_action_margin": config.min_action_margin,
    }


def _config_from_dict(payload: dict[str, Any]) -> CalibratedFusionConfig:
    return CalibratedFusionConfig(
        c3_weight=float(payload.get("c3_weight", 0.0)),
        temperature=float(payload.get("temperature", 1.0)),
        label_biases={str(key): float(value) for key, value in payload.get("label_biases", {}).items()},
        min_action_confidence=float(payload.get("min_action_confidence", 0.0)),
        min_action_margin=float(payload.get("min_action_margin", 0.0)),
    )


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0):
        return None
    return int(value)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _all_latencies(robustness: dict[str, dict]) -> list[float]:
    latencies = []
    for scenario in robustness.values():
        for report in scenario.values():
            latency = report.get("latency", {})
            if "median_ms" in latency:
                latencies.append(float(latency["median_ms"]))
    return latencies


if __name__ == "__main__":
    main()
