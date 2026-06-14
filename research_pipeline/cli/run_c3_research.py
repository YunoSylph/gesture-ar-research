from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import asdict
from typing import Any

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.evaluation.ablation import benchmark_policy_manifest, score_candidate
from research_pipeline.evaluation.robustness import (
    PerturbationConfig,
    benchmark_robustness_manifest,
    summarize_robustness,
)
from research_pipeline.interaction.fsm import ContextPolicyConfig
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.hybrid import (
    CachedArtifactPredictor,
    GeometryPriorRecognizer,
    HybridConfig,
    HybridRecognizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated C3 ablation and interaction-policy research benchmark.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    start = time.perf_counter()
    artifact = load_artifact(project_path(config["model_path"]))
    base_hybrid = dict(config.get("base_hybrid", {}))
    calibration_scenarios = [_scenario_from_dict(item) for item in config.get("calibration_robustness", [])]
    evaluation_scenarios = [_scenario_from_dict(item) for item in config.get("evaluation_robustness", [])]
    if not calibration_scenarios:
        calibration_scenarios = [_scenario_from_dict({"name": "clean", "kind": "clean"})]
    if not evaluation_scenarios:
        evaluation_scenarios = calibration_scenarios

    calibration = _calibrate(
        artifact,
        config,
        base_hybrid=base_hybrid,
        scenarios=calibration_scenarios,
    )
    best_config = calibration["best"]["config"]
    ablation = _run_ablation(
        artifact,
        config,
        best_config=best_config,
        scenarios=evaluation_scenarios,
    )
    policy = _run_policy_ablation(
        artifact,
        config,
        best_config=best_config,
    )

    report = {
        "method": {
            "name": "C3 calibrated ablation",
            "description": "Calibration on public train split, final ablation on public test split.",
        },
        "calibration": calibration,
        "ablation": ablation,
        "policy_ablation": policy,
        "config": config,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_report(config.get("output_report", "artifacts/reports/c3_research_ablation.json"), report)
    summary = ablation["summary"]
    best_macro = summary.get("c3_hybrid", {}).get("perturbed_macro_f1_mean", 0.0)
    direct_macro = summary.get("c1t_direct", {}).get("perturbed_macro_f1_mean", 0.0)
    print(f"best_c3_perturbed_macro_f1={best_macro:.4f} direct={direct_macro:.4f}")


def _calibrate(
    artifact: dict[str, Any],
    config: dict[str, Any],
    *,
    base_hybrid: dict[str, Any],
    scenarios: list[PerturbationConfig],
) -> dict[str, Any]:
    candidates = _candidate_configs(config, base_hybrid)
    ranked = []
    objective = config.get("objective", {})
    for index, candidate in enumerate(candidates, start=1):
        hybrid = HybridRecognizer(artifact, HybridConfig(**candidate))
        report = benchmark_robustness_manifest(
            project_path(config["calibration_manifest"]),
            {"c3_hybrid": hybrid.predict},
            scenarios,
            seed=int(config.get("seed", 42)) + index * 101,
            max_records=int(config["max_calibration_records"]) if config.get("max_calibration_records") else None,
        )
        summary = summarize_robustness(report)["c3_hybrid"]
        score = score_candidate(
            summary,
            false_action_penalty=float(objective.get("false_action_penalty", 0.25)),
            drop_penalty=float(objective.get("drop_penalty", 0.1)),
        )
        ranked.append(
            {
                "rank": 0,
                "score": score,
                "config": candidate,
                "summary": summary,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {
        "manifest": config["calibration_manifest"],
        "scenarios": [asdict(item) for item in scenarios],
        "num_candidates": len(ranked),
        "best": ranked[0],
        "top_candidates": ranked[: min(8, len(ranked))],
    }


def _run_ablation(
    artifact: dict[str, Any],
    config: dict[str, Any],
    *,
    best_config: dict[str, Any],
    scenarios: list[PerturbationConfig],
) -> dict[str, Any]:
    direct = CachedArtifactPredictor(artifact)
    geometry = GeometryPriorRecognizer(HybridConfig(**best_config))
    fusion_config = dict(best_config)
    fusion_config["enable_safety_gate"] = False
    tcn_geometry = HybridRecognizer(artifact, HybridConfig(**fusion_config))
    c3 = HybridRecognizer(artifact, HybridConfig(**best_config))
    robustness = benchmark_robustness_manifest(
        project_path(config["evaluation_manifest"]),
        {
            "c1t_direct": direct.predict,
            "geometry_only": geometry.predict,
            "tcn_geometry_fusion": tcn_geometry.predict,
            "c3_hybrid": c3.predict,
        },
        scenarios,
        seed=int(config.get("seed", 42)),
        max_records=int(config["max_evaluation_records"]) if config.get("max_evaluation_records") else None,
    )
    return {
        "manifest": config["evaluation_manifest"],
        "scenarios": [asdict(item) for item in scenarios],
        "robustness": robustness,
        "summary": summarize_robustness(robustness),
    }


def _run_policy_ablation(artifact: dict[str, Any], config: dict[str, Any], *, best_config: dict[str, Any]) -> dict[str, Any]:
    scenario = _scenario_from_dict(config.get("policy_scenario", {"name": "clean", "kind": "clean"}))
    direct = CachedArtifactPredictor(artifact)
    c3 = HybridRecognizer(artifact, HybridConfig(**best_config))
    policy_config = ContextPolicyConfig(**config.get("c2_policy", {}))
    return benchmark_policy_manifest(
        project_path(config["evaluation_manifest"]),
        {
            "c1t_direct": direct.predict,
            "c3_hybrid": c3.predict,
        },
        scenario,
        seed=int(config.get("seed", 42)),
        max_records=int(config["max_policy_records"]) if config.get("max_policy_records") else None,
        c2_policy=policy_config,
        frames_per_clip=int(config.get("policy_frames_per_clip", 3)),
    )


def _candidate_configs(config: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("candidates"):
        return [{**base, **candidate} for candidate in config["candidates"]]
    grid = config.get("grid", {})
    keys = list(grid)
    values = [grid[key] for key in keys]
    candidates = []
    for combo in itertools.product(*values):
        candidate = dict(base)
        candidate.update({key: value for key, value in zip(keys, combo)})
        candidates.append(candidate)
    return candidates or [dict(base)]


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


if __name__ == "__main__":
    main()
