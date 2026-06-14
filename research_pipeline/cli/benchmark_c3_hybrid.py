from __future__ import annotations

import argparse
import time
from dataclasses import asdict

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.evaluation.robustness import (
    PerturbationConfig,
    benchmark_robustness_manifest,
    summarize_robustness,
)
from research_pipeline.models.artifacts import load_artifact
from research_pipeline.models.hybrid import CachedArtifactPredictor, HybridConfig, HybridRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C3 hybrid recognizer against direct temporal inference.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    artifact = load_artifact(project_path(config["model_path"]))
    direct = CachedArtifactPredictor(artifact)
    hybrid_config = HybridConfig(**config.get("hybrid", {}))
    hybrid = HybridRecognizer(artifact, hybrid_config)
    scenarios = [_scenario_from_dict(item) for item in config.get("robustness", _default_scenarios())]

    start = time.perf_counter()
    robustness = benchmark_robustness_manifest(
        project_path(config["manifest"]),
        {
            "c1t_direct": direct.predict,
            "c3_hybrid": hybrid.predict,
        },
        scenarios,
        seed=int(config.get("seed", 42)),
        max_records=int(config["max_records"]) if config.get("max_records") else None,
    )
    elapsed = time.perf_counter() - start
    report = {
        "method": {
            "name": "C3 Hybrid",
            "description": "Cached temporal TCN probabilities fused with geometry-aware safety priors.",
            "hybrid_config": asdict(hybrid_config),
        },
        "robustness": robustness,
        "summary": summarize_robustness(robustness),
        "config": config,
        "elapsed_seconds": elapsed,
    }
    write_json_report(config.get("output_report", "artifacts/reports/c3_hybrid_robustness.json"), report)
    clean = robustness.get("clean", {}).get("c3_hybrid")
    if clean:
        confusion = clean["recognition"]["confusion_matrix"]
        num_samples = sum(sum(row) for row in confusion)
        write_json_report(
            config.get("output_clean_report", "artifacts/reports/c3_hybrid_clean_recognition.json"),
            {
                "recognition": clean["recognition"],
                "risk": clean["risk"],
                "latency": {
                    "num_samples": num_samples,
                    "note": "C3 latency is reported in the robustness benchmark elapsed_seconds; it reuses cached TCN inference plus lightweight geometry fusion.",
                },
                "config": config,
            },
        )
    summary = report["summary"]
    direct_macro = summary.get("c1t_direct", {}).get("perturbed_macro_f1_mean", 0.0)
    hybrid_macro = summary.get("c3_hybrid", {}).get("perturbed_macro_f1_mean", 0.0)
    print(f"c1t_perturbed_macro_f1={direct_macro:.4f} c3_perturbed_macro_f1={hybrid_macro:.4f}")


def _scenario_from_dict(payload: dict) -> PerturbationConfig:
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


def _default_scenarios() -> list[dict]:
    return [
        {"name": "clean", "kind": "clean"},
        {"name": "noise_mild", "kind": "gaussian_noise", "sigma": 0.006},
        {"name": "frame_drop_15", "kind": "frame_drop", "drop_rate": 0.15},
        {"name": "landmark_mask_10", "kind": "landmark_mask", "mask_rate": 0.10},
        {"name": "temporal_jitter_2", "kind": "temporal_jitter", "jitter": 2},
        {"name": "combined_mild", "kind": "combined", "sigma": 0.004, "drop_rate": 0.10, "mask_rate": 0.06},
    ]


if __name__ == "__main__":
    main()
