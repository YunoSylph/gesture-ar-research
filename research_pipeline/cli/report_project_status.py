from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import project_path, write_json_report


DEFAULT_RECOGNITION_REPORTS = {
    "C0 rule": "artifacts/reports/ipn_c0_full_recognition.json",
    "C1 random forest": "artifacts/reports/ipn_c1_rf_full_recognition.json",
    "C1-T compact TCN": "artifacts/reports/ipn_c1t_tcn_full_recognition.json",
    "C1-T compact TCN validated": "artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json",
    "C6 augmented TCN": "artifacts/reports/ipn_c1t_tcn_augmented_recognition.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact current-state report for the thesis project.")
    parser.add_argument("--output", default="artifacts/reports/project_stage_status.json")
    args = parser.parse_args()

    recognition = {
        name: _recognition_row(project_path(path))
        for name, path in DEFAULT_RECOGNITION_REPORTS.items()
    }
    report = {
        "recognition_public_ipn": recognition,
        "interaction": {
            "desktop_ar_demo": _exists("demo/ar_interaction_app/src/main.tsx"),
            "live_backend": _exists("research_pipeline/serve/live_backend.py"),
            "task_report": _json_if_exists("artifacts/reports/live_task_report.json"),
            "recognition_risk": _json_if_exists("artifacts/reports/recognition_risk_analysis.json"),
            "scenario_config": _exists("configs/interaction/ar_task_scenarios.yaml"),
        },
        "c3_hybrid_research": {
            "robustness": _json_if_exists("artifacts/reports/c3_hybrid_robustness.json"),
            "calibrated_ablation": _json_if_exists("artifacts/reports/c3_research_ablation.json"),
        },
        "c4_action_safe_research": {
            "action_replay": _json_if_exists("artifacts/reports/c4_action_safe_research.json"),
            "task_benchmark": _json_if_exists("artifacts/reports/c4_task_benchmark.json"),
            "task_failure_analysis": _json_if_exists("artifacts/reports/c4_task_failure_analysis.json"),
        },
        "c6_recognition_research": {
            "augmented_robustness": _json_if_exists("artifacts/reports/c6_augmented_robustness.json"),
            "ensemble_calibrated": _json_if_exists("artifacts/reports/c6_ensemble_calibrated_recognition.json"),
            "summary": _file_status("artifacts/reports/c6_recognition_upgrade.md"),
        },
        "thesis_outputs": {
            "experiment_chapter": _file_status("artifacts/reports/thesis_experiment_chapter.md"),
        },
        "portability": {
            "onnx_full": _file_status("artifacts/export/ipn_c1t_tcn_full.onnx"),
            "mobile_bundle": _file_status("artifacts/mobile/gesture_mobile_bundle/bundle_manifest.json"),
            "ios_swift_contract": {
                "labels": _exists("ios_demo/GestureAR/Sources/GestureLabels.swift"),
                "preprocessing": _exists("ios_demo/GestureAR/Sources/LandmarkPreprocessor.swift"),
                "context_policy": _exists("ios_demo/GestureAR/Sources/ContextPolicy.swift"),
            },
            "coreml": "deferred_to_macos_or_linux",
        },
        "local_domain": _json_if_exists("artifacts/reports/domain_readiness.json"),
        "remaining_requires_local_videos": [
            "zero-shot evaluation on phone_rear_ar",
            "C2 threshold calibration on local validation clips",
            "optional TCN fine-tuning for C2+Local",
            "final Direct vs C2 user/task sessions on real phone/webcam clips",
        ],
        "remaining_requires_macos_or_ios_device": [
            "Core ML .mlpackage conversion verification",
            "Xcode RealityKit app assembly",
            "on-device latency and portability measurements",
        ],
    }
    write_json_report(args.output, report)
    print(f"wrote project status to {project_path(args.output)}")


def _recognition_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload is None:
        return {"status": "missing", "path": str(path)}
    recognition = payload.get("recognition", {})
    latency = payload.get("latency", {})
    return {
        "status": "ready",
        "path": str(path),
        "accuracy": recognition.get("accuracy"),
        "macro_f1": recognition.get("macro_f1"),
        "weighted_f1": recognition.get("weighted_f1"),
        "p95_latency_ms": latency.get("offline_latency_ms_p95"),
        "num_samples": latency.get("num_samples"),
    }


def _json_if_exists(path: str) -> dict[str, Any] | str:
    payload = _load_json(project_path(path))
    return payload if payload is not None else "missing"


def _file_status(path: str) -> dict[str, Any]:
    file_path = project_path(path)
    return {
        "exists": file_path.exists(),
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
    }


def _exists(path: str) -> bool:
    return project_path(path).exists()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    main()
