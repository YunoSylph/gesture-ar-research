from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research_pipeline.cli.common import load_yaml, project_path
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, ContextPolicyConfig
from research_pipeline.labels import FINAL_GESTURES, TARGET_LABELS
from research_pipeline.models.artifacts import artifact_feature_flags, load_artifact
from research_pipeline.models.preprocessing_contract import feature_layout_contract
from research_pipeline.utils.errors import PipelineError


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the mobile/iOS portability contract bundle.")
    parser.add_argument("--model-path", default="artifacts/models/ipn_c1t_tcn_full.pkl")
    parser.add_argument("--onnx-path", default="artifacts/export/ipn_c1t_tcn_full.onnx")
    parser.add_argument("--c2-config", default="configs/interaction/c2.yaml")
    parser.add_argument("--output-dir", default="artifacts/mobile/gesture_mobile_bundle")
    args = parser.parse_args()

    model_path = project_path(args.model_path)
    onnx_path = project_path(args.onnx_path)
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(model_path)
    target_length = int(artifact.get("target_length", 32))
    input_dim = int(artifact.get("tcn_config", {}).get("input_dim", 74))
    c2_config = load_yaml(project_path(args.c2_config))
    policy = ContextPolicyConfig(**c2_config.get("policy", {}))

    labels_payload = {
        "labels": [
            {
                "index": gesture.index,
                "target_label": gesture.target_label,
                "ipn_name": gesture.ipn_name,
                "ipn_id": gesture.ipn_id,
                "semantics": gesture.semantics,
                "action": ACTION_BY_LABEL.get(gesture.target_label, "none"),
            }
            for gesture in FINAL_GESTURES
        ],
        "directed_label_policy": {
            "swipe_left": "screen-space left action",
            "swipe_right": "screen-space right action",
            "mirror_rule": "swap swipe_left/swipe_right only when augmenting mirrored training samples",
        },
    }
    include_mv, mv_coords = artifact_feature_flags(artifact)
    preprocessing_payload = feature_layout_contract(
        target_length=target_length, multiview_coords=mv_coords, include_multiview=include_mv
    )
    preprocessing_payload["model_input_shape"] = [1, target_length, input_dim]
    preprocessing_payload["runtime_contract"] = {
        "desktop": "MediaPipe Python HandLandmarker -> same preprocessing -> ONNX/PyTorch classifier",
        "ios": "ARKit rear-camera frame -> MediaPipe/Vision hand landmarks -> same preprocessing -> Core ML classifier",
    }
    if preprocessing_payload["feature_dim"] != input_dim:
        raise PipelineError(
            f"feature contract dim {preprocessing_payload['feature_dim']} != model input_dim {input_dim}; "
            "the artifact's feature flags disagree with its tcn_config."
        )
    c2_payload = {
        "activation_threshold": policy.activation_threshold,
        "stable_frames": policy.stable_frames,
        "cooldown_ms": policy.cooldown_ms,
        "no_gesture_reset_frames": policy.no_gesture_reset_frames,
        "action_by_label": ACTION_BY_LABEL,
    }
    portability_payload = {
        "model_path": str(model_path),
        "model_type": artifact["model_type"],
        "target_labels": list(TARGET_LABELS),
        "onnx": _file_info(onnx_path),
        "coreml_stage": {
            "status": "conversion_deferred_to_macos_or_linux",
            "recommended_output": "artifacts/export/GestureClassifier.mlpackage",
            "minimum_deployment_target": "iOS15",
            "compute_precision": "FLOAT16",
        },
        "phone_ar_contract": {
            "camera_source": "rear_world_camera",
            "use_same_frame_for_ar_and_gesture_recognition": True,
            "ar_renderer": "RealityKit",
            "tracking": "ARKit world tracking",
            "gesture_domain": "phone_rear_ar",
            "local_adaptation_role": "target-domain validation and optional fine-tuning/calibration",
        },
    }

    _write_json(output_dir / "labels.json", labels_payload)
    _write_json(output_dir / "preprocessing_contract.json", preprocessing_payload)
    _write_json(output_dir / "c2_policy.json", c2_payload)
    _write_json(output_dir / "portability_contract.json", portability_payload)
    _write_json(
        output_dir / "bundle_manifest.json",
        {
            "files": [
                "labels.json",
                "preprocessing_contract.json",
                "c2_policy.json",
                "portability_contract.json",
            ],
            "summary": "Mobile contract bundle for the iPhone/iPad AR portability stage.",
        },
    )
    print(f"wrote mobile bundle to {output_dir}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    main()
