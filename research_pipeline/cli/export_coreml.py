from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.models.artifacts import artifact_feature_flags, load_artifact
from research_pipeline.models.coreml_export import convert_tcn_artifact_to_coreml

DEFAULT_MODEL = "artifacts/models/ipn_c1t_tcn_full_validated_mv.pkl"
DEFAULT_OUTPUT = "artifacts/export/GestureClassifier.mlpackage"


def _example_from_manifest(manifest_path: Path, artifact: dict) -> np.ndarray | None:
    """Build one preprocessed feature window from a manifest's first tensor (real parity input)."""

    from research_pipeline.data.manifest import read_jsonl
    from research_pipeline.data.schema import resolve_path
    from research_pipeline.data.tensors import load_landmark_npz
    from research_pipeline.features.preprocessing import preprocess_dual_view

    records = read_jsonl(manifest_path)
    if not records or not records[0].tensor_path:
        return None
    include_mv, mv_coords = artifact_feature_flags(artifact)
    target_length = int(artifact.get("target_length", 32))
    tensor = load_landmark_npz(resolve_path(records[0].tensor_path, manifest_path.parent))
    sequence = preprocess_dual_view(
        tensor, target_length=target_length, include_multiview=include_mv, multiview_coords=mv_coords
    )
    return sequence.features[None, :, :].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trained TCN artifact to a Core ML .mlpackage for iPhone.")
    parser.add_argument("--config", help="Optional YAML overriding the options below.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-deployment-target", default="iOS15")
    parser.add_argument("--compute-precision", default="FLOAT16", choices=["FLOAT16", "FLOAT32"])
    parser.add_argument("--input-name", default="landmarks")
    parser.add_argument("--sample-manifest", help="Landmarks manifest to draw a real parity example from.")
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}
    model_path = project_path(config.get("model_path", args.model_path))
    output_path = project_path(config.get("output_path", args.output_path))
    deployment = config.get("minimum_deployment_target", args.minimum_deployment_target)
    precision = config.get("compute_precision", args.compute_precision)
    input_name = config.get("input_name", args.input_name)
    sample_manifest = config.get("sample_manifest", args.sample_manifest)

    artifact = load_artifact(model_path)
    example = _example_from_manifest(Path(sample_manifest), artifact) if sample_manifest else None

    result = convert_tcn_artifact_to_coreml(
        artifact,
        output_path,
        input_name=input_name,
        minimum_deployment_target=deployment,
        compute_precision=precision,
        example_input=example,
    )

    report = {"stage": "coreml_export", "status": "converted", "model_path": str(model_path), **result}
    write_json_report(str(output_path) + ".contract.json", report)
    parity = result["parity"]
    print(
        f"converted -> {output_path} | argmax_match={parity['argmax_match']} "
        f"max_abs_logit_diff={parity['max_abs_logit_diff']:.4f}"
    )


if __name__ == "__main__":
    main()
