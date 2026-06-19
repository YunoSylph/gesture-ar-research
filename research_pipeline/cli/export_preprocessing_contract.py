from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.cli.common import project_path
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.models.artifacts import artifact_feature_flags, load_artifact, predict_with_artifact
from research_pipeline.models.preprocessing_contract import feature_layout_contract, golden_sample

DEFAULT_MODEL = "artifacts/models/ipn_c1t_tcn_full_validated_mv.pkl"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the exact on-device feature contract + landmarks->features golden vectors."
    )
    parser.add_argument("--manifest", required=True, help="Landmarks manifest to draw golden samples from.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="artifacts/mobile/preprocessing")
    parser.add_argument("--max-samples", type=int, default=7, help="At most one golden sample per target class.")
    args = parser.parse_args()

    artifact = load_artifact(project_path(args.model_path))
    include_mv, mv_coords = artifact_feature_flags(artifact)
    if not include_mv:
        raise SystemExit("Model is not a multiview model; this contract describes the mv feature layout.")
    target_length = int(artifact.get("target_length", 32))

    contract = feature_layout_contract(target_length=target_length, multiview_coords=mv_coords)

    manifest_path = Path(project_path(args.manifest))
    records = read_jsonl(manifest_path)
    seen: set[str] = set()
    golden = []
    for record in records:
        if record.target_label in seen or not record.tensor_path:
            continue
        tensor = load_landmark_npz(resolve_path(record.tensor_path, manifest_path.parent))
        sample = golden_sample(
            tensor,
            sample_id=record.sample_id,
            target_label=record.target_label,
            target_length=target_length,
            multiview_coords=mv_coords,
        )
        prediction = predict_with_artifact(artifact, tensor)
        sample["expected_label"] = prediction.label
        sample["expected_scores"] = prediction.scores
        golden.append(sample)
        seen.add(record.target_label)
        if len(golden) >= args.max_samples:
            break

    out_dir = Path(project_path(args.output_dir))
    _write_json(out_dir / "feature_contract.json", contract)
    _write_json(out_dir / "golden_samples.json", {"model_path": str(args.model_path), "samples": golden})
    print(f"feature_dim={contract['feature_dim']} | golden_samples={len(golden)} -> {out_dir}")


if __name__ == "__main__":
    main()
