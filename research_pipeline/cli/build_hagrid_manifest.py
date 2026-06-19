from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.labels import remap_hagrid_label


# HaGRID ships one annotation JSON per gesture class (<gesture>.json), each keyed
# by image id; images live under <images_root>/<gesture>/<image_id>.jpg. Records
# are tagged as static poses so downstream extraction replicates the detected
# pose across the temporal window.
def build_hagrid_manifest(
    annotations_dir: Path,
    images_root: Path | None = None,
    *,
    split: str = "train",
    fold_non_target_as_no_gesture: bool = False,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for json_path in sorted(annotations_dir.glob("*.json")):
        gesture = json_path.stem
        target_label = remap_hagrid_label(
            gesture, fold_non_target_as_no_gesture=fold_non_target_as_no_gesture
        )
        if target_label is None:
            continue
        entries = json.loads(json_path.read_text(encoding="utf-8"))
        for image_id, entry in entries.items():
            user_id = entry.get("user_id", "unknown") if isinstance(entry, dict) else "unknown"
            raw_video_path = str(images_root / gesture / f"{image_id}.jpg") if images_root else ""
            records.append(
                ManifestRecord(
                    sample_id=f"hagrid_{gesture}_{image_id}",
                    source_dataset="hagrid",
                    public_label=gesture,
                    target_label=target_label,
                    participant_id=str(user_id or "unknown"),
                    session_id="hagrid",
                    repetition_id=str(image_id),
                    split_group=split,
                    fps=30.0,
                    clip_start_ms=0,
                    clip_end_ms=0,
                    raw_video_path=raw_video_path,
                    camera_device="hagrid",
                    notes="generated_by_build_hagrid_manifest;static_pose",
                    extras={"capture_domain": "hagrid_static", "static_pose": True},
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a project manifest from HaGRID annotations.")
    parser.add_argument("--annotations-dir", required=True, help="Dir with per-gesture HaGRID <gesture>.json files.")
    parser.add_argument("--images-root", help="Root with <gesture>/<image_id>.jpg images.")
    parser.add_argument("--split", default="train", choices=["train", "test", "local"])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fold-non-target-as-no-gesture",
        action="store_true",
        help="Map non-target HaGRID poses to no_gesture static hard negatives.",
    )
    args = parser.parse_args()
    records = build_hagrid_manifest(
        Path(args.annotations_dir),
        Path(args.images_root) if args.images_root else None,
        split=args.split,
        fold_non_target_as_no_gesture=args.fold_non_target_as_no_gesture,
    )
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
