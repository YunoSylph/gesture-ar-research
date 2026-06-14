from __future__ import annotations

import argparse
from pathlib import Path

from research_pipeline.data.manifest import read_capture_csv, write_jsonl
from research_pipeline.data.schema import ManifestRecord, REQUIRED_MANIFEST_FIELDS
from research_pipeline.labels import validate_target_label


def build_local_manifest_records(manifest_csv: str | Path, video_dir: str | Path) -> list[ManifestRecord]:
    video_dir = Path(video_dir)
    records: list[ManifestRecord] = []
    known_csv_fields = set(REQUIRED_MANIFEST_FIELDS) | {"file_name", "video_path", "label"}
    for index, row in enumerate(read_capture_csv(manifest_csv), start=1):
        label = row.get("target_label") or row.get("label") or ""
        validate_target_label(label)
        file_name = row.get("video_path") or row.get("file_name") or row.get("raw_video_path") or ""
        raw_video_path = str(Path(file_name) if Path(file_name).is_absolute() else video_dir / file_name)
        sample_id = row.get("sample_id") or f"local_{index:04d}_{label}"
        extras = {key: value for key, value in row.items() if key not in known_csv_fields and value not in {None, ""}}
        extras.setdefault("capture_domain", row.get("capture_domain") or "phone_rear_ar")
        extras.setdefault("camera_view", row.get("camera_view") or "rear_world")
        extras.setdefault("coordinate_semantics", row.get("coordinate_semantics") or "screen_space")
        extras.setdefault("adaptation_role", row.get("adaptation_role") or "target_domain")
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                source_dataset="local_phone",
                public_label=row.get("public_label") or label,
                target_label=label,
                participant_id=row.get("participant_id") or "local_user",
                session_id=row.get("session_id") or "session_01",
                repetition_id=row.get("repetition_id") or str(index),
                split_group=row.get("split_group") or "local",
                hand_recorded=row.get("hand_recorded") or "unknown",
                fps=float(row.get("fps") or 30.0),
                width=int(float(row.get("width") or 0)),
                height=int(float(row.get("height") or 0)),
                camera_device=row.get("camera_device") or "phone",
                background_tag=row.get("background_tag") or "unknown",
                lighting_tag=row.get("lighting_tag") or "unknown",
                clip_start_ms=int(float(row.get("clip_start_ms") or 0)),
                clip_end_ms=int(float(row.get("clip_end_ms") or 0)),
                raw_video_path=raw_video_path,
                notes=row.get("notes") or "generated_by_ingest_local_videos",
                extras=extras,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a local capture CSV into canonical JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="CSV with at least file_name/video_path and target_label.")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = build_local_manifest_records(args.manifest, args.video_dir)
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
