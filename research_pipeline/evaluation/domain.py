from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import ManifestRecord


def summarize_domain_manifests(manifest_paths: list[str | Path]) -> dict[str, Any]:
    records_by_manifest: dict[str, list[ManifestRecord]] = {}
    all_records: list[ManifestRecord] = []
    missing_raw_videos: list[dict[str, str]] = []

    for manifest_path in manifest_paths:
        path = Path(manifest_path)
        records = read_jsonl(path)
        records_by_manifest[str(path)] = records
        all_records.extend(records)
        for record in records:
            if record.source_dataset != "local_phone" or not record.raw_video_path:
                continue
            if not _path_exists(record.raw_video_path, path.parent):
                missing_raw_videos.append(
                    {
                        "sample_id": record.sample_id,
                        "target_label": record.target_label,
                        "raw_video_path": record.raw_video_path,
                    }
                )

    by_source = Counter(record.source_dataset for record in all_records)
    by_label = Counter(record.target_label for record in all_records)
    by_split = Counter(record.split_group for record in all_records)
    by_capture_domain = Counter(_capture_domain(record) for record in all_records)
    by_camera_view = Counter(_camera_view(record) for record in all_records)
    per_manifest = {
        manifest: {
            "records": len(records),
            "source_dataset": dict(Counter(record.source_dataset for record in records)),
            "target_label": dict(Counter(record.target_label for record in records)),
            "capture_domain": dict(Counter(_capture_domain(record) for record in records)),
        }
        for manifest, records in records_by_manifest.items()
    }
    local_records = [record for record in all_records if record.source_dataset == "local_phone"]

    return {
        "total_records": len(all_records),
        "by_source_dataset": dict(by_source),
        "by_target_label": dict(by_label),
        "by_split_group": dict(by_split),
        "by_capture_domain": dict(by_capture_domain),
        "by_camera_view": dict(by_camera_view),
        "per_manifest": per_manifest,
        "local_phone": {
            "planned_records": len(local_records),
            "missing_raw_video_count": len(missing_raw_videos),
            "ready_for_landmark_extraction": bool(local_records) and not missing_raw_videos,
            "pending_examples": missing_raw_videos[:20],
        },
        "domain_transfer_status": _domain_transfer_status(local_records, missing_raw_videos),
    }


def _capture_domain(record: ManifestRecord) -> str:
    return str(record.extras.get("capture_domain") or record.source_dataset)


def _camera_view(record: ManifestRecord) -> str:
    return str(record.extras.get("camera_view") or record.camera_device or "unknown")


def _path_exists(value: str, base_dir: Path) -> bool:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, base_dir / path]
    return any(candidate.exists() for candidate in candidates)


def _domain_transfer_status(local_records: list[ManifestRecord], missing_raw_videos: list[dict[str, str]]) -> str:
    if not local_records:
        return "public_only_no_local_plan"
    if missing_raw_videos:
        return "local_plan_ready_waiting_for_videos"
    return "local_videos_ready_for_extraction"
