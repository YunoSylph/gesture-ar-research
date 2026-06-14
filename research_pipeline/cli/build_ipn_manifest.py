from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.labels import remap_ipn_label


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def infer_public_label(path: Path) -> str:
    candidates = [path.stem, *[part for part in reversed(path.parts[-5:])]]
    for candidate in candidates:
        mapped = remap_ipn_label(candidate)
        if mapped is not None:
            return candidate
    match = re.search(r"(?:class|gesture|label)[_-]?(\d+)", path.as_posix(), flags=re.IGNORECASE)
    return match.group(1) if match else path.parent.name


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts or "training" in parts:
        return "train"
    if "test" in parts or "validation" in parts or "val" in parts:
        return "test"
    return "unknown"


def infer_participant(path: Path) -> str:
    match = re.search(r"(?:subject|subj|participant|p)[_-]?(\d+)", path.as_posix(), flags=re.IGNORECASE)
    return f"p{match.group(1)}" if match else "unknown"


def build_manifest(root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for video_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES):
        public_label = infer_public_label(video_path)
        target_label = remap_ipn_label(public_label)
        if target_label is None:
            continue
        sample_id = video_path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        records.append(
            ManifestRecord(
                sample_id=sample_id,
                source_dataset="ipn_hand",
                public_label=str(public_label),
                target_label=target_label,
                participant_id=infer_participant(video_path),
                session_id="ipn",
                repetition_id=video_path.stem,
                split_group=infer_split(video_path),
                raw_video_path=str(video_path),
                camera_device="ipn_hand",
                notes="generated_by_build_ipn_manifest",
            )
        )
    return records


def _metadata_by_video(annotations_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = annotations_dir / "metadata.csv"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["Video Name"]: row for row in csv.DictReader(handle)}


def _find_video_path(video_root: Path | None, video_name: str) -> str:
    if video_root is None:
        return ""
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        direct = video_root / f"{video_name}{suffix}"
        if direct.exists():
            return str(direct)
    matches = list(video_root.rglob(f"{video_name}.*"))
    return str(matches[0]) if matches else str(video_root / f"{video_name}.mp4")


def build_manifest_from_annotations(annotations_dir: Path, video_root: Path | None = None) -> list[ManifestRecord]:
    metadata = _metadata_by_video(annotations_dir)
    records: list[ManifestRecord] = []
    split_files = (("train", annotations_dir / "Annot_TrainList.txt"), ("test", annotations_dir / "Annot_TestList.txt"))
    for split, path in split_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                video_name, public_label, class_id, start_frame, end_frame, duration = [
                    item.strip() for item in text.split(",")
                ]
                target_label = remap_ipn_label(public_label)
                if target_label is None:
                    continue
                meta = metadata.get(video_name, {})
                sample_id = f"{video_name}_{int(start_frame):06d}_{int(end_frame):06d}_{public_label}"
                fps = 30.0
                clip_start_ms = round(int(start_frame) * 1000 / fps)
                clip_end_ms = round(int(end_frame) * 1000 / fps)
                hand = (meta.get("Hand") or "unknown").strip().lower()
                if hand not in {"left", "right"}:
                    hand = "unknown"
                records.append(
                    ManifestRecord(
                        sample_id=sample_id,
                        source_dataset="ipn_hand",
                        public_label=public_label,
                        target_label=target_label,
                        participant_id=video_name.split("_")[0],
                        session_id=video_name,
                        repetition_id=f"{start_frame}-{end_frame}",
                        split_group=split,
                        hand_recorded=hand,
                        fps=fps,
                        width=640,
                        height=480,
                        camera_device="ipn_hand",
                        background_tag=(meta.get("Background") or "unknown").strip().lower(),
                        lighting_tag=(meta.get("Illumination") or "unknown").strip().lower(),
                        clip_start_ms=clip_start_ms,
                        clip_end_ms=clip_end_ms,
                        raw_video_path=_find_video_path(video_root, video_name),
                        notes=f"annotation_file={path.name};line={line_number};class_id={class_id};duration_frames={duration}",
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a project manifest from an IPN Hand video root.")
    parser.add_argument("--root")
    parser.add_argument("--annotations-dir")
    parser.add_argument("--video-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.annotations_dir:
        records = build_manifest_from_annotations(
            Path(args.annotations_dir),
            Path(args.video_root) if args.video_root else (Path(args.root) if args.root else None),
        )
    elif args.root:
        records = build_manifest(Path(args.root))
    else:
        raise SystemExit("Provide --annotations-dir or --root.")
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
