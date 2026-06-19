from __future__ import annotations

import argparse
from pathlib import Path

from research_pipeline.data.manifest import write_jsonl
from research_pipeline.data.schema import ManifestRecord
from research_pipeline.labels import remap_jester_label


# 20BN-Jester is recorded at 12 fps; each video_id is one short clip stored as a
# folder of zero-padded JPEG frames (<frames_root>/<video_id>/00001.jpg ...).
JESTER_FPS = 12.0
FRAME_GLOB = "*.jpg"
# OpenCV opens such a folder as an image sequence via a printf pattern, so the
# existing extract_landmarks (cv2.VideoCapture + CAP_PROP_POS_FRAMES) reuses
# unchanged once raw_video_path points at this pattern.
FRAME_PATTERN = "%05d.jpg"


def _read_split_csv(path: Path) -> list[tuple[str, str]]:
    """Read a Jester ``video_id;label`` split file (tolerates comma-separated)."""

    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.split(";") if ";" in text else text.split(",")
            if len(parts) < 2:
                continue
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def _frame_span(frames_root: Path | None, video_id: str, fps: float) -> tuple[str, int]:
    """Return (raw_video_path pattern, clip_end_ms) when the frame folder exists."""

    if frames_root is None:
        return "", 0
    folder = frames_root / video_id
    if not folder.is_dir():
        return "", 0
    frame_count = sum(1 for _ in folder.glob(FRAME_GLOB))
    clip_end_ms = round(frame_count * 1000 / fps) if frame_count else 0
    return str(folder / FRAME_PATTERN), clip_end_ms


def build_jester_manifest(
    annotations_dir: Path,
    frames_root: Path | None = None,
    *,
    include_motion_equivalents: bool = False,
    fold_non_target_as_no_gesture: bool = False,
    fps: float = JESTER_FPS,
) -> list[ManifestRecord]:
    """Build canonical manifest records from Jester annotation split files.

    The Jester ``validation`` split is the labelled evaluation split (the real
    ``test`` split ships without labels), so it is mapped to ``split_group=test``
    to match how the rest of the pipeline consumes held-out data.
    """

    split_files = (
        ("train", annotations_dir / "jester-v1-train.csv"),
        ("test", annotations_dir / "jester-v1-validation.csv"),
    )
    records: list[ManifestRecord] = []
    for split, path in split_files:
        if not path.exists():
            continue
        for video_id, public_label in _read_split_csv(path):
            target_label = remap_jester_label(
                public_label,
                include_motion_equivalents=include_motion_equivalents,
                fold_non_target_as_no_gesture=fold_non_target_as_no_gesture,
            )
            if target_label is None:
                continue
            raw_video_path, clip_end_ms = _frame_span(frames_root, video_id, fps)
            records.append(
                ManifestRecord(
                    sample_id=f"jester_{video_id}",
                    source_dataset="jester",
                    public_label=public_label,
                    target_label=target_label,
                    participant_id="unknown",
                    session_id="jester",
                    repetition_id=str(video_id),
                    split_group=split,
                    fps=fps,
                    clip_start_ms=0,
                    clip_end_ms=clip_end_ms,
                    raw_video_path=raw_video_path,
                    camera_device="jester_webcam",
                    notes="generated_by_build_jester_manifest",
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a project manifest from 20BN-Jester annotations.")
    parser.add_argument("--annotations-dir", required=True, help="Dir with jester-v1-{train,validation}.csv.")
    parser.add_argument("--frames-root", help="Root of Jester frame folders (<root>/<video_id>/00001.jpg).")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-motion-equivalents",
        action="store_true",
        help="Fold two-finger slide / full-hand zoom into swipe/zoom targets.",
    )
    parser.add_argument(
        "--fold-non-target-as-no-gesture",
        action="store_true",
        help="Map non-command Jester gestures to no_gesture hard negatives.",
    )
    args = parser.parse_args()
    records = build_jester_manifest(
        Path(args.annotations_dir),
        Path(args.frames_root) if args.frames_root else None,
        include_motion_equivalents=args.include_motion_equivalents,
        fold_non_target_as_no_gesture=args.fold_non_target_as_no_gesture,
    )
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
