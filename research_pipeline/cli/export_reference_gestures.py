from __future__ import annotations

import argparse
import csv
from pathlib import Path

from research_pipeline.cli.common import project_path
from research_pipeline.data.manifest import read_jsonl
from research_pipeline.labels import FINAL_GESTURES, TARGET_LABELS


def _duration_ms(record) -> int:
    return int(record.clip_end_ms - record.clip_start_ms)


def _source_path(record) -> Path:
    return project_path(record.raw_video_path)


def _select_records(records, *, labels: list[str], per_label: int, min_ms: int, max_ms: int, preferred_ms: int):
    selected = {}
    for label in labels:
        candidates = [
            record
            for record in records
            if record.target_label == label
            and record.raw_video_path
            and _source_path(record).exists()
            and min_ms <= _duration_ms(record) <= max_ms
        ]
        candidates.sort(key=lambda item: (abs(_duration_ms(item) - preferred_ms), item.participant_id, item.sample_id))
        output = []
        used_participants: set[str] = set()
        for record in candidates:
            if record.participant_id in used_participants:
                continue
            output.append(record)
            used_participants.add(record.participant_id)
            if len(output) >= per_label:
                break
        if len(output) < per_label:
            for record in candidates:
                if record in output:
                    continue
                output.append(record)
                if len(output) >= per_label:
                    break
        selected[label] = output
    return selected


def _write_clip(record, output_path: Path, *, pad_ms: int) -> tuple[int, int]:
    import cv2

    source = _source_path(record)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or record.fps or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or record.width or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or record.height or 480)
    start_ms = max(0, int(record.clip_start_ms) - pad_ms)
    end_ms = max(start_ms + 1, int(record.clip_end_ms) + pad_ms)
    start_frame = max(0, int(round(start_ms * fps / 1000.0)))
    end_frame = max(start_frame + 1, int(round(end_ms * fps / 1000.0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    current = start_frame
    while current <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1
        current += 1

    writer.release()
    cap.release()
    if written == 0:
        raise RuntimeError(f"No frames were written for {record.sample_id}")
    return start_ms, end_ms


def _write_readme(output_dir: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Local Gesture Reference Clips",
        "",
        "These clips are duplicated from the IPN Hand subset and define the exact gesture semantics for local recording.",
        "Record local clips with the same target labels and the same gesture meaning. Do not introduce new gesture variants.",
        "",
        "Target labels:",
        "",
    ]
    for item in FINAL_GESTURES:
        lines.append(f"- `{item.target_label}`: {item.semantics}; IPN source class `{item.ipn_name}`; interaction `{item.interaction}`.")
    lines.extend(["", "Generated clips:", ""])
    for row in rows:
        lines.append(f"- `{row['target_label']}`: `{row['clip_path']}` from `{row['sample_id']}`")
    lines.extend(
        [
            "",
            "Recommended local capture: 5-10 clips per target label, 2-4 seconds each, one gesture per clip.",
            "Keep the hand fully visible and use the same label names as above.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export short IPN reference clips for local gesture capture.")
    parser.add_argument("--manifest", default="data/interim/manifests/ipn_subset.jsonl")
    parser.add_argument("--output-dir", default="data/reference_gestures/ipn_hand")
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--min-ms", type=int, default=700)
    parser.add_argument("--max-ms", type=int, default=6500)
    parser.add_argument("--preferred-ms", type=int, default=2200)
    parser.add_argument("--pad-ms", type=int, default=200)
    parser.add_argument("--labels", nargs="*", default=list(TARGET_LABELS))
    args = parser.parse_args()

    manifest_path = project_path(args.manifest)
    output_dir = project_path(args.output_dir)
    records = read_jsonl(manifest_path)
    selected = _select_records(
        records,
        labels=args.labels,
        per_label=args.per_label,
        min_ms=args.min_ms,
        max_ms=args.max_ms,
        preferred_ms=args.preferred_ms,
    )

    rows: list[dict[str, str]] = []
    for label, label_records in selected.items():
        if not label_records:
            print(f"warning: no reference candidates for {label}")
            continue
        for index, record in enumerate(label_records, start=1):
            output_path = output_dir / label / f"{label}_ref_{index:02d}.mp4"
            start_ms, end_ms = _write_clip(record, output_path, pad_ms=args.pad_ms)
            rows.append(
                {
                    "target_label": label,
                    "clip_path": str(output_path.relative_to(output_dir)),
                    "sample_id": record.sample_id,
                    "public_label": record.public_label,
                    "participant_id": record.participant_id,
                    "source_video": record.raw_video_path,
                    "clip_start_ms": str(start_ms),
                    "clip_end_ms": str(end_ms),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reference_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_label",
                "clip_path",
                "sample_id",
                "public_label",
                "participant_id",
                "source_video",
                "clip_start_ms",
                "clip_end_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    _write_readme(output_dir, rows)
    print(f"exported={len(rows)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
