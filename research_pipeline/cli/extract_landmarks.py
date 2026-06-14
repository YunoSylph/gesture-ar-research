from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from research_pipeline.data.manifest import read_jsonl, write_jsonl
from research_pipeline.data.synthetic import synthetic_landmarks
from research_pipeline.data.tensors import LandmarkTensor
from research_pipeline.data.tensors import save_landmark_npz
from research_pipeline.utils.errors import DependencyMissingError


def _frame_range_from_record(record, target_length: int) -> np.ndarray:
    fps = record.fps or 30.0
    start_frame = max(0, int(round(record.clip_start_ms * fps / 1000.0)))
    end_frame = max(start_frame, int(round(record.clip_end_ms * fps / 1000.0)))
    if end_frame == start_frame:
        return np.full((target_length,), start_frame, dtype=np.int64)
    return np.linspace(start_frame, end_frame, target_length).round().astype(np.int64)


def _extract_one_video_clip(video_path: str, frame_indices: np.ndarray, detector) -> LandmarkTensor:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    landmarks = np.zeros((len(frame_indices), 21, 3), dtype=np.float32)
    mask = np.zeros((len(frame_indices),), dtype=bool)
    confidence = np.zeros((len(frame_indices),), dtype=np.float32)
    handedness_score = np.zeros((len(frame_indices),), dtype=np.float32)

    for out_index, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(image)
        if not result.hand_landmarks:
            continue
        hand_index = 0
        if result.handedness:
            scores = [category[0].score for category in result.handedness if category]
            hand_index = int(np.argmax(scores))
            handedness_score[out_index] = float(scores[hand_index])
        else:
            handedness_score[out_index] = 1.0
        points = result.hand_landmarks[hand_index]
        landmarks[out_index] = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32)
        mask[out_index] = True
        confidence[out_index] = handedness_score[out_index]
    cap.release()
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=mask,
        frame_confidence=confidence,
        handedness_score=handedness_score,
        coord_space="image_normalized_xyz",
    )


def _new_hands_detector(model_asset_path: str):
    try:
        import cv2  # noqa: F401
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise DependencyMissingError(
            "MediaPipe/OpenCV extraction requires mediapipe and opencv-python. "
            "Use --backend synthetic for smoke data or install requirements/windows-train.txt."
        ) from exc
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_asset_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    return vision.HandLandmarker.create_from_options(options)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical [T,21,3] landmark NPZ shards.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest")
    parser.add_argument("--backend", choices=["auto", "synthetic", "mediapipe"], default="auto")
    parser.add_argument("--target-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split")
    parser.add_argument("--model-asset", default="models/mediapipe/hand_landmarker.task")
    parser.add_argument("--resume", action="store_true", help="Reuse existing NPZ shards when present.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    if args.split:
        records = [record for record in records if record.split_group == args.split]
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    output_manifest = Path(args.output_manifest) if args.output_manifest else Path(args.manifest).with_name(
        f"{Path(args.manifest).stem}_landmarks.jsonl"
    )
    hands = None
    if args.backend in {"auto", "mediapipe"}:
        needs_mediapipe = args.backend == "mediapipe" or any(
            record.raw_video_path and Path(record.raw_video_path).exists() for record in records
        )
        if needs_mediapipe:
            hands = _new_hands_detector(args.model_asset)

    for index, record in enumerate(records, start=1):
        tensor_path = output_dir / f"{record.sample_id}.npz"
        record.tensor_path = os.path.relpath(tensor_path, output_manifest.parent)
        if args.resume and tensor_path.exists():
            if index % args.progress_every == 0:
                print(f"reused {index}/{len(records)} clips")
            continue
        backend = args.backend
        if backend == "auto":
            backend = "mediapipe" if record.raw_video_path and Path(record.raw_video_path).exists() else "synthetic"
        if backend == "synthetic":
            tensor = synthetic_landmarks(
                record.target_label,
                length=args.target_length,
                seed=args.seed,
                sample_id=record.sample_id,
            )
        else:
            if hands is None:
                hands = _new_hands_detector(args.model_asset)
            tensor = _extract_one_video_clip(
                record.raw_video_path,
                _frame_range_from_record(record, args.target_length),
                hands,
            )
        save_landmark_npz(tensor_path, tensor, sample_id=record.sample_id, target_label=record.target_label)
        if index % args.progress_every == 0:
            print(f"extracted {index}/{len(records)} clips")
    if hands is not None:
        hands.close()
    write_jsonl(output_manifest, records)
    print(f"wrote {len(records)} tensors to {output_dir}")
    print(f"wrote updated manifest to {output_manifest}")


if __name__ == "__main__":
    main()
