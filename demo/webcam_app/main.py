from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from research_pipeline.evaluation.interaction import ReplayFrame, compute_interaction_metrics, replay_predictions
from research_pipeline.interaction.fsm import ContextPolicyConfig
from research_pipeline.utils.errors import DependencyMissingError


def _run_prerecorded(config: dict) -> None:
    timeline_path = Path(config["timeline"])
    frames: list[ReplayFrame] = []
    with timeline_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            frames.append(
                ReplayFrame(
                    timestamp_ms=int(payload["timestamp_ms"]),
                    label=payload["label"],
                    confidence=float(payload.get("confidence", 1.0)),
                    expected_action=payload.get("expected_action", ""),
                )
            )
    events = replay_predictions(frames, ContextPolicyConfig(**config.get("policy", {})))
    metrics = compute_interaction_metrics(frames, events)
    print(f"prerecorded demo: frames={len(frames)} events={len(events)} success={metrics['task_success_rate']:.3f}")


def _run_webcam(_config: dict) -> None:
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise DependencyMissingError(
            "Webcam demo requires opencv-python and mediapipe. "
            "Install requirements/windows-train.txt or use mode: prerecorded."
        ) from exc
    raise NotImplementedError("Live webcam UI hook is reserved for the dataset/model integration stage.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop gesture demo runner.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    mode = config.get("mode", "webcam")
    if mode == "prerecorded":
        _run_prerecorded(config)
    elif mode == "webcam":
        _run_webcam(config)
    else:
        raise ValueError(f"Unsupported demo mode '{mode}'.")


if __name__ == "__main__":
    main()

