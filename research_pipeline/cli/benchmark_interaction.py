from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.evaluation.interaction import ReplayFrame, compute_interaction_metrics, replay_predictions
from research_pipeline.interaction.fsm import ContextPolicyConfig


def _load_frames(path: Path) -> list[ReplayFrame]:
    frames: list[ReplayFrame] = []
    with path.open("r", encoding="utf-8") as handle:
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
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-aware interaction replay benchmark.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    frames = _load_frames(project_path(config["timeline"]))
    policy = ContextPolicyConfig(**config.get("policy", {}))
    events = replay_predictions(frames, policy)
    report = {
        "interaction": compute_interaction_metrics(frames, events),
        "events": [asdict(event) for event in events],
        "config": config,
    }
    write_json_report(config.get("output_report", "artifacts/reports/interaction.json"), report)
    print(f"events={len(events)} precision={report['interaction']['action_precision']:.4f}")


if __name__ == "__main__":
    main()
