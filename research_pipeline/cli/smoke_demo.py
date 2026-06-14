from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from research_pipeline.evaluation.interaction import ReplayFrame, compute_interaction_metrics, replay_predictions
from research_pipeline.interaction.fsm import ContextPolicyConfig


def main() -> None:
    root = Path("artifacts/smoke/demo")
    root.mkdir(parents=True, exist_ok=True)
    frames = [
        ReplayFrame(0, "no_gesture", 0.95),
        ReplayFrame(100, "swipe_right", 0.74),
        ReplayFrame(200, "swipe_right", 0.81, "navigate_next"),
        ReplayFrame(520, "no_gesture", 0.91),
        ReplayFrame(650, "click_2f", 0.70),
        ReplayFrame(760, "click_2f", 0.83, "select_confirm"),
    ]
    timeline = root / "interaction_timeline.jsonl"
    with timeline.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(asdict(frame), sort_keys=True))
            handle.write("\n")
    events = replay_predictions(frames, ContextPolicyConfig(stable_frames=2, cooldown_ms=250))
    metrics = compute_interaction_metrics(frames, events)
    report = root / "interaction_report.json"
    with report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump({"interaction": metrics, "events": [asdict(event) for event in events]}, handle, indent=2)
        handle.write("\n")
    print(f"smoke_demo ok: events={len(events)} precision={metrics['action_precision']:.3f}")


if __name__ == "__main__":
    main()
