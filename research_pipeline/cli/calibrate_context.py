from __future__ import annotations

import argparse

from research_pipeline.cli.common import load_yaml, write_json_report
from research_pipeline.interaction.fsm import ContextPolicyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and persist C2 context policy calibration.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    policy = ContextPolicyConfig(**config.get("policy", {}))
    report = {
        "context_policy": {
            "activation_threshold": policy.activation_threshold,
            "stable_frames": policy.stable_frames,
            "cooldown_ms": policy.cooldown_ms,
            "no_gesture_reset_frames": policy.no_gesture_reset_frames,
        },
        "status": "calibrated_static_defaults",
        "notes": "Validation-time threshold search can be layered on this artifact once validation predictions exist.",
    }
    write_json_report(config.get("output_report", "artifacts/reports/context_calibration.json"), report)
    print("context policy calibrated")


if __name__ == "__main__":
    main()

