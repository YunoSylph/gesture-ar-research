from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_pipeline.cli.common import load_yaml, project_path
from research_pipeline.interaction.contract import golden_validation_traces, validation_contract
from research_pipeline.interaction.fsm import ContextPolicyConfig


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the on-device acceptance-policy (TARC/validation) contract + golden decision traces."
    )
    parser.add_argument("--c2-config", help="Optional YAML with a 'policy' block overriding the defaults.")
    parser.add_argument("--output-dir", default="artifacts/mobile/validation")
    args = parser.parse_args()

    config = ContextPolicyConfig()
    if args.c2_config:
        policy_cfg = load_yaml(project_path(args.c2_config)).get("policy", {})
        config = ContextPolicyConfig(**policy_cfg)

    out_dir = Path(project_path(args.output_dir))
    contract = validation_contract(config)
    traces = golden_validation_traces(config)
    _write_json(out_dir / "validation_contract.json", contract)
    _write_json(out_dir / "golden_traces.json", {"scenarios": traces})
    total_events = sum(len(trace["events"]) for trace in traces)
    print(f"validation contract + {len(traces)} golden traces ({total_events} events) -> {out_dir}")


if __name__ == "__main__":
    main()
