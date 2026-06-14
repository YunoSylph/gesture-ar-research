from __future__ import annotations

import argparse

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.evaluation.recognition import benchmark_recognition_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recognition benchmark and write JSON metrics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    metrics, latency = benchmark_recognition_manifest(
        project_path(config["manifest"]),
        model_path=project_path(config["model_path"]) if config.get("model_path") else None,
        variant=config.get("variant", "artifact"),
        target_length=int(config.get("target_length", 32)),
    )
    report = {"recognition": metrics.to_dict(), "latency": latency, "config": config}
    write_json_report(config.get("output_report", "artifacts/reports/recognition.json"), report)
    print(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")


if __name__ == "__main__":
    main()

