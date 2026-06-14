from __future__ import annotations

import argparse
import platform

from research_pipeline.cli.common import load_yaml, project_path, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Core ML conversion stage metadata.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_path = project_path(config.get("model_path", "artifacts/models/ipn_c1t_tcn_full.pkl"))
    output_path = project_path(config.get("output_path", "artifacts/export/GestureClassifier.mlpackage"))
    is_windows = platform.system().lower() == "windows"
    report = {
        "stage": "coreml_export",
        "status": "deferred_on_windows" if is_windows else "ready_for_conversion",
        "minimum_deployment_target": config.get("minimum_deployment_target", "iOS15"),
        "convert_to": config.get("convert_to", "mlprogram"),
        "compute_precision": config.get("compute_precision", "FLOAT16"),
        "model_path": str(model_path),
        "input_name": config.get("input_name", "landmarks"),
        "input_shape": [1, int(config.get("target_length", 32)), 74],
        "output_path": str(output_path),
        "notes": "Core ML conversion is intentionally separated from the Windows training mainline.",
    }
    write_json_report(str(output_path) + ".contract.json", report)
    print(report["status"])


if __name__ == "__main__":
    main()
