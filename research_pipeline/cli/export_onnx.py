from __future__ import annotations

import argparse
import platform

from research_pipeline.cli.common import load_yaml, project_path, write_json_report
from research_pipeline.models.artifacts import load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX or write an export contract report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_path = project_path(config["model_path"])
    output_path = project_path(config.get("output_path", "artifacts/export/model.onnx"))
    artifact = load_artifact(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if artifact["model_type"] == "c1t_tcn_torch":
        import torch

        from research_pipeline.models.tcn import TCNConfig, build_tcn

        tcn_config = TCNConfig(**artifact["tcn_config"])
        model = build_tcn(tcn_config)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        dummy = torch.zeros(
            1,
            int(artifact.get("target_length", 32)),
            int(tcn_config.input_dim),
            dtype=torch.float32,
        )
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=[config.get("input_name", "landmarks")],
            output_names=["logits"],
            dynamic_axes={
                config.get("input_name", "landmarks"): {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=int(config.get("opset_version", 18)),
        )
        try:
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            checked = True
        except Exception:
            checked = False
        report = {
            "stage": "onnx_export",
            "status": "exported",
            "model_type": artifact["model_type"],
            "model_path": str(model_path),
            "output_path": str(output_path),
            "platform": platform.platform(),
            "onnx_checked": checked,
        }
        write_json_report(str(output_path) + ".contract.json", report)
        print(f"exported ONNX model to {output_path}")
        return

    report = {
        "stage": "onnx_export",
        "status": "contract_written",
        "model_type": artifact["model_type"],
        "model_path": str(model_path),
        "output_path": str(output_path),
        "platform": platform.platform(),
        "notes": (
            "Prototype/sklearn artifacts are not ONNX graphs. Train the torch TCN backend to emit a real .onnx model."
        ),
    }
    write_json_report(str(output_path) + ".contract.json", report)
    print(f"wrote ONNX export contract to {output_path}.contract.json")


if __name__ == "__main__":
    main()
