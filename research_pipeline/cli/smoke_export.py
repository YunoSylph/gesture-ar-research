from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.models.artifacts import save_artifact


def main() -> None:
    root = Path("artifacts/smoke/export")
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "c0_rule.pkl"
    save_artifact(
        model_path,
        {
            "model_type": "c0_rule",
            "target_length": 32,
            "seed": 13,
            "params": {},
        },
    )
    reports = {
        "onnx": {
            "stage": "onnx_export",
            "status": "contract_written",
            "model_path": str(model_path),
            "notes": "Smoke verifies export metadata path without requiring torch/onnx.",
        },
        "coreml": {
            "stage": "coreml_export",
            "status": "deferred_on_windows",
            "notes": "Core ML conversion remains a macOS/Linux portability stage.",
        },
    }
    for name, payload in reports.items():
        path = root / f"{name}_contract.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("smoke_export ok: wrote ONNX/CoreML contract reports")


if __name__ == "__main__":
    main()

