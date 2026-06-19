from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from research_pipeline.labels import TARGET_LABELS
from research_pipeline.utils.errors import DependencyMissingError, PipelineError


def _require_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - exercised only without coremltools
        raise DependencyMissingError(
            "CoreML export requires coremltools (install requirements/macos-arm64.txt on macOS/Linux)."
        ) from exc
    return ct


def convert_tcn_artifact_to_coreml(
    artifact: dict[str, Any],
    output_path: str | Path,
    *,
    input_name: str = "landmarks",
    minimum_deployment_target: str = "iOS15",
    compute_precision: str = "FLOAT16",
    example_input: np.ndarray | None = None,
) -> dict[str, Any]:
    """Convert a trained C1-T TCN artifact into a Core ML .mlpackage for iPhone.

    The exported model takes the project's preprocessed feature window
    ([1, target_length, input_dim]) and returns class logits; the landmark ->
    feature preprocessing is reproduced on-device (see export_mobile_bundle's
    preprocessing_contract). Returns conversion metadata including a numerical
    parity check between the PyTorch and Core ML outputs.
    """

    if artifact.get("model_type") != "c1t_tcn_torch":
        raise PipelineError(
            f"CoreML export supports c1t_tcn_torch artifacts, got '{artifact.get('model_type')}'."
        )

    import torch

    from research_pipeline.models.tcn import TCNConfig, build_tcn

    ct = _require_coremltools()

    config = TCNConfig(**artifact["tcn_config"])
    model = build_tcn(config)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    target_length = int(artifact.get("target_length", 32))
    input_dim = int(config.input_dim)
    if example_input is None:
        example_input = np.random.default_rng(0).standard_normal((1, target_length, input_dim)).astype(np.float32)
    example_input = np.asarray(example_input, dtype=np.float32)
    if example_input.shape != (1, target_length, input_dim):
        raise PipelineError(
            f"example_input shape {example_input.shape} != expected (1, {target_length}, {input_dim})."
        )

    example_t = torch.from_numpy(example_input)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_t)
        # Freeze to inline parameters/buffers. Without this, identical constants
        # in the graph (e.g. equal BatchNorm num_batches_tracked across layers)
        # get deduplicated by TorchScript and trip a coremltools lowering
        # assertion ("tensor value not consistent between torch ir and
        # state_dict") for some trained weights, e.g. MPS-trained models.
        traced = torch.jit.freeze(traced.eval())
        torch_logits = model(example_t).cpu().numpy().reshape(-1)

    precision = (
        ct.precision.FLOAT16 if str(compute_precision).upper() == "FLOAT16" else ct.precision.FLOAT32
    )
    deployment = getattr(ct.target, minimum_deployment_target)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=(1, target_length, input_dim), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=deployment,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))

    prediction = mlmodel.predict({input_name: example_input})
    output_name = next(iter(prediction))
    coreml_logits = np.asarray(prediction[output_name], dtype=np.float32).reshape(-1)

    labels = list(artifact.get("labels", TARGET_LABELS))
    parity = {
        "output_name": output_name,
        "torch_argmax_label": labels[int(np.argmax(torch_logits))] if labels else int(np.argmax(torch_logits)),
        "argmax_match": bool(np.argmax(torch_logits) == np.argmax(coreml_logits)),
        "max_abs_logit_diff": float(np.max(np.abs(torch_logits - coreml_logits))),
    }
    return {
        "output_path": str(output_path),
        "input_name": input_name,
        "input_shape": [1, target_length, input_dim],
        "labels": labels,
        "minimum_deployment_target": minimum_deployment_target,
        "compute_precision": compute_precision,
        "parity": parity,
    }
