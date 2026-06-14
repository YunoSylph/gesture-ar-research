from __future__ import annotations

import time
from pathlib import Path

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import load_landmark_npz
from research_pipeline.evaluation.metrics import RecognitionMetrics, compute_recognition_metrics
from research_pipeline.models.artifacts import load_artifact, predict_with_artifact
from research_pipeline.models.rule_based import RuleBasedRecognizer
from research_pipeline.utils.errors import SchemaError


def benchmark_recognition_manifest(
    manifest_path: str | Path,
    *,
    model_path: str | Path | None = None,
    variant: str = "artifact",
    target_length: int = 32,
) -> tuple[RecognitionMetrics, dict]:
    records = read_jsonl(manifest_path)
    base_dir = Path(manifest_path).parent
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []
    artifact = load_artifact(model_path) if model_path else None
    rule = RuleBasedRecognizer()

    for record in records:
        if not record.tensor_path:
            raise SchemaError(f"Record '{record.sample_id}' has no tensor_path.")
        tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
        start = time.perf_counter()
        if variant == "c0" or artifact is None:
            prediction = rule.predict(tensor)
        else:
            prediction = predict_with_artifact(artifact, tensor)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        y_true.append(record.target_label)
        y_pred.append(prediction.label)

    metrics = compute_recognition_metrics(y_true, y_pred)
    latency_report = {
        "offline_latency_ms_median": _percentile(latencies_ms, 50),
        "offline_latency_ms_p95": _percentile(latencies_ms, 95),
        "num_samples": len(records),
        "target_length": target_length,
    }
    return metrics, latency_report


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    import numpy as np

    return float(np.percentile(values, q))

