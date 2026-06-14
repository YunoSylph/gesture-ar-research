from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.evaluation.error_analysis import analyze_recognition_risk
from research_pipeline.evaluation.metrics import compute_recognition_metrics
from research_pipeline.models.common import Prediction


@dataclass(slots=True)
class PerturbationConfig:
    name: str
    kind: str = "clean"
    sigma: float = 0.0
    drop_rate: float = 0.0
    mask_rate: float = 0.0
    jitter: int = 0
    translation: float = 0.0
    scale: float = 0.0


Recognizer = Callable[[LandmarkTensor], Prediction]


def benchmark_robustness_manifest(
    manifest_path: str | Path,
    recognizers: dict[str, Recognizer],
    scenarios: list[PerturbationConfig],
    *,
    seed: int = 42,
    max_records: int | None = None,
) -> dict:
    records = read_jsonl(manifest_path)
    if max_records is not None:
        records = records[:max_records]
    base_dir = Path(manifest_path).parent
    report: dict[str, dict] = {}

    for scenario_index, scenario in enumerate(scenarios):
        y_true: list[str] = []
        predictions: dict[str, list[str]] = {name: [] for name in recognizers}
        rng = np.random.default_rng(seed + scenario_index * 997)

        for record in records:
            tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
            perturbed = perturb_tensor(tensor, scenario, rng)
            y_true.append(record.target_label)
            for name, recognizer in recognizers.items():
                predictions[name].append(recognizer(perturbed).label)

        scenario_report: dict[str, dict] = {}
        for name, y_pred in predictions.items():
            metrics = compute_recognition_metrics(y_true, y_pred)
            payload = {"recognition": metrics.to_dict()}
            scenario_report[name] = {
                "recognition": metrics.to_dict(),
                "risk": analyze_recognition_risk(payload),
            }
        report[scenario.name] = scenario_report
    return report


def perturb_tensor(tensor: LandmarkTensor, config: PerturbationConfig, rng: np.random.Generator) -> LandmarkTensor:
    landmarks = tensor.landmarks.astype(np.float32).copy()
    sequence_mask = tensor.sequence_mask.astype(bool).copy()
    frame_confidence = tensor.frame_confidence.astype(np.float32).copy()
    handedness = tensor.handedness_score.astype(np.float32).copy()
    world = tensor.world_landmarks.astype(np.float32).copy() if tensor.world_landmarks is not None else None

    if config.kind == "clean":
        pass
    elif config.kind == "gaussian_noise":
        landmarks += rng.normal(0.0, config.sigma, size=landmarks.shape).astype(np.float32)
    elif config.kind == "frame_drop":
        drops = rng.random(landmarks.shape[0]) < config.drop_rate
        sequence_mask[drops] = False
        frame_confidence[drops] *= 0.1
    elif config.kind == "landmark_mask":
        mask = rng.random(landmarks.shape[:2]) < config.mask_rate
        wrist = landmarks[:, 0:1, :]
        landmarks[mask] = np.repeat(wrist, landmarks.shape[1], axis=1)[mask]
        frame_confidence *= np.float32(max(0.0, 1.0 - config.mask_rate * 0.6))
    elif config.kind == "temporal_jitter":
        idx = np.arange(landmarks.shape[0])
        offsets = rng.integers(-config.jitter, config.jitter + 1, size=idx.shape[0])
        jittered = np.clip(idx + offsets, 0, landmarks.shape[0] - 1)
        landmarks = landmarks[jittered]
        sequence_mask = sequence_mask[jittered]
        frame_confidence = frame_confidence[jittered]
        if handedness.shape == (landmarks.shape[0],):
            handedness = handedness[jittered]
        if world is not None:
            world = world[jittered]
    elif config.kind == "translation":
        delta = rng.normal(0.0, config.translation, size=(1, 1, 2)).astype(np.float32)
        landmarks[:, :, :2] += delta
    elif config.kind == "scale":
        factor = np.float32(1.0 + rng.normal(0.0, config.scale))
        wrist = landmarks[:, 0:1, :]
        landmarks = wrist + (landmarks - wrist) * factor
    elif config.kind == "combined":
        landmarks += rng.normal(0.0, config.sigma, size=landmarks.shape).astype(np.float32)
        drops = rng.random(landmarks.shape[0]) < config.drop_rate
        sequence_mask[drops] = False
        frame_confidence[drops] *= 0.1
        mask = rng.random(landmarks.shape[:2]) < config.mask_rate
        wrist = landmarks[:, 0:1, :]
        landmarks[mask] = np.repeat(wrist, landmarks.shape[1], axis=1)[mask]
    else:
        raise ValueError(f"Unknown perturbation kind '{config.kind}'.")

    landmarks[:, :, :2] = np.clip(landmarks[:, :, :2], -0.5, 1.5)
    return LandmarkTensor(
        landmarks=landmarks.astype(np.float32),
        sequence_mask=sequence_mask,
        frame_confidence=frame_confidence.astype(np.float32),
        handedness_score=handedness,
        coord_space=tensor.coord_space,
        world_landmarks=world,
    )


def summarize_robustness(report: dict) -> dict:
    clean = report.get("clean", {})
    summary: dict[str, dict] = {}
    methods = sorted({method for scenario in report.values() for method in scenario})
    for method in methods:
        clean_metrics = clean.get(method, {}).get("recognition", {})
        clean_macro = float(clean_metrics.get("macro_f1", 0.0))
        scenario_macros = [
            float(scenario[method]["recognition"]["macro_f1"])
            for name, scenario in report.items()
            if name != "clean" and method in scenario
        ]
        scenario_false_actions = [
            float(scenario[method]["risk"].get("no_gesture_false_action_rate", 0.0))
            for name, scenario in report.items()
            if name != "clean" and method in scenario
        ]
        mean_macro = float(np.mean(scenario_macros)) if scenario_macros else clean_macro
        mean_false_action = float(np.mean(scenario_false_actions)) if scenario_false_actions else 0.0
        summary[method] = {
            "clean_macro_f1": clean_macro,
            "perturbed_macro_f1_mean": mean_macro,
            "macro_f1_drop": clean_macro - mean_macro,
            "perturbed_no_gesture_false_action_rate_mean": mean_false_action,
        }
    return summary
