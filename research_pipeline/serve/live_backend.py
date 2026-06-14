from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from research_pipeline.data.manifest import read_jsonl
from research_pipeline.data.schema import resolve_path
from research_pipeline.data.tensors import LandmarkTensor, load_landmark_npz
from research_pipeline.features.preprocessing import clip_feature_summary, preprocess_dual_view
from research_pipeline.interaction.action_safe import ActionSafePolicy, ActionSafePolicyConfig
from research_pipeline.interaction.fsm import ACTION_BY_LABEL, ContextAwarePolicy, ContextPolicyConfig
from research_pipeline.interaction.task_aware import TaskAwareActionSafePolicy, TaskAwarePolicyConfig, load_task_scenarios
from research_pipeline.labels import TARGET_LABELS
from research_pipeline.models.calibrated import CalibratedFusionConfig, calibrated_fusion_prediction
from research_pipeline.models.c6_ensemble import C6EnsembleConfig, C6EnsembleRecognizer
from research_pipeline.models.common import Prediction, prediction_from_scores
from research_pipeline.models.hybrid import GeometryPriorRecognizer, HybridConfig, fuse_hybrid_predictions
from research_pipeline.models.rule_based import RuleBasedRecognizer
from research_pipeline.utils.errors import DependencyMissingError, PipelineError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD_ARTIFACTS = {
    "c1_rf": PROJECT_ROOT / "artifacts/models/ipn_c1_rf_full.pkl",
    "c1t_tcn": PROJECT_ROOT / "artifacts/models/ipn_c1t_tcn_full.pkl",
    "c1t_tcn_validated": PROJECT_ROOT / "artifacts/models/ipn_c1t_tcn_full_validated.pkl",
    "c1t_tcn_augmented": PROJECT_ROOT / "artifacts/models/ipn_c1t_tcn_augmented.pkl",
    "onnx": PROJECT_ROOT / "artifacts/export/ipn_c1t_tcn_full.onnx",
}
DEFAULT_LIVE_HYBRID_CONFIG = HybridConfig(
    neural_weight=0.96,
    geometry_weight=0.08,
    enable_safety_gate=True,
    action_threshold=0.44,
    no_gesture_margin=0.03,
    low_quality_confidence=0.42,
    low_quality_no_gesture_boost=0.12,
    weak_motion_no_gesture_boost=0.06,
    swipe_motion_min=0.055,
    zoom_delta_min=0.035,
    click_distance_max=0.075,
)
DEFAULT_ACTION_SAFE_CONFIG = ActionSafePolicyConfig(
    default_threshold=0.70,
    label_thresholds={
        "point_2f": 0.62,
        "click_2f": 0.75,
        "swipe_left": 0.75,
        "swipe_right": 0.75,
        "zoom_in": 0.75,
        "zoom_out": 0.75,
    },
    default_stable_frames=1,
    label_stable_frames={
        "click_2f": 2,
        "swipe_left": 2,
        "swipe_right": 2,
        "zoom_in": 2,
        "zoom_out": 2,
    },
    cooldown_ms=250,
    no_gesture_reset_frames=3,
    min_score_margin=0.0,
)
DEFAULT_TARC_ACTION_CONFIG = ActionSafePolicyConfig(
    default_threshold=0.62,
    label_thresholds={
        "point_2f": 0.54,
        "click_2f": 0.78,
        "swipe_left": 0.62,
        "swipe_right": 0.62,
        "zoom_in": 0.62,
        "zoom_out": 0.62,
    },
    default_stable_frames=1,
    label_stable_frames={
        "click_2f": 3,
        "swipe_left": 2,
        "swipe_right": 2,
        "zoom_in": 2,
        "zoom_out": 2,
    },
    cooldown_ms=420,
    no_gesture_reset_frames=3,
    min_score_margin=0.04,
)
DEFAULT_TASK_AWARE_CONFIG = TaskAwarePolicyConfig(
    base=DEFAULT_TARC_ACTION_CONFIG,
    expected_threshold_delta=-0.06,
    unexpected_threshold_delta=0.16,
    idle_threshold_delta=0.22,
    expected_stable_frames=2,
)
DEFAULT_C6_CALIBRATION_CONFIG = CalibratedFusionConfig(
    c3_weight=0.15,
    temperature=1.25,
    label_biases={
        "click_2f": 0.18,
        "no_gesture": 0.12,
        "swipe_left": 0.0,
        "zoom_in": 0.0,
        "zoom_out": 0.22,
    },
)
DEFAULT_REPLAY_MANIFEST = PROJECT_ROOT / "data/interim/manifests/ipn_test_full_landmarks.jsonl"
DEFAULT_MEDIAPIPE_MODEL = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
DEFAULT_LIVE_LOG_DIR = PROJECT_ROOT / "artifacts/live_sessions"
DEFAULT_TASK_SCENARIOS_PATH = PROJECT_ROOT / "configs/interaction/ar_task_scenarios.yaml"
DEFAULT_TASK_SCENARIOS = load_task_scenarios(DEFAULT_TASK_SCENARIOS_PATH) if DEFAULT_TASK_SCENARIOS_PATH.exists() else {}


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - np.max(values)
    exp = np.exp(values)
    return (exp / np.maximum(exp.sum(), 1e-12)).astype(np.float32)


class LivePredictor:
    def __init__(self, method: str, *, target_length: int = 32):
        self.method = method
        self.target_length = target_length
        self.rule = RuleBasedRecognizer()
        self.artifact: dict[str, Any] | None = None
        self.onnx_session = None
        self.torch_model = None
        self.torch = None
        self.device = None
        self.hybrid_config: HybridConfig | None = None
        self.geometry_prior: GeometryPriorRecognizer | None = None
        self.c6_recognizer: C6EnsembleRecognizer | None = None

        if method == "c0":
            return
        if method in {"onnx", "c3"}:
            self._load_onnx(DEFAULT_METHOD_ARTIFACTS["onnx"])
            if method == "c3":
                self.hybrid_config = DEFAULT_LIVE_HYBRID_CONFIG
                self.geometry_prior = GeometryPriorRecognizer(self.hybrid_config)
            return
        if method == "c6_ensemble":
            self.c6_recognizer = C6EnsembleRecognizer(
                C6EnsembleConfig(
                    model_paths=[
                        str(DEFAULT_METHOD_ARTIFACTS["c1t_tcn_validated"]),
                        str(DEFAULT_METHOD_ARTIFACTS["c1t_tcn_augmented"]),
                    ],
                    hybrid=DEFAULT_LIVE_HYBRID_CONFIG,
                    calibration=DEFAULT_C6_CALIBRATION_CONFIG,
                )
            )
            return
        if method in {"c1_rf", "c1t_tcn"}:
            self._load_artifact(DEFAULT_METHOD_ARTIFACTS[method])
            return
        raise PipelineError(f"Unknown live recognizer method '{method}'.")

    def _load_artifact(self, path: Path) -> None:
        from research_pipeline.models.artifacts import load_artifact

        self.artifact = load_artifact(path)
        if self.artifact["model_type"] == "c1t_tcn_torch":
            import torch

            from research_pipeline.models.tcn import TCNConfig, build_tcn

            config = TCNConfig(**self.artifact["tcn_config"])
            model = build_tcn(config)
            model.load_state_dict(self.artifact["state_dict"])
            model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(self.device)
            self.torch = torch
            self.torch_model = model

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyMissingError("ONNX live inference requires onnxruntime-gpu or onnxruntime.") from exc
        available_providers = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if os.environ.get("GESTURE_AR_ONNX_CUDA", "").lower() in {"1", "true", "yes"} and "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.onnx_session = ort.InferenceSession(str(path), sess_options=options, providers=providers)

    def predict(self, tensor: LandmarkTensor) -> Prediction:
        if not tensor.sequence_mask.any():
            return prediction_from_scores({"no_gesture": 1.0})
        if self.method == "c0":
            return self.rule.predict(tensor)

        sequence = preprocess_dual_view(tensor, target_length=self.target_length)
        if self.method == "c6_ensemble":
            assert self.c6_recognizer is not None
            return self.c6_recognizer.predict(tensor)

        if self.method in {"onnx", "c3"}:
            assert self.onnx_session is not None
            logits = self.onnx_session.run(None, {"landmarks": sequence.features[None, :, :].astype(np.float32)})[0][0]
            probabilities = softmax(logits)
            neural = prediction_from_scores({label: float(probabilities[index]) for index, label in enumerate(TARGET_LABELS)})
            if self.method == "c3":
                assert self.geometry_prior is not None and self.hybrid_config is not None
                geometry = self.geometry_prior.predict(tensor)
                return fuse_hybrid_predictions(neural, geometry, tensor, self.hybrid_config)
            return neural

        assert self.artifact is not None
        if self.artifact["model_type"] == "c1_random_forest":
            estimator = self.artifact["estimator"]
            summary = clip_feature_summary(sequence)[None, :]
            scores = {label: 0.0 for label in TARGET_LABELS}
            probabilities = estimator.predict_proba(summary)[0]
            labels = self.artifact["labels"]
            for cls, score in zip(estimator.classes_, probabilities):
                scores[labels[int(cls)]] = float(score)
            return prediction_from_scores(scores)

        if self.artifact["model_type"] == "c1t_tcn_torch":
            assert self.torch is not None and self.torch_model is not None
            with self.torch.no_grad():
                x = self.torch.from_numpy(sequence.features[None, :, :].astype(np.float32)).to(self.device)
                logits = self.torch_model(x)
                probabilities = self.torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            labels = self.artifact["labels"]
            return prediction_from_scores({labels[index]: float(score) for index, score in enumerate(probabilities)})

        raise PipelineError(f"Unsupported live artifact type '{self.artifact['model_type']}'.")


class FrameLandmarker:
    def __init__(self, model_asset_path: Path = DEFAULT_MEDIAPIPE_MODEL):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise DependencyMissingError("Webcam mode requires opencv-python and mediapipe.") from exc

        self.cv2 = cv2
        self.mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_asset_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, bool, float]:
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms = max(self._timestamp_ms + 1, int(time.perf_counter() * 1000))
        result = self.detector.detect_for_video(image, self._timestamp_ms)
        if not result.hand_landmarks:
            return np.zeros((21, 3), dtype=np.float32), False, 0.0
        hand_index = 0
        confidence = 1.0
        if result.handedness:
            scores = [category[0].score for category in result.handedness if category]
            hand_index = int(np.argmax(scores))
            confidence = float(scores[hand_index])
        points = result.hand_landmarks[hand_index]
        landmarks = np.array([[point.x, point.y, point.z] for point in points], dtype=np.float32)
        return landmarks, True, confidence

    def close(self) -> None:
        self.detector.close()


def tensor_from_window(window: Iterable[tuple[np.ndarray, bool, float]]) -> LandmarkTensor:
    items = list(window)
    if not items:
        return LandmarkTensor(
            landmarks=np.zeros((1, 21, 3), dtype=np.float32),
            sequence_mask=np.zeros((1,), dtype=bool),
            frame_confidence=np.zeros((1,), dtype=np.float32),
            handedness_score=np.zeros((1,), dtype=np.float32),
        )
    landmarks = np.stack([item[0] for item in items]).astype(np.float32)
    mask = np.array([item[1] for item in items], dtype=bool)
    confidence = np.array([item[2] for item in items], dtype=np.float32)
    return LandmarkTensor(
        landmarks=landmarks,
        sequence_mask=mask,
        frame_confidence=confidence,
        handedness_score=confidence.copy(),
        coord_space="image_normalized_xyz",
    )


def prediction_payload(
    *,
    method: str,
    source: str,
    prediction: Prediction,
    timestamp_ms: int,
    event: Any | None,
    sample_id: str = "",
    target_label: str = "",
    detection_rate: float = 0.0,
    preview_image: str | None = None,
    landmarks: list[list[float]] | None = None,
    pointer: dict[str, float] | None = None,
    action_override: str | None = None,
    fps: float | None = None,
    processing_ms: float | None = None,
    camera: dict[str, Any] | None = None,
    session_id: str = "",
    log_path: str = "",
    task: str = "",
    policy_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = action_override or (event.action if event is not None else "idle")
    return {
        "type": "prediction",
        "timestamp_ms": timestamp_ms,
        "method": method,
        "source": source,
        "gesture": prediction.label,
        "confidence": prediction.confidence,
        "scores": prediction.scores,
        "event": asdict(event) if event is not None else None,
        "action": action,
        "sample_id": sample_id,
        "target_label": target_label,
        "detection_rate": detection_rate,
        "preview_image": preview_image,
        "landmarks": landmarks,
        "pointer": pointer,
        "fps": fps,
        "processing_ms": processing_ms,
        "camera": camera,
        "session_id": session_id,
        "log_path": log_path,
        "task": task,
        "policy_context": policy_context,
    }


def encode_preview_frame(cv2, frame: np.ndarray, *, width: int = 960, quality: int = 82) -> str:
    height, current_width = frame.shape[:2]
    if current_width > width:
        scale = width / current_width
        frame = cv2.resize(frame, (width, max(1, int(height * scale))))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 35, 95))])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def encode_jpeg_bytes(cv2, frame: np.ndarray, *, width: int = 960, quality: int = 82) -> bytes:
    height, current_width = frame.shape[:2]
    if current_width > width:
        scale = width / current_width
        frame = cv2.resize(frame, (width, max(1, int(height * scale))))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 35, 95))])
    return encoded.tobytes() if ok else b""


def measured_fps(timestamps: deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0
    return (len(timestamps) - 1) / elapsed


class LiveSessionLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        source: str,
        method: str,
        interaction_mode: str,
        task: str,
        max_bytes: int = 50_000_000,
    ) -> None:
        self.enabled = enabled
        self.source = source
        self.method = method
        self.interaction_mode = interaction_mode
        self.task = task
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.truncated = False
        self.session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1) * 1000):03d}_{source}_{method}_{task}"
        self.path = DEFAULT_LIVE_LOG_DIR / f"{self.session_id}.jsonl"
        self.handle = None
        if enabled:
            DEFAULT_LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a", encoding="utf-8")

    @property
    def public_path(self) -> str:
        return str(self.path) if self.enabled else ""

    def write(self, payload: dict[str, Any], *, extra: dict[str, Any] | None = None) -> None:
        if self.handle is None:
            return
        if self.truncated:
            return
        record = {key: value for key, value in payload.items() if key not in {"preview_image", "scores", "log_path"}}
        record["source"] = self.source
        record["method"] = self.method
        record["interaction_mode"] = self.interaction_mode
        record["task"] = self.task
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if self.bytes_written + encoded_size > self.max_bytes:
            truncation = {
                "type": "log_truncated",
                "session_id": self.session_id,
                "timestamp_ms": payload.get("timestamp_ms", 0),
                "max_bytes": self.max_bytes,
            }
            self.handle.write(json.dumps(truncation, ensure_ascii=False) + "\n")
            self.truncated = True
            self.handle.flush()
            return
        self.handle.write(line)
        self.bytes_written += encoded_size
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class CameraHub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.frame: np.ndarray | None = None
        self.error = ""
        self.config: tuple[int, int, int, bool, int] | None = None
        self.frame_timestamp = 0.0
        self.frame_size: tuple[int, int] = (0, 0)
        self.capture_timestamps: deque[float] = deque(maxlen=60)
        self.backend_name = ""

    def start(self, *, camera_index: int, width: int, height: int, mirror: bool, target_fps: int = 30) -> None:
        target_fps = max(1, min(60, int(target_fps)))
        config = (camera_index, width, height, mirror, target_fps)
        with self.lock:
            if self.running and self.config == config:
                return
            self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self.lock:
            self.running = True
            self.frame = None
            self.error = ""
            self.config = config
            self.frame_timestamp = 0.0
            self.frame_size = (0, 0)
            self.capture_timestamps.clear()
            self.backend_name = ""
        self.thread = threading.Thread(target=self._capture_loop, args=config, daemon=True)
        self.thread.start()

    def latest(self) -> np.ndarray | None:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def latest_error(self) -> str:
        with self.lock:
            return self.error

    def status(self) -> dict[str, Any]:
        with self.lock:
            age_ms = (time.perf_counter() - self.frame_timestamp) * 1000 if self.frame_timestamp else None
            requested = self.config or (0, 0, 0, False, 0)
            return {
                "running": self.running,
                "error": self.error,
                "camera_index": requested[0],
                "requested_width": requested[1],
                "requested_height": requested[2],
                "mirror": requested[3],
                "target_fps": requested[4],
                "width": self.frame_size[0],
                "height": self.frame_size[1],
                "capture_fps": measured_fps(self.capture_timestamps),
                "frame_age_ms": age_ms,
                "backend": self.backend_name,
            }

    def _capture_loop(self, camera_index: int, width: int, height: int, mirror: bool, target_fps: int) -> None:
        try:
            import cv2
        except ImportError:
            with self.lock:
                self.error = "OpenCV is not installed."
                self.running = False
            return

        backends = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("dshow", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("msmf", cv2.CAP_MSMF))
        backends.append(("default", None))

        cap = None
        backend_name = ""
        for name, backend in backends:
            candidate = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if candidate.isOpened():
                cap = candidate
                backend_name = name
                break
            candidate.release()
        if cap is None or not cap.isOpened():
            with self.lock:
                self.error = f"Cannot open camera index {camera_index}."
                self.running = False
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self.lock:
            self.backend_name = backend_name

        try:
            while True:
                with self.lock:
                    if not self.running or self.config != (camera_index, width, height, mirror, target_fps):
                        break
                ok, frame = cap.read()
                if ok:
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    now = time.perf_counter()
                    with self.lock:
                        self.frame = frame
                        self.error = ""
                        self.frame_timestamp = now
                        self.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                        self.capture_timestamps.append(now)
                else:
                    with self.lock:
                        self.error = "Camera frame read failed."
                    time.sleep(0.05)
        finally:
            cap.release()


camera_hub = CameraHub()


def direct_action_for_prediction(prediction: Prediction) -> str:
    if prediction.label == "no_gesture":
        return "idle"
    return ACTION_BY_LABEL.get(prediction.label, "idle")


def create_interaction_policy(
    interaction_mode: str,
    policy_config: ContextPolicyConfig,
    task: str,
) -> ContextAwarePolicy | ActionSafePolicy | TaskAwareActionSafePolicy | None:
    if interaction_mode == "direct":
        return None
    if interaction_mode in {"c4_task", "c4_task_aware"}:
        return TaskAwareActionSafePolicy(DEFAULT_TASK_SCENARIOS.get(task), DEFAULT_TASK_AWARE_CONFIG)
    if interaction_mode == "c4":
        return ActionSafePolicy(DEFAULT_ACTION_SAFE_CONFIG)
    return ContextAwarePolicy(policy_config)


def policy_event(policy: ContextAwarePolicy | ActionSafePolicy | TaskAwareActionSafePolicy, prediction: Prediction, timestamp_ms: int):
    return policy.update(prediction, timestamp_ms)


def policy_context(policy: ContextAwarePolicy | ActionSafePolicy | TaskAwareActionSafePolicy | None) -> dict[str, Any] | None:
    if isinstance(policy, TaskAwareActionSafePolicy):
        return policy.context()
    return None


class LivePredictionStabilizer:
    """Smooth live webcam predictions and suppress accidental click dominance."""

    def __init__(self, *, history_size: int = 6):
        self.history: deque[Prediction] = deque(maxlen=history_size)
        self.stable_prediction: Prediction = prediction_from_scores({"no_gesture": 1.0})

    def update(self, prediction: Prediction, tensor: LandmarkTensor) -> Prediction:
        sanitized = self._sanitize(prediction, tensor)
        self.history.append(sanitized)
        if sanitized.label == "no_gesture":
            self.stable_prediction = sanitized
            return sanitized

        weighted_scores = {label: 0.0 for label in TARGET_LABELS}
        counts = {label: 0 for label in TARGET_LABELS}
        for item in self.history:
            counts[item.label] += 1
            for label in TARGET_LABELS:
                weighted_scores[label] += float(item.scores.get(label, 0.0)) * max(0.15, item.confidence)

        candidate = max(weighted_scores, key=weighted_scores.get)
        needed = 3 if candidate == "click_2f" else 2
        if candidate != "no_gesture" and counts.get(candidate, 0) < needed:
            return self._hold_or_idle()

        candidate_prediction = prediction_from_scores(weighted_scores)
        if candidate == "click_2f" and (candidate_prediction.confidence < 0.62 or not self._click_geometry_ok(tensor)):
            return self._hold_or_idle()

        self.stable_prediction = candidate_prediction
        return candidate_prediction

    def _sanitize(self, prediction: Prediction, tensor: LandmarkTensor) -> Prediction:
        if prediction.label != "click_2f":
            return prediction
        if prediction.confidence < 0.58 or not self._click_geometry_ok(tensor):
            scores = dict(prediction.scores)
            scores["click_2f"] = min(scores.get("click_2f", 0.0), scores.get("no_gesture", 0.0) * 0.75)
            scores["no_gesture"] = max(scores.get("no_gesture", 0.0), prediction.confidence + 0.02)
            return prediction_from_scores(scores)
        return prediction

    def _click_geometry_ok(self, tensor: LandmarkTensor) -> bool:
        valid = tensor.sequence_mask.astype(bool)
        landmarks = tensor.landmarks[valid] if valid.shape[0] == tensor.landmarks.shape[0] else tensor.landmarks
        if landmarks.shape[0] < 3:
            return False
        index_middle = np.linalg.norm(landmarks[:, 8, :2] - landmarks[:, 12, :2], axis=1)
        recent = index_middle[-min(6, index_middle.shape[0]) :]
        return float(np.min(recent)) <= 0.065

    def _hold_or_idle(self) -> Prediction:
        if self.stable_prediction.label != "click_2f" and self.stable_prediction.confidence >= 0.45:
            return self.stable_prediction
        return prediction_from_scores({"no_gesture": 1.0})


def pointer_from_landmarks(landmarks: np.ndarray, valid: bool) -> dict[str, float] | None:
    if not valid:
        return None
    # Index fingertip gives a natural AR cursor while still being cheap to compute.
    x = float(np.clip(landmarks[8, 0], 0.0, 1.0))
    y = float(np.clip(landmarks[8, 1], 0.0, 1.0))
    return {"x": x, "y": y}


async def stream_replay(
    websocket: WebSocket,
    method: str,
    interval_ms: int,
    interaction_mode: str,
    policy_config: ContextPolicyConfig,
    logger: LiveSessionLogger,
    task: str,
) -> None:
    await websocket.send_json({"type": "status", "message": "loading model"})
    predictor = LivePredictor(method)
    await websocket.send_json({"type": "status", "message": "streaming dataset"})
    policy = create_interaction_policy(interaction_mode, policy_config, task)
    records = read_jsonl(DEFAULT_REPLAY_MANIFEST)
    base_dir = DEFAULT_REPLAY_MANIFEST.parent
    start = time.perf_counter()
    frame_times: deque[float] = deque(maxlen=30)
    index = 0
    try:
        while True:
            frame_started = time.perf_counter()
            frame_times.append(frame_started)
            record = records[index % len(records)]
            tensor = load_landmark_npz(resolve_path(record.tensor_path, base_dir))
            prediction = predictor.predict(tensor)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            event = None if policy is None else policy_event(policy, prediction, timestamp_ms)
            action_override = direct_action_for_prediction(prediction) if interaction_mode == "direct" else None
            processing_ms = (time.perf_counter() - frame_started) * 1000
            payload = prediction_payload(
                method=method,
                source="replay",
                prediction=prediction,
                timestamp_ms=timestamp_ms,
                event=event,
                action_override=action_override,
                sample_id=record.sample_id,
                target_label=record.target_label,
                detection_rate=float(tensor.sequence_mask.mean()),
                fps=measured_fps(frame_times),
                processing_ms=processing_ms,
                session_id=logger.session_id,
                log_path=logger.public_path,
                task=task,
                policy_context=policy_context(policy),
            )
            logger.write(payload)
            await websocket.send_json(payload)
            index += 1
            await asyncio.sleep(max(0.0, interval_ms / 1000.0 - (time.perf_counter() - frame_started)))
    finally:
        logger.close()


async def stream_webcam(
    websocket: WebSocket,
    method: str,
    camera_index: int,
    frame_interval_ms: int,
    interaction_mode: str,
    policy_config: ContextPolicyConfig,
    logger: LiveSessionLogger,
    *,
    camera_width: int,
    camera_height: int,
    capture_fps: int,
    preview_width: int,
    jpeg_quality: int,
    mirror: bool,
    send_preview: bool,
    task: str,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise DependencyMissingError("Webcam mode requires opencv-python.") from exc

    await websocket.send_json({"type": "status", "message": "loading model"})
    predictor = LivePredictor(method)
    await websocket.send_json({"type": "status", "message": "opening camera"})
    policy = create_interaction_policy(interaction_mode, policy_config, task)
    landmarker = FrameLandmarker()
    camera_hub.start(camera_index=camera_index, width=camera_width, height=camera_height, mirror=mirror, target_fps=capture_fps)
    for _ in range(60):
        if camera_hub.latest() is not None:
            break
        if camera_hub.latest_error():
            await websocket.send_json({"type": "error", "message": camera_hub.latest_error()})
            landmarker.close()
            logger.close()
            return
        await asyncio.sleep(0.05)
    if camera_hub.latest() is None:
        await websocket.send_json({"type": "error", "message": "Camera did not produce frames."})
        landmarker.close()
        logger.close()
        return

    window: deque[tuple[np.ndarray, bool, float]] = deque(maxlen=32)
    stabilizer = LivePredictionStabilizer()
    start = time.perf_counter()
    frame_times: deque[float] = deque(maxlen=30)
    await websocket.send_json({"type": "status", "message": "streaming camera"})
    try:
        while True:
            frame_started = time.perf_counter()
            frame_times.append(frame_started)
            frame = camera_hub.latest()
            if frame is None:
                await websocket.send_json({"type": "error", "message": camera_hub.latest_error() or "Camera frame read failed."})
                await asyncio.sleep(0.2)
                continue
            landmarks, valid, confidence = landmarker.detect(frame)
            window.append((landmarks, valid, confidence))
            tensor = tensor_from_window(window)
            prediction = stabilizer.update(predictor.predict(tensor), tensor)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            event = None if policy is None else policy_event(policy, prediction, timestamp_ms)
            action_override = direct_action_for_prediction(prediction) if interaction_mode == "direct" else None
            processing_ms = (time.perf_counter() - frame_started) * 1000
            payload = prediction_payload(
                method=method,
                source="webcam",
                prediction=prediction,
                timestamp_ms=timestamp_ms,
                event=event,
                action_override=action_override,
                detection_rate=float(tensor.sequence_mask.mean()),
                preview_image=encode_preview_frame(cv2, frame, width=preview_width, quality=jpeg_quality) if send_preview else None,
                landmarks=landmarks[:, :2].round(5).tolist() if valid else [],
                pointer=pointer_from_landmarks(landmarks, valid),
                fps=measured_fps(frame_times),
                processing_ms=processing_ms,
                camera=camera_hub.status(),
                session_id=logger.session_id,
                log_path=logger.public_path,
                task=task,
                policy_context=policy_context(policy),
            )
            logger.write(
                payload,
                extra={
                    "camera_index": camera_index,
                    "camera_width": camera_width,
                    "camera_height": camera_height,
                    "capture_fps": capture_fps,
                    "preview_width": preview_width,
                    "jpeg_quality": jpeg_quality,
                },
            )
            await websocket.send_json(payload)
            await asyncio.sleep(max(0.0, frame_interval_ms / 1000.0 - (time.perf_counter() - frame_started)))
    finally:
        landmarker.close()
        logger.close()


app = FastAPI(title="Gesture AR Live Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "methods": ["c1t_tcn", "c6_ensemble"],
        "interaction_modes": ["direct", "c4_task_aware"],
        "ablation_methods": ["c0", "c1_rf", "onnx", "c3"],
        "ablation_interaction_modes": ["c2", "c4"],
        "replay_manifest": str(DEFAULT_REPLAY_MANIFEST),
    }


@app.get("/api/methods")
def methods() -> dict[str, Any]:
    return {
        "methods": [
            {"id": "c1t_tcn", "label": "Baseline TCN", "artifact": str(DEFAULT_METHOD_ARTIFACTS["c1t_tcn"])},
            {
                "id": "c6_ensemble",
                "label": "Robust C6 Recognizer",
                "artifact": (
                    f"{DEFAULT_METHOD_ARTIFACTS['c1t_tcn_validated']} + "
                    f"{DEFAULT_METHOD_ARTIFACTS['c1t_tcn_augmented']}"
                ),
            },
        ],
        "ablations": [
            {"id": "c0", "label": "C0 Rule", "artifact": "rule-based"},
            {"id": "c1_rf", "label": "C1 RF", "artifact": str(DEFAULT_METHOD_ARTIFACTS["c1_rf"])},
            {"id": "onnx", "label": "ONNX", "artifact": str(DEFAULT_METHOD_ARTIFACTS["onnx"])},
            {"id": "c3", "label": "C3 Hybrid", "artifact": str(DEFAULT_METHOD_ARTIFACTS["onnx"])},
        ],
    }


@app.get("/api/camera/status")
def camera_status() -> dict[str, Any]:
    return camera_hub.status()


async def mjpeg_frames(
    *,
    camera_index: int,
    camera_width: int,
    camera_height: int,
    preview_width: int,
    jpeg_quality: int,
    mirror: bool,
    fps: int,
):
    try:
        import cv2
    except ImportError as exc:
        raise DependencyMissingError("Video feed requires opencv-python.") from exc

    camera_hub.start(camera_index=camera_index, width=camera_width, height=camera_height, mirror=mirror, target_fps=fps)
    sleep_seconds = 1.0 / max(1, min(30, fps))
    while True:
        frame = camera_hub.latest()
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        encoded = encode_jpeg_bytes(cv2, frame, width=preview_width, quality=jpeg_quality)
        if encoded:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\n\r\n" + encoded + b"\r\n"
        await asyncio.sleep(sleep_seconds)


@app.get("/video_feed")
def video_feed(
    camera: int = 0,
    camera_width: int = 1280,
    camera_height: int = 720,
    preview_width: int = 960,
    jpeg_quality: int = 82,
    mirror: bool = True,
    fps: int = 30,
):
    return StreamingResponse(
        mjpeg_frames(
            camera_index=camera,
            camera_width=camera_width,
            camera_height=camera_height,
            preview_width=preview_width,
            jpeg_quality=jpeg_quality,
            mirror=mirror,
            fps=fps,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    params = websocket.query_params
    method = params.get("method", "onnx")
    source = params.get("source", "replay")
    interval_ms = max(16, int(params.get("interval_ms", "80")))
    camera_index = int(params.get("camera", "0"))
    interaction_mode = params.get("interaction", "direct")
    preview_width = int(params.get("preview_width", "960"))
    jpeg_quality = int(params.get("jpeg_quality", "82"))
    camera_width = int(params.get("camera_width", "1280"))
    camera_height = int(params.get("camera_height", "720"))
    capture_fps = max(1, min(60, int(params.get("capture_fps", "30"))))
    mirror = params.get("mirror", "true").lower() not in {"0", "false", "no"}
    log_enabled = params.get("log", "true").lower() not in {"0", "false", "no"}
    send_preview = params.get("preview", "false").lower() not in {"0", "false", "no"}
    task = params.get("task", "object")
    max_log_mb = float(params.get("max_log_mb", "50"))
    policy_config = ContextPolicyConfig(
        activation_threshold=float(params.get("threshold", "0.62")),
        stable_frames=int(params.get("stable_frames", "2")),
        cooldown_ms=int(params.get("cooldown_ms", "250")),
        no_gesture_reset_frames=int(params.get("reset_frames", "3")),
    )
    logger = LiveSessionLogger(
        enabled=log_enabled,
        source=source,
        method=method,
        interaction_mode=interaction_mode,
        task=task,
        max_bytes=max(1_000_000, int(max_log_mb * 1_000_000)),
    )
    try:
        if source == "webcam":
            await stream_webcam(
                websocket,
                method,
                camera_index,
                interval_ms,
                interaction_mode,
                policy_config,
                logger,
                camera_width=camera_width,
                camera_height=camera_height,
                capture_fps=capture_fps,
                preview_width=preview_width,
                jpeg_quality=jpeg_quality,
                mirror=mirror,
                send_preview=send_preview,
                task=task,
            )
        else:
            await stream_replay(websocket, method, interval_ms, interaction_mode, policy_config, logger, task)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        logger.close()
