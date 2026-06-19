from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_pipeline.labels import validate_target_label
from research_pipeline.utils.errors import SchemaError


REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "source_dataset",
    "public_label",
    "target_label",
    "participant_id",
    "session_id",
    "repetition_id",
    "split_group",
    "hand_recorded",
    "handedness_detected",
    "mirrored",
    "fps",
    "width",
    "height",
    "camera_device",
    "background_tag",
    "lighting_tag",
    "clip_start_ms",
    "clip_end_ms",
    "raw_video_path",
    "tensor_path",
    "notes",
)

HAND_VALUES = {"left", "right", "unknown"}
SOURCE_VALUES = {"ipn_hand", "local_phone", "synthetic", "jester", "hagrid"}


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    source_dataset: str
    public_label: str
    target_label: str
    participant_id: str
    session_id: str
    repetition_id: str
    split_group: str
    hand_recorded: str = "unknown"
    handedness_detected: str = "unknown"
    mirrored: bool = False
    fps: float = 30.0
    width: int = 0
    height: int = 0
    camera_device: str = "unknown"
    background_tag: str = "unknown"
    lighting_tag: str = "unknown"
    clip_start_ms: int = 0
    clip_end_ms: int = 0
    raw_video_path: str = ""
    tensor_path: str = ""
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {field_name: getattr(self, field_name) for field_name in REQUIRED_MANIFEST_FIELDS}
        payload.update(self.extras)
        return payload


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise SchemaError(f"Cannot parse boolean value '{value}'.")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def manifest_record_from_dict(payload: dict[str, Any], *, strict: bool = True) -> ManifestRecord:
    missing = [field_name for field_name in REQUIRED_MANIFEST_FIELDS if field_name not in payload]
    if missing and strict:
        raise SchemaError(f"Manifest record is missing required fields: {missing}")

    values = {field_name: payload.get(field_name, "") for field_name in REQUIRED_MANIFEST_FIELDS}
    values["mirrored"] = _bool_value(values["mirrored"])
    values["fps"] = _coerce_float(values["fps"], 30.0)
    values["width"] = _coerce_int(values["width"], 0)
    values["height"] = _coerce_int(values["height"], 0)
    values["clip_start_ms"] = _coerce_int(values["clip_start_ms"], 0)
    values["clip_end_ms"] = _coerce_int(values["clip_end_ms"], 0)
    extras = {key: value for key, value in payload.items() if key not in REQUIRED_MANIFEST_FIELDS}
    record = ManifestRecord(**values, extras=extras)
    validate_manifest_record(record)
    return record


def validate_manifest_record(record: ManifestRecord) -> None:
    if not record.sample_id:
        raise SchemaError("sample_id must be non-empty.")
    if record.source_dataset not in SOURCE_VALUES:
        raise SchemaError(f"source_dataset '{record.source_dataset}' is not supported.")
    validate_target_label(record.target_label)
    if record.hand_recorded not in HAND_VALUES:
        raise SchemaError(f"hand_recorded must be one of {HAND_VALUES}.")
    if record.handedness_detected not in HAND_VALUES:
        raise SchemaError(f"handedness_detected must be one of {HAND_VALUES}.")
    if record.fps < 0:
        raise SchemaError("fps must be non-negative.")
    if record.clip_end_ms and record.clip_end_ms < record.clip_start_ms:
        raise SchemaError("clip_end_ms must be greater than or equal to clip_start_ms.")


def resolve_path(path: str, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate

