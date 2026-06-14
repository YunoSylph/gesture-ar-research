from __future__ import annotations

from dataclasses import dataclass

from research_pipeline.utils.errors import SchemaError


@dataclass(frozen=True)
class GestureClass:
    index: int
    target_label: str
    semantics: str
    ipn_name: str
    ipn_id: int
    interaction: str


FINAL_GESTURES: tuple[GestureClass, ...] = (
    GestureClass(0, "no_gesture", "absence of command", "No gesture", 0, "suppress/reset"),
    GestureClass(1, "point_2f", "two-finger pointing", "Point-2f", 2, "pointer/hover"),
    GestureClass(2, "click_2f", "two-finger click", "Click-2f", 4, "select/confirm"),
    GestureClass(3, "swipe_left", "leftward motion", "Th-left", 7, "previous/rotate left"),
    GestureClass(4, "swipe_right", "rightward motion", "Th-right", 8, "next/rotate right"),
    GestureClass(5, "zoom_in", "pinch/open zoom in", "Zoom-in", 12, "transform zoom in"),
    GestureClass(6, "zoom_out", "pinch/close zoom out", "Zoom-o", 13, "transform zoom out"),
)

TARGET_LABELS: tuple[str, ...] = tuple(item.target_label for item in FINAL_GESTURES)
TARGET_TO_INDEX: dict[str, int] = {item.target_label: item.index for item in FINAL_GESTURES}
INDEX_TO_TARGET: dict[int, str] = {item.index: item.target_label for item in FINAL_GESTURES}
IPN_ID_TO_TARGET: dict[int, str] = {item.ipn_id: item.target_label for item in FINAL_GESTURES}
IPN_ID_TO_TARGET.update(
    {
        # Official IPN annotation classIdx.txt is 1-based and uses 14 entries:
        # 1 D0X, 3 B0B, 5 G02, 8 G05, 9 G06, 13 G10, 14 G11.
        1: "no_gesture",
        3: "point_2f",
        5: "click_2f",
        8: "swipe_left",
        9: "swipe_right",
        13: "zoom_in",
        14: "zoom_out",
    }
)
IPN_NAME_TO_TARGET: dict[str, str] = {
    item.ipn_name.lower().replace("_", "-"): item.target_label for item in FINAL_GESTURES
}
IPN_NAME_TO_TARGET.update(
    {
        "d0x": "no_gesture",
        "b0b": "point_2f",
        "g02": "click_2f",
        "g05": "swipe_left",
        "g06": "swipe_right",
        "g10": "zoom_in",
        "g11": "zoom_out",
        "no gesture": "no_gesture",
        "no_gesture": "no_gesture",
        "non-gesture": "no_gesture",
        "zoom-o": "zoom_out",
        "zoom-out": "zoom_out",
        "th left": "swipe_left",
        "th-left": "swipe_left",
        "throw left": "swipe_left",
        "th right": "swipe_right",
        "th-right": "swipe_right",
        "throw right": "swipe_right",
    }
)

MIRROR_LABEL_SWAP: dict[str, str] = {
    "swipe_left": "swipe_right",
    "swipe_right": "swipe_left",
}


def normalize_label_text(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def validate_target_label(label: str) -> str:
    if label not in TARGET_TO_INDEX:
        raise SchemaError(f"Unknown target label '{label}'. Expected one of {TARGET_LABELS}.")
    return label


def label_to_index(label: str) -> int:
    return TARGET_TO_INDEX[validate_target_label(label)]


def index_to_label(index: int) -> str:
    try:
        return INDEX_TO_TARGET[int(index)]
    except KeyError as exc:
        raise SchemaError(f"Unknown label index '{index}'.") from exc


def swap_mirrored_label(label: str) -> str:
    validate_target_label(label)
    return MIRROR_LABEL_SWAP.get(label, label)


def remap_ipn_label(public_label: str | int) -> str | None:
    """Map an IPN Hand id/name to the final thesis gesture dictionary."""

    if isinstance(public_label, int):
        return IPN_ID_TO_TARGET.get(public_label)

    text = str(public_label).strip()
    if text == "":
        return None
    if text.isdigit():
        return IPN_ID_TO_TARGET.get(int(text))
    normalized = text.lower().replace("_", "-")
    if normalized in IPN_NAME_TO_TARGET:
        return IPN_NAME_TO_TARGET[normalized]
    canonical = normalize_label_text(text)
    return canonical if canonical in TARGET_TO_INDEX else None
