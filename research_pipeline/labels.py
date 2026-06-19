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


# ---------------------------------------------------------------------------
# Jester (20BN-Jester) -> 7-class action vocabulary
# ---------------------------------------------------------------------------
# Jester is a large, dynamic, easily downloadable webcam gesture dataset and the
# practical primary source for the rear-camera recognizer's *dynamic* classes
# (swipe/zoom) after EgoGesture was set aside for download infeasibility. Only a
# subset of Jester's 27 classes maps to this project's vocabulary:
#   - swipe_left/right and zoom_in/out have clear equivalents;
#   - no_gesture is covered by Jester's idle/"other" classes;
#   - point_2f and click_2f have NO Jester equivalent and must come from another
#     source (HaGRID static poses or a local rear-camera set).
# Jester is also frontal / palm-to-camera, whereas the rear deployment shows the
# back of the hand. That orientation gap is closed with augmentation plus a local
# fine-tune set, not by this label mapping.

JESTER_NAME_TO_TARGET: dict[str, str] = {
    "swiping left": "swipe_left",
    "swiping right": "swipe_right",
    "zooming in with two fingers": "zoom_in",
    "zooming out with two fingers": "zoom_out",
    "no gesture": "no_gesture",
    "doing other things": "no_gesture",
}

# Distinct Jester gestures that share screen-space motion / zoom semantics with a
# target class. Folded into that target only when explicitly requested, since
# they are visually different actions (two-finger slide, full-hand zoom).
JESTER_MOTION_EQUIVALENT_TO_TARGET: dict[str, str] = {
    "sliding two fingers left": "swipe_left",
    "sliding two fingers right": "swipe_right",
    "zooming in with full hand": "zoom_in",
    "zooming out with full hand": "zoom_out",
}

# Jester classes with no equivalent in the 7-class vocabulary. Returned as None
# by default (excluded); training may instead fold them into no_gesture as hard
# negatives to suppress false actions on non-command motion.
JESTER_NON_TARGET: tuple[str, ...] = (
    "swiping up",
    "swiping down",
    "pushing hand away",
    "pulling hand in",
    "sliding two fingers up",
    "sliding two fingers down",
    "pushing two fingers away",
    "pulling two fingers in",
    "rolling hand forward",
    "rolling hand backward",
    "turning hand clockwise",
    "turning hand counterclockwise",
    "thumb up",
    "thumb down",
    "shaking hand",
    "stop sign",
    "drumming fingers",
)

# Target classes Jester can supply directly, and those it cannot (honest coverage).
JESTER_COVERED_TARGETS: frozenset[str] = frozenset(JESTER_NAME_TO_TARGET.values())
JESTER_MISSING_TARGETS: tuple[str, ...] = tuple(
    label for label in TARGET_LABELS if label not in JESTER_COVERED_TARGETS
)


def _normalize_jester_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def remap_jester_label(
    label: str,
    *,
    include_motion_equivalents: bool = False,
    fold_non_target_as_no_gesture: bool = False,
) -> str | None:
    """Map a Jester class name to the final 7-class action vocabulary.

    Returns None when the Jester class has no project equivalent, unless
    ``fold_non_target_as_no_gesture`` is set, in which case recognised
    non-command Jester gestures (including unused motion-equivalents) are treated
    as ``no_gesture`` hard negatives. ``include_motion_equivalents`` instead maps
    the motion-equivalent classes onto their swipe/zoom target.
    """

    normalized = _normalize_jester_text(label)
    if normalized == "":
        return None
    if normalized in JESTER_NAME_TO_TARGET:
        return JESTER_NAME_TO_TARGET[normalized]
    if include_motion_equivalents and normalized in JESTER_MOTION_EQUIVALENT_TO_TARGET:
        return JESTER_MOTION_EQUIVALENT_TO_TARGET[normalized]
    if normalized in JESTER_NON_TARGET or normalized in JESTER_MOTION_EQUIVALENT_TO_TARGET:
        return "no_gesture" if fold_non_target_as_no_gesture else None
    return None


# ---------------------------------------------------------------------------
# HaGRID / HaGRIDv2 -> 7-class action vocabulary
# ---------------------------------------------------------------------------
# HaGRID is a large, easily downloadable *static-image* hand-pose dataset. It is
# used only to supply the two-finger pose class point_2f (Jester has no
# point/click), via the two-finger poses two_up / peace (and their inverted
# variants). HaGRID's own "point" class is a SINGLE-finger point, so it is NOT
# mapped to point_2f. Being static, HaGRID cannot supply the dynamic classes
# (swipe_*/zoom_*) or click_2f; those come from Jester and a local rear-camera
# set. Downstream, HaGRID images are turned into static-pose clips (one detected
# frame replicated over the window) before feature extraction.

HAGRID_NAME_TO_TARGET: dict[str, str] = {
    "two_up": "point_2f",
    "two_up_inverted": "point_2f",
    "peace": "point_2f",
    "peace_inverted": "point_2f",
    "no_gesture": "no_gesture",
}

# Full HaGRIDv2 class vocabulary, so non-target folding only applies to genuine
# HaGRID classes rather than arbitrary strings.
HAGRID_KNOWN_CLASSES: frozenset[str] = frozenset(
    {
        "call", "dislike", "fist", "four", "like", "mute", "ok", "palm", "peace",
        "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2",
        "three3", "three_gun", "two_up", "two_up_inverted", "one", "grabbing",
        "grip", "hand_heart", "hand_heart2", "holy", "little_finger",
        "middle_finger", "point", "take_picture", "thumb_index", "thumb_index2",
        "timeout", "xsign", "no_gesture",
    }
)

HAGRID_COVERED_TARGETS: frozenset[str] = frozenset(HAGRID_NAME_TO_TARGET.values())
HAGRID_MISSING_TARGETS: tuple[str, ...] = tuple(
    label for label in TARGET_LABELS if label not in HAGRID_COVERED_TARGETS
)


def remap_hagrid_label(label: str, *, fold_non_target_as_no_gesture: bool = False) -> str | None:
    """Map a HaGRID gesture class name to the final 7-class action vocabulary.

    Returns None for HaGRID classes with no project equivalent, unless
    ``fold_non_target_as_no_gesture`` is set, in which case any recognised HaGRID
    class is treated as a ``no_gesture`` static hard negative.
    """

    normalized = normalize_label_text(label)
    if normalized in HAGRID_NAME_TO_TARGET:
        return HAGRID_NAME_TO_TARGET[normalized]
    if fold_non_target_as_no_gesture and normalized in HAGRID_KNOWN_CLASSES:
        return "no_gesture"
    return None
