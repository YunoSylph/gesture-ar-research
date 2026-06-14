# Live AR Interface Refinement Report

## Problem Found In Manual Testing

Manual webcam testing showed that the live demo was still too close to a raw recognition lab screen:

- too many AR tasks with overlapping semantics;
- too many controls exposed at once;
- unstable gesture switching in live webcam mode;
- `click_2f` could dominate and interrupt other gestures;
- the first cube object was too large for a camera-backed AR view;
- gesture names and execution style were not visible enough during the task.

## Implemented Changes

### Reduced Live Task Set

The live UI now exposes four clear tasks:

| Task | Purpose |
|---|---|
| Object: select and scale | Point at cube, short click, zoom in, zoom out |
| List: scroll and open | Swipe through a list and open a highlighted row |
| Cards: browse and inspect | Browse AR cards, open one, inspect it |
| Sorting: move item | Pick a virtual object, move it to a bin, drop it |

The larger scenario library remains available internally for benchmarks, but it is no longer pushed into the main demo UI.

### Live Recognition Stabilization

The backend now applies a `LivePredictionStabilizer` in webcam mode:

- short rolling prediction history;
- minimum vote count before changing the stable gesture;
- `click_2f` geometry check based on index-middle fingertip distance;
- higher live thresholds and stable-frame requirements for click;
- longer cooldown for action execution in the TARC controller.

This targets the most visible failure mode: click interrupting swipes, zooms, and pointing.

### Interface Simplification

The default UI now exposes:

- one task selector;
- `Start Task` / `Reset Task`;
- telemetry;
- an `Advanced Controls` drawer for model/source/debug controls.

The old replay test buttons were removed from the main control flow.

### Gesture Guidance

A new `Guide` page explains the six active gestures with visual cards, names, AR effect, and execution cues. The live task overlay also shows the current expected gesture with a short performance hint.

### AR Object Scale

The first object-control cube was reduced in base geometry and initial scale, leaving more of the camera preview visible.

## Expected Effect

These changes do not claim a new offline model metric. They are live-system improvements aimed at reducing interface ambiguity and action jitter. The most important qualitative improvement should be fewer accidental click actions and clearer task execution.

## Remaining Risk

If webcam lighting, framing, or hand pose differs strongly from IPN-style training data, recognition can still be imperfect. The current stabilizer makes live interaction more usable, but the strongest final validation still requires local webcam/phone recordings or a short user study.
