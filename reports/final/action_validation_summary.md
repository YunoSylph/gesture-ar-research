# Action Validation Summary

Last updated: 2026-06-18

## Implemented Validation Layer

`research_pipeline/interaction/gesture_validation.py` defines the shared `GestureValidationLayer` used by replay and live paths. It converts raw recognizer/controller output into a proposal with:

- `proposal_label`;
- `proposal_state`;
- `proposal_confidence`;
- active/background flags;
- ready/accepted/rejected flags;
- rejection reason;
- lock progress;
- cooldown remaining;
- risk cost;
- last accepted action.

The layer has explicit states:

```text
idle, background, tracking, candidate, preparing, ready, locked, cooldown, release_required, rejected
```

## Relationship To TARC

TARC only receives proposals whose state is `ready` or `locked`. Unstable candidates, low-confidence frames, cooldown frames, and release-required frames are rejected before they can become task actions.

This separates three concepts:

- raw model label;
- validated gesture proposal;
- task-accepted AR action.

## Live Demo Integration

The live backend attaches `validation_context` to websocket payloads. The React UI can show:

- expected gesture;
- proposal state;
- candidate label;
- lock progress;
- rejection reason;
- cooldown;
- last accepted action.

The UI guide follows the gesture contract:

- `point_2f` is a stable visible-hand cursor state;
- `click_2f` is a short click/tap with release before the next click;
- swipe is a wide horizontal movement;
- zoom is handled by the current live-controller contract and must match the on-screen guide.

## Current Action-Level Evidence

In the latest pseudo-continuous replay with real extracted IPN landmarks and the multi-view C6 ensemble, direct mapping accepted many noisy actions and produced high false-action cost. Validation and TARC reduced accepted false actions:

| Method | Accepted | Rejected | False-action cost | Graded completion | Confident completion |
|---|---:|---:|---:|---:|---:|
| `direct_c6` | 3218 | 0 | 4061.25 | 0.058 | 0.000 |
| `c6_smoothing` | 3202 | 0 | 4038.25 | 0.056 | 0.000 |
| `c6_validation_confidence_only` | 1037 | 462 | 900.25 | 0.198 | 0.000 |
| `c6_validation_confidence_stability` | 1010 | 572 | 858.75 | 0.208 | 0.000 |
| `c6_validation_confidence_stability_cooldown` | 560 | 2355 | 167.50 | 0.578 | 0.667 |
| `c6_validation_tarc` | 461 | 2543 | 101.50 | 0.669 | 0.875 |

The meaningful result is architectural: the same recognizer becomes safer for AR tasks when its output is treated as a proposal and filtered through confidence, stability, cooldown, and task-aware risk logic. The project should claim reduced false actions and improved graded completion, not perfect live control or SOTA recognition.
