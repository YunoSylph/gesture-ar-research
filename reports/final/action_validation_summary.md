# Action Validation Summary

Last updated: 2026-06-15

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

TARC must only receive proposals whose state is `ready` or `locked`. Unstable candidates, cooldown frames, low-confidence frames, and release-required frames are rejected before they can become task actions.

This separates three concepts that were previously easy to mix together:

- raw model label;
- validated gesture proposal;
- task-accepted AR action.

## Live Demo Integration

The live backend now attaches `validation_context` to websocket payloads. The React UI can show:

- expected gesture;
- proposal state;
- candidate label;
- lock progress;
- rejection reason;
- cooldown;
- last accepted action.

The UI guide now follows the gesture contract:

- `point_2f` is a stable visible-hand cursor state;
- `click_2f` is open/armed -> short pinch/tap -> lock -> release;
- swipe is a wide horizontal whole-hand movement;
- zoom is a clear hand-scale change, not arbitrary finger folding.

## Current Action-Level Evidence

In the latest pseudo-continuous smoke run, direct mapping accepted many actions and produced high false-action cost. Validation and TARC reduced accepted false actions:

| Method | Accepted | Rejected | False-action cost | Task success |
|---|---:|---:|---:|---:|
| `direct_c6` | 504 | 0 | 925.0 | 0.0 |
| `c6_smoothing` | 495 | 0 | 905.0 | 0.0 |
| `c6_validation_confidence_only` | 54 | 297 | 76.0 | 0.0 |
| `c6_validation_confidence_stability` | 45 | 327 | 62.0 | 0.0 |
| `c6_validation_confidence_stability_cooldown` | 11 | 460 | 7.0 | 0.0 |
| `c6_validation_tarc` | 8 | 472 | 2.0 | 0.0 |

This is a validation smoke result, not final C6 evidence. The meaningful observation is architectural: the proposed gating path sharply reduces false-action cost in the same replay conditions, while task success remains unresolved and must be improved with real C6 outputs and task-tuned action timing.
