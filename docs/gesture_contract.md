# Gesture Contract

Last audited: 2026-06-14

This document is the source of truth for the project gesture vocabulary. Dataset mapping, controller logic, UI guide visuals, task scenarios, and evaluation must use this contract. If a gesture is changed here, the implementation and documentation must be updated together.

The seven labels come from the selected IPN Hand subset:

| Target label | IPN class | AR action | Risk cost | Role | Event type |
|---|---|---|---:|---|---|
| `no_gesture` | `No gesture` / `D0X` | `idle` | 0.00 | background | continuous state |
| `point_2f` | `Point-2f` / `B0B` | `pointer_hover` | 0.25 | pointer state | continuous state |
| `click_2f` | `Click-2f` / `G02` | `select_confirm` | 2.00 | command | discrete event |
| `swipe_left` | `Th-left` / `G05` | `navigate_previous` | 1.00 | command | discrete event |
| `swipe_right` | `Th-right` / `G06` | `navigate_next` | 1.00 | command | discrete event |
| `zoom_in` | `Zoom-in` / `G10` | `zoom_in` | 1.25 | transform command | discrete event or bounded continuous state |
| `zoom_out` | `Zoom-o` / `G11` | `zoom_out` | 1.25 | transform command | discrete event or bounded continuous state |

Risk costs are defined in `configs/interaction/action_risk_costs.yaml`.

## Global Rules

- A raw recognizer label is an action proposal, not a command.
- `select_confirm` has the highest cost and must require arming, temporal stability, release, and cooldown.
- `point_2f` is a state. It may drive cursor motion, but it should not complete task steps that require a discrete command.
- Swipe and zoom actions must have cooldowns so one physical motion cannot trigger repeated events.
- TARC may lower the threshold for the current expected task label and raise it for unexpected labels, but it should not accept gestures that violate physical constraints.
- UI guide visuals must show start, active, lock, and release timing when the controller requires those stages.
- The current live controller uses landmark geometry more heavily than C6 for webcam control. Evaluation must log both raw model output and controller output.

## `no_gesture`

Semantic meaning: idle or non-command background movement.

Physical movement: no visible command gesture. The hand may be absent, outside the frame, not confidently tracked, or resting without an intended AR command.

Expected start condition: no valid hand, low landmark confidence, insufficient valid frames, or explicit release after a command.

Active condition: no accepted gesture proposal; system remains idle.

End/release condition: a valid hand appears and a command or pointer state passes the relevant physical checks.

Mapped AR action: `idle`.

Risk cost: 0.00.

Role: idle/background.

Event type: continuous state.

False positives can be caused by:

- low-light tracking noise;
- a partially visible hand being classified as a command;
- a sliding window that still contains previous command frames;
- forced fallback to `point_2f` whenever any hand is visible.

Controller should reject:

- any action proposal when detection confidence is too low;
- any action proposal from too few valid frames;
- command proposals immediately after a release if cooldown is active;
- high-risk commands when task context expects idle.

## `point_2f`

Semantic meaning: intentional pointer or hover state.

Physical movement: index and middle fingers are presented as a stable two-finger pointing gesture. The live controller currently approximates this as "a visible stable hand" more than as a strict two-finger pose; this is a known mismatch to resolve.

Expected start condition: valid hand landmarks, stable tracking, and no active discrete command. Ideally the two extended fingers should be visible and separable.

Active condition: the pointer follows the index fingertip or a stable hand target. Movement should be smoothed and should not complete a discrete command by itself.

End/release condition: hand leaves the frame, confidence drops, fingers leave the pointing pose, or a discrete command is armed.

Mapped AR action: `pointer_hover`.

Risk cost: 0.25.

Role: pointer state.

Event type: continuous state.

False positives can be caused by:

- accepting any visible hand as `point_2f`;
- low-confidence landmarks;
- confusing release frames from click, swipe, or zoom with pointing;
- pointer jitter when the index fingertip is unstable.

Controller should reject:

- low-confidence or incomplete landmarks;
- frames with high motion that are better explained as a swipe;
- closed/pinched hand states that are likely part of click;
- point-driven task completion without a required click/confirm step.

## `click_2f`

Semantic meaning: confirm, select, pick, drop, open, close, or lock.

Physical movement: a short intentional click gesture. The live operational contract should be: stable pointer/open state, then a brief close/tap motion, then release back to open. Current code arms click when recent finger spacing is open and locks a click when index-middle or thumb-index distance becomes small with low whole-hand motion.

Expected start condition: a visible stable hand has first entered an open or pointer-like state. Click must not start from an already closed hand.

Active condition: a short close/tap gesture is detected for the required number of stable frames. The hand should remain mostly stationary so a swipe is not confused with a click.

End/release condition: the hand reopens or returns to pointer/idle. Cooldown starts after one accepted click.

Mapped AR action: `select_confirm`.

Risk cost: 2.00.

Role: command.

Event type: discrete event.

False positives can be caused by:

- click class dominance in the neural recognizer;
- landmark collapse when fingers overlap;
- normal hand pose changes while pointing;
- continuous closed-hand frames being interpreted as repeated clicks;
- UI guide visuals showing an unnatural pinch that does not match the controller.

Controller should reject:

- click without prior open-state arming;
- held closed hand without release;
- click during high whole-hand motion;
- repeated click during cooldown;
- low confidence or low score margin;
- click when task context expects a different high-confidence action, unless explicitly allowed.

## `swipe_left`

Semantic meaning: previous item, scroll up, rotate/nudge left, or move to a previous state.

Physical movement: one clean horizontal movement of the visible hand to the left in screen/AR coordinates.

Expected start condition: valid hand tracking and no active locked command. The hand should begin from a relatively stable position.

Active condition: horizontal displacement exceeds a motion threshold and is more horizontal than vertical. Current live controller uses index-tip displacement over the recent landmark window and adapts the threshold to jitter.

End/release condition: the lateral motion stops, the label is locked for a short hold, and cooldown prevents repeats.

Mapped AR action: `navigate_previous`.

Risk cost: 1.00.

Role: command.

Event type: discrete event.

False positives can be caused by:

- pointer drift;
- camera shake or low-FPS motion aliasing;
- reaching for the mouse/keyboard while the hand is visible;
- mirrored camera assumptions;
- an expected-label gate that accepts the motion even if the user intended pointing.

Controller should reject:

- displacement below threshold;
- movement that is mostly vertical;
- high jitter with no coherent direction;
- repeated swipes inside cooldown;
- swipe when a task step expects click or zoom and the confidence is weak.

## `swipe_right`

Semantic meaning: next item, scroll down, rotate/nudge right, or move a virtual item to the next target.

Physical movement: one clean horizontal movement of the visible hand to the right in screen/AR coordinates.

Expected start condition: valid hand tracking and no active locked command. The hand should begin from a relatively stable position.

Active condition: horizontal displacement exceeds a motion threshold and is more horizontal than vertical. Current live controller uses index-tip displacement over the recent landmark window and adapts the threshold to jitter.

End/release condition: the lateral motion stops, the label is locked for a short hold, and cooldown prevents repeats.

Mapped AR action: `navigate_next`.

Risk cost: 1.00.

Role: command.

Event type: discrete event.

False positives can be caused by:

- pointer drift;
- camera mirroring confusion;
- hand entering the frame from the side;
- user repositioning before click or zoom;
- low frame rate causing one fast motion to look like a large displacement.

Controller should reject:

- displacement below threshold;
- movement that is mostly vertical;
- high jitter with no coherent direction;
- repeated swipes inside cooldown;
- swipe when a task step expects click or zoom and the confidence is weak.

## `zoom_in`

Semantic meaning: increase scale, magnify details, or move closer in an AR inspection task.

Physical movement: there are two competing interpretations that must be reconciled. IPN labels describe zoom as a hand gesture class (`Zoom-in`), while the current live controller interprets zoom-in as the hand growing in the frame, usually by moving closer to the camera. The operational live contract is currently scale growth, not a finger-only pinch animation.

Expected start condition: valid hand tracking, stable visible hand, and no active swipe/click. The hand should start at a measurable scale.

Active condition: hand scale increases beyond the threshold and the motion is not mainly lateral. Current live controller uses palm-scale delta over the recent window.

End/release condition: scale change stops, the action locks briefly, and cooldown prevents repeated zoom events.

Mapped AR action: `zoom_in`.

Risk cost: 1.25.

Role: transform command.

Event type: currently discrete event; may become bounded continuous state if the UI supports smooth continuous scaling.

False positives can be caused by:

- the user moving the hand toward the camera while pointing;
- tracking scale changes caused by wrist rotation;
- landmark detection resizing the hand due to partial visibility;
- UI guide showing finger pinch/open while the controller expects frame-scale growth.

Controller should reject:

- scale delta below threshold;
- scale change dominated by lateral motion;
- zoom during high jitter or low confidence;
- zoom when task context expects click and confidence is weak;
- repeated zoom events without cooldown or release.

## `zoom_out`

Semantic meaning: decrease scale, fit an object, reset view, or move farther away in an AR inspection task.

Physical movement: there are two competing interpretations that must be reconciled. IPN labels describe zoom as a hand gesture class (`Zoom-o`), while the current live controller interprets zoom-out as the hand shrinking in the frame, usually by moving away from the camera. The operational live contract is currently scale shrinkage.

Expected start condition: valid hand tracking, stable visible hand, and no active swipe/click. The hand should start at a measurable scale.

Active condition: hand scale decreases beyond the threshold and the motion is not mainly lateral. Current live controller uses palm-scale delta over the recent window.

End/release condition: scale change stops, the action locks briefly, and cooldown prevents repeated zoom events.

Mapped AR action: `zoom_out`.

Risk cost: 1.25.

Role: transform command.

Event type: currently discrete event; may become bounded continuous state if the UI supports smooth continuous scaling.

False positives can be caused by:

- the hand moving away while the user is simply repositioning;
- partial hand visibility reducing estimated palm scale;
- motion blur or low-confidence landmarks;
- mismatch between IPN reference movement, UI guide, and live controller scale logic.

Controller should reject:

- scale delta below threshold;
- scale change dominated by lateral motion;
- zoom during high jitter or low confidence;
- zoom when task context expects click and confidence is weak;
- repeated zoom events without cooldown or release.

## Known Contract Mismatches To Resolve

- `point_2f`: current live logic often treats any visible valid hand as pointing. The contract requires an intentional pointer state.
- `click_2f`: UI visuals must show open -> short close/tap -> lock -> release, not an unnatural folded-finger pose.
- `zoom_in` and `zoom_out`: IPN/reference semantics and live scale-change semantics must be explicitly aligned. Until this is solved, evaluation must report raw C6 zoom output and live-controller zoom output separately.
- Backend task scenarios contain more steps than some simplified UI tasks. The UI and `configs/interaction/ar_task_scenarios.yaml` should be aligned before claiming task-aware validation quality from live sessions.
- Gesture cards should be generated from reference examples or controller states, not from abstract decorative hand drawings.
