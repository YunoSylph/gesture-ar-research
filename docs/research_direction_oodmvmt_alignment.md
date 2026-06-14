# Research Direction: OO-dMVMT Alignment

Last audited: 2026-06-14

This project should be defended as a reproducible pipeline for continuous gesture-to-action validation in AR-like interaction, not as a new state-of-the-art gesture classifier. The reference paper is Cunico et al. 2023, "OO-dMVMT: A Deep Multi-View Multi-Task Classification Framework for Real-Time 3D Hand Gesture Classification and Segmentation". The paper is used as methodological support for moving from pre-segmented clip classification to online recognition, segmentation, decision latency, false positives, and real-time validation.

## What OO-dMVMT Contributes

OO-dMVMT frames mid-air hand gesture recognition as an online problem over continuous 3D hand-pose streams. Its key lesson for this project is that high clip-level classification quality is not enough for mixed-reality interaction. A useful AR recognizer must also decide when a gesture starts, when it ends, how late the decision is made, and how many false positives it produces during non-gesture movement.

The paper is especially relevant because it treats:

- continuous streams rather than only isolated gesture clips;
- segmentation and classification as coupled online tasks;
- false positives as a first-class metric because a false positive may trigger an unwanted action;
- decision latency as part of system quality;
- multiple pose/motion views as a way to represent local temporal dynamics;
- task outputs that can be enabled or disabled depending on whether they are meaningful for the current input.

The current thesis project should not copy OO-dMVMT directly. Instead, it should use this framing to justify a task-aware continuous gesture-to-action validation layer on top of a public-data recognizer.

## Current Project Alignment

The current repository already contains several components that align with this direction:

- `research_pipeline/features/preprocessing.py` converts MediaPipe-style 21-landmark streams into a dual-view representation: normalized pose plus motion features.
- `research_pipeline/cli/extract_landmarks.py` builds landmark tensors from IPN Hand clips, using clip boundaries from the public manifest.
- `research_pipeline/models/tcn.py` and `research_pipeline/models/torch_training.py` implement a temporal TCN recognizer over fixed-length 32-frame sequences.
- `research_pipeline/models/c6_ensemble.py` defines the robust C6 recognizer as an ensemble plus calibrated geometry fusion.
- `research_pipeline/models/hybrid.py` adds geometry-aware safety gates for weak motion, zoom scale deltas, low confidence, and click distance.
- `research_pipeline/serve/live_backend.py` contains a live webcam stream, MediaPipe-style landmark extraction, a 32-frame sliding window, a live landmark controller, and JSONL logging.
- `research_pipeline/interaction/action_safe.py` treats classifier output as a proposal that can be rejected by confidence, stability, margin, and cooldown constraints.
- `research_pipeline/interaction/task_aware.py` implements TARC: expected task actions lower thresholds for the current step and raise them for unexpected actions.
- `research_pipeline/cli/benchmark_c4_tasks.py` already compares a direct classifier-to-action baseline against stronger recognizers and the proposed task-aware controller in replayed AR task scenarios.
- `configs/interaction/action_risk_costs.yaml` assigns different costs to AR actions, with `select_confirm` as the highest-risk action.
- `demo/ar_interaction_app/src/main.tsx` exposes the practical AR-style UI, live webcam source, direct/TARC interaction modes, task scenarios, and a gesture guide.

This means the project already has the skeleton of a research contribution: classifier output is not treated as a final command, but as a proposal that is filtered by stability, context, and action risk.

## What Does Not Yet Match OO-dMVMT

The current system still has several important gaps:

- IPN Hand is used mainly as pre-segmented clips. This supports offline recognition but does not directly measure continuous segmentation.
- The TCN/C6 recognizer produces one label for a fixed 32-frame window. It has no explicit onset head, offset head, or temporal segmentation target.
- The live webcam backend feeds a sliding window into C6, but the final live gesture is produced mostly by `LiveLandmarkGestureController`, not directly by the neural model.
- `LivePredictionStabilizer` exists and is tested, but the current webcam path uses `LiveLandmarkGestureController` instead.
- The live controller uses heuristic event logic: point, click, swipe, and zoom are inferred from landmark geometry and expected task label gates.
- The UI gesture guide is a synthetic CSS visualization. It is not yet a verified rendering of the actual IPN reference motion or the controller's event contract.
- Some task definitions in the backend scenarios are richer than the simplified UI tasks, which can make expected-label gating and user-facing task instructions drift apart.
- There is no complete public-data online replay dataset with explicit frame-level ground truth, segment boundaries, onset/offset error, and false positives per minute.

These gaps do not invalidate the project, but they define the next research work. The scientific claim should be made at the pipeline level, not by pretending that C6 alone solves continuous AR control.

## Why Direct Clip-Level Classification Is Insufficient

A direct clip-level classifier answers a limited question: "Which class is this already-segmented gesture clip?" Live AR asks a harder question: "Is the user's current continuous movement a command, which command is it, is the decision stable enough, and is it safe to execute in this task state?"

The difference matters because a webcam stream contains:

- transitions into and out of gestures;
- pauses, hesitation, and partial gestures;
- non-command hand motion;
- hand scale changes caused by depth and framing;
- tracking dropouts and landmark jitter;
- motion blur and lighting changes;
- overlapping semantics such as pointing, selecting, and moving the whole hand.

A 32-frame sliding window can land before the gesture starts, in the middle of the gesture, during release, or across two different motions. The same offline classifier can therefore show strong benchmark metrics and still create label jitter, click false positives, action switching, or missed releases in live interaction.

## Why an Online Proposal and Validation Layer Is Needed

The proposed architecture should be described as a multi-stage validation pipeline:

1. Offline recognizer: estimates class probabilities from a landmark window.
2. Online proposal layer: converts raw predictions and landmark dynamics into candidate gestures, while tracking stability, release, cooldown, and no-hand states.
3. Action mapping: maps a candidate gesture to an AR action only when the gesture contract allows it.
4. TARC validation: uses task context and risk to accept, reject, or delay the action.
5. AR task execution: applies accepted actions to the scene and records task-level outcomes.

This is the central research idea: a gesture is not an immediate command. It is an action proposal. The proposal must pass confidence, stability, context, and risk filtering.

## Why the Live Webcam Demo Is Not the Main Scientific Proof

The live webcam demo is valuable as an engineering demonstration, but it is a weak primary proof because it is not fully reproducible:

- camera quality, lighting, placement, and user behavior vary;
- the user's webcam may be covered, low-light, or noisy;
- MediaPipe tracking quality changes with hardware and browser/backend load;
- there is usually no frame-level ground truth for every live session;
- UI impressions can conflate model quality, controller behavior, and task design.

The main proof should therefore be a reproducible public-data online replay protocol. A replay can construct continuous streams from public landmark clips, insert idle gaps and distractors, preserve logged model/controller outputs, and evaluate identical trials across direct baseline, robust recognizer, and TARC.

Live webcam sessions should be used as a secondary validation layer: they show feasibility, expose practical failure modes, and support qualitative analysis, but they should not be the only evidence for the thesis claim.

## Main Claims That Are Defensible

The project can claim:

- The TCN/C6 recognizer achieves measurable offline recognition quality on the selected IPN Hand gesture subset.
- Offline recognition quality does not by itself guarantee reliable live AR interaction.
- A task-aware confidence/stability/risk validation layer can reduce false AR actions compared with direct classifier-to-action mapping.
- False-action cost, unintended action rate, label switch rate, and decision latency are more relevant to AR interaction than accuracy alone.
- A reproducible online replay benchmark is an appropriate bridge between public gesture recognition data and AR-style task interaction.
- The live UI demonstrates the intended interaction loop, but the scientific evidence should be grounded in replay metrics and ablations.

## Claims That Should Not Be Made

The project should not claim:

- New state-of-the-art gesture recognition on all hand-gesture datasets.
- Robust deployment across all cameras, users, phones, headsets, and lighting conditions.
- That C6 alone is sufficient for live AR control.
- That IPN Hand pre-segmented metrics prove continuous gesture segmentation quality.
- That the current webcam demo is equivalent to validated phone AR, Quest, HoloLens, or iOS ARKit deployment.
- That visual gesture guide cards are ground truth unless they are derived from the gesture contract and reference examples.

## Minimal Methodology Upgrade

The next development stage should be:

1. Treat `docs/gesture_contract.md` as the source of truth for labels, physical motion, release conditions, mapped actions, risk, and rejection rules.
2. Add an online replay evaluator that emits frame-level labels, segment boundaries, proposal latency, label switches, and false positives per minute.
3. Compare at least three methods:
   - direct classifier-to-action baseline;
   - robust recognizer plus generic stability validation;
   - robust recognizer plus TARC task-aware validation.
4. Report both recognition metrics and AR action metrics.
5. Use the live webcam demo only after the replay protocol is stable, as an engineering demonstration and failure-analysis source.

This framing keeps the work honest and makes the contribution stronger: the thesis is not "we trained the best gesture model", but "we built and evaluated a reproducible continuous gesture-to-action validation pipeline for AR tasks."
