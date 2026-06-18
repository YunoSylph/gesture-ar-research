# Evaluation Protocol

Last audited: 2026-06-14

This project must be evaluated at three separate levels. A strong offline recognizer is useful, but it does not prove reliable AR interaction. The thesis evidence should show how recognition quality, online proposal quality, and action-level validation relate to each other.

## Level A: Offline Recognition

Question: given a pre-segmented public gesture clip, does the recognizer classify it correctly?

Unit of evaluation: one labeled clip or landmark tensor.

Primary data source: IPN Hand subset manifests and extracted MediaPipe-style landmark tensors.

Methods to compare:

- baseline clip recognizer, such as `c1t_tcn`;
- robust recognizer, such as `c6_ensemble`;
- optional ablations only when they answer a specific research question.

Metrics:

- accuracy;
- macro F1;
- weighted F1;
- per-class precision, recall, and F1;
- confusion matrix;
- optional balanced accuracy;
- offline inference latency per clip.

Existing implementation:

- `research_pipeline/evaluation/metrics.py` computes accuracy, macro F1, weighted F1, balanced accuracy, per-class metrics, and confusion matrix.
- `research_pipeline/evaluation/recognition.py` evaluates manifest records.
- `configs/eval/c6_ensemble_calibrated_recognition.yaml` defines C6 recognition evaluation and robustness scenarios.

What this level proves:

- the recognizer learned the selected public gesture vocabulary;
- C6 can be compared against a simpler baseline on reproducible benchmark splits;
- weak classes and confusion patterns can be identified.

What this level does not prove:

- continuous segmentation quality;
- correct gesture onset/offset timing;
- low false positives during idle motion;
- reliable live webcam AR control;
- safe action execution.

## Level B: Online Continuous Recognition and Proposal

Question: in a continuous stream, when does the system propose a gesture, how stable is that proposal, and how often does it fire during non-command motion?

Unit of evaluation: frames, windows, and gesture segments in a continuous stream.

Recommended primary data source: reproducible public-data online replay. The replay should concatenate public landmark clips with no-gesture gaps and idle distractors, then evaluate every method on the same constructed stream. Live webcam recordings may be used as secondary failure analysis, but should not be the main proof unless they are annotated.

Methods to compare:

- raw sliding-window classifier output;
- robust C6 output;
- C6 plus generic temporal stabilization;
- live landmark proposal controller;
- live landmark proposal controller with task context available only as context, not as hidden ground-truth labels.

Metrics:

- frame-level accuracy;
- segment-level F1;
- gesture onset error in milliseconds or frames;
- gesture offset error in milliseconds or frames;
- decision latency from true onset or target time;
- false positives per minute;
- false negatives per gesture;
- label switch rate;
- no_gesture/action confusion;
- click false positives per minute;
- proposal duration and lock duration;
- raw-to-controller disagreement rate.

Required logging fields:

- `stream_id`;
- `frame_id`;
- `timestamp_ms`;
- `true_label`, when available;
- `true_segment_id`, `true_onset_ms`, and `true_offset_ms`, when available;
- raw model label, confidence, and score distribution;
- controller proposal label and confidence;
- controller mode, candidate label, progress, stable frames, required frames, cooldown state, and click armed state;
- detection confidence and valid landmark ratio;
- expected task label only when evaluating task-aware methods;
- processing latency and measured FPS.

Definitions:

- Frame-level accuracy counts whether the frame's predicted/proposed label matches the frame annotation.
- Segment-level F1 counts whether a predicted gesture segment overlaps a true segment enough to be considered a detection.
- Onset error is predicted onset minus true onset.
- Offset error is predicted offset minus true offset.
- Decision latency is the time between true onset or target time and the first accepted stable proposal.
- False positives per minute counts non-idle proposals during idle/no-gesture periods.
- Label switch rate counts changes between non-idle labels per minute or per active segment.
- No_gesture/action confusion counts idle frames classified as actions and action frames suppressed as idle.

What this level proves:

- whether offline labels transfer to continuous streams;
- whether a sliding 32-frame window is stable enough for live interaction;
- whether controller logic reduces jitter and false positives;
- whether click dominance or zoom/swipe confusion is present before AR action mapping.

What this level does not prove:

- task completion quality;
- whether an accepted action is useful or harmful in a specific AR scenario;
- full camera/device robustness.

## Level C: Action-Level AR Validation

Question: when gesture proposals are mapped to AR actions, does the system accept the right actions, reject unsafe actions, and complete the task reliably?

Unit of evaluation: accepted/rejected AR actions and task trials.

Primary data source: reproducible AR task replay over public-data gesture streams. Live webcam sessions can be analyzed with the same logging schema when available.

Methods to compare:

- direct classifier-to-action baseline;
- robust recognizer direct-to-action baseline;
- robust recognizer plus generic action-safe policy;
- proposed TARC pipeline.

Metrics:

- accepted actions;
- rejected actions;
- false-action cost;
- missed-action cost;
- false-action cost rate;
- missed-action cost rate;
- unintended actions;
- action switch rate;
- task success;
- required action recall;
- task completion time;
- action precision and recall;
- weighted action precision and recall;
- click precision;
- pointer jitter;
- active action rate per minute;
- correction count per task.

Risk model:

Action costs are defined in `configs/interaction/action_risk_costs.yaml`:

| Action | Cost | Interpretation |
|---|---:|---|
| `idle` | 0.00 | no command |
| `pointer_hover` | 0.25 | low-cost cursor movement |
| `navigate_previous` | 1.00 | medium-cost navigation or rotation |
| `navigate_next` | 1.00 | medium-cost navigation or rotation |
| `zoom_in` | 1.25 | higher-cost object transform |
| `zoom_out` | 1.25 | higher-cost object transform |
| `select_confirm` | 2.00 | highest-cost committing action |

Existing implementation:

- `research_pipeline/evaluation/action_risk.py` computes weighted action precision/recall, false-action cost, and missed-action cost.
- `research_pipeline/evaluation/task_benchmark.py` summarizes task-level metrics.
- `research_pipeline/evaluation/live_sessions.py` summarizes live records and evaluates configured task scenarios.
- `research_pipeline/cli/benchmark_c4_tasks.py` builds task replay trials and compares direct, robust, and TARC-style methods.
- `configs/eval/official_method_benchmark.yaml` defines the compact method set: baseline direct, robust recognizer direct, and proposed TARC.

What this level proves:

- whether TARC reduces unwanted AR actions compared with direct classifier-to-action mapping;
- whether risk-weighted false actions decrease without destroying task completion;
- whether the controller improves interaction reliability beyond offline recognition quality;
- whether high-risk commands such as click/select are controlled better than low-risk pointer movement.

What this level does not prove:

- real-world deployment robustness across devices;
- phone rear-camera or headset readiness;
- user-study usability unless human trials are explicitly collected and annotated.

## Required Baseline Structure

The main thesis comparison should use a compact method set:

| Method | Description | Purpose |
|---|---|---|
| M1 Baseline Direct | baseline recognizer output mapped directly to AR action | shows cost of direct classifier-to-action mapping |
| M2 Robust Direct | robust C6 recognizer mapped directly to AR action | separates recognition improvement from controller validation |
| M3 Proposed TARC | robust recognizer plus confidence/stability/context/risk validation | tests the main research claim |

Extra methods can be reported in appendices, but the main narrative should stay compact. Too many similar methods weaken the research story.

## Public-Data Online Replay Protocol

The next evaluator should construct continuous streams as follows:

1. Select public landmark clips by label from the evaluation manifest.
2. Concatenate gesture clips according to task scenarios.
3. Insert `no_gesture` gaps and idle distractor clips between command clips.
4. Preserve frame timestamps and segment boundaries.
5. Run every method on the same stream.
6. Log raw recognizer output, proposal output, TARC decisions, and accepted AR actions.
7. Compute Level B and Level C metrics from the same run.

This protocol is reproducible and supports ablation. It also follows the OO-dMVMT lesson that continuous recognition should be judged by segmentation quality, false positives, delay, and real-time behavior, not by isolated clip accuracy alone.

## Live Webcam Evaluation Protocol

Live webcam sessions should be used after replay evaluation is stable.

Minimum requirements:

- record JSONL logs with raw model output and controller output;
- record camera FPS, processing latency, valid landmark ratio, and confidence;
- optionally record synchronized video for manual annotation;
- annotate at least intended action segments if live sessions are used for quantitative claims;
- report webcam results separately from public replay results.

Live demo claims should be phrased as engineering validation and qualitative failure analysis unless annotated ground truth is available.

### Reproducible aggregation (implemented)

A single session is noisy and not reproducible, so live evidence is aggregated over a *set* of
sessions into the same action-level metrics as the replay. Run the protocol as follows:

1. Run N sessions per task (recommended N >= 5) with a fixed task scenario, logging one JSONL per
   session into `artifacts/live_sessions/`.
2. Provide a task-scenario JSON so each session can be scored against ground-truth expected actions
   (`expected_actions` with `target_ms`/`required`); without it only session-quality metrics
   aggregate.
3. Aggregate:
   `python -m research_pipeline.cli.aggregate_live_sessions --scenarios <scenarios.json>`.

`research_pipeline/evaluation/live_protocol.py` (`aggregate_session_reports`) produces, over the
session set: per-task task-success rate, mean cost-weighted action precision/recall, required-action
recall, median decision latency, and pooled session quality (FPS, p95 processing latency, detection
coverage, confidence), plus an overall summary and the number of scored task runs. This makes live
behaviour a trackable, reproducible measurement that *complements* the replay ablation; it does not
replace it, and the replay ablation remains the primary thesis proof until annotated live sessions
are collected at scale.

## Current Gaps To Close

- No complete Level B evaluator currently computes onset/offset error, segment-level F1, false positives per minute, and label switch rate.
- The live webcam path logs raw C6 output, but the user-facing gesture comes from `LiveLandmarkGestureController`; these outputs need separate metrics.
- The gesture guide must be aligned with `docs/gesture_contract.md`.
- UI task sequences and backend task scenario definitions must be aligned before live task metrics are interpreted.
- Zoom semantics are not fully aligned between IPN labels, UI guidance, and live scale-change logic.
- The live demo should not be used as the primary proof until annotated live sessions exist; the reproducible aggregation tooling now exists (`cli/aggregate_live_sessions`), but the annotated sessions themselves still need to be collected at scale.

## Reporting Template

Every experiment report should include:

- dataset and manifest paths;
- model artifacts and git commit;
- method definitions;
- task scenario file;
- risk-cost file;
- offline recognition metrics;
- online proposal metrics;
- action-level AR metrics;
- confidence intervals or repeated trials where applicable;
- explicit statement of what the experiment does and does not prove.
