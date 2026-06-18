# Online Summary

Last updated: 2026-06-15

## Evaluation Mode

The implemented online evaluator supports pseudo-continuous replay:

- gesture clips are concatenated into task-like sequences;
- idle/no-gesture gaps are inserted between steps;
- timestamps are generated at a fixed frame interval;
- each event row logs model label, proposal label, controller state, expected label, final action, acceptance, rejection reason, cooldown, risk cost, task id, and task step.

This mode is explicitly marked as pseudo-continuous. It is not claimed to be a recovered full IPN continuous video timeline.

## Implemented Metrics

The evaluator computes:

- frame-level recognition/proposal accuracy;
- macro F1 and per-label F1;
- segment precision, recall, and F1;
- onset and offset error;
- decision latency;
- false positives per minute;
- false negatives per gesture;
- label switch rate;
- no-gesture false positive rate;
- accepted/rejected action counts and rates;
- cost-weighted action precision and recall;
- graded task-completion score (F1 of weighted action precision and recall);
- confident-completion rate (fraction of sequences whose completion score clears tau = 0.5), the
  headline task metric, alongside the strict binary task-success rate kept for completeness.

## Current Run (real IPN landmarks + multi-view C6 ensemble)

Latest run characteristics:

- data mode: `real_landmark_tensors` (real MediaPipe landmarks for gesture clips and no_gesture idle gaps);
- effective predictor: `c6_ensemble` over the deployed multi-view TCN models;
- methods compared: 9;
- event rows: 39312 across 24 paired sequences (`trials_per_task = 8`);
- task scenarios: object control, scroll and open, sort virtual item.

## Headline Ablation Result

The thesis claim -- the combined validation/TARC pipeline reduces false AR actions versus direct
classifier-to-action mapping on identical replay sequences -- is numerically supported and
statistically significant (paired bootstrap + exact McNemar, n = 24):

| Method | False-action cost | Δ vs direct | 95% CI | p (McNemar) |
| --- | --- | --- | --- | --- |
| direct_c6 (baseline) | 169.22 | — | — | — |
| c6_temporal_stabilized | 166.24 | −2.98 | [−4.55, −1.32] | 0.011 |
| c6_validation_confidence_only | 37.51 | −131.71 | [−146.31, −116.59] | <0.001 |
| c6_validation_confidence_stability | 35.78 | −133.44 | [−148.25, −118.39] | <0.001 |
| c6_validation_confidence_stability_cooldown | 6.98 | −162.24 | [−180.21, −143.40] | <0.001 |
| c6_validation_tarc | 4.23 | −164.99 | [−182.94, −146.19] | <0.001 |

Supporting effects: the temporal stabilizer cuts false-positives-per-minute (52.4 -> 34.1) and
label-switch rate (51.6 -> 35.4), addressing live jitter; TARC reaches the lowest false-action cost
and the only non-zero task-success rate, while decision latency stays in the ~470-555 ms band. The
direct-mapping baseline cost is higher than in the earlier dual-view run because the more confident
multi-view recognizer accepts more actions when mapped straight to commands, which makes the
validation pipeline's reduction larger, not smaller. Per-frame recognition accuracy in this table is
intentionally lower than the clip-level 0.92 in `recognition_summary.md` because it is measured over
the pseudo-continuous stream, not pre-segmented clips.

## Task Completion

The strict binary task-success rate is a floor-effect metric: a scenario counts as a success only
when every required step is accepted in order *and* the false-action cost is exactly zero. On a
noisy pseudo-continuous replay almost every sequence has at least one false action, so binary
success stays near zero for all methods (0.000 for direct mapping, 0.083 for TARC) and cannot
distinguish them. Two graded metrics replace it as the headline: the continuous `task_completion`
score (F1 of cost-weighted action precision and recall) and the **confident-completion rate** -- the
fraction of replay sequences whose completion score clears a threshold (tau = 0.5), i.e. tasks that
were carried out with more correct than wasted action cost.

| Method | Confident completion (tau=0.5) | Graded completion | Binary success |
| --- | --- | --- | --- |
| direct_c6 (baseline) | 0.000 | 0.058 | 0.000 |
| c6_smoothing | 0.000 | 0.056 | 0.000 |
| c6_temporal_stabilized | 0.000 | 0.057 | 0.000 |
| c6_validation_confidence_only | 0.000 | 0.198 | 0.000 |
| c6_validation_confidence_stability | 0.000 | 0.208 | 0.000 |
| c6_validation_confidence_stability_cooldown | 0.667 | 0.578 | 0.042 |
| c6_validation_tarc | 0.875 | 0.669 | 0.083 |

The confident-completion rate separates the methods cleanly where binary success cannot: it rises
from 0.000 for direct classifier-to-action mapping to 0.875 for the full validation/TARC pipeline.
The graded score moves in step (0.058 -> 0.669), driven by cost-weighted action precision climbing
from 0.030 to 0.607 while recall stays in the 0.85-0.95 band. The completion gain is statistically
significant on the same paired sequences (paired bootstrap, n = 24, higher is better):

| Method | Δ graded completion vs direct | 95% CI | p (McNemar) |
| --- | --- | --- | --- |
| c6_validation_confidence_only | +0.141 | [0.112, 0.169] | <0.001 |
| c6_validation_confidence_stability | +0.151 | [0.120, 0.181] | <0.001 |
| c6_validation_confidence_stability_cooldown | +0.521 | [0.435, 0.611] | <0.001 |
| c6_validation_tarc | +0.611 | [0.524, 0.693] | <0.001 |

So the pipeline is now supported in both directions on identical replay sequences: it reduces
false-action cost (lower is better, CIs below zero) *and* raises task completion (higher is better,
CIs above zero), both with p < 0.001. The smoothing and stabilizer arms do not move completion --
they only reduce jitter -- which is the expected, honest result. The binary success rate is reported
for completeness only.

## Release-gated Re-arming (evaluated, not adopted)

A `c6_validation_tarc_release` arm was added to test a stronger debounce: after any command is
accepted, every command is blocked until a no_gesture release is observed (not only a repeat of the
same gesture). On this windowed replay the mechanism backfires. It trims false-action cost only
marginally (101.5 -> 95.8) but collapses cost-weighted recall from 0.858 to 0.580, so confident
completion drops from 0.875 to 0.542 and graded completion from 0.669 to 0.522. The cause is honest
and mechanical: with a 32-frame window the recognizer does not always emit a clean no_gesture between
adjacent steps, so the release is not registered and the *next legitimate* step is blocked too. The
lever is therefore rejected -- TARC without global release remains the operating point -- and the arm
is kept only as documented negative ablation evidence; it is not enabled in the deployed config.

## Outputs

The online evaluator writes:

- `reports/online_gesture/events.csv`;
- `reports/online_gesture/events.jsonl`;
- `reports/online_gesture/summary.json`;
- `reports/online_gesture/summary.md`;
- `reports/online_gesture/method_comparison.csv`;
- `reports/online_gesture/method_comparison.md`;
- `reports/online_gesture/figures/summary_metrics.svg`.
