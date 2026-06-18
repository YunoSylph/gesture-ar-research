# Ablation Summary

Last updated: 2026-06-18

## Implemented Ablation Modes

The online benchmark compares the same replay sequences under several controller policies:

- `direct_c6`: raw C6 prediction mapped directly to action;
- `c6_smoothing`: temporal score smoothing before direct mapping;
- `c6_temporal_stabilized`: additional label stabilizer for jitter reduction;
- `c6_validation_confidence_only`: confidence gate only;
- `c6_validation_confidence_stability`: confidence plus short temporal stability;
- `c6_validation_confidence_stability_cooldown`: confidence, stability, cooldown, and release checks;
- `c6_validation_tarc`: full validation plus task-aware expected gesture and risk policy;
- `c6_validation_tarc_release`: stricter global release gate, evaluated as a negative ablation;
- `landmark_controller` and `landmark_controller_tarc`: engineering baselines.

## Current Comparison Table

Latest real-landmark pseudo-continuous run with the deployed multi-view C6 ensemble:

| Method | Accuracy | Segment F1 | FP/min | Switch/min | Accepted | Rejected | False-action cost | Graded completion | Confident completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_c6` | 0.3576 | 0.4898 | 52.45 | 51.62 | 3218 | 0 | 4061.25 | 0.058 | 0.000 |
| `c6_smoothing` | 0.3363 | 0.5414 | 40.79 | 41.21 | 3202 | 0 | 4038.25 | 0.056 | 0.000 |
| `c6_temporal_stabilized` | 0.3075 | 0.5645 | 34.13 | 35.38 | 3172 | 0 | 3989.75 | 0.057 | 0.000 |
| `c6_validation_confidence_only` | 0.3519 | 0.4948 | 51.20 | 51.20 | 1037 | 462 | 900.25 | 0.198 | 0.000 |
| `c6_validation_confidence_stability` | 0.3555 | 0.4932 | 51.62 | 52.03 | 1010 | 572 | 858.75 | 0.208 | 0.000 |
| `c6_validation_confidence_stability_cooldown` | 0.3574 | 0.4898 | 52.45 | 51.62 | 560 | 2355 | 167.50 | 0.578 | 0.667 |
| `c6_validation_tarc` | 0.3567 | 0.4898 | 52.45 | 51.62 | 461 | 2543 | 101.50 | 0.669 | 0.875 |
| `c6_validation_tarc_release` | 0.3571 | 0.4898 | 52.45 | 51.62 | 437 | 2638 | 95.75 | 0.522 | 0.542 |
| `landmark_controller` | 0.1218 | 0.1552 | 138.20 | 138.20 | 4018 | 0 | 1650.50 | 0.085 | 0.000 |
| `landmark_controller_tarc` | 0.1401 | 0.2676 | 67.85 | 69.51 | 1494 | 2313 | 391.75 | 0.238 | 0.083 |

## Interpretation

The ablation supports an action-validation claim, not a frame-recognition claim. Smoothing and stabilizing reduce jitter metrics but do not solve task completion by themselves. The main gain appears when the system stops treating every classifier output as an immediate AR command.

The strongest operating point is `c6_validation_tarc`:

```text
direct_c6 mean false-action cost: 169.22 per sequence-task
c6_validation_tarc mean false-action cost: 4.23 per sequence-task
direct_c6 graded completion: 0.058
c6_validation_tarc graded completion: 0.669
```

The paired comparison reports a mean false-action-cost reduction of `-164.99` with a 95% bootstrap CI of `[-182.94, -146.19]` and `p < 0.001`. Graded task completion improves by `+0.611`, also with `p < 0.001`.

`c6_validation_tarc_release` is deliberately not adopted: although it trims cost slightly further, it blocks legitimate next steps when the 32-frame window does not emit a clean `no_gesture` release between adjacent gestures, reducing completion from `0.669` to `0.522`.
