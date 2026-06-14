# Ablation Summary

Last updated: 2026-06-15

## Implemented Ablation Modes

The online benchmark compares:

- `direct_c6`: raw prediction mapped directly to action;
- `c6_smoothing`: temporal score smoothing before direct action mapping;
- `c6_validation_confidence_only`: confidence gate only;
- `c6_validation_confidence_stability`: confidence plus temporal stability;
- `c6_validation_confidence_stability_cooldown`: confidence, stability, cooldown, and release checks;
- `c6_validation_tarc`: validation plus task-aware expected gesture and risk policy;
- `landmark_controller`: live landmark controller baseline when dependencies are available;
- `landmark_controller_tarc`: landmark controller plus validation/TARC.

## Current Comparison Table

Latest smoke run:

| Method | Effective predictor | Accuracy | Segment F1 | FP/min | Switch/min | Accepted | Rejected | False-action cost | Missed-action cost | Task success |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_c6` | `rule_based` | 0.290323 | 0.457143 | 41.893590 | 33.514872 | 504 | 0 | 925.0 | 3.0 | 0.0 |
| `c6_smoothing` | `rule_based` | 0.290323 | 0.484848 | 36.307778 | 27.929060 | 495 | 0 | 905.0 | 3.0 | 0.0 |
| `c6_validation_confidence_only` | `rule_based` | 0.287250 | 0.457143 | 41.893590 | 33.514872 | 54 | 297 | 76.0 | 3.0 | 0.0 |
| `c6_validation_confidence_stability` | `rule_based` | 0.290323 | 0.457143 | 41.893590 | 33.514872 | 45 | 327 | 62.0 | 3.0 | 0.0 |
| `c6_validation_confidence_stability_cooldown` | `rule_based` | 0.290323 | 0.457143 | 41.893590 | 33.514872 | 11 | 460 | 7.0 | 5.0 | 0.0 |
| `c6_validation_tarc` | `rule_based` | 0.290323 | 0.457143 | 41.893590 | 33.514872 | 8 | 472 | 2.0 | 5.0 | 0.0 |
| `landmark_controller` | `rule_based` | 0.164363 | 0.434783 | 16.757436 | 22.343248 | 645 | 0 | 181.75 | 5.25 | 0.0 |
| `landmark_controller_tarc` | `rule_based` | 0.164363 | 0.476190 | 11.171624 | 16.757436 | 257 | 379 | 63.0 | 8.5 | 0.0 |

## Interpretation

The current smoke run shows action-level improvement, not recognition-level improvement. Validation does not improve frame accuracy in the fallback run, because the same effective predictor is used and the proposal stream still records rejected candidate labels. The improvement appears in accepted actions and false-action cost.

The strongest current action-level reduction is:

```text
direct_c6 false-action cost: 925.0
c6_validation_tarc false-action cost: 2.0
```

This supports the intended research hypothesis only at smoke-test level. Final claims require real C6 artifacts and processed landmark tensors.
