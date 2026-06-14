# C6 Recognition Upgrade

## Motivation

The previous C3 layer produced only a small recognition gain over the validated C1-TCN backbone. The weak points were concentrated in rare classes: `swipe_left`, `zoom_out`, and `click_2f`, while `no_gesture` false actions remained important for AR safety.

## Implemented Upgrade

C6 adds a stronger recognition stack:

- augmented TCN backbone trained with a wider/deeper TCN, `avgmax` temporal pooling, class-balanced sampling, focal loss, label smoothing, and online feature/time augmentation;
- ensemble inference over the original validated TCN and the augmented TCN;
- calibrated C5/C6 score fusion over neural ensemble scores and C3 geometry-aware scores;
- class-bias calibration for weak classes and `no_gesture` safety.

## Main Results

| Method | Clean Accuracy | Clean Macro F1 | Robust Macro F1 Mean | Robust False Action Rate |
|---|---:|---:|---:|---:|
| C1-TCN validated | 0.907 | 0.850 | 0.826 | 0.097 |
| C3 hybrid validated | 0.908 | 0.851 | 0.828 | 0.094 |
| Augmented TCN | 0.909 | 0.856 | 0.835 | 0.090 |
| Augmented C3 hybrid | 0.913 | 0.866 | 0.838 | 0.086 |
| C6 ensemble calibrated | 0.930 | 0.887 | 0.859 | 0.067 |

## Per-Class Clean F1 Changes

| Class | C1-TCN | C6 Ensemble | Delta |
|---|---:|---:|---:|
| `click_2f` | 0.789 | 0.797 | +0.007 |
| `no_gesture` | 0.936 | 0.957 | +0.021 |
| `point_2f` | 0.961 | 0.957 | -0.004 |
| `swipe_left` | 0.756 | 0.914 | +0.158 |
| `swipe_right` | 0.862 | 0.839 | -0.023 |
| `zoom_in` | 0.883 | 0.913 | +0.030 |
| `zoom_out` | 0.764 | 0.832 | +0.068 |

## Interpretation

The upgrade is now research-significant: it improves the exact weak classes that limited the earlier implementation, while also lowering false actions for `no_gesture`. The remaining weakness is the `swipe_right` trade-off, where the stronger correction for `swipe_left` slightly reduces `swipe_right` F1. This is acceptable for the current AR demo, but the next research iteration should target symmetric directional calibration.

## Reproduction

```powershell
.\.venv311\Scripts\python.exe -m research_pipeline.cli.train --config configs\train\ipn_c1t_tcn_augmented.yaml
.\.venv311\Scripts\python.exe -m research_pipeline.cli.benchmark_recognition --config configs\eval\ipn_c1t_tcn_augmented.yaml
.\.venv311\Scripts\python.exe -m research_pipeline.cli.benchmark_c3_hybrid --config configs\eval\c6_augmented_robustness.yaml
.\.venv311\Scripts\python.exe -m research_pipeline.cli.run_c5_calibrated_recognition --config configs\eval\c6_ensemble_calibrated_recognition.yaml
```
