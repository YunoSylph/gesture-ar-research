# Recognition Summary

Last updated: 2026-06-15

## What Is Implemented

The repository contains an IPN Hand based recognition stack with:

- selected label mapping for `no_gesture`, `point_2f`, `click_2f`, `swipe_left`, `swipe_right`, `zoom_in`, and `zoom_out`;
- MediaPipe 21-landmark tensor extraction from the public IPN Hand videos;
- a temporal TCN recognizer, trained in two variants that form the robust C6 ensemble;
- a rule-based fallback recognizer for smoke tests;
- live backend integration that logs raw model output and controller output separately.

## Offline Recognition Results (real IPN Hand test split)

Source data: 1033 test clips, MediaPipe landmarks extracted from the official IPN Hand `.avi` videos
(`data/interim/manifests/ipn_test_full_landmarks.jsonl`, `data_mode = real_landmark_tensors`).
These are clip-level (pre-segmented) classification metrics.

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced Acc |
| --- | --- | --- | --- | --- |
| C1-T TCN (validated, dual-view) | 0.9071 | 0.8502 | 0.9109 | 0.8966 |
| C1-T TCN (augmented, dual-view) | 0.9090 | 0.8565 | — | — |
| C1-T TCN (validated + multi-view) | 0.9197 | 0.8623 | 0.9203 | 0.8724 |

Per-class F1 (validated model):

| Class | F1 | Precision | Recall | Support |
| --- | --- | --- | --- | --- |
| no_gesture | 0.936 | 0.968 | 0.906 | 509 |
| point_2f | 0.961 | 0.996 | 0.928 | 264 |
| zoom_in | 0.883 | 0.831 | 0.942 | 52 |
| swipe_right | 0.862 | 0.825 | 0.904 | 52 |
| click_2f | 0.789 | 0.726 | 0.865 | 52 |
| zoom_out | 0.764 | 0.724 | 0.808 | 52 |
| swipe_left | 0.756 | 0.640 | 0.923 | 52 |

The weakest classes (`swipe_left`, `zoom_out`, `click_2f`) are the low-support, motion- or
finger-distance-dependent gestures, consistent with the OO-dMVMT observation that dynamic/fine
gestures are the hardest to detect cleanly.

### Multi-view feature block (OO-dMVMT-inspired)

Following OO-dMVMT (Cunico et al. 2023, Sec. 3.1), the per-frame feature vector can be extended with
a multi-view block: Joint Collection Distances (all pairwise joint distances, scale-normalised) plus
slow and fast per-joint motion. This is appended to the existing pose+motion stream
(`feature_set: dual_view_multiview`), recorded in the model artifact, and applied identically at
training and inference. Re-training the validated TCN with the block (same hyper-parameters)
moves clip-level accuracy 0.9071 → 0.9197 and macro F1 0.8502 → 0.8623. The gain is concentrated
in exactly the motion/distance-dependent weak classes the reference paper targets, with two
directional regressions, reported honestly:

| Class | F1 (dual-view) | F1 (multi-view) | Δ |
| --- | --- | --- | --- |
| `swipe_left` | 0.756 | 0.847 | +0.091 |
| `zoom_out` | 0.764 | 0.809 | +0.045 |
| `click_2f` | 0.789 | 0.810 | +0.021 |
| `no_gesture` | 0.936 | 0.944 | +0.008 |
| `point_2f` | 0.961 | 0.969 | +0.008 |
| `swipe_right` | 0.862 | 0.819 | −0.043 |
| `zoom_in` | 0.883 | 0.838 | −0.045 |

At the C6 ensemble level the multi-view models are the deployed configuration: clean accuracy rises
0.9235 → 0.9274, the safety-config ECE improves 0.0202 → 0.0146, and the mean perturbed false-action
rate drops 0.074 → 0.068, with macro F1 unchanged within noise. Reproduce:
`python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_full_validated_mv.yaml`
then `... benchmark_recognition --config configs/eval/ipn_c1t_tcn_full_validated_mv.yaml`.

## Confidence Calibration (real IPN Hand test split)

The recognition stack is described as "calibrated", so the study measures calibration directly
instead of assuming it, and the candidate-selection objective now co-optimises it. The metrics below
are the deployed multi-view ensemble, on the clean evaluation scenario of the C6 fusion run
(`artifacts/reports/c6_ensemble_calibrated_recognition_mv.json`, 1033 test clips, 15 equal-width
reliability bins): expected calibration error (ECE), maximum calibration error (MCE), the
multi-class Brier score, and the signed gap between mean confidence and accuracy (overconfidence).
`c1t_direct` is the two-TCN ensemble before fusion; `c5_safety` is the fusion configuration
selected for the safety objective; `c5_macro` is the configuration selected for macro F1.

| Method | Accuracy | Macro F1 | ECE | MCE | Brier | Overconfidence |
| --- | --- | --- | --- | --- | --- | --- |
| c1t_direct (ensemble) | 0.9255 | 0.8731 | 0.0207 | 0.7162 | 0.1186 | −0.0155 |
| c3_hybrid | 0.9284 | 0.8771 | 0.0658 | 0.3619 | 0.1225 | −0.0658 |
| c5_macro | 0.9206 | 0.8669 | 0.0261 | 0.3316 | 0.1229 | +0.0089 |
| c5_safety | 0.9274 | 0.8778 | 0.0146 | 0.3125 | 0.1157 | +0.0097 |

The selection objective ranks candidates by macro F1 plus a weak-class bonus, minus a no-gesture
false-action penalty, minus an `ece_penalty` term on the expected calibration error (configured in
the `objective` block). The ECE term matters: without it the safety configuration reached the best
accuracy but the *worst* calibration (on the dual-view ensemble its ECE was 0.087, Brier 0.125),
because the bias and temperature search optimised macro F1 and the no-gesture margin alone. With
`ece_penalty = 0.5` the selected safety configuration is the best-calibrated of all variants
(ECE 0.0146, Brier 0.1157, both below the raw ensemble) while keeping the best accuracy, and its
overconfidence is close to zero rather than under-confident. The cost is small and is reported, not
hidden: macro F1 sits within noise of the raw ensemble. The high MCE reflects sparse
extreme-confidence bins and is not a headline result.

Reproduce: `python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition_mv.yaml`.

## Clip-Level vs Online Frame-Level

These offline numbers are clip-level classification on pre-segmented clips. They are intentionally
kept separate from the online action-validation metrics in `reports/online_gesture/`, whose
frame-level recognition figures are lower because they are computed over a pseudo-continuous stream
(idle gaps, window misalignment, transitions). Do not conflate the two: clip accuracy ~0.91 measures
the recognizer; the online tables measure the gesture-to-action pipeline.

## Reproduce

```bash
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_full_validated.yaml
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_augmented.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_full_validated.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/c6_augmented_robustness.yaml
```
