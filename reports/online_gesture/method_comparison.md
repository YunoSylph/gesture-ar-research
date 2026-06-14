# Online Gesture Method Comparison

| method | effective_predictor | recognition_accuracy | macro_f1 | segment_f1 | false_positives_per_minute | label_switch_rate | decision_latency_ms | accepted_actions | rejected_actions | false_action_cost | missed_action_cost | task_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_c6 | rule_based | 0.290323 | 0.192456 | 0.457143 | 41.89359 | 33.514872 | 449.625 | 504 | 0 | 925.0 | 3.0 | 0.0 |
| c6_smoothing | rule_based | 0.290323 | 0.186942 | 0.484848 | 36.307778 | 27.92906 | 462.0 | 495 | 0 | 905.0 | 3.0 | 0.0 |
| c6_validation_confidence_only | rule_based | 0.28725 | 0.184948 | 0.457143 | 41.89359 | 33.514872 | 449.625 | 54 | 297 | 76.0 | 3.0 | 0.0 |
| c6_validation_confidence_stability | rule_based | 0.290323 | 0.190999 | 0.457143 | 41.89359 | 33.514872 | 449.625 | 45 | 327 | 62.0 | 3.0 | 0.0 |
| c6_validation_confidence_stability_cooldown | rule_based | 0.290323 | 0.192456 | 0.457143 | 41.89359 | 33.514872 | 449.625 | 11 | 460 | 7.0 | 5.0 | 0.0 |
| c6_validation_tarc | rule_based | 0.290323 | 0.192456 | 0.457143 | 41.89359 | 33.514872 | 449.625 | 8 | 472 | 2.0 | 5.0 | 0.0 |
| landmark_controller | rule_based | 0.164363 | 0.064602 | 0.434783 | 16.757436 | 22.343248 | 731.5 | 645 | 0 | 181.75 | 5.25 | 0.0 |
| landmark_controller_tarc | rule_based | 0.164363 | 0.064402 | 0.47619 | 11.171624 | 16.757436 | 455.4 | 257 | 379 | 63.0 | 8.5 | 0.0 |

## Limitations

- Full continuous IPN timeline is not available: the manifest stores selected/remapped clips with timestamps, but raw videos are absent and gaps may contain omitted IPN classes or unannotated motion. The evaluator therefore runs in explicitly marked pseudo-continuous mode.
- OO-dMVMT is used as methodological direction for online classification/segmentation metrics; this benchmark does not compare against OO-dMVMT numeric results.
- Configured C6 model artifacts were not found: ['C:\\Users\\Maksim Iuzhakov\\Desktop\\Hand Gestures Project\\gesture-ar-research\\artifacts\\models\\ipn_c1t_tcn_full_validated.pkl', 'C:\\Users\\Maksim Iuzhakov\\Desktop\\Hand Gestures Project\\gesture-ar-research\\artifacts\\models\\ipn_c1t_tcn_augmented.pkl'].
- Using rule_based predictor fallback for evaluator smoke run.
- C6 artifacts are unavailable in this run; C6-named baselines use effective predictor 'rule_based'.
- 12 task replay clips referenced missing tensor files and were replaced by synthetic landmarks.
- This run used synthetic fallback landmarks because processed tensors were not available; do not interpret recognition metrics as public benchmark results.
