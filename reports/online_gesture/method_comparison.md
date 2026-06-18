# Online Gesture Method Comparison

| method | effective_predictor | recognition_accuracy | macro_f1 | segment_f1 | false_positives_per_minute | label_switch_rate | decision_latency_ms | accepted_actions | rejected_actions | false_action_cost | missed_action_cost | action_precision | action_recall | task_completion | confident_completion | task_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_c6 | c6_ensemble | 0.357601 | 0.334014 | 0.489796 | 52.447552 | 51.615052 | 468.988235 | 3218 | 0 | 4061.25 | 5.5 | 0.029776 | 0.953509 | 0.05752 | 0.0 | 0.0 |
| c6_smoothing | c6_ensemble | 0.33631 | 0.315048 | 0.541353 | 40.792541 | 41.208791 | 508.588235 | 3202 | 0 | 4038.25 | 8.0 | 0.029222 | 0.931579 | 0.056425 | 0.0 | 0.0 |
| c6_temporal_stabilized | c6_ensemble | 0.307463 | 0.286238 | 0.564516 | 34.132534 | 35.381285 | 555.036145 | 3172 | 0 | 3989.75 | 9.5 | 0.029338 | 0.920092 | 0.056637 | 0.0 | 0.0 |
| c6_validation_confidence_only | c6_ensemble | 0.351877 | 0.327882 | 0.494845 | 51.198801 | 51.198801 | 475.588235 | 1037 | 462 | 900.25 | 13.75 | 0.113961 | 0.88325 | 0.198483 | 0.0 | 0.0 |
| c6_validation_confidence_stability | c6_ensemble | 0.35554 | 0.331921 | 0.493151 | 51.615052 | 52.031302 | 475.976471 | 1010 | 572 | 858.75 | 13.75 | 0.120787 | 0.88325 | 0.208324 | 0.0 | 0.0 |
| c6_validation_confidence_stability_cooldown | c6_ensemble | 0.357372 | 0.333752 | 0.489796 | 52.447552 | 51.615052 | 469.376471 | 560 | 2355 | 167.5 | 17.75 | 0.487536 | 0.849916 | 0.5783 | 0.666667 | 0.041667 |
| c6_validation_tarc | c6_ensemble | 0.356685 | 0.333101 | 0.489796 | 52.447552 | 51.615052 | 469.764706 | 461 | 2543 | 101.5 | 16.75 | 0.607375 | 0.85825 | 0.669003 | 0.875 | 0.083333 |
| c6_validation_tarc_release | c6_ensemble | 0.357143 | 0.333547 | 0.489796 | 52.447552 | 51.615052 | 469.376471 | 437 | 2638 | 95.75 | 50.5 | 0.541589 | 0.580221 | 0.521683 | 0.541667 | 0.083333 |
| landmark_controller | c6_ensemble | 0.121795 | 0.077115 | 0.155172 | 138.195138 | 138.195138 | 601.071429 | 4018 | 0 | 1650.5 | 39.75 | 0.045421 | 0.667356 | 0.084807 | 0.0 | 0.0 |
| landmark_controller_tarc | c6_ensemble | 0.14011 | 0.099582 | 0.267559 | 67.848818 | 69.51382 | 531.0 | 1494 | 2313 | 391.75 | 53.5 | 0.152873 | 0.552235 | 0.237638 | 0.083333 | 0.0 |

## Paired Comparison vs direct_c6 (lower is better)

Per-(sequence, task) paired bootstrap. `delta` = method - baseline; a `delta_ci_high` below 0 means the reduction is significant at the chosen level. `p_value` is the exact McNemar test.

| method | metric | baseline_mean | method_mean | delta | delta_ci_low | delta_ci_high | prob_improvement | p_value | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c6_smoothing | false_action_cost | 169.2188 | 168.2604 | -0.9583 | -1.9375 | 0.0104 | 0.9730 | 0.4049 | 24 |
| c6_smoothing | false_actions | 130.3333 | 129.7500 | -0.5833 | -1.2500 | 0.0417 | 0.9545 | 0.5034 | 24 |
| c6_smoothing | task_completion_score | 0.0575 | 0.0564 | -0.0011 | -0.0048 | 0.0010 | 0.3530 | 0.6776 | 24 |
| c6_temporal_stabilized | false_action_cost | 169.2188 | 166.2396 | -2.9792 | -4.5521 | -1.3229 | 1.0000 | 0.0106 | 24 |
| c6_temporal_stabilized | false_actions | 130.3333 | 128.5000 | -1.8333 | -2.8333 | -0.8750 | 1.0000 | 0.0013 | 24 |
| c6_temporal_stabilized | task_completion_score | 0.0575 | 0.0566 | -0.0009 | -0.0043 | 0.0017 | 0.3200 | 0.0347 | 24 |
| c6_validation_confidence_only | false_action_cost | 169.2188 | 37.5104 | -131.7083 | -146.3052 | -116.5932 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_only | false_actions | 130.3333 | 39.6667 | -90.6667 | -99.1260 | -81.5406 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_only | task_completion_score | 0.0575 | 0.1985 | 0.1410 | 0.1122 | 0.1689 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability | false_action_cost | 169.2188 | 35.7812 | -133.4375 | -148.2508 | -118.3940 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability | false_actions | 130.3333 | 38.5417 | -91.7917 | -100.3354 | -82.7917 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability | task_completion_score | 0.0575 | 0.2083 | 0.1508 | 0.1200 | 0.1809 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability_cooldown | false_action_cost | 169.2188 | 6.9792 | -162.2396 | -180.2091 | -143.3953 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability_cooldown | false_actions | 130.3333 | 19.9583 | -110.3750 | -121.0031 | -99.0406 | 1.0000 | 0.0000 | 24 |
| c6_validation_confidence_stability_cooldown | task_completion_score | 0.0575 | 0.5783 | 0.5208 | 0.4354 | 0.6106 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc | false_action_cost | 169.2188 | 4.2292 | -164.9896 | -182.9385 | -146.1857 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc | false_actions | 130.3333 | 15.7917 | -114.5417 | -124.6677 | -103.9167 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc | task_completion_score | 0.0575 | 0.6690 | 0.6115 | 0.5242 | 0.6927 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc_release | false_action_cost | 169.2188 | 3.9896 | -165.2292 | -183.1589 | -146.4656 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc_release | false_actions | 130.3333 | 15.6667 | -114.6667 | -124.8333 | -103.9583 | 1.0000 | 0.0000 | 24 |
| c6_validation_tarc_release | task_completion_score | 0.0575 | 0.5217 | 0.4642 | 0.3562 | 0.5806 | 1.0000 | 0.0000 | 24 |
| landmark_controller | false_action_cost | 169.2188 | 68.7708 | -100.4479 | -119.3669 | -81.9766 | 1.0000 | 0.0000 | 24 |
| landmark_controller | false_actions | 130.3333 | 164.5000 | 34.1667 | 21.6656 | 47.3771 | 0.0000 | 0.0000 | 24 |
| landmark_controller | task_completion_score | 0.0575 | 0.0848 | 0.0273 | 0.0121 | 0.0416 | 1.0000 | 0.0227 | 24 |
| landmark_controller_tarc | false_action_cost | 169.2188 | 16.3229 | -152.8958 | -170.0047 | -134.7469 | 1.0000 | 0.0000 | 24 |
| landmark_controller_tarc | false_actions | 130.3333 | 59.7500 | -70.5833 | -79.8344 | -61.2500 | 1.0000 | 0.0000 | 24 |
| landmark_controller_tarc | task_completion_score | 0.0575 | 0.2376 | 0.1801 | 0.1180 | 0.2431 | 1.0000 | 0.0000 | 24 |

## Limitations

- Real extracted landmark tensors are available for every manifest clip, but the dataset stores segmented clips rather than the original uncut IPN recordings. The evaluator therefore builds an explicitly marked pseudo-continuous stream by concatenating real gesture clips with real no_gesture idle gaps; this is real-clip replay, not the original continuous IPN timeline.
- OO-dMVMT is used as methodological direction for online classification/segmentation metrics; this benchmark does not compare against OO-dMVMT numeric results.
- Gesture and idle segments use real extracted MediaPipe landmarks from IPN Hand clips; the only synthetic element is the concatenation order (pseudo-continuous replay).
