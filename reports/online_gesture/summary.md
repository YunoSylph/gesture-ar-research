# Online Gesture Evaluation Summary

Mode: `pseudo_continuous`
Data mode: `synthetic_fallback_pseudo_continuous`
Manifest: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\data\interim\manifests\ipn_test_full_landmarks.jsonl`
Predictor: `rule_based`

## Data Availability

- `manifest_records`: 1033
- `manifest_path`: C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\data\interim\manifests\ipn_test_full_landmarks.jsonl
- `sessions`: 52
- `label_counts`: {'no_gesture': 509, 'zoom_in': 52, 'zoom_out': 52, 'point_2f': 264, 'swipe_right': 52, 'click_2f': 52, 'swipe_left': 52}
- `manifest_has_clip_timestamps`: True
- `ipn_annotation_order_recoverable`: True
- `continuous_timeline_available`: False
- `continuous_timeline_reason`: Full continuous IPN timeline is not available: the manifest stores selected/remapped clips with timestamps, but raw videos are absent and gaps may contain omitted IPN classes or unannotated motion. The evaluator therefore runs in explicitly marked pseudo-continuous mode.
- `raw_videos_found`: 0
- `tensor_paths_declared`: 1033
- `tensor_files_found`: 0
- `model_paths_declared`: 2
- `model_artifacts_found`: 0

## Limitations

- Full continuous IPN timeline is not available: the manifest stores selected/remapped clips with timestamps, but raw videos are absent and gaps may contain omitted IPN classes or unannotated motion. The evaluator therefore runs in explicitly marked pseudo-continuous mode.
- OO-dMVMT is used as methodological direction for online classification/segmentation metrics; this benchmark does not compare against OO-dMVMT numeric results.
- Configured C6 model artifacts were not found: ['C:\\Users\\Maksim Iuzhakov\\Desktop\\Hand Gestures Project\\gesture-ar-research\\artifacts\\models\\ipn_c1t_tcn_full_validated.pkl', 'C:\\Users\\Maksim Iuzhakov\\Desktop\\Hand Gestures Project\\gesture-ar-research\\artifacts\\models\\ipn_c1t_tcn_augmented.pkl'].
- Using rule_based predictor fallback for evaluator smoke run.
- C6 artifacts are unavailable in this run; C6-named baselines use effective predictor 'rule_based'.
- 12 task replay clips referenced missing tensor files and were replaced by synthetic landmarks.
- This run used synthetic fallback landmarks because processed tensors were not available; do not interpret recognition metrics as public benchmark results.

## Metrics

- `accepted_action_count`: 8
- `accepted_action_rate_per_minute`: 22.343248
- `accepted_action_ratio`: 0.015873
- `decision_latency_ms_mean`: 449.625000
- `decision_latency_ms_median`: 264.000000
- `false_negatives_per_gesture`: 0.333333
- `false_positives_per_minute`: 41.893590
- `frame_accuracy`: 0.290323
- `frame_accuracy_model`: 0.290323
- `frame_accuracy_proposal`: 0.290323
- `label_switch_rate_per_minute`: 33.514872
- `macro_f1`: 0.192456
- `macro_f1_model`: 0.192456
- `macro_f1_proposal`: 0.192456
- `matched_segments`: 8
- `no_gesture_false_positive_rate`: 0.962963
- `offset_error_ms_mean`: 148.500000
- `offset_error_ms_median`: 0.000000
- `onset_error_ms_mean`: 301.125000
- `onset_error_ms_median`: 478.500000
- `predicted_segments`: 23
- `recognition_accuracy`: 0.290323
- `rejected_action_count`: 472
- `rejected_action_rate_per_minute`: 1318.251641
- `rejected_action_ratio`: 0.936508
- `segment_f1`: 0.457143
- `segment_precision`: 0.347826
- `segment_recall`: 0.666667
- `true_segments`: 12

## Outputs

- `events_csv`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\events.csv`
- `events_jsonl`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\events.jsonl`
- `summary_json`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\summary.json`
- `summary_md`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\summary.md`
- `method_comparison_csv`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\method_comparison.csv`
- `method_comparison_md`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\method_comparison.md`
- `figures_dir`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\figures`
- `summary_figure`: `C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\reports\online_gesture\figures\summary_metrics.svg`
