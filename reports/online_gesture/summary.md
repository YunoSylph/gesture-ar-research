# Online Gesture Evaluation Summary

Mode: `pseudo_continuous`
Data mode: `real_landmark_tensors`
Manifest: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\data\interim\manifests\ipn_test_full_landmarks.jsonl`
Predictor: `c6_ensemble`

## Data Availability

- `manifest_records`: 1033
- `manifest_path`: C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\data\interim\manifests\ipn_test_full_landmarks.jsonl
- `sessions`: 52
- `label_counts`: {'no_gesture': 509, 'zoom_in': 52, 'zoom_out': 52, 'point_2f': 264, 'swipe_right': 52, 'click_2f': 52, 'swipe_left': 52}
- `manifest_has_clip_timestamps`: True
- `ipn_annotation_order_recoverable`: True
- `continuous_timeline_available`: False
- `continuous_timeline_reason`: Real extracted landmark tensors are available for every manifest clip, but the dataset stores segmented clips rather than the original uncut IPN recordings. The evaluator therefore builds an explicitly marked pseudo-continuous stream by concatenating real gesture clips with real no_gesture idle gaps; this is real-clip replay, not the original continuous IPN timeline.
- `raw_videos_found`: 1033
- `tensor_paths_declared`: 1033
- `tensor_files_found`: 1033
- `model_paths_declared`: 2
- `model_artifacts_found`: 2

## Limitations

- Real extracted landmark tensors are available for every manifest clip, but the dataset stores segmented clips rather than the original uncut IPN recordings. The evaluator therefore builds an explicitly marked pseudo-continuous stream by concatenating real gesture clips with real no_gesture idle gaps; this is real-clip replay, not the original continuous IPN timeline.
- OO-dMVMT is used as methodological direction for online classification/segmentation metrics; this benchmark does not compare against OO-dMVMT numeric results.
- Gesture and idle segments use real extracted MediaPipe landmarks from IPN Hand clips; the only synthetic element is the concatenation order (pseudo-continuous replay).

## Metrics

- `accepted_action_count`: 461
- `accepted_action_rate_per_minute`: 191.891442
- `accepted_action_ratio`: 0.143168
- `decision_latency_ms_mean`: 469.764706
- `decision_latency_ms_median`: 429.000000
- `false_negatives_per_gesture`: 0.250000
- `false_positives_per_minute`: 52.447552
- `frame_accuracy`: 0.356685
- `frame_accuracy_model`: 0.357601
- `frame_accuracy_proposal`: 0.356685
- `label_switch_rate_per_minute`: 51.615052
- `macro_f1`: 0.333101
- `macro_f1_model`: 0.334014
- `macro_f1_proposal`: 0.333101
- `matched_segments`: 72
- `no_gesture_false_positive_rate`: 0.800154
- `offset_error_ms_mean`: 393.250000
- `offset_error_ms_median`: 429.000000
- `onset_error_ms_mean`: 449.625000
- `onset_error_ms_median`: 462.000000
- `predicted_segments`: 198
- `recognition_accuracy`: 0.356685
- `rejected_action_count`: 2543
- `rejected_action_rate_per_minute`: 1058.524809
- `rejected_action_ratio`: 0.789752
- `segment_f1`: 0.489796
- `segment_precision`: 0.363636
- `segment_recall`: 0.750000
- `true_segments`: 96

## Outputs

- `events_csv`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\events.csv`
- `events_jsonl`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\events.jsonl`
- `summary_json`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\summary.json`
- `summary_md`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\summary.md`
- `method_comparison_csv`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\method_comparison.csv`
- `method_comparison_md`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\method_comparison.md`
- `figures_dir`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\figures`
- `summary_figure`: `C:\Users\Maksim Iuzhakov\Desktop\Another_one_bite\gesture-ar-research-2nd\reports\online_gesture\figures\summary_metrics.svg`
