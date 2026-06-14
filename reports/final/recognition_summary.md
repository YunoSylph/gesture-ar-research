# Recognition Summary

Last updated: 2026-06-15

## What Is Implemented

The repository contains an IPN Hand based recognition stack with:

- selected label mapping for `no_gesture`, `point_2f`, `click_2f`, `swipe_left`, `swipe_right`, `zoom_in`, and `zoom_out`;
- MediaPipe-style 21-landmark tensor representation;
- temporal TCN/C6 recognizer interfaces;
- rule-based fallback recognizer for smoke tests;
- live backend integration that logs raw model output and controller output separately.

## Current Evaluation Status

The online evaluator currently found the manifest but did not find the referenced processed tensor files or C6 model artifacts in the workspace used for this run:

- manifest records: 1033;
- raw videos found: 0;
- tensor files found: 0;
- C6 model artifacts found: 0.

Because of this, the latest online comparison used synthetic fallback landmarks and the rule-based fallback predictor. These numbers are not final C6 recognition results and should not be used as public benchmark performance.

## Required Final Recognition Run

For the thesis-grade recognition table, restore or regenerate:

- `data/interim/manifests/ipn_test_full_landmarks.jsonl` tensor targets;
- landmark `.npz` tensors referenced by the manifest;
- C6 model artifacts configured in `configs/eval/online_gesture.yaml`.

Then run:

```bash
python -m research_pipeline.cli.benchmark_online_gesture --config configs/eval/online_gesture.yaml --output-dir reports/online_gesture
```

The final report should include offline accuracy, macro F1, weighted F1, per-class F1, and confusion matrix from the C6 benchmark, and should keep those results separate from online action-validation metrics.
