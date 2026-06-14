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
- accepted/rejected action counts and rates.

## Current Smoke Run

Latest run characteristics:

- data mode: `synthetic_fallback_pseudo_continuous`;
- effective predictor: `rule_based`;
- methods compared: 8;
- event rows: 5208;
- task scenarios: object control, scroll and open, sort virtual item.

The run confirms that the online logging, method comparison, and task replay execute end to end. It does not yet prove final C6 performance because real tensors and C6 artifacts were unavailable.

## Outputs

The online evaluator writes:

- `reports/online_gesture/events.csv`;
- `reports/online_gesture/events.jsonl`;
- `reports/online_gesture/summary.json`;
- `reports/online_gesture/summary.md`;
- `reports/online_gesture/method_comparison.csv`;
- `reports/online_gesture/method_comparison.md`;
- `reports/online_gesture/figures/summary_metrics.svg`.
