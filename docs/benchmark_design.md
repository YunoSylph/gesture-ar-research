# Benchmark Design

Проект разделяет два уровня оценки.

## Recognition

Вход: manifest с NPZ tensors и model artifact.

Метрики:

- accuracy;
- macro F1;
- weighted F1;
- balanced accuracy;
- per-class precision/recall/F1;
- confusion matrix;
- median/p95 offline latency.

## Interaction Replay

Вход: JSONL timeline с `timestamp_ms`, `label`, `confidence`, optional `expected_action`.

Метрики:

- task success rate;
- unintended action rate;
- false trigger rate per minute;
- action precision/recall;
- corrections per task.

Этот уровень нужен, чтобы C2 оценивался как interaction layer, а не просто как еще один classifier.

