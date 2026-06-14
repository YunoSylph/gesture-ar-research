# C4 Action-Safe Interaction Research

## Почему нужна реконцептуализация

Текущий C3 Hybrid почти не улучшает recognition-level macro F1 относительно C1-T TCN. Это слабый центральный результат для магистерской работы, потому что прирост классификатора слишком мал и легко выглядит как нулевая результативность.

Новая постановка переносит вклад проекта на AR interaction risk:

```text
Can a calibrated action-safety layer reduce unintended AR actions while preserving usable gesture recall?
```

Для AR это более сильная исследовательская цель: ложный `click`, `zoom` или `navigate` портит взаимодействие сильнее, чем отдельный пропущенный жест.

## Метод

`C4 Action-Safe Interaction = C3 recognizer + calibrated action-risk controller`

Компоненты:

- `C3 recognizer`: temporal TCN + geometry-aware safety prior.
- `Action proposal`: predicted gesture is treated as a candidate command, not as an immediate action.
- `Per-action thresholds`: high-risk actions can require higher confidence than pointer movement.
- `Temporal stability`: action is emitted only after repeated stable prediction.
- `Cooldown`: repeated accidental triggers are suppressed.
- `Abstention`: uncertain predictions become no-action instead of an AR command.
- `Calibration protocol`: thresholds are selected on a stratified public train subset, then evaluated on public test.
- `Risk cost matrix`: false actions are weighted by AR cost, so `select_confirm` is treated as riskier than `pointer_hover`.
- `Bootstrap CI`: scenario-level confidence intervals are produced for core action and risk metrics.

## Команды

```powershell
python -m research_pipeline.cli.run_c4_action_safe_research --config configs/eval/c4_action_safe_research.yaml
python -m research_pipeline.cli.generate_c4_research_assets
python -m research_pipeline.cli.benchmark_c4_tasks --config configs/eval/c4_task_benchmark.yaml
python -m research_pipeline.cli.generate_c4_task_assets
python -m research_pipeline.cli.analyze_c4_task_failures
```

## Результат

Evaluation на IPN public test в сценариях `clean`, `noise_mild`, `noise_strong`, `frame_drop_30`, `landmark_mask_20`, `temporal_jitter_2`, `combined_mild`:

| Method | Action Precision | Action Recall | Unintended Action Rate | False Action Cost Rate |
|---|---:|---:|---:|---:|
| `c1t_direct` | 0.8318 | 0.8779 | 0.1682 | 0.2885 |
| `c3_direct` | 0.8338 | 0.8776 | 0.1662 | 0.2856 |
| `c3_c2_default` | 0.9495 | 0.9141 | 0.0505 | 0.0808 |
| `c4_balanced` | 0.9497 | 0.9220 | 0.0503 | 0.0808 |
| `c4_safety` | 0.9670 | 0.8855 | 0.0330 | 0.0495 |

Главный защищаемый вывод:

- Обычный C3 recognizer не даёт сильного прироста как classifier.
- Основной вклад находится на interaction layer.
- `C4 Safety` снижает unintended action rate с `0.1682` до `0.0330` относительно direct TCN.
- Относительно сильного baseline `C3 + C2 default` risk падает с `0.0505` до `0.0330`.
- Weighted false action cost rate падает с `0.0808` у `C3 + C2 default` до `0.0495` у `C4 Safety`.
- Цена safety-режима: action recall снижается с `0.9141` до `0.8855`, что является явным precision-recall trade-off.

Это превращает работу в исследование безопасного AR-взаимодействия, а не в гонку за небольшим приростом accuracy.

## Task-level AR benchmark

Следующий слой оценки проверяет не отдельные gesture clips, а выполнение типовых AR-задач из `configs/interaction/ar_task_scenarios.yaml`: object control, scroll list, spatial browser, virtual sorting, placement, inspection, measurement, assembly, docking и другие. Каждый trial синтезирует последовательность expected actions из публичных IPN clips и добавляет `no_gesture` distractors, чтобы измерять ложные действия в паузах.

В этот этап добавлен новый исследовательский вариант:

```text
C4 Task-Aware = C3 recognizer + C4 risk controller + current AR task-step hint
```

Идея: если интерфейс знает текущий шаг сценария, он может временно снизить threshold для ожидаемого жеста и поднять threshold для неожиданных действий. Это не использует локальные видео пользователя и соответствует пошаговым AR-интерфейсам, где система знает, ждёт ли она `select`, `navigate`, `zoom` или pointer action.

Task-level результаты на 13 AR-сценариях, 5 perturbation-сценариях и 8 trials per task:

| Method | Task Success | Precision | Recall | Unintended | False Cost |
|---|---:|---:|---:|---:|---:|
| `c1t_direct` | 0.4962 | 0.8893 | 0.8553 | 0.1107 | 0.1167 |
| `c3_direct` | 0.4865 | 0.8854 | 0.8489 | 0.1146 | 0.1219 |
| `c3_c2_default` | 0.4058 | 0.9102 | 0.8146 | 0.0898 | 0.0905 |
| `c4_balanced` | 0.4077 | 0.9108 | 0.8177 | 0.0892 | 0.0905 |
| `c4_safety` | 0.3269 | 0.9198 | 0.7807 | 0.0764 | 0.0758 |
| `c4_task_aware` | 0.4058 | 0.9430 | 0.8166 | 0.0531 | 0.0527 |

Главный новый вывод:

- `C4 Safety` даёт минимизацию риска, но чрезмерно снижает completion rate.
- `C4 Task-Aware` сохраняет task success уровня `C3 + C2 default`, но снижает weighted false action cost rate с `0.0905` до `0.0527`.
- Относительно `c1t_direct` false action cost падает с `0.1167` до `0.0527`, а unintended action rate с `0.1107` до `0.0531`.
- Это сильнее защищается как магистерский вклад: комбинируется recognizer, risk-aware controller и task-context gating.

## Артефакты

- `artifacts/reports/c4_action_safe_research.json`
- `artifacts/reports/c4_action_safe_tables.md`
- `artifacts/reports/c4_tables/c4_summary.csv`
- `artifacts/reports/c4_tables/c4_by_scenario.csv`
- `artifacts/reports/c4_tables/c4_calibration.csv`
- `artifacts/reports/c4_tables/c4_bootstrap_ci.csv`
- `artifacts/figures/c4_unintended_action_rate.png`
- `artifacts/figures/c4_false_action_cost_rate.png`
- `artifacts/figures/c4_precision_recall_tradeoff.png`
- `artifacts/reports/c4_task_benchmark.json`
- `artifacts/reports/c4_task_benchmark_tables.md`
- `artifacts/reports/c4_task_failure_analysis.md`
- `artifacts/reports/c4_task_tables/c4_task_summary.csv`
- `artifacts/reports/c4_task_tables/c4_task_by_task.csv`
- `artifacts/reports/c4_task_tables/c4_task_by_scenario.csv`
- `artifacts/reports/c4_task_tables/c4_task_bootstrap_ci.csv`
- `artifacts/figures/c4_task_success_rate.png`
- `artifacts/figures/c4_task_false_action_cost_rate.png`
- `artifacts/figures/c4_task_unintended_action_rate.png`
- `artifacts/figures/c4_task_success_by_scenario.png`

## Что писать в магистерской

Основная формулировка:

```text
The proposed contribution is not a standalone classifier, but a calibrated risk-aware AR interaction pipeline. It combines temporal landmark recognition with geometry-aware priors and an action-safety controller that explicitly optimizes unintended action risk.
```

Сильная структура главы экспериментов:

1. Recognition baseline: показать, что C1-T уже силён, а C3 даёт только небольшой classifier-level gain.
2. Failure analysis: показать, что classifier accuracy не отражает качество AR-взаимодействия.
3. C4 policy calibration: описать stratified train calibration и два operating points.
4. Task-risk evaluation: сравнить direct, C3 direct, C3+C2 default, C4 balanced, C4 safety, C4 task-aware.
5. Trade-off discussion: completion rate против unintended/cost risk.
6. Proposed AR system design: показать, что task-aware gating естественно встраивается в интерфейс с выбранным сценарием и текущим шагом.

## Ограничения

Без локального phone-rear датасета нельзя честно доказывать перенос на заднюю камеру телефона. Поэтому текущий автономный результат защищает public-data AR interaction-risk методологию. Phone-rear в дальнейшем должен идти как отдельный domain-shift validation, а не смешиваться с webcam-like IPN в одну неразличимую выборку.
