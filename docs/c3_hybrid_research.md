# C3 Hybrid Research Track

## Переформулировка без локальных видео

Без локально снятых phone rear роликов нельзя строго доказать перенос на заднюю камеру телефона. Поэтому исследовательское ядро проекта переносится в автономную public-data-first постановку:

```text
Can a hybrid temporal + geometry + context method improve robust AR gesture interaction on public landmark data?
```

Главная проверяемая гипотеза:

> C3 Hybrid, объединяющий temporal TCN probabilities и geometry-aware safety priors, повышает устойчивость gesture recognition и снижает риск ложных AR-действий по сравнению с прямым C1-T gesture-to-action mapping.

## Метод

`C3 Hybrid = C1-T Temporal TCN + lightweight geometric prior + safety gate`

Компоненты:

- `C1-T`: compact temporal TCN по dual-view landmark representation.
- `Geometric prior`: интерпретируемые признаки motion, wrist displacement, palm scale delta, fingertip distance и landmark confidence.
- `Safety gate`: мягко усиливает `no_gesture`, когда действие имеет недостаточную уверенность или плохо подтверждается геометрией.
- `C2 interaction layer`: отдельный context-aware слой для task-level replay и live AR интерфейса.

Это не заменяет temporal model, а добавляет второй источник информации там, где для AR важнее не активировать случайное действие.

## Автономные эксперименты

Основной benchmark:

```powershell
python -m research_pipeline.cli.benchmark_c3_hybrid --config configs/eval/c3_hybrid_robustness.yaml
```

Он сравнивает:

- `c1t_direct`: прямой temporal TCN;
- `c3_hybrid`: temporal TCN + geometry-aware fusion.

Сценарии:

- clean IPN test;
- mild/strong landmark Gaussian noise;
- 15%/30% frame drop;
- 10%/20% landmark masking;
- temporal jitter;
- translation shift;
- scale shift;
- combined mild degradation.

Метрики:

- accuracy;
- macro F1;
- weighted F1;
- balanced accuracy;
- no-gesture false-action rate;
- false swipe rate;
- directed confusion для `left/right` и `zoom_in/out`.

## Текущий результат

Первый полный прогон на IPN test показывает небольшой, но измеримый выигрыш:

| Метод | Clean Macro F1 | Mean Perturbed Macro F1 | Macro F1 Drop | Mean Perturbed False Action Rate |
|---|---:|---:|---:|---:|
| `c1t_direct` | 0.8502 | 0.8263 | 0.0239 | 0.0967 |
| `c3_hybrid` | 0.8513 | 0.8283 | 0.0231 | 0.0945 |

Интерпретация: C3 не даёт драматического скачка на clean public benchmark, но улучшает устойчивость и slightly снижает риск ложных действий. Для магистерской это полезнее, чем очередная архитектура classifier, потому что метрика привязана к AR interaction risk.

## Calibration + Ablation

Последующий исследовательский этап добавляет отдельный pipeline:

```powershell
python -m research_pipeline.cli.run_c3_research --config configs/eval/c3_research_ablation.yaml
```

Он делает две вещи:

1. Подбирает параметры C3 на public train/calibration subset, чтобы не использовать test для настройки.
2. Проверяет ablation на public test subset.

Лучший найденный calibration config:

```text
neural_weight=0.96
geometry_weight=0.08
action_threshold=0.44
no_gesture_margin=0.03
enable_safety_gate=true
```

Recognition ablation на test:

| Вариант | Clean Macro F1 | Mean Perturbed Macro F1 | Mean Perturbed False Action Rate |
|---|---:|---:|---:|
| `geometry_only` | 0.0873 | 0.0855 | 0.7600 |
| `c1t_direct` | 0.8502 | 0.8189 | 0.1018 |
| `tcn_geometry_fusion` | 0.8513 | 0.8181 | 0.1018 |
| `c3_hybrid` | 0.8522 | 0.8195 | 0.0995 |

Interaction-policy ablation на clean test replay:

| Вариант | Policy | Action Precision | Action Recall | Unintended Action Rate |
|---|---|---:|---:|---:|
| `c1t_direct` | direct | 0.8546 | 0.9084 | 0.1454 |
| `c3_hybrid` | direct | 0.8577 | 0.9084 | 0.1423 |
| `c1t_direct` | C2 | 0.9613 | 0.9485 | 0.0387 |
| `c3_hybrid` | C2 | 0.9667 | 0.9427 | 0.0333 |

Главный вывод ablation:

- `geometry_only` не является достаточным recognizer.
- Простая late fusion без safety gate почти не улучшает robustness.
- `C3 safety gate` даёт небольшой, но стабильный gain по устойчивости и false-action rate.
- Самый сильный interaction-level эффект даёт связка `C3 + C2`: unintended action rate падает до `0.0333`.

Это делает исследование защищаемым: вклад каждого компонента измеряется отдельно, а главный итог связан с AR interaction risk, не только с accuracy.

## Как развивать дальше

1. Расширить task-level replay benchmark сценариями AR Scroll List / Spatial Browser / Virtual Sorting.
2. Проверить sensitivity C2 threshold и cooldown на interaction-level метриках.
3. Добавить отдельную таблицу statistical significance / bootstrap CI для robustness gain.
4. Подготовить текстовый раздел "Limitations" про отсутствие локального phone-rear датасета.
5. После появления локальных видео провести domain-shift проверку без смешивания несовместимых ракурсов в одну модель.

## Thesis-ready assets

Графики и таблицы для главы экспериментов генерируются без повторного обучения:

```powershell
python -m research_pipeline.cli.generate_c3_research_assets
```

Выходные файлы:

- `artifacts/reports/c3_research_tables.md`;
- `artifacts/reports/c3_tables/c3_robustness_summary.csv`;
- `artifacts/reports/c3_tables/c3_robustness_by_scenario.csv`;
- `artifacts/reports/c3_tables/c3_ablation_summary.csv`;
- `artifacts/reports/c3_tables/c3_policy_ablation.csv`;
- `artifacts/figures/c3_robustness_macro_f1.png`;
- `artifacts/figures/c3_ablation_perturbed_macro_f1.png`;
- `artifacts/figures/c3_policy_unintended_action_rate.png`.

Live-интерфейс теперь также использует `C3 Hybrid + C2 Gate` как основной режим для webcam AR задач, чтобы демонстрационная часть соответствовала исследовательской постановке.

## Реконцептуализация после C3

Главное ограничение C3: recognition-level прирост слишком мал, чтобы быть центральным научным результатом. Поэтому C3 теперь трактуется как recognizer component, а основной исследовательский вклад переносится на `C4 Action-Safe Interaction`.

Новая постановка и результаты:

- `docs/c4_action_safe_research.md`;
- `artifacts/reports/c4_action_safe_research.json`;
- `artifacts/reports/c4_action_safe_tables.md`.

Ключевой результат C4: unintended action rate на public test снижается с `0.1402` у direct TCN до `0.0138` у safety operating point, при явной цене в recall. Это значительно сильнее и честнее для AR-темы, чем пытаться защищать небольшой прирост macro F1.
