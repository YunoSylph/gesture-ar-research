# Материал для отправки в ChatGPT: Hand Gestures Project

Дата подготовки: 2026-06-13.

Этот текст предназначен для вставки в ChatGPT как единый контекст проекта. Он описывает текущее состояние репозитория `Hand Gestures Project` на основе фактически просмотренных файлов, конфигов, отчетов, артефактов и тестов. В тексте намеренно разделены доказанные результаты, текущие ограничения, планы и разные версии benchmark-отчетов, чтобы не смешивать исследовательские постановки.

## Как использовать этот материал

Скопируй весь документ в ChatGPT и добавь свой вопрос после него. Пример формулировки:

```text
Ниже дан полный контекст моего проекта. Проанализируй проект как исследование для магистерской работы: проверь научную постановку, вклад, корректность экспериментов, ограничения, структуру главы экспериментов и что нужно улучшить. Не выдумывай того, чего нет в контексте.
```

Если ChatGPT будет просить код, в первую очередь стоит отправлять файлы, указанные в разделе "Ключевые файлы для дополнительной отправки".

## Короткая суть проекта

Проект называется `Hand Gestures Research Pipeline`. Это Windows-first исследовательский pipeline и демонстрационный прототип для темы:

```text
Context-Aware Temporal Landmark Gesture Recognition for AR Interaction
```

Практическая цель: распознавать жесты руки по MediaPipe-style landmarks и использовать их как команды для AR-интерфейса.

Научная цель в текущей, более сильной формулировке: исследовать не только точность классификатора жестов, а риск-ориентированное AR-взаимодействие, где выход recognizer рассматривается как предложение действия, а не как немедленная команда.

Самый защищаемый тезис проекта:

```text
Проект предлагает risk-aware, task-contextual gesture interaction pipeline для AR. Улучшения на уровне чистой классификации есть, но главный вклад находится в C4/TARC interaction layer, который снижает unintended actions и weighted false action cost по сравнению с прямым gesture-to-action control.
```

Важно: проект не должен описываться как "мы просто сделали классификатор жестов". Это уже многоуровневый исследовательский стенд:

- сбор и нормализация public IPN Hand данных;
- manifest/NPZ data contract;
- temporal landmark recognizers;
- geometry-aware safety prior;
- calibrated recognition fusion;
- risk-aware AR action controller;
- task-aware AR interaction policy;
- offline recognition, robustness, action-risk и task-level benchmarks;
- FastAPI live backend;
- React/Three.js AR-style UI с webcam и dataset replay;
- подготовленные отчеты, фигуры, модели и экспортные артефакты.

## Что проект делает

Проект берет видеоклипы жестов, извлекает 21 hand landmark на кадр, нормализует последовательность до фиксированной длины 32 кадра, строит dual-view признаки и обучает/оценивает модели распознавания 7 финальных классов жестов.

Финальные классы жестов:

| Индекс | Label | Семантика | IPN соответствие | AR action |
|---:|---|---|---|---|
| 0 | `no_gesture` | отсутствие команды | `D0X` / no gesture | `idle` |
| 1 | `point_2f` | наведение двумя пальцами | `B0B` / Point-2f | `pointer_hover` |
| 2 | `click_2f` | подтверждение двумя пальцами | `G02` / Click-2f | `select_confirm` |
| 3 | `swipe_left` | движение влево | `G05` / Th-left | `navigate_previous` |
| 4 | `swipe_right` | движение вправо | `G06` / Th-right | `navigate_next` |
| 5 | `zoom_in` | zoom-in жест | `G10` / Zoom-in | `zoom_in` |
| 6 | `zoom_out` | zoom-out жест | `G11` / Zoom-o | `zoom_out` |

В live UI распознанный жест управляет AR-сценой:

- `point_2f` двигает AR-курсор;
- `click_2f` выбирает, открывает, подтверждает, берет или фиксирует объект;
- `swipe_left` и `swipe_right` выполняют навигацию, поворот, сдвиг, скролл или перемещение;
- `zoom_in` и `zoom_out` меняют масштаб или приближают/уменьшают объект;
- `no_gesture` означает отсутствие команды.

Проект поддерживает 13 AR task-сценариев: `Object Control`, `Gallery Navigation`, `AR Scroll List`, `Spatial Browser`, `Virtual Sorting`, `Target Selection`, `Object Placement`, `Object Inspection`, `Distance Measure`, `Assembly Assist`, `Info Panel`, `Precision Docking`, `Guided Tour`.

## Главная исследовательская реконцептуализация

Ранний C3 Hybrid recognizer давал только небольшой прирост macro F1 относительно сильного C1-T TCN baseline. Поэтому текущая сильная постановка проекта - не "мы улучшили classifier accuracy", а:

```text
Can a gesture-driven AR system reduce unintended high-cost actions while preserving enough task completion quality?
```

Причина: для AR ошибки несимметричны. Ложный `select_confirm`, `zoom` или navigation event часто хуже, чем пропущенный кадр жеста. Поэтому проект вводит action-risk costs, policy thresholds, stability, cooldown, abstention и task-context gating.

## Структура репозитория

Основные директории:

| Путь | Назначение |
|---|---|
| `research_pipeline/` | Основной Python package: data schema, preprocessing, models, evaluation, interaction policy, live backend, CLI |
| `configs/` | YAML-конфиги для datasets, training, evaluation, export, interaction |
| `data/` | Raw/interim/processed данные, reference gestures, interaction gesture examples |
| `artifacts/` | Обученные модели, ONNX, reports, figures, screenshots, live sessions, mobile bundle |
| `demo/ar_interaction_app/` | React + Three.js интерфейс Gesture AR |
| `demo/webcam_app/` | Python webcam demo entrypoint |
| `ios_demo/` | iOS/Swift contract skeleton для будущего RealityKit/mobile слоя |
| `docs/` | Документация, исследовательские записки, generated `.docx` reports и картинки |
| `tests/` | Unit/integration tests |
| `scripts/` | PowerShell launch scripts |
| `tools/` | Скрипты генерации отчетов и проверки manifest |

Основные entrypoints из README:

```powershell
python -m research_pipeline.cli.build_ipn_manifest --root <IPN_ROOT> --output data/interim/manifests/ipn_all.jsonl
python -m research_pipeline.cli.remap_ipn_subset --input data/interim/manifests/ipn_all.jsonl --output data/interim/manifests/ipn_subset.jsonl
python -m research_pipeline.cli.extract_landmarks --manifest data/interim/manifests/ipn_subset.jsonl --output-dir data/processed/public_landmarks
python -m research_pipeline.cli.train --config configs/train/c1t_tcn_public.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/recognition.yaml
python -m research_pipeline.cli.benchmark_interaction --config configs/eval/interaction.yaml
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
python -m research_pipeline.cli.export_onnx --config configs/export/onnx.yaml
python -m research_pipeline.cli.export_coreml --config configs/export/coreml.yaml
```

## Данные

### Public IPN Hand data

В workspace есть подготовленные IPN Hand assets:

- annotations: `data/raw/ipn_hand/annotations`;
- videos: `data/raw/ipn_hand/videos/videos`;
- 200 `.avi` videos;
- 5 video archives в `data/raw/ipn_hand/videos`;
- копия/репозиторий IPN Hand в `data/raw/ipn_hand/repo/IPN-hand`.

Построенный thesis subset manifest:

- `data/interim/manifests/ipn_subset.jsonl`;
- 3438 target-class segments.

Полный landmark benchmark:

- train: `data/interim/manifests/ipn_train_full_landmarks.jsonl` - 2405 records;
- test: `data/interim/manifests/ipn_test_full_landmarks.jsonl` - 1033 records;
- всего: 3438 records.

Initial benchmark subset:

- train: `data/interim/manifests/ipn_train_initial_landmarks.jsonl` - 175 records;
- test: `data/interim/manifests/ipn_test_initial_landmarks.jsonl` - 70 records.

Processed NPZ landmarks:

- `data/processed/public_landmarks_full/train` - 2405 files;
- `data/processed/public_landmarks_full/test` - 1033 files;
- `data/processed/public_landmarks_initial/train` - 175 files;
- `data/processed/public_landmarks_initial/test` - 70 files;
- `data/processed/public_landmarks_sample` - 20 files.

### Local phone data

Локальный phone rear AR dataset пока не записан как реальные видео. Есть только план/шаблон:

- `data/interim/manifests/local_phone_plan.jsonl` - 35 planned records;
- `data/raw/local_phone/local_capture_template.csv` - template;
- в `data/raw/local_phone` нет реальных `.mp4` клипов, только CSV-шаблон.

Следовательно, проект не доказывает перенос на заднюю камеру телефона. Любые claims о phone-rear AR transfer должны формулироваться как future/domain-shift validation, а не как уже доказанный результат.

### Reference gestures

Есть reference clips по 3 на каждый из 7 классов:

- `data/reference_gestures/ipn_hand`;
- `data/interaction_gesture_examples`;
- всего по 3 клипа для `no_gesture`, `point_2f`, `click_2f`, `swipe_left`, `swipe_right`, `zoom_in`, `zoom_out`.

Эти reference clips используются как словарь команд и помощь для будущей локальной записи. Для phone rear AR нужно копировать семантику действия, а не exact webcam palm orientation.

## Data contract

### Manifest JSONL

Manifest record описан в `research_pipeline/data/schema.py`. Required fields:

```text
sample_id, source_dataset, public_label, target_label, participant_id,
session_id, repetition_id, split_group, hand_recorded, handedness_detected,
mirrored, fps, width, height, camera_device, background_tag, lighting_tag,
clip_start_ms, clip_end_ms, raw_video_path, tensor_path, notes
```

Валидация:

- `sample_id` должен быть непустым;
- `source_dataset` должен быть одним из `ipn_hand`, `local_phone`, `synthetic`;
- `target_label` должен входить в финальный словарь жестов;
- `hand_recorded` и `handedness_detected` должны быть `left`, `right` или `unknown`;
- `fps` неотрицательный;
- `clip_end_ms` не меньше `clip_start_ms`.

Чтение/запись JSONL реализованы в `research_pipeline/data/manifest.py`.

### Landmark NPZ

Landmark tensor описан в `research_pipeline/data/tensors.py`:

```text
landmarks: [T, 21, 3]
sequence_mask: [T]
frame_confidence: [T]
handedness_score: [T] или [1]
coord_space: обычно image_normalized_xyz
world_landmarks: optional [T, 21, 3]
```

Тензор сохраняется через `save_landmark_npz`, загружается через `load_landmark_npz`, shape проверяется через `validate_landmark_tensor`.

### Preprocessing

Основной preprocessing находится в `research_pipeline/features/preprocessing.py`.

Pipeline:

1. Последовательность ресэмплируется до `target_length=32`.
2. Pose stream центрируется по wrist landmark и нормализуется palm scale.
3. Motion stream сохраняет глобальную динамику: centroid, wrist xy, velocity, hand size, hand size delta, confidence.
4. Итоговые признаки `features` объединяют pose + motion.
5. Для TCN input dimension по умолчанию 74.

`clip_feature_summary` строит summary vector для classical models: mean, std, first, last, delta по valid frames.

## Модельные слои

### C0 Rule

Rule-based recognizer. Используется как слабый baseline и smoke/export artifact.

### C1 Random Forest

Classical baseline на summary features. Обучение через `research_pipeline/models/classical.py`, inference через artifact wrapper.

### C1-T Temporal Prototype

Temporal prototype baseline на flattened sequence features. Используется в initial/smoke experiments.

### C1-T Compact TCN

Основной temporal neural baseline. Реализация в `research_pipeline/models/tcn.py`:

- compact TCN over `[B,T,F]`;
- temporal blocks with Conv1d, BatchNorm1d, GELU, Dropout;
- dilations по блокам;
- pooling: `avg` или `avgmax`;
- default input dim 74, num classes 7.

Training в `research_pipeline/models/torch_training.py`:

- stratified validation split;
- PyTorch DataLoader;
- optional class-balanced sampler;
- optional focal loss;
- optional label smoothing;
- optional online augmentation: noise, scale, feature dropout, temporal shift, time mask.

### Validated TCN

Контрольная/validated ветка:

- model: `artifacts/models/ipn_c1t_tcn_full_validated.pkl`;
- ONNX: `artifacts/export/ipn_c1t_tcn_full_validated.onnx`;
- report: `artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json`.

### Augmented TCN

Усиленная C6 ветка:

- config: `configs/train/ipn_c1t_tcn_augmented.yaml`;
- model: `artifacts/models/ipn_c1t_tcn_augmented.pkl`;
- training: 140 epochs, batch 64, lr 0.0008, weight decay 0.008, validation split 0.15, early stopping 18;
- architecture: channels `[96,96,128,128]`, kernel size 5, dropout 0.20, pooling `avgmax`;
- balanced sampler, focal gamma 1.2, label smoothing 0.03;
- augmentation: noise sigma 0.010, scale std 0.015, feature dropout 0.035, temporal shift 2, time mask.

### C3 Hybrid

Реализация: `research_pipeline/models/hybrid.py`.

Идея:

```text
C3 Hybrid = temporal TCN probabilities + geometry-aware safety prior + safety gate
```

C3 не заменяет TCN. Он берет neural prediction и добавляет lightweight geometry prior:

- wrist displacement;
- centroid motion;
- palm scale delta;
- fingertip distance;
- mean landmark confidence.

Safety gate может усилить `no_gesture`, если:

- top action confidence ниже threshold;
- `no_gesture` близко к top score;
- landmark confidence низкий;
- predicted swipe не подтвержден движением;
- predicted zoom не подтвержден scale delta;
- predicted click не подтвержден index-middle distance.

Калиброванная C3 конфигурация из reports/configs:

```text
neural_weight=0.96
geometry_weight=0.08
action_threshold=0.44
no_gesture_margin=0.03
enable_safety_gate=true
```

### C5 Calibrated Recognition

Реализация: `research_pipeline/models/calibrated.py` и CLI `research_pipeline/cli/run_c5_calibrated_recognition.py`.

Идея: calibrated score fusion между C1/C6 neural scores и C3 scores:

- `c3_weight`;
- temperature calibration;
- label biases;
- optional action abstention by confidence/margin.

### C6 Ensemble Calibrated Recognizer

Реализация: `research_pipeline/models/c6_ensemble.py`.

Идея:

```text
C6 = validated TCN + augmented TCN ensemble + C3 geometry fusion + C5/C6 calibration
```

Конфиг:

- `configs/eval/c6_ensemble_calibrated_recognition.yaml`;
- model paths:
  - `artifacts/models/ipn_c1t_tcn_full_validated.pkl`;
  - `artifacts/models/ipn_c1t_tcn_augmented.pkl`;
- calibration includes `c3_weight=0.15`, `temperature=1.25`, label biases for `click_2f`, `no_gesture`, `swipe_left`, `zoom_in`, `zoom_out`.

C6 сейчас является основным robust recognizer в live backend и UI.

## Interaction policy layers

### Gesture-to-action mapping

Mapping реализован в `research_pipeline/interaction/fsm.py`:

```text
point_2f -> pointer_hover
click_2f -> select_confirm
swipe_left -> navigate_previous
swipe_right -> navigate_next
zoom_in -> zoom_in
zoom_out -> zoom_out
no_gesture -> no direct action / idle
```

### C2 ContextAwarePolicy

Реализация: `research_pipeline/interaction/fsm.py`.

Назначение: базовый context-aware gate:

- activation threshold;
- stable frames;
- cooldown;
- reset after no_gesture frames.

### C4 ActionSafePolicy

Реализация: `research_pipeline/interaction/action_safe.py`.

Главная идея: classifier output не команда, а proposal. Policy может abstain.

Проверки:

- per-label confidence thresholds;
- temporal stability;
- cooldown;
- no_gesture reset;
- score margin.

Default C4 safety config в live backend:

```text
default_threshold=0.70
point_2f=0.62
click_2f/swipe_left/swipe_right/zoom_in/zoom_out=0.75
default_stable_frames=1
high-risk stable_frames=2
cooldown_ms=250
no_gesture_reset_frames=3
```

### C4 Task-Aware / TARC

Реализация: `research_pipeline/interaction/task_aware.py`.

Идея:

```text
TaskAwareActionSafePolicy = C4 risk controller + current AR task-step hint
```

Если интерфейс знает текущий ожидаемый шаг, policy временно:

- снижает threshold для expected action;
- повышает threshold для unexpected actions;
- повышает thresholds в idle phase;
- может уменьшать required stable frames для expected action.

Это реалистично для guided AR workflows, где система знает следующий допустимый жест. Это не open-world gesture control.

В live backend `c4_task_aware` является основным interaction mode.

## Action risk costs

Файл: `configs/interaction/action_risk_costs.yaml`.

```text
idle: 0.0
pointer_hover: 0.25
navigate_previous: 1.0
navigate_next: 1.0
zoom_in: 1.25
zoom_out: 1.25
select_confirm: 2.0
```

Смысл: `select_confirm` имеет самый высокий риск, pointer movement - низкий. Поэтому проект оценивает не только false action count, но и weighted false action cost.

## Evaluation layers

### Recognition benchmark

Реализация: `research_pipeline/evaluation/recognition.py` и `research_pipeline/evaluation/metrics.py`.

Метрики:

- accuracy;
- macro F1;
- weighted F1;
- balanced accuracy;
- confusion matrix;
- per-class precision/recall/F1;
- offline latency median/p95.

### Robustness benchmark

Реализация: `research_pipeline/evaluation/robustness.py`.

Perturbations:

- clean;
- Gaussian landmark noise;
- frame drop;
- landmark masking;
- temporal jitter;
- translation shift;
- scale shift;
- combined mild degradation.

### Action-risk benchmark

Реализация: `research_pipeline/evaluation/action_risk.py`.

Метрики:

- action precision/recall;
- unintended action rate;
- weighted action precision/recall;
- false action cost rate;
- missed action cost rate;
- false trigger rate per minute;
- no_gesture false-action risk.

### Task-level AR benchmark

Реализация:

- `research_pipeline/cli/benchmark_c4_tasks.py`;
- `research_pipeline/evaluation/task_benchmark.py`;
- scenario config: `configs/interaction/ar_task_scenarios.yaml`.

Benchmark строит trials из public IPN test clips, добавляет `no_gesture` distractors и оценивает выполнение AR-сценариев целиком, а не отдельные clips.

Важно: task-level replay является synthetic/scripted benchmark на public clips, не human-in-the-loop user study.

## Основные результаты

Все числа ниже взяты из подготовленных reports/CSV в `artifacts/reports` и документации проекта. Если в будущем отчеты будут перегенерированы, числа нужно обновить.

### Initial public benchmark

Initial subset: train 175 clips, test 70 clips, target length 32.

| Variant | Accuracy | Macro F1 | Weighted F1 | p95 latency |
|---|---:|---:|---:|---:|
| C0 rule | 0.2000 | 0.1358 | 0.1358 | 0.272 ms |
| C1 random forest | 0.7286 | 0.7214 | 0.7214 | 31.558 ms |
| C1-T temporal prototype | 0.6857 | 0.6859 | 0.6859 | 0.293 ms |
| C1-T compact TCN | 0.7857 | 0.7705 | 0.7705 | 5.087 ms |

### Full recognition benchmark

Full test set: 1033 clips.

| Variant | Accuracy | Macro F1 | Weighted F1 | p95 latency |
|---|---:|---:|---:|---:|
| C0 rule | 0.1820 | 0.0874 | 0.2158 | 0.237 ms |
| C1 random forest | 0.8955 | 0.7987 | 0.8930 | 56.409 ms |
| C1-T compact TCN | 0.9061 | 0.8504 | 0.9093 | 3.912 ms in current JSON, README also mentions 5.228 ms |
| C1-T compact TCN validated | 0.9071 | 0.8502 | 0.9109 | 4.638 ms |
| Augmented TCN | 0.9090 | 0.8565 | 0.9121 | 10.952 ms |

Примечание: для `ipn_c1t_tcn_full_recognition.json` текущий JSON показывает p95 `3.9117600026656874 ms`, а `docs/setup_training_summary.md` указывает `5.228 ms`. В документе нужно считать JSON более актуальным источником для конкретного артефакта, но расхождение стоит помнить.

### C3 robustness

Report/table: `artifacts/reports/c3_tables/c3_robustness_summary.csv`.

| Method | Clean Macro F1 | Perturbed Macro F1 Mean | Macro F1 Drop | Perturbed False Action Rate Mean |
|---|---:|---:|---:|---:|
| `c1t_direct` | 0.8502 | 0.8263 | 0.0239 | 0.0967 |
| `c3_hybrid` | 0.8513 | 0.8283 | 0.0231 | 0.0945 |

Вывод: C3 дает небольшой, но стабильный recognition/robustness gain. Этого недостаточно как центральный научный вклад, но полезно как компонент.

### C3 ablation

Report/table: `artifacts/reports/c3_tables/c3_ablation_summary.csv`.

| Method | Clean Macro F1 | Perturbed Macro F1 Mean | Macro F1 Drop | Perturbed False Action Rate Mean |
|---|---:|---:|---:|---:|
| `c1t_direct` | 0.8502 | 0.8189 | 0.0312 | 0.1018 |
| `c3_hybrid` | 0.8522 | 0.8195 | 0.0327 | 0.0995 |
| `geometry_only` | 0.0873 | 0.0855 | 0.0018 | 0.7600 |
| `tcn_geometry_fusion` | 0.8513 | 0.8181 | 0.0332 | 0.1018 |

Interaction-policy ablation:

| Method | Policy | Action Precision | Action Recall | Unintended Action Rate |
|---|---|---:|---:|---:|
| `c1t_direct` | direct | 0.8546 | 0.9084 | 0.1454 |
| `c3_hybrid` | direct | 0.8577 | 0.9084 | 0.1423 |
| `c1t_direct` | C2 | 0.9613 | 0.9485 | 0.0387 |
| `c3_hybrid` | C2 | 0.9667 | 0.9427 | 0.0333 |

Вывод: geometry-only не является рабочим recognizer; safety gate и C2/C4 interaction layer дают главный практический эффект.

### C4 Action-Safe Interaction

Report/table: `artifacts/reports/c4_tables/c4_summary.csv`.

Evaluation scenarios: clean, mild/strong noise, frame_drop_30, landmark_mask_20, temporal_jitter_2, combined_mild.

| Method | Action Precision | Action Recall | Unintended Action Rate | False Action Cost Rate |
|---|---:|---:|---:|---:|
| `c1t_direct` | 0.8318 | 0.8779 | 0.1682 | 0.2885 |
| `c3_direct` | 0.8338 | 0.8776 | 0.1662 | 0.2856 |
| `c3_c2_default` | 0.9495 | 0.9141 | 0.0505 | 0.0808 |
| `c4_balanced` | 0.9497 | 0.9220 | 0.0503 | 0.0808 |
| `c4_safety` | 0.9670 | 0.8855 | 0.0330 | 0.0495 |

Главный вывод: `C4 Safety` существенно снижает unintended action rate и false action cost, но платит снижением recall. Это корректный precision/recall/safety trade-off.

### Full C4 task-level benchmark

Report/table: `artifacts/reports/c4_task_tables/c4_task_summary.csv`.

Это полный benchmark с несколькими методами. Его не нужно смешивать с compact `official_method_benchmark`.

| Method | Task Success | Precision | Recall | Unintended | False Cost |
|---|---:|---:|---:|---:|---:|
| `c1t_direct` | 0.4962 | 0.8893 | 0.8553 | 0.1107 | 0.1167 |
| `c3_direct` | 0.4865 | 0.8854 | 0.8489 | 0.1146 | 0.1219 |
| `c3_c2_default` | 0.4058 | 0.9102 | 0.8146 | 0.0898 | 0.0905 |
| `c4_balanced` | 0.4077 | 0.9108 | 0.8177 | 0.0892 | 0.0905 |
| `c4_safety` | 0.3269 | 0.9198 | 0.7807 | 0.0764 | 0.0758 |
| `c4_task_aware` | 0.4058 | 0.9430 | 0.8166 | 0.0531 | 0.0527 |

Вывод для этой версии benchmark: `c4_task_aware` сохраняет task success уровня `c3_c2_default`, но снижает false action cost с 0.0905 до 0.0527 и unintended action rate с 0.0898 до 0.0531.

### Official compact benchmark M1/M2/M3

Report: `artifacts/reports/official_method_benchmark.json`.

Этот benchmark ближе к тому, что hardcoded в UI как official method set. Он сравнивает:

- `baseline_direct` - baseline direct control;
- `robust_recognizer_direct` - robust C6 recognizer direct;
- `proposed_tarc` - proposed task-aware risk-calibrated controller.

Summary:

| Method | Task Success | Action Precision | Action Recall | Unintended | False Cost |
|---|---:|---:|---:|---:|---:|
| `baseline_direct` | 0.5269 | 0.8917 | 0.8598 | 0.1063 | 0.1103 |
| `robust_recognizer_direct` | 0.5519 | 0.9171 | 0.8663 | 0.0809 | 0.0848 |
| `proposed_tarc` | 0.5308 | 0.9744 | 0.8568 | 0.0237 | 0.0251 |

Вывод для compact benchmark: `proposed_tarc` почти сохраняет task success относительно baseline direct, но резко снижает unintended action rate и false action cost. Это самая сильная thesis-friendly формулировка текущего результата.

### C6 recognition upgrade

Report/table: `artifacts/reports/c6_tables/summary.csv`.

| Method | Clean Accuracy | Clean Macro F1 | Robust Macro F1 Mean | Robust False Action Rate |
|---|---:|---:|---:|---:|
| C1-TCN validated | 0.907 | 0.850 | 0.826 | 0.097 |
| C3 hybrid validated | 0.908 | 0.851 | 0.828 | 0.094 |
| Augmented TCN | 0.909 | 0.856 | 0.835 | 0.090 |
| Augmented C3 hybrid | 0.913 | 0.866 | 0.838 | 0.086 |
| C6 ensemble calibrated | 0.930 | 0.887 | 0.859 | 0.067 |

Per-class clean F1 changes:

| Class | C1-TCN | C6 Ensemble | Delta |
|---|---:|---:|---:|
| `click_2f` | 0.7895 | 0.7966 | +0.0071 |
| `no_gesture` | 0.9360 | 0.9573 | +0.0213 |
| `point_2f` | 0.9608 | 0.9567 | -0.0041 |
| `swipe_left` | 0.7559 | 0.9143 | +0.1584 |
| `swipe_right` | 0.8624 | 0.8393 | -0.0231 |
| `zoom_in` | 0.8829 | 0.9126 | +0.0297 |
| `zoom_out` | 0.7636 | 0.8319 | +0.0682 |

Вывод: C6 дает уже существенный recognition upgrade, особенно для слабых классов `swipe_left` и `zoom_out`, и снижает robust false action rate. Trade-off: `swipe_right` F1 немного падает.

### Domain readiness

Report: `artifacts/reports/domain_readiness.json`.

Факты:

- `ipn_hand`: 3438 records;
- `phone_rear_ar`: 35 planned records;
- total records in combined readiness report: 3473;
- `domain_transfer_status`: `local_plan_ready_waiting_for_videos`.

Интерпретация: public pipeline готов, phone domain только спланирован.

### Live session reports

Есть live session traces в `artifacts/live_sessions/*.jsonl`.

`artifacts/reports/live_session_summary.json` показывает одну idle/no-hand summary:

- method `onnx`;
- confidence_mean 1.0;
- detection_rate_mean 0.0;
- gesture_counts: `no_gesture: 25`;
- action_counts: `idle: 25`.

`artifacts/reports/live_task_report.json` содержит live task report для webcam ONNX placement session:

- method `onnx`;
- confidence_mean 0.8302;
- detection_rate_mean 0.8163;
- task order includes `placement`;
- содержит proxy coverage и ground-truth scenario metrics when scenario file provided.

Live reports полезны для triage, но не являются полноценным user study.

## Backend и UI

### FastAPI live backend

Файл: `research_pipeline/serve/live_backend.py`.

Backend:

- `GET /api/health`;
- `GET /api/methods`;
- `GET /api/camera/status`;
- `GET /video_feed`;
- `WebSocket /ws/stream`.

Поддерживает источники:

- `replay` - поток подготовленных IPN test landmark tensors из `data/interim/manifests/ipn_test_full_landmarks.jsonl`;
- `webcam` - локальная камера через OpenCV + MediaPipe Hand Landmarker.

Методы:

- primary `c1t_tcn`;
- primary `c6_ensemble`;
- ablation `c0`;
- ablation `c1_rf`;
- ablation `onnx`;
- ablation `c3`.

Текущее `/api/health` возвращает:

```text
methods: ["c1t_tcn", "c6_ensemble"]
interaction_modes: ["direct", "c4_task_aware"]
ablation_methods: ["c0", "c1_rf", "onnx", "c3"]
ablation_interaction_modes: ["c2", "c4"]
```

Важно: один unit test сейчас ожидает `c3` в `health()["methods"]`, но backend сейчас кладет `c3` в `ablation_methods`. Это текущая несогласованность тестового ожидания и API-контракта.

Live backend:

- держит model objects alive in process;
- для ONNX может использовать CUDAExecutionProvider только если env `GESTURE_AR_ONNX_CUDA` разрешает и provider доступен;
- MediaPipe модель: `models/mediapipe/hand_landmarker.task`;
- webcam window собирается в deque длиной 32;
- index fingertip landmark используется для AR pointer;
- live records логируются в `artifacts/live_sessions`.

### React + Three.js UI

Путь: `demo/ar_interaction_app`.

Технологии:

- React 19;
- TypeScript;
- Vite;
- Three.js;
- lucide-react icons.

Основные файлы:

- `demo/ar_interaction_app/src/main.tsx` - около 2167 строк;
- `demo/ar_interaction_app/src/styles.css` - около 1033 строк.

UI страницы:

- `Live`;
- `Tables`;
- `Charts`.

Default state:

- method: `c6_ensemble`;
- source: `webcam`;
- interaction mode: `c4_task_aware`;
- task: `object`;
- camera: 1280x720;
- target FPS 30;
- preview width 960;
- JPEG quality 84.

UI функции:

- выбор AR task;
- Start/Stop/Reset task;
- telemetry: backend, FPS, processing ms, detection, gesture;
- advanced controls;
- recognizer selection;
- gesture test pad;
- dataset replay vs camera stream;
- direct control vs TARC controller;
- camera index/FPS/preview/JPEG controls;
- Three.js scene поверх webcam feed;
- landmark overlay и pointer;
- experiment results tables/charts.

В `main.tsx` есть hardcoded visual result values для charts/tables. Для научного анализа предпочтительнее брать source-of-truth из `artifacts/reports/*.json` и CSV, а UI считать presentation layer.

## Mobile / iOS artifacts

Есть:

- `artifacts/mobile/gesture_mobile_bundle`;
- `ios_demo/GestureAR/Sources/LandmarkPreprocessor.swift`;
- `ios_demo/GestureAR/Sources/GestureLabels.swift`;
- `ios_demo/GestureAR/Sources/ContextPolicy.swift`;
- `ios_demo/GestureAR/README.md`.

Мобильный слой сейчас лучше описывать как contract skeleton / portability bundle, а не как уже полностью валидированный on-device RealityKit deployment.

Core ML export вынесен в отдельную стадию, и документация указывает, что финальная Core ML/RealityKit latency validation еще не завершена.

## Команды запуска

### Быстрый запуск AR demo

```powershell
cd "C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project"
powershell -ExecutionPolicy Bypass -File .\scripts\start_ar_demo.ps1
```

С `-Restart`, если порты заняты:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ar_demo.ps1 -Restart
```

### Ручной backend

```powershell
.\.venv311\Scripts\Activate.ps1
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
```

### Ручной frontend

```powershell
cd demo/ar_interaction_app
npm install
npm run dev -- --port 5173
```

Обычно открыть:

```text
http://127.0.0.1:5173
```

### Основные research commands

```powershell
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_augmented.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_augmented.yaml
python -m research_pipeline.cli.benchmark_c3_hybrid --config configs/eval/c6_augmented_robustness.yaml
python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition.yaml
python -m research_pipeline.cli.run_c4_action_safe_research --config configs/eval/c4_action_safe_research.yaml
python -m research_pipeline.cli.benchmark_c4_tasks --config configs/eval/c4_task_benchmark.yaml
python -m research_pipeline.cli.analyze_c4_task_failures
python -m research_pipeline.cli.build_experiment_chapter
python -m research_pipeline.cli.analyze_recognition_risk
python -m research_pipeline.cli.report_project_status
```

### Smoke/test commands from README

```powershell
python -m research_pipeline.cli.smoke_public
python -m research_pipeline.cli.smoke_demo
python -m research_pipeline.cli.smoke_export
python -m pytest -q
```

## Тесты

Tests находятся в `tests/`.

Что покрывается:

- `test_manifest.py` - manifest schema/IO;
- `test_preprocessing.py` - resampling, preprocessing, feature contract;
- `test_labels.py` - label mapping/remapping;
- `test_torch_training.py` - training helper behavior;
- `test_hybrid_robustness.py` - perturbation shape, hybrid safety gate;
- `test_calibrated_recognition.py` - class bias and abstention in calibrated fusion;
- `test_action_risk.py` - weighted action risk metrics;
- `test_action_safe_policy.py` - thresholds, stable frames, margin;
- `test_task_aware_policy.py` - expected/unexpected task-aware thresholds;
- `test_task_benchmark.py` - task metrics and weighted costs;
- `test_live_sessions.py` - live/session summaries and task scenario evaluation;
- `test_live_backend_contract.py` - backend method/mode/status contract;
- `test_smoke_pipeline.py` - synthetic train + benchmark smoke integration.

Текущий запуск 2026-06-13:

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
```

Результат:

```text
30 passed, 1 failed
```

Падение:

```text
tests/unit/test_live_backend_contract.py::test_live_backend_exposes_c3_hybrid_method
AssertionError: assert 'c3' in ['c1t_tcn', 'c6_ensemble']
```

Причина по коду: `health()["methods"]` сейчас содержит только primary methods `c1t_tcn` и `c6_ensemble`, а `c3` находится в `health()["ablation_methods"]`. Кроме того, `methods()["methods"]` сейчас содержит primary methods, а `c3` находится в `methods()["ablations"]`. Нужно решить, что считать правильным API-контрактом: обновить tests или вернуть `c3` в primary method list.

Системный Python `C:\Python314\python.exe` не имеет `pytest`, поэтому тесты нужно запускать через `.venv311`.

## Ограничения проекта

1. Нет реального local phone rear dataset. Есть только manifest plan и CSV template. Поэтому нельзя честно утверждать, что transfer на заднюю камеру телефона уже доказан.
2. Task-level replay является synthetic/scripted на public clips. Это не заменяет live user study.
3. C3 recognition-level gain сам по себе мал. C3 нужно подавать как supporting robustness/safety component, а не как главный thesis result.
4. Live webcam demo чувствителен к освещению, видимости руки, privacy shutter, camera index и стабильности MediaPipe.
5. iOS/RealityKit/Core ML on-device latency validation еще не завершена.
6. UI содержит hardcoded presentation numbers. Для научного анализа лучше ссылаться на reports в `artifacts/reports`.
7. Есть текущая несогласованность одного backend contract test с фактическим `/api/health` contract.
8. Некоторые reports имеют разные версии/постановки benchmark. Важно не смешивать full C4 task benchmark и official compact M1/M2/M3 benchmark.

## Что исследуется

Проект исследует несколько связанных вопросов:

1. Можно ли построить воспроизводимый public-data-first pipeline для temporal landmark gesture recognition?
2. Насколько compact TCN лучше rule/classical baselines на IPN Hand target subset?
3. Может ли geometry-aware prior улучшить robustness и false-action behavior?
4. Почему classifier accuracy недостаточна для AR interaction quality?
5. Может ли action-safe policy снизить unintended/high-cost AR actions?
6. Может ли task-aware controller сохранить task completion, одновременно снижая false action cost?
7. Как robust recognition upgrade C6 влияет на слабые классы и downstream AR task metrics?
8. Как live webcam prototype демонстрирует исследовательский pipeline в интерактивном AR-style UI?

## Что тестируется и измеряется

Проект измеряет:

- recognition accuracy/macro F1/weighted F1/balanced accuracy;
- per-class F1 и confusion matrix;
- offline median/p95 latency;
- robustness under perturbations;
- no_gesture false-action risk;
- false swipe rate;
- action precision/recall;
- weighted action precision/recall;
- unintended action rate;
- false action cost rate;
- missed action cost rate;
- false trigger rate per minute;
- task success rate;
- task-level latency absolute median/p95;
- live detection rate, FPS, processing latency, gesture/action counts.

## Как проект лучше формулировать в магистерской

Рекомендуемая структура исследования:

1. Введение: AR gesture interaction и проблема unintended commands.
2. Related work: gesture recognition, hand landmarks, temporal models, AR interaction safety.
3. Dataset and data contract: IPN Hand subset, manifest/NPZ, preprocessing.
4. Recognition baselines: C0, C1 RF, C1-T TCN, C6.
5. Risk analysis: почему accuracy не равна safe AR control.
6. C3 Hybrid: geometry-aware safety prior как supporting component.
7. C4 Action-Safe Interaction: per-action thresholds, stability, cooldown, abstention, cost matrix.
8. TARC / C4 Task-Aware: scenario-aware threshold adaptation for guided AR workflows.
9. Experiments:
   - recognition benchmark;
   - robustness benchmark;
   - C3 ablation;
   - C4 action-risk benchmark;
   - task-level benchmark;
   - compact official M1/M2/M3 benchmark;
   - live demo observations.
10. Limitations:
   - no phone-rear validation yet;
   - scripted replay vs user study;
   - webcam/MediaPipe sensitivity;
   - no final on-device iOS latency.
11. Conclusion: strongest claim about risk-aware task-contextual AR interaction pipeline.

## Ключевые файлы для дополнительной отправки

Если ChatGPT попросит код, отправлять лучше не весь проект, а эти файлы:

Data/schema:

- `research_pipeline/labels.py`;
- `research_pipeline/data/schema.py`;
- `research_pipeline/data/manifest.py`;
- `research_pipeline/data/tensors.py`;
- `research_pipeline/features/preprocessing.py`.

Models:

- `research_pipeline/models/tcn.py`;
- `research_pipeline/models/torch_training.py`;
- `research_pipeline/models/hybrid.py`;
- `research_pipeline/models/calibrated.py`;
- `research_pipeline/models/c6_ensemble.py`;
- `research_pipeline/models/artifacts.py`.

Interaction:

- `research_pipeline/interaction/fsm.py`;
- `research_pipeline/interaction/action_safe.py`;
- `research_pipeline/interaction/task_aware.py`;
- `configs/interaction/action_risk_costs.yaml`;
- `configs/interaction/ar_task_scenarios.yaml`.

Evaluation:

- `research_pipeline/evaluation/metrics.py`;
- `research_pipeline/evaluation/recognition.py`;
- `research_pipeline/evaluation/robustness.py`;
- `research_pipeline/evaluation/action_risk.py`;
- `research_pipeline/evaluation/task_benchmark.py`;
- `research_pipeline/evaluation/live_sessions.py`;
- `research_pipeline/cli/run_c4_action_safe_research.py`;
- `research_pipeline/cli/benchmark_c4_tasks.py`;
- `research_pipeline/cli/run_c5_calibrated_recognition.py`.

Backend/UI:

- `research_pipeline/serve/live_backend.py`;
- `demo/ar_interaction_app/src/main.tsx`;
- `demo/ar_interaction_app/src/styles.css`.

Reports/configs:

- `docs/c3_hybrid_research.md`;
- `docs/c4_action_safe_research.md`;
- `docs/c6_recognition_upgrade.md`;
- `docs/project_research_assessment.md`;
- `docs/setup_training_summary.md`;
- `artifacts/reports/c6_tables/summary.csv`;
- `artifacts/reports/c4_tables/c4_summary.csv`;
- `artifacts/reports/c4_task_tables/c4_task_summary.csv`;
- `artifacts/reports/official_method_benchmark.json`.

Tests:

- `tests/unit/test_action_safe_policy.py`;
- `tests/unit/test_task_aware_policy.py`;
- `tests/unit/test_action_risk.py`;
- `tests/unit/test_task_benchmark.py`;
- `tests/unit/test_live_backend_contract.py`;
- `tests/integration/test_smoke_pipeline.py`.

## Вопросы, которые полезно задать ChatGPT после этого контекста

1. Насколько научно корректна текущая формулировка вклада?
2. Как лучше сформулировать research questions и hypotheses?
3. Какие эксперименты выглядят убедительно, а какие требуют доработки?
4. Как не переутверждать phone-rear AR transfer?
5. Как описать C3, C4, C6 и TARC в единой архитектуре?
6. Какие таблицы/фигуры оставить в магистерской, а какие убрать?
7. Как объяснить разницу между recognition accuracy и AR interaction safety?
8. Как лучше оформить limitations?
9. Какие тесты или validation experiments добавить перед защитой?
10. Как исправить текущую несогласованность live backend contract test?

## Краткий итог для ChatGPT

Это проект не только про распознавание жестов, а про безопасное жестовое управление AR-интерфейсом. Recognition pipeline построен на IPN Hand landmarks, а центральный исследовательский вклад - C4/TARC layer, где classifier output проходит через risk-aware и task-aware policy перед тем, как стать AR action. C6 усиливает recognizer и улучшает weak classes. Самые сильные результаты - снижение unintended action rate и false action cost в task-level AR benchmark. Самое важное ограничение - отсутствие реальных local phone rear videos и полноценного user study.
