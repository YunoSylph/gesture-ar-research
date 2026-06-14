# Live Gesture Diagnosis

Дата аудита: 2026-06-14

Цель документа: зафиксировать, насколько текущие dataset, recognizer, live controller, TARC и UI согласованы между собой. Это аудит без разрушительных изменений: код интерфейса и backend-логика не перестраивались.

## Главный вывод

Система может показывать хорошие offline recognition metrics и при этом плохо работать в live AR, потому что offline и live задачи сейчас имеют разную природу:

- offline recognizer обучается и оценивается на pre-segmented IPN Hand клипах;
- live UI получает continuous webcam stream без явных границ жеста;
- C6 в webcam path используется как `model_prediction`, но live-команда в основном формируется `LiveLandmarkGestureController`;
- TARC ожидает стабильные action proposals, но получает proposals из отдельного landmark-first controller, а не напрямую из C6;
- UI guide и task definitions не полностью совпадают с backend scenario definitions и фактической controller logic.

Иначе говоря, текущий live-режим - это уже не чистый "C6 recognizer -> AR command". Это гибрид:

```text
camera frame
-> MediaPipe landmarks
-> 32-frame sliding tensor
-> C6/raw model prediction for logging/research signal
-> LiveLandmarkGestureController as live action-proposal source
-> TARC/ActionSafePolicy confidence/stability/context/risk gate
-> UI task state and Three.js AR action
```

Такую архитектуру можно защитить как научный вклад, но только если явно измерять разницу между raw model output, live controller output и final TARC action.

## Текущий Data Flow

### Webcam live path

Файл: `research_pipeline/serve/live_backend.py`

1. Frontend открывает WebSocket:

```text
/ws/stream?method=c6_ensemble&source=webcam&interaction=c4_task_aware&task=<task>
```

2. Backend запускает `stream_webcam`.
3. `CameraHub` читает кадры камеры, по умолчанию frontend просит `1920x1080`, `30 FPS`, mirrored view.
4. `FrameLandmarker` запускает MediaPipe HandLandmarker и возвращает 21 landmark.
5. Последние 32 кадров складываются в `tensor_from_window`.
6. `LivePredictor(method)` строит `model_prediction`.
7. Если выбран `c4_task_aware`, backend берет `expected_label` из `TaskAwareActionSafePolicy.context()`.
8. `LiveLandmarkGestureController.update(model_prediction, tensor, expected_label=...)` строит live `prediction`.
9. TARC/ActionSafePolicy принимает или отклоняет эту `prediction`.
10. Payload уходит в UI: `gesture`, `scores`, `action`, `policy_context`, `control_context`, landmarks, pointer и telemetry.
11. Raw C6 label/confidence пишутся только в JSONL logger extra как `model_gesture` и `model_confidence`; в WebSocket payload UI их сейчас не показывает.

Критично: в webcam path AR-команда идет из `LiveLandmarkGestureController`, а не напрямую из C6. Даже `interaction=direct` применяет `direct_action_for_prediction(prediction)` к controller prediction.

### Replay path

Файл: `research_pipeline/serve/live_backend.py`

`stream_replay` читает уже готовые landmark tensors из manifest и вызывает `predictor.predict(tensor)`. В этом path нет camera transitions, MediaPipe frame drops, live hand acquisition, pointer jitter или непрерывного пользовательского поведения.

`LivePredictionStabilizer` сейчас определен и покрыт unit-тестами, но не подключен к production `stream_webcam` или `stream_replay`. Фактическая live-стабилизация идет через `LiveLandmarkGestureController` и TARC.

## Где Используется C6

Файлы:

- `research_pipeline/models/c6_ensemble.py`
- `research_pipeline/models/hybrid.py`
- `research_pipeline/serve/live_backend.py`

`C6EnsembleRecognizer` объединяет:

- validated TCN artifact;
- augmented TCN artifact;
- geometry prior;
- calibrated fusion.

В live backend `method=c6_ensemble` пытается загрузить:

```text
artifacts/models/ipn_c1t_tcn_full_validated.pkl
artifacts/models/ipn_c1t_tcn_augmented.pkl
```

Сейчас после очистки репозитория `artifacts/` локально отсутствует. Это правильно для GitHub, но для новой строгой диагностики raw C6 live behavior нужно заново восстановить/пересоздать model artifacts.

В webcam mode C6 output:

- вычисляется как `model_prediction`;
- передается в `LiveLandmarkGestureController`;
- добавляется в controller scores только малой примесью `0.035 * model_score`;
- логируется в session JSONL extra как `model_gesture` / `model_confidence`;
- не является главным live action proposal.

Следствие: offline C6 metrics не доказывают live AR usability напрямую. Их нужно трактовать как recognition benchmark, а live usability измерять отдельно.

## Где Используется Landmark-First Controller

Файл: `research_pipeline/serve/live_backend.py`

`LiveLandmarkGestureController` является фактическим источником live gesture proposal в webcam mode.

Фактическая логика:

| Label | Code-level live condition |
|---|---|
| `no_gesture` | Недостаточно landmarks: `valid_ratio < 0.42`, `valid_frames < 3` или `confidence < 0.35`. |
| `point_2f` | Любая достаточно видимая рука без активного command candidate. Не проверяется именно two-finger pointing pose. |
| `click_2f` | Сначала open state: recent close/open ratio `> 0.64`; затем close state: `min(index-middle, thumb-index) / palm_scale <= 0.46`, low motion `< 0.09`, 2 candidate frames, lock-hold 5 frames, cooldown 20 frames. |
| `swipe_left/right` | За последние окна index fingertip должен сместиться по X больше `max(0.16, jitter * 3.8)`, движение должно быть в основном горизонтальным, 2 candidate frames, cooldown 14 frames. |
| `zoom_in/out` | Изменение `palm_scale` больше `0.20` и больше `scale_noise * 2.0`, при motion `< 0.18`, 2 candidate frames, cooldown 14 frames. |

Important mismatch: controller zoom is distance-to-camera / apparent hand-scale change. IPN labels describe `Zoom-in` / `Zoom-o` as pinch/open and pinch/close classes. Это не одно и то же физическое движение.

## Где Используется TARC

Файлы:

- `research_pipeline/interaction/task_aware.py`
- `research_pipeline/interaction/action_safe.py`
- `research_pipeline/serve/live_backend.py`
- `configs/interaction/ar_task_scenarios.yaml`

TARC (`TaskAwareActionSafePolicy`) берет текущий expected action из сценария и:

- снижает threshold для expected label;
- повышает threshold для unexpected labels;
- меняет required stable frames для expected label;
- считает false events, если action не совпала с ожидаемым шагом.

В текущем webcam flow `expected_label` используется дважды:

1. TARC context передается в `LiveLandmarkGestureController`, и controller уже на уровне candidate generation suppresses unrelated candidates.
2. Затем TARC повторно фильтрует готовую controller prediction.

Это может быть полезным, но это также риск: если сценарий backend не совпадает с UI task state, expected-label focus будет подавлять жест, который пользователь видит как правильный.

## Какие Жесты Ожидаются В UI

Файл: `demo/ar_interaction_app/src/main.tsx`

UI показывает 6 action gestures:

| UI gesture | UI text |
|---|---|
| `point_2f` | "Hold index and middle fingers visible." |
| `click_2f` | "Briefly pinch index to thumb, then open again." |
| `swipe_left` | "Move the whole visible hand horizontally left." |
| `swipe_right` | "Move the whole visible hand horizontally right." |
| `zoom_in` | "Move the hand closer so it grows in frame." |
| `zoom_out` | "Move the hand back so it shrinks in frame." |

UI gesture cards are CSS-drawn abstractions in `demo/ar_interaction_app/src/styles.css`, not rendered from real MediaPipe landmarks or reference video frames.

Current guide text is closer to controller logic than older text, but the visual still can mislead:

- `click_2f` visual animates thumb/index bending inside a synthetic hand. It does not show the required controller timing: open/armed -> short close -> lock -> release/cooldown.
- `zoom_in/out` visual shows a scale animation, which matches controller hand-scale logic, but it does not match the IPN `Zoom-in` / `Zoom-o` pinch/open class description.
- `point_2f` visual/text suggest a two-finger pose, while controller treats any visible hand as `point_2f`.

## Какие Жесты Есть В Reference Examples

Папки:

- `data/reference_gestures/ipn_hand`
- `data/interaction_gesture_examples`

Есть 7 labels:

```text
no_gesture
point_2f
click_2f
swipe_left
swipe_right
zoom_in
zoom_out
```

Каждый action label имеет по 3 reference clips из IPN subset. Manifest связывает labels с public IPN classes:

| Target label | IPN class |
|---|---|
| `no_gesture` | `D0X` / No gesture |
| `point_2f` | `B0B` / Point-2f |
| `click_2f` | `G02` / Click-2f |
| `swipe_left` | `G05` / Th-left |
| `swipe_right` | `G06` / Th-right |
| `zoom_in` | `G10` / Zoom-in |
| `zoom_out` | `G11` / Zoom-o |

Проверка metadata reference clips через `ffprobe`:

- resolution: `640x480`;
- frame rate: около `30 FPS`;
- длительность: примерно `2.5-3.9 s`;
- frame count: примерно `74-118`.

Это pre-segmented clips. Live backend использует sliding `deque(maxlen=32)`, то есть примерно 1.07 секунды при 30 FPS, причем окно может попадать на начало, середину, конец, паузу или переход между жестами.

## Обнаруженные Расхождения

### 1. Dataset Mismatch

IPN Hand дает сегментированные gesture clips. Live AR получает continuous stream. Это объясняет label jitter:

- live window не знает true gesture boundaries;
- transitions между idle/point/click/swipe попадают в те же 32 кадра, что и собственно gesture;
- часть жестов короткая, а окно может захватить слишком много до/после движения;
- lighting, blur, hand scale, camera angle и partial occlusion не представлены как отдельная live-control задача в offline metric.

Existing `docs/live_model_assessment.md` уже фиксирует прошлое наблюдение на пользовательском webcam видео: raw `c6_ensemble` был нестабилен и не выдавал `point_2f`, а controller давал более пригодное распределение для live control. Но это нужно воспроизвести заново после восстановления model artifacts.

### 2. Label / Gesture Mismatch

Самые рискованные места:

- `click_2f`: IPN/reference говорит "two-finger click"; hybrid geometry suppresses click по `index-middle` distance; live controller разрешает close по `min(index-middle, thumb-index)`; UI text говорит index-to-thumb pinch; CSS visual показывает синтетический thumb/index bend. Это четыре похожих, но не идентичных определения.
- `zoom_in/out`: IPN/reference описывает pinch/open и pinch/close zoom gestures; live controller и UI используют hand-scale change, то есть движение руки ближе/дальше к камере.
- `point_2f`: reference/UI ожидают two-finger pointing; live controller превращает любую видимую руку в `point_2f`, чтобы обеспечить cursor tracking.
- `swipe_left/right`: reference labels являются IPN `Th-left/Th-right`; live controller использует screen-space index fingertip `dx` after mirroring. Нужно отдельно проверить direction convention на реальных live clips.

### 3. Model / Live Mismatch

C6 может быть сильным offline recognizer, но текущий live path не использует его как прямой controller:

- raw model output в WebSocket UI не виден;
- `gesture` в payload - это controller label;
- `scores` в payload в основном synthetic controller scores плюс малая примесь model scores;
- `direct` interaction mode для webcam не означает "raw C6 direct"; это direct mapping от controller prediction.

Это может вводить пользователя и исследовательский отчет в заблуждение, если не разделить:

```text
raw recognizer prediction
live controller proposal
final TARC action
```

### 4. Controller / TARC Mismatch

TARC предполагает, что incoming prediction уже является meaningful proposal. Но в webcam path proposal формируется эвристическим controller. При этом expected label из TARC влияет на controller candidate generation.

Это нормально как метод, но опасно при ошибочном scenario state:

- если expected label неверный, controller подавляет фактически выполненный жест;
- если click not armed, TARC не поможет, потому что click proposal не появится;
- if controller emits `point_2f` during cooldown, TARC может считать задачу стоящей на expected step, но пользователь видит движение как уже выполненное.

### 5. UI / Backend Task Mismatch

Это критическое практическое расхождение.

UI live tasks в `main.tsx`:

```text
object:   point -> click -> zoom_in -> zoom_out
scroll:   swipe_right -> swipe_left -> click
transfer: point -> click -> swipe_right -> click
```

Backend TARC scenarios в `configs/interaction/ar_task_scenarios.yaml`:

```text
object:   pointer_hover -> select_confirm -> zoom_in -> zoom_out
scroll:   navigate_next -> navigate_next -> navigate_previous -> select_confirm
transfer: pointer_hover -> select_confirm -> navigate_next -> select_confirm -> navigate_previous
```

`object` совпадает по шагам. `scroll` и `transfer` не совпадают.

Последствие: frontend может считать следующий правильный шаг одним жестом, а backend policy context будет ожидать другой. Так как UI overlay использует `policy_context.expected_label`, пользователь может видеть подсказку, которая идет не из frontend task definition, а из backend YAML. Это создает ощущение "я делаю правильно, но система не принимает".

### 6. Gesture Guide Visual Mismatch

Guide должен показывать не абстрактную красивую руку, а фактический gesture contract.

Сейчас:

- карточки не основаны на reference clips;
- click visual не показывает arm/lock/release timing;
- zoom visual соответствует controller, но не IPN reference semantics;
- нет явного сравнения "what model was trained on" vs "what live controller accepts".

## Чего Сейчас Не Хватает Для Строгой Верификации

Не нужно выдумывать результаты. Для доказательной диагностики нужно заново собрать или восстановить:

1. Model artifacts:

```text
artifacts/models/ipn_c1t_tcn_full_validated.pkl
artifacts/models/ipn_c1t_tcn_augmented.pkl
```

2. Live session JSONL logs:

```text
artifacts/live_sessions/*.jsonl
```

3. Для пользовательского webcam video:

```text
C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\20260614_15_02_10_615.mp4
```

файл локально существует, но в чистом репозитории нет текущих model artifacts и нет fresh output reports. Старые численные наблюдения есть в `docs/live_model_assessment.md`, но их нужно воспроизвести pipeline-командой перед включением в финальную магистерскую отчетность.

4. Ground-truth для live tasks:

- timestamp, когда пользователь начинает/заканчивает gesture;
- expected step;
- accepted action;
- mistake/false action;
- completion time.

Без такой разметки нельзя честно утверждать, что live TARC лучше direct live controller.

## Минимальный Научно Обоснованный План Модернизации

Это план следующего этапа, не выполненный в рамках данного audit-only изменения.

### P0. Align Contracts Before UI Polish

1. Сделать один источник истины для live task steps.
   - Либо UI импортирует/получает backend scenarios.
   - Либо YAML генерируется из UI-visible task definitions.
   - `scroll` и `transfer` должны совпадать немедленно.

2. Разделить в telemetry:
   - `raw_model_gesture`;
   - `raw_model_confidence`;
   - `controller_gesture`;
   - `controller_mode`;
   - `final_action`;
   - `expected_label`.

3. Переименовать live labels в UI:
   - "Recognizer" для C6 offline/model signal;
   - "Live Controller" для landmark-first event logic;
   - "TARC" для final action gate.

### P1. Build Measurement Before More Heuristics

Добавить report для live recordings:

```text
raw label switches / minute
controller label switches / minute
confidence jitter
click false positives
click armed -> locked conversion rate
false commands / minute
missed expected actions
task completion time
pointer jitter
MediaPipe detection coverage
```

Нужно сравнить минимум:

```text
raw C6 direct
landmark controller direct
landmark controller + TARC
```

### P2. Fix Gesture Contract

Создать явную таблицу "model reference vs live controller vs UI guide":

| Gesture | IPN/reference | Live controller accepts | UI should show |
|---|---|---|---|
| `point_2f` | two-finger pointing | visible hand / stable pointer | either true two-finger point or rename live state to `pointer_tracking` |
| `click_2f` | must be verified from clips | open -> short close -> lock -> release | reference video or landmark animation with arm/lock/release timing |
| `swipe_left/right` | IPN Th-left/right | wide horizontal index-tip movement | whole-hand lateral movement with direction convention |
| `zoom_in/out` | IPN pinch/open/close | hand scale grows/shrinks | either change controller to pinch-spread or rename UI action to distance zoom |

Самый важный выбор: либо live controller должен следовать IPN gesture semantics, либо thesis должен честно назвать live controller отдельным continuous AR control layer с собственной operational semantics.

### P3. Replace Misleading Visual Cards

Gesture Guide должен показывать:

- короткие reference MP4 или extracted frame strips из `data/interaction_gesture_examples`;
- для live controller - отдельную simplified landmark animation, построенную по тем же признакам, что controller реально проверяет;
- текст timing: "prepare", "hold until lock", "release/cooldown".

CSS-hand visuals лучше оставить только как fallback, не как основной instructional source.

### P4. Decide On Data Strategy

Дообучение имеет смысл только после P1/P2.

Если проблема в segmentation/continuous mismatch, второй pre-segmented dataset сам по себе не решит live AR. Нужны:

- local calibration recordings с теми же gestures и камерой, что live demo;
- или dataset с continuous/egocentric hand-control sequences;
- или обучение/калибровка event detector поверх landmarks, отдельно от offline class recognizer.

Рекомендуемый минимум для магистерского проекта:

```text
public IPN benchmark -> C6 recognition evidence
local live recordings -> continuous AR interaction evidence
ablation -> raw C6 vs landmark controller vs TARC
metrics -> recognition F1 + live false actions/min + task success + pointer jitter
```

## Что Делать Следующим

Следующий кодовый этап должен быть не "перерисовать UI", а "свести контракты":

1. Синхронизировать UI task definitions и YAML backend scenarios.
2. Добавить raw-vs-controller telemetry в payload и live logs.
3. Добавить CLI report для raw/controller/TARC divergence на live recordings.
4. После этого менять Guide visual так, чтобы он был основан на фактическом gesture contract и reference examples.

