# Phone AR Transfer Plan

## Суть переноса

IPN Hand не является AR-датасетом. Он используется как публичный источник жестов для обучения compact temporal landmark recognizer. Перенос на телефон проверяет другой вопрос: сохраняется ли качество этого recognizer, когда те же жесты выполняются в реальном `phone_rear_ar` домене.

Финальная мобильная цепочка:

```text
iPhone rear camera frame
-> ARKit/RealityKit world tracking and rendering
-> hand landmarks from the same camera frame
-> [T,21,3] preprocessing to [1,32,74]
-> Core ML gesture classifier
-> C2 context-aware policy
-> AR object action
```

Это одна система по данным и управлению: меняется только runtime shell. На Windows это webcam/Three.js/ONNX, на iPhone это rear camera/RealityKit/Core ML.

## Почему local phone видео не смешиваются вслепую

Локальные ролики с задней камеры могут иметь другой ракурс ладони, масштаб, фон и направление движения относительно камеры. Поэтому они фиксируются как отдельный домен:

```text
source_dataset=local_phone
capture_domain=phone_rear_ar
camera_view=rear_world
coordinate_semantics=screen_space
adaptation_role=target_domain
viewpoint_policy=natural_rear_camera
palm_orientation_policy=not_forced
```

Это решает проблему несовпадения webcam-reference и задней камеры телефона: пользователь не должен выкручивать кисть, чтобы повторить сторону ладони из IPN. Reference задаёт класс команды, а phone rear клип фиксирует естественное AR-исполнение той же команды.

Порядок работы:

1. Public model обучается на IPN.
2. Public model оценивается на IPN test.
3. Те же веса проверяются на `phone_rear_ar` без дообучения.
4. Только затем выполняется local calibration или fine-tuning.
5. Отдельно сравниваются Direct и C2 task-level метрики.

## Правило направленных жестов

`swipe_left` и `swipe_right` размечаются в экранных координатах. Для пользователя это означает:

- `swipe_left`: действие объекта/карусели влево на экране;
- `swipe_right`: действие вправо на экране.

Mirroring augmentation при обучении меняет местами только эти два класса. `no_gesture`, `point_2f`, `click_2f`, `zoom_in`, `zoom_out` не меняют label при зеркалировании.

Сторона ладони не меняет label. Если задняя камера видит тыльную сторону руки, это всё ещё тот же жест, но в другом capture domain. Такой материал используется для zero-shot проверки и последующей калибровки/fine-tuning, а не смешивается с IPN как визуально идентичный webcam-клип.

## Что уже можно делать без локальных видео

- обучать и валидировать C1-T на IPN;
- экспортировать ONNX;
- писать Core ML conversion contract;
- генерировать mobile bundle;
- запускать desktop AR demo;
- строить domain readiness report по manifest-плану;
- готовить iOS Swift preprocessing/FSM слой.

## Что начнется после локальных видео

- extraction landmarks из `phone_rear_ar`;
- zero-shot report: IPN model on phone domain;
- C2 threshold calibration на local validation;
- optional TCN fine-tuning;
- final portability report и iPhone RealityKit demo.
