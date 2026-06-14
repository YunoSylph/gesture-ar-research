# Local Capture Protocol

Локальные клипы используются только для adaptation/demo слоя, а не как основной источник обучения. Они должны описывать отдельный домен `phone_rear_ar`, чтобы не смешивать IPN/webcam и реальную телефонную AR-съёмку вслепую.

## Главный принцип phone AR

Финальная мобильная демонстрация использует заднюю камеру как общий источник:

```text
задняя камера iPhone -> hand landmarks -> classifier/C2 -> RealityKit AR object
```

Жесты размечаются в экранных координатах. `swipe_left` означает действие влево на экране/в AR-интерфейсе, а `swipe_right` означает действие вправо. Это важнее, чем анатомическая сторона ладони относительно камеры.

Важное уточнение по стороне ладони: reference-клипы из IPN/webcam нужны как словарь команд, а не как требование показать ту же сторону кисти. Для `phone_rear_ar` жест выполняется естественно перед задней камерой телефона. Если камера видит тыльную сторону руки вместо ладони, это не ошибка и не повод выворачивать кисть. Такой клип остаётся тем же target label, но относится к отдельному домену `phone_rear_ar`.

Подробный разбор этой проблемы и решение зафиксированы в:

```text
docs/phone_rear_gesture_resolution.md
```

## Эталонные жесты

Перед записью откройте референсы:

```text
data/reference_gestures/ipn_hand
```

Там лежат по 3 клипа на каждый финальный класс:

- `no_gesture`
- `point_2f`
- `click_2f`
- `swipe_left`
- `swipe_right`
- `zoom_in`
- `zoom_out`

Повторять нужно именно эти семантики. Не добавляйте новые жесты, не заменяйте `point_2f` на один палец и не меняйте направление `swipe_left/right`.

При этом не копируйте webcam-ракурс буквально. Копируется команда:

- `point_2f`: указание/наведение двумя пальцами в AR-сцене;
- `click_2f`: короткое подтверждение;
- `swipe_left/right`: экранное направление;
- `zoom_in/out`: разведение/сведение пальцев.

Сторона кисти относительно камеры не является label.

## Минимум

- 25 клипов: 5 классов x 5 повторов.
- 2-4 секунды на клип.
- 30 fps, 720p или 1080p.
- Один жест на клип.

## Целевой набор

- 50 клипов: 5 классов x 10 повторов.
- 2-3 сессии.
- 2-3 фона.
- 2-3 световых условия.
- Указать `hand_recorded`, `background_tag`, `lighting_tag`, `camera_device`.

## CSV поля

Минимальные поля для `ingest_local_videos`:

```text
file_name,target_label,participant_id,session_id,repetition_id,hand_recorded,background_tag,lighting_tag
```

Дополнительные поля из manifest schema можно добавлять сразу в CSV. Для phone AR используйте доменные поля:

```text
capture_domain=phone_rear_ar
camera_view=rear_world
coordinate_semantics=screen_space
adaptation_role=target_domain
viewpoint_policy=natural_rear_camera
palm_orientation_policy=not_forced
```

Пример шаблона:

```text
data/raw/local_phone/local_capture_template.csv
```

Сейчас шаблон можно прогнать через `ingest_local_videos` даже без самих видео, чтобы получить manifest-план. Landmark extraction и C2+Local запускаются только после появления файлов.
