# Gesture AR Research Project Overview

Дата состояния: 2026-06-14

## Что это за проект

Проект реализует исследовательскую систему распознавания жестов руки для AR-взаимодействия:

- публичный benchmark на IPN Hand;
- MediaPipe landmark extraction;
- temporal TCN / robust C6 recognition pipeline;
- live webcam backend;
- React + Three.js AR interface;
- task-aware risk-calibrated control layer для снижения ложных AR-команд.

Главная практическая идея текущей версии: сегментная модель жестов не используется как единственный live-контроллер. Для реального webcam AR добавлен `LiveLandmarkGestureController`, который работает поверх landmarks и вводит фиксацию жеста:

```text
camera frame
-> MediaPipe landmarks
-> live landmark gesture controller
-> preparing / locked / cooldown
-> Direct or TARC task-aware action policy
-> AR task interaction
```

Модель `Robust C6` сохраняется как исследовательский recognizer для benchmark, логирования и сравнения, но live-команды стабилизируются отдельным continuous-control слоем.

## Где смотреть код

- Backend live-потока: `research_pipeline/serve/live_backend.py`
- Модели и fusion: `research_pipeline/models/`
- Interaction policies: `research_pipeline/interaction/`
- Веб-интерфейс: `demo/ar_interaction_app/src/main.tsx`
- Стили интерфейса: `demo/ar_interaction_app/src/styles.css`
- Unit tests live-контракта: `tests/unit/test_live_backend_contract.py`

## Как запустить

Открыть PowerShell в корне проекта:

```powershell
cd "C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research"
powershell -ExecutionPolicy Bypass -File .\scripts\start_ar_demo.ps1 -Restart
```

Обычно интерфейс будет доступен на:

```text
http://127.0.0.1:5173
```

Подробная инструкция: `START_HERE.md`.

## Текущее состояние live UI

Интерфейс оставлен компактным:

- `Live`: camera-backed AR task surface;
- `Guide`: визуальные подсказки жестов;
- `Results`: графики и таблицы экспериментов.

Live-задачи:

- `1. Object control`;
- `2. Scroll and open`;
- `3. Sort virtual item`.

Жесты в live-режиме теперь имеют фиксацию:

- `point_2f`: рука видна, управление курсором;
- `click_2f`: открытая рука -> короткий pinch/tap -> открыть обратно;
- `swipe_left/right`: широкое горизонтальное движение;
- `zoom_in/out`: изменение масштаба руки в кадре.

Overlay показывает lock bar. Команда считается принятой после состояния `locked`, а не после первого случайного кадра.

## Научная постановка

Текущий исследовательский вклад лучше формулировать как комбинированный метод:

```text
public-data recognizer
+ robust landmark geometry
+ expected-gesture task focus
+ lock-hold live fixation
+ TARC risk-aware action policy
```

Эта постановка честно разделяет:

- offline recognition quality на IPN Hand;
- live AR interaction reliability;
- риск ложных команд в AR-задачах.

Ключевые документы:

- `deep-research-report.md`
- `docs/c3_hybrid_research.md`
- `docs/c4_action_safe_research.md`
- `docs/c6_recognition_upgrade.md`
- `docs/live_model_assessment.md`
- `docs/project_research_assessment.md`
- `docs/system_ui_plan.md`

## Текущие ограничения

- IPN Hand является сегментным webcam dataset и не идеально соответствует continuous AR control.
- Live-взаимодействие зависит от качества MediaPipe landmarks, света, фона, motion blur и положения руки.
- Для полноценной магистерской оценки нужен отдельный live evaluation protocol: false commands per minute, task completion, click precision, pointer jitter.
- Следующий сильный шаг: собрать небольшую локальную calibration/evaluation выборку и сравнить Direct, landmark-controller и TARC на одинаковых live-сценариях.

## Проверка текущего состояния

Последняя проверка перед коммитом:

```powershell
npm run build
python -m pytest -q
python -m py_compile research_pipeline\serve\live_backend.py
```

Ожидаемое состояние: frontend собирается, backend тесты проходят, live demo запускается через `scripts/start_ar_demo.ps1`.
