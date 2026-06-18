# Быстрый запуск Gesture AR

## 1. Запуск одной командой

Откройте PowerShell в корне проекта:

```powershell
cd "C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research"
powershell -ExecutionPolicy Bypass -File .\scripts\start_ar_demo.ps1
```

Скрипт запустит:

- `.venv-gesture-ar` и live-зависимости из `requirements/live.txt`, если окружение еще не создано;
- `npm install` для frontend, если `node_modules` отсутствует;
- Python backend на `http://127.0.0.1:8000`;
- React/Three.js интерфейс на первом свободном порту `5173-5179`;
- браузер с точным адресом интерфейса.

Если порты заняты старым запуском:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ar_demo.ps1 -Restart
```

Логи лежат в `artifacts/logs`, frontend пишет в файл вида `ar_frontend_5173.log`.

## 2. Ручной запуск

Терминал 1:

```powershell
cd "C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research"
python -m venv .venv-gesture-ar
.\.venv-gesture-ar\Scripts\Activate.ps1
pip install -r requirements\live.txt
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
```

Терминал 2:

```powershell
cd "C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\gesture-ar-research\demo\ar_interaction_app"
npm install
npm run dev -- --port 5173
```

Откройте URL, который напечатает Vite, обычно `http://127.0.0.1:5173`.

## 3. Как пользоваться интерфейсом

1. Проверьте, что в блоке `Telemetry` backend показывает `backend ready`.
2. Выберите задачу в блоке `AR Task`.
3. Нажмите `Start Task`.
4. Камера должна стать фоном AR-сцены, а объект должен отображаться поверх видео.
5. Выполняйте жест, который подсвечен в правом верхнем overlay, и доводите lock bar до `locked`.
6. Смотрите `Telemetry`: текущий жест, `FPS`, `Proc`, `Detect` и состояние камеры.
7. По умолчанию выбран усиленный метод `Robust C6` и режим `TARC`. Если нужно вручную выбрать другую модель, direct mode, camera index, FPS или тестовые кнопки жестов, откройте `Advanced Controls`.

## 3.1. AR task-сценарии

В блоке `AR Task` теперь оставлены только 3 понятных live-сценария:

- `1. Object control`: наведение на AR-модуль, короткий click, приближение руки, отведение руки назад.
- `2. Scroll and open`: горизонтальный swipe right/left по AR-списку и короткий click по строке.
- `3. Sort virtual item`: выбор предмета, перенос в правый контейнер и сброс.

Кнопка `Start Task` переводит ввод в `Camera Stream`, запускает live-сессию и начинает проверять шаги сценария по действиям, которые приходят из backend через WebSocket. Если камера закрыта шторкой, task останется в состоянии ожидания и покажет низкое обнаружение руки.

Вкладка `Guide` в интерфейсе показывает названия жестов, визуальные подсказки и действие каждого жеста. Подробная инструкция: `docs/ar_interface_user_guide.md`.

Техническая оценка live-качества на записанном webcam-видео и объяснение, почему добавлен landmark-first controller: `docs/live_model_assessment.md`.

Примеры жестов, нотация и CSV-последовательности задач: `data/interaction_gesture_examples`. Быстрая памятка по связке жестов с задачами: `data/interaction_gesture_examples/TASK_INTERACTIONS.md`.

## 4. Если камера не работает

- Проверьте `http://127.0.0.1:8000/api/health`.
- Проверьте разрешение камеры в Windows Privacy settings.
- Закройте Zoom/Teams/браузеры, если они заняли камеру.
- Попробуйте другой индекс в поле `Camera`: `0`, `1`, `2`.
- Посмотрите `artifacts/logs/ar_backend.log`.

## 5. Live-логи и отчёты

Каждая backend-сессия сохраняется в:

```text
artifacts/live_sessions/*.jsonl
```

Сводку по последней сессии можно построить так:

```powershell
python -m research_pipeline.cli.summarize_live_session
```

Отчёт будет записан в `artifacts/reports/live_session_summary.json`.

Task-level отчёт по AR-задачам:

```powershell
python -m research_pipeline.cli.report_live_tasks
```

Он создаёт `artifacts/reports/live_task_report.json` с FPS/latency, coverage по действиям, pointer coverage, warnings и ground-truth метриками по сценариям из `configs/interaction/ar_task_scenarios.yaml`.

## 5.1. Автономные recognition/interaction benchmarks

Без локально снятых видео исследовательское ядро развивается через C6 Ensemble и C4 task-aware AR interaction. C6 усиливает распознавание: validated TCN + augmented TCN + calibrated C3/C5 fusion.

```powershell
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_augmented.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_augmented.yaml
python -m research_pipeline.cli.benchmark_c3_hybrid --config configs/eval/c6_augmented_robustness.yaml
python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition.yaml
python -m research_pipeline.cli.benchmark_c3_hybrid --config configs/eval/c3_hybrid_robustness.yaml
python -m research_pipeline.cli.run_c3_research --config configs/eval/c3_research_ablation.yaml
python -m research_pipeline.cli.generate_c3_research_assets
python -m research_pipeline.cli.run_c4_action_safe_research --config configs/eval/c4_action_safe_research.yaml
python -m research_pipeline.cli.generate_c4_research_assets
python -m research_pipeline.cli.benchmark_c4_tasks --config configs/eval/c4_task_benchmark.yaml
python -m research_pipeline.cli.generate_c4_task_assets
python -m research_pipeline.cli.analyze_c4_task_failures
python -m research_pipeline.cli.build_experiment_chapter
python -m research_pipeline.cli.analyze_recognition_risk
python -m research_pipeline.cli.report_project_status
```

Основной отчёт:

```text
artifacts/reports/c3_hybrid_robustness.json
artifacts/reports/c3_research_ablation.json
artifacts/reports/c3_research_tables.md
artifacts/reports/c3_tables/*.csv
artifacts/figures/c3_*.png
artifacts/reports/c4_action_safe_research.json
artifacts/reports/c4_action_safe_tables.md
artifacts/reports/c4_tables/*.csv
artifacts/figures/c4_*.png
artifacts/reports/c4_task_benchmark.json
artifacts/reports/c4_task_benchmark_tables.md
artifacts/reports/c4_task_failure_analysis.md
artifacts/reports/thesis_experiment_chapter.md
artifacts/reports/c4_task_tables/*.csv
artifacts/figures/c4_task_*.png
artifacts/reports/c6_recognition_upgrade.md
artifacts/reports/c6_ensemble_calibrated_recognition.json
docs/c6_recognition_upgrade.md
configs/interaction/action_risk_costs.yaml
```

Итог C6: clean macro F1 вырос с `0.850` до `0.887`, средний robustness macro F1 с `0.826` до `0.859`, а robust false action rate снизился примерно с `0.097` до `0.067`.

Task-level C4 benchmark проверяет уже не одиночные жесты, а выполнение AR-сценариев `object`, `scroll`, `browser`, `transfer`, `placement`, `measurement`, `docking` и других. Вариант `c4_task_aware` использует текущий ожидаемый шаг интерфейса как контекст: он сохраняет task success уровня `C3 + C2 default`, но снижает false action cost rate с `0.0905` до `0.0527`. В живом интерфейсе этот метод доступен как режим `C4 Task`.

Описание исследовательской постановки: `docs/c3_hybrid_research.md` и `docs/c4_action_safe_research.md`.

## 6. Без локальных видео: текущие отчёты

Эти команды уже работают без ваших записей:

```powershell
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_full_validated.yaml
python -m research_pipeline.cli.analyze_recognition_risk
python -m research_pipeline.cli.report_domain_readiness --manifests data/interim/manifests/ipn_train_full_landmarks.jsonl data/interim/manifests/ipn_test_full_landmarks.jsonl data/interim/manifests/local_phone_plan.jsonl
python -m research_pipeline.cli.export_mobile_bundle
python -m research_pipeline.cli.report_project_status
```

Основные результаты лежат в:

```text
artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json
artifacts/reports/recognition_risk_analysis.json
artifacts/reports/domain_readiness.json
artifacts/mobile/gesture_mobile_bundle
artifacts/reports/project_stage_status.json
```

## 7. Жесты

- Эталонные клипы для локальной записи лежат в `data/reference_gestures/ipn_hand`.
- `point_2f`: AR-курсор по указательному пальцу.
- `click_2f`: выбор/подтверждение.
- `swipe_left`, `swipe_right`: навигация.
- `zoom_in`, `zoom_out`: масштабирование объекта.

Локальные phone AR ролики должны попадать в `data/raw/local_phone/videos` и соответствовать manifest-плану `data/interim/manifests/local_phone_plan.jsonl`.

Важно: для задней камеры телефона не нужно выворачивать кисть, чтобы повторить webcam-ракурс эталона. Reference задаёт смысл команды, а не сторону ладони. Подробно: `docs/phone_rear_gesture_resolution.md`.
