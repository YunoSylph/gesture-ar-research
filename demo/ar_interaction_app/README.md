# Gesture AR Interaction App

React + Three.js desktop demo surface for the final system layer.

```powershell
cd demo/ar_interaction_app
npm install
npm run dev
```

Current scope:

- recognizer method selection;
- full-benchmark metrics panel;
- gesture action dispatch;
- interactive Three.js object rendered over the camera layer;
- dataset replay stream through the Python backend;
- real webcam AR background with landmark overlay through the Python backend;
- multiple AR tasks: object control, gallery navigation, target selection;
- experiment results page.

Backend:

```powershell
cd ../..
.\.venv311\Scripts\Activate.ps1
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
```

UI terms:

- `Dataset File`: streams saved IPN full-test landmark tensors through the selected method.
- `Camera Stream`: opens local camera index `0` by default, extracts MediaPipe landmarks, and places the AR object over the live camera frame.
- `Direct Control`: maps the current classifier output directly to an AR action. In camera mode, the index fingertip drives the AR cursor position.
- `C2 Gate`: routes recognizer output through confidence and stability checks before an action reaches the scene.
- `Start Camera` / `Start Dataset`: connects the selected backend WebSocket source.
- `Run Test`: runs a deterministic manual gesture script inside the UI without the backend.
- `Target FPS`, `Preview px`, `JPEG`: control live camera smoothness and transport quality.

Live sessions are logged to `artifacts/live_sessions/*.jsonl`. Summarize the latest one:

```powershell
python -m research_pipeline.cli.summarize_live_session
python -m research_pipeline.cli.report_live_tasks
```

Task scenarios live in `configs/interaction/ar_task_scenarios.yaml`. They define expected AR actions and timing windows for object control, gallery navigation, and target selection.

Next integration step: collect real successful local sessions and use them for C2/local adaptation analysis.
