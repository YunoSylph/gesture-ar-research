# System UI and AR Interaction Plan

The current system UI is implemented in:

```text
demo/ar_interaction_app
```

It provides:

- recognizer method selection: `Baseline TCN` and `Robust C6`;
- action policy selection: `TARC` and `Direct`;
- benchmark telemetry from the full IPN run;
- live landmark-controller action dispatch for `point_2f`, `click_2f`, `swipe_left`, `swipe_right`, `zoom_in`, `zoom_out`;
- a Three.js AR object rendered over the live camera layer;
- real webcam background with MediaPipe landmark overlay and smoothed fingertip cursor mapping;
- three live AR tasks: object control, list scroll/open, and virtual item sorting;
- visual gesture guide cards with pose names and execution cues;
- experiment results page with method tables and charts;
- debug scene controls kept behind `Advanced Controls`.

Run it:

```powershell
cd demo/ar_interaction_app
npm install
npm run dev
```

Current URL:

```text
http://127.0.0.1:5173
```

## Next Engineering Step

The local backend bridge is now implemented:

```powershell
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

WebSocket:

```text
ws://127.0.0.1:8000/ws/stream?method=c6_ensemble&source=webcam&interaction=c4_task_aware
```

Camera video stream:

```text
http://127.0.0.1:8000/video_feed?camera=0&camera_width=1920&camera_height=1080&preview_width=1280&fps=30
```

Implemented flow:

```text
webcam frame
-> MediaPipe landmark extractor
-> preprocessing contract [32,74]
-> selected recognizer backend for research logging
-> landmark-first live controller with expected-gesture focus and lock-hold states
-> Direct action mapping or TARC task-aware policy
-> websocket action and fingertip pointer payload
-> camera-backed Three.js AR scene
```

Supported backend source:

- `Camera Stream` / `webcam`: uses OpenCV camera index `0`, MediaPipe HandLandmarker, a sliding 32-frame window, and the selected recognizer. The UI receives `preview_image`, normalized landmark points, and a fingertip `pointer` coordinate.
- Camera transport defaults to 1920x1080 capture, 1280px JPEG preview, target 30 FPS, mirrored view, and JSONL session logging.

Supported interaction modes:

- `Direct` / `direct`: maps each current live-controller label to an AR action immediately.
- `TARC` / `c4_task_aware`: applies task-aware thresholds, expected-gesture focus, lock-hold states, click arming, gesture-specific risk calibration, smoothing, and cooldowns before emitting actions.

Supported methods:

- `c1t_tcn`
- `c6_ensemble`

The UI is connected through `Start Task`. The scene test pad is available only inside `Advanced Controls` for deterministic UI-local checks.

## Remaining Work

- Add per-method backend switching without reconnect flicker.
- Add local adaptation model slot.
- Collect real successful local task sessions and compare Direct vs TARC with the scripted scenario windows.
