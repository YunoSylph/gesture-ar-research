# System UI and AR Interaction Plan

The current system UI is implemented in:

```text
demo/ar_interaction_app
```

It provides:

- recognizer method selection: `C0 Rule`, `C1 RF`, `C1-T TCN`, `ONNX`;
- benchmark telemetry from the full IPN run;
- gesture action dispatch for `point_2f`, `click_2f`, `swipe_left`, `swipe_right`, `zoom_in`, `zoom_out`;
- a Three.js AR object rendered over the live camera layer;
- real webcam background with MediaPipe landmark overlay and fingertip cursor mapping;
- dataset replay through the backend;
- AR task modes: object control, gallery navigation, target selection;
- experiment results page;
- direct and C2-gated interaction modes;
- manual scene test controls for deterministic UI-only checks.

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
ws://127.0.0.1:8000/ws/stream?method=onnx&source=webcam&interaction=direct
```

Camera video stream:

```text
http://127.0.0.1:8000/video_feed?camera=0&preview_width=960&fps=15
```

Implemented flow:

```text
webcam frame
-> MediaPipe landmark extractor
-> preprocessing contract [32,74]
-> selected recognizer backend
-> direct action mapping or C2 ContextAwarePolicy
-> websocket action and fingertip pointer payload
-> camera-backed Three.js AR scene
```

Supported backend sources:

- `Dataset File` / `replay`: streams real full-test IPN NPZ tensors through the selected recognizer.
- `Camera Stream` / `webcam`: uses OpenCV camera index `0`, MediaPipe HandLandmarker, a sliding 32-frame window, and the selected recognizer. The UI receives `preview_image`, normalized landmark points, and a fingertip `pointer` coordinate.
- Camera transport defaults to 1280x720 capture, 960px JPEG preview, target 12 FPS, mirrored view, and JSONL session logging.

Supported interaction modes:

- `Direct Control` / `direct`: maps each current recognizer label to an AR action immediately.
- `C2 Gate` / `c2`: applies activation threshold, stable frame count, cooldown, and no-gesture reset before emitting actions.

Supported methods:

- `c0`
- `c1_rf`
- `c1t_tcn`
- `onnx`

The UI is connected through `Start Camera` or `Start Dataset`. `Run Test` is separate: it only runs a deterministic UI-local gesture sequence for fast visual testing.

## Remaining Work

- Add per-method backend switching without reconnect flicker.
- Add local adaptation model slot.
- Collect real successful local task sessions and compare Direct Control vs C2 Gate with the scripted scenario windows.
