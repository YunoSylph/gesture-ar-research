# GestureAR — iOS on-device app

iPhone app for the rear-camera gesture pipeline. Same recognizer as the desktop demo;
the capture front-end (rear 0.5× ultra-wide camera) and the AR runtime (Core ML +
SceneKit) change. Full architecture, datasets and training:
[`docs/mobile_app_handoff.md`](../../docs/mobile_app_handoff.md).

```text
rear 0.5× ultra-wide camera (matches the training data)
  → MediaPipe Hands → 21 landmarks (image-normalized x,y,z)
  → LandmarkPreprocessor          [1, 32, 326]
  → GestureModel (Core ML, FP16)  → 7-class logits
  → ContextAwareGesturePolicy     (confidence / stability / cooldown / release)
  → ARGestureAction               → SceneKit 3D scene (per-task interaction)
  ‖ hand skeleton overlay (21 joints + bones) over the camera
```

## Two layers

**`Sources/GestureARCore/`** — SwiftPM library, no iOS-only deps, so it `swift test`s on
macOS (a verified 1:1 port of the Python reference):

```bash
cd ios_demo/GestureAR && swift test
```

- `LandmarkPreprocessor.swift` — exact **326-dim** mv features (vs `golden_samples.json`, < 1e-4).
- `ContextPolicy.swift` — acceptance policy (vs `golden_traces.json`); optional **per-class floors**.
- `GestureRecognizer.swift` — sliding 32-frame window → preprocess → classify → gate.
- `ARInteractionState.swift`, `GestureLabels.swift`, `GestureModel.swift`, `Types.swift`.

**`App/`** — Xcode-only app target (NOT built by `swift test`; validate in Xcode):

- `GestureARApp.swift` — `@main`; minimal task selector + `ARTaskView`.
- `AppViewModel.swift` — rear 0.5× capture + MediaPipe `HandLandmarker` + `GestureRecognizer`;
  publishes hand/skeleton state and emits `ARGestureAction` events.
- `ARViewContainer.swift` — layered `UIViewController` (camera → transparent `SCNView` →
  skeleton overlay) + `SkeletonOverlayView`.
- `GestureScene.swift` — `SceneController`: per-task SceneKit scene + gesture handling.
- `TaskModel.swift` — `ARTask` enum + `HandSkeleton`.

## The experience

Three demo tasks, each a distinct SceneKit scene rendered over the live camera with the
MediaPipe hand skeleton drawn on top:

- **Object Control** — swipe ← → to rotate, zoom in/out to scale, click to recolor.
- **AR Scrolling** — swipe ← → to move focus through a vertical card stack, click to open.
- **Sorting Objects** — swipe ← → to move, click to pick up & swap with a neighbor.

Minimal, non-fullscreen UI: a compact top bar (back, title, hand indicator) and a bottom
hint / action toast. The objects use lit PBR materials so they read as solid 3D.

> **3D layer = SceneKit, on purpose.** A non-AR RealityKit `ARView` renders an opaque
> background that hides the camera; a transparent `SCNView` (`isOpaque = false`) composites
> reliably over the camera preview. Objects are currently **device-relative** (held in front
> of the phone), because the 0.5× ultra-wide has no ARKit world tracking. World-anchored AR
> (objects fixed in the room) needs the 1× wide camera with `ARWorldTrackingConfiguration` —
> a different FOV than the 0.5× training data, so re-validate recognition.

## Build & run

Everything is wired in `project.yml`: the trained `GestureClassifier.mlpackage`, the
`hand_landmarker.task`, and **MediaPipe** (via the `SwiftTasksVision` SPM wrapper — no
CocoaPods).

```bash
brew install xcodegen
cd ios_demo/GestureAR
xcodegen generate
open GestureAR.xcodeproj   # set a signing team, select your iPhone, Run
```

The first build downloads the MediaPipe binary frameworks (large). The app builds and runs
in the Simulator (task selector + all three 3D scenes render), but the **Simulator has no
camera**, so hand detection / gestures only work on a real iPhone — select your device, Run,
and grant the camera prompt. The camera-usage string is generated from `project.yml`.

Notes:
- The model (`GestureClassifier.mlmodelc`) and `hand_landmarker.task` are embedded
  automatically; no manual Xcode steps.
- MediaPipe orientation: frames are delivered portrait-upright and passed to `MPImage`
  with `.up`. Adjust in `AppViewModel.processFrame` if detection looks rotated.
- Core ML input is `landmarks`, shape `[1, 32, 326]`. To refresh the model, re-export
  `rear_c1t_tcn_mv.pkl` and copy `GestureClassifier.mlpackage` here.
- Test launch hooks (inert in normal use): `SIMCTL_CHILD_GAR_TASK=0|1|2` boots straight into
  a task; `SIMCTL_CHILD_GAR_NOCAM=1` skips the camera path (for Simulator screenshots).

## click_2f calibration note

`click_2f` is under-supported (only 7 local training samples) and over-fires on `point_2f`.
Click vs point are **not separable by a simple geometry/motion rule** (index–middle distance
and fingertip motion overlap), so no hand-crafted discriminator ships. Mitigation: a raised
per-class floor — `config.perClassActivation = [.click2f: 0.9]` — so the high-risk select
only commits on very confident clicks. Robust fix: more `click_2f` / `point_2f` rear clips →
re-merge → retrain. A dwell-to-confirm UX on the focused object is a good alternative.

## Regenerating golden test resources

```bash
python -m research_pipeline.cli.export_preprocessing_contract \
  --manifest data/interim/manifests/local_phone_capture_landmarks_filtered.jsonl \
  --model-path artifacts/models/rear_c1t_tcn_mv.pkl \
  --output-dir artifacts/mobile/preprocessing
python -m research_pipeline.cli.export_validation_contract --output-dir artifacts/mobile/validation
# copy the *.json into Tests/GestureARCoreTests/Resources/
```
