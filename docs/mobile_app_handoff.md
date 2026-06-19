# Mobile (iOS) variation — architecture, datasets, training & app handoff

This document consolidates everything about the **smartphone variation** of the
gesture-AR project: how the desktop research pipeline was ported to an on-device
iPhone app, what was trained on which data, how the model is exported, and how the
shipped iOS app is structured. It is the single entry point for picking up the
mobile work. Research framing and the desktop validation results live in
`docs/final_research_claim.md` and `docs/handoff_mobile.md`; this file is the
mobile-specific companion.

---

## 1. What the mobile variation is

A one-handed, in-air **gesture controller for AR on iPhone**, using the **rear
0.5× ultra-wide camera** as the gesture input. The user raises their free hand in
front of the phone; MediaPipe extracts hand landmarks; a Core ML temporal model
classifies the gesture; an acceptance policy gates it into an AR action that drives
a 3D scene. Three demo tasks show the interaction model: **Object Control**,
**AR Scrolling**, and **Sorting Objects**.

The research contribution is unchanged from the desktop project: the value is the
**reproducible gesture→action validation pipeline** (confidence / stability /
cooldown / release + task-aware acceptance), not a new SOTA recognizer. Only the
*capture front-end* (rear phone camera) and the *runtime* (Core ML + SceneKit)
change for mobile.

### End-to-end pipeline

```text
rear 0.5× ultra-wide camera (matches the recorded training domain)
  → MediaPipe Hands  → 21 landmarks (image-normalized x,y,z)
  → LandmarkPreprocessor          → [1, 32, 326] feature window
  → GestureModel (Core ML, FP16)  → 7-class logits → softmax
  → ContextAwareGesturePolicy     → (confidence / stability / cooldown / release)
  → ARGestureAction               → SceneKit 3D scene (per-task interaction)
  ‖ hand skeleton overlay (21 joints + bones) drawn over the camera
```

---

## 2. System architecture (four layers)

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 1. RESEARCH PIPELINE (Python, GPU/CPU)                                 │
│    datasets → MediaPipe-21 landmarks → 326-dim features → TCN ensemble │
│    → calibrated fusion (ECE) → rear_c1t_tcn_mv.pkl                      │
├──────────────────────────────────────────────────────────────────────┤
│ 2. EXPORT / CONTRACTS (Python, coremltools)                            │
│    torch TCN → Core ML .mlpackage (FP16, iOS15), parity-verified       │
│    + feature contract + golden vectors + policy contract + traces      │
├──────────────────────────────────────────────────────────────────────┤
│ 3. GestureARCore (SwiftPM library, macOS-testable)                     │
│    LandmarkPreprocessor · GestureModel · ContextPolicy ·               │
│    GestureRecognizer · GestureLabels · ARInteractionState              │
│    — verified 1:1 against the Python golden vectors via `swift test`   │
├──────────────────────────────────────────────────────────────────────┤
│ 4. GestureAR app (Xcode-only: AVFoundation + MediaPipe + SceneKit)     │
│    AppViewModel · ARViewContainer · GestureScene · GestureARApp        │
└──────────────────────────────────────────────────────────────────────┘
```

The split between layers 3 and 4 is deliberate: **GestureARCore has no iOS-only
dependencies**, so the numerically-critical preprocessing and policy code is
unit-tested on macOS (`swift test`) against the Python reference, while the
camera/ML/AR glue that needs a device lives in the app target.

---

## 3. Datasets

The rear phone camera is a different capture geometry than the desktop webcam
(IPN Hand): hand at arm's length, smaller/variable scale, moving handheld
background, ultra-wide viewpoint. The IPN-trained model collapses to `no_gesture`
zero-shot on rear clips, confirming a real domain gap — so the rear recognizer was
**retrained on a rear-appropriate mix**:

| Source | Role | Classes contributed | Notes |
|---|---|---|---|
| **Jester / 20BN** | primary dynamic | `no_gesture`, `swipe_left`, `swipe_right`, `zoom_in`, `zoom_out` | large, dynamic; ~4k/class after balancing |
| **HaGRID** | static poses | `point_2f` (two_up / peace) | single images → replicated-frame clips |
| **Local iPhone rear clips** | domain + rare class | `click_2f`, orientation fine-tune | 50 clips, rear 0.5×, back-of-hand |
| IPN Hand | desktop baseline only | (all 7) | used to *measure* the domain gap, not shipped for rear |

Label adapters map each source onto the **7-class action vocabulary** in
`research_pipeline/labels.py` (`remap_jester_label`, `remap_hagrid_label`).
`cli/merge_datasets` caps per-class/per-source counts, reports coverage, warns on
missing classes and preserves domain tags. The local clips are filtered by
detection coverage (`cli/filter_by_coverage`): `click_2f` extracts cleanly
(~98% frame detection); fast handheld swipes motion-blur/leave-frame and are
better sourced from Jester.

> Datasets themselves are **not** in git (`.gitignore` excludes `data/raw/`,
> `local_vids/`, the ~28 GB IPN download). Manifests under
> `data/interim/manifests/` are tracked.

---

## 4. Features & model

**Feature window — 326 dims/frame, 32 frames** (the OO-dMVMT multi-view block):

| Block | Dims | Description |
|---|---|---|
| pose | 63 | 21 landmarks × (x, y, z) |
| motion | 11 | frame-to-frame motion summary |
| JCD | 210 | joint–joint pairwise distance matrix (upper triangle) |
| slow motion | 21 | low-temporal-rate velocity |
| fast motion | 21 | high-temporal-rate velocity |
| **total** | **326** | row-major `[time × dim]` |

**Model:** compact temporal CNN (TCN) ensemble, `feature_set:
dual_view_multiview`, ~650K params. Class-balanced / focal training; C6 calibrated
fusion with ECE in the candidate-selection objective. Configs:
`configs/train/rear_c1t_tcn_mv.yaml` (mirrors the validated IPN config).

**Rear model:** `artifacts/models/rear_c1t_tcn_mv.pkl` — held-out val accuracy
**≈ 0.82**. `point_2f`, `zoom_in/out`, `swipe_left/right`, `no_gesture` all work
(a real contrast with the IPN zero-shot collapse). **`click_2f` is under-supported
(7 samples) and confuses with `point_2f`** — the main quality gap (see §8).

Training runs on CUDA (RTX laptop) or MPS; the loop auto-selects CUDA→MPS→CPU. The
~650K-param TCN is ~10× faster on MPS than CPU (~25 min full run). The reason heavy
runs stay on GPU is the *data* side (Jester ~22 GB + landmark extraction over
~148K clips), not the optimizer. Runbook: `docs/rear_training_runbook.md`.

---

## 5. Export & on-device contracts

`cli/export_coreml` → `models/coreml_export`: converts the TCN artifact to a Core
ML `.mlpackage` (mlprogram / **FP16** / iOS15) and verifies **torch ↔ Core ML
numerical parity**. The converter reads the real `input_dim` (326, not 74). The
traced graph is `jit.freeze`d before conversion — otherwise identical BatchNorm
`num_batches_tracked` constants get deduplicated and trip a coremltools lowering
assertion (notably on MPS-trained weights).

Three contracts keep Swift in lock-step with Python:

- `cli/export_preprocessing_contract` → 326-dim feature layout + landmark→feature
  **golden vectors** (`golden_samples.json`); Swift parity asserted < 1e-4.
- `cli/export_validation_contract` → acceptance-policy spec +
  **golden decision traces** (`golden_traces.json`); Swift reproduces them 100%.
- `cli/export_mobile_bundle` → packaged bundle using the accurate 326-dim contract.

Core ML input is `landmarks`, shape `[1, 32, 326]`; output is 7 logits in
`GestureLabel` order.

---

## 6. iOS app architecture

### GestureARCore (`Sources/GestureARCore/`, SwiftPM, `swift test` on macOS)

| File | Responsibility |
|---|---|
| `LandmarkPreprocessor.swift` | exact 326-dim feature extraction; golden-parity tested |
| `GestureModel.swift` | Core ML wrapper; `[1,32,326]` → softmax → `GesturePrediction` |
| `ContextPolicy.swift` | acceptance FSM (confidence / stability / cooldown / release); per-class floors |
| `GestureRecognizer.swift` | sliding 32-frame window → preprocess → classify → gate |
| `GestureLabels.swift` | 7-class vocabulary ↔ `ARGestureAction` mapping |
| `ARInteractionState.swift` | reusable focus / zoom / select / point state machine |
| `Types.swift` | `MediaPipeFrame` and shared value types |

### GestureAR app (`App/`, Xcode-only)

| File | Responsibility |
|---|---|
| `GestureARApp.swift` | `@main`; task selector + `ARTaskView` (camera + scene + minimal UI) |
| `AppViewModel.swift` | owns the `AVCaptureSession` (rear 0.5× ultra-wide), MediaPipe `HandLandmarker`, and `GestureRecognizer`; publishes `handDetected`, `skeleton`, `lastAction`; emits `ARGestureAction` events |
| `ARViewContainer.swift` | `UIViewController` stacking three layers: camera preview → transparent `SCNView` → hand-skeleton overlay; also the `SkeletonOverlayView` (CAShapeLayer, aspect-fill mapped) |
| `GestureScene.swift` | `SceneController` — builds the per-task SceneKit scene (lit PBR objects) and applies gesture actions |
| `TaskModel.swift` | `ARTask` enum (titles/subtitles/hints/icons) + `HandSkeleton` value type |

### Why SceneKit (not a non-AR RealityKit `ARView`)

A non-AR RealityKit `ARView` renders an **opaque Metal background** that covers the
camera preview beneath it (setting `backgroundColor = .clear` on the UIView does
not help). **SceneKit `SCNView` with `isOpaque = false` and a nil scene background
composites reliably over the camera layer** — this is what makes the camera visible
behind the 3D objects. (This was the root cause of an earlier black-screen bug,
alongside the camera session never being configured and the camera permission never
being requested — all fixed.)

### Layer order (bottom → top)

```text
AVCaptureVideoPreviewLayer   (live camera)
SCNView (transparent)        (lit 3D task objects)
SkeletonOverlayView          (21 joints + bones over the hand)
SwiftUI overlay              (minimal top bar + bottom hint/action toast)
```

---

## 7. Gesture vocabulary & per-task interaction

Seven classes → `ARGestureAction`:

| Gesture | Action | Object Control | AR Scrolling | Sorting Objects |
|---|---|---|---|---|
| `swipe_left` | navigatePrevious | rotate −Y | focus previous | move/swap left |
| `swipe_right` | navigateNext | rotate +Y | focus next | move/swap right |
| `zoom_in` | zoomIn | scale up | — | — |
| `zoom_out` | zoomOut | scale down | — | — |
| `click_2f` | selectConfirm | recolor + pulse | open card | pick up & swap |
| `point_2f` | pointerHover | nudge | — | — |
| `no_gesture` | — | idle | idle | idle |

Each task centers its focused element and highlights it (emission glow + scale).

---

## 8. State — done & verified

- ✅ **Python pipeline**: dataset adapters, merge/balance, 326-dim features, TCN
  training, calibration, Core ML export with parity — unit-tested (`pytest`).
- ✅ **Rear model trained & exported**: `rear_c1t_tcn_mv.pkl` (~0.82 val) →
  `GestureClassifier.mlpackage` (FP16, parity-verified).
- ✅ **GestureARCore**: preprocessing + policy + interaction state, `swift test`
  green against Python golden vectors/traces.
- ✅ **iOS app builds & runs**: task selector + all three SceneKit scenes verified
  in the Simulator (objects render over the transparent layer; camera shows through
  on device); minimal UI respects safe areas; camera permission requested.
- ✅ **MediaPipe** linked via `SwiftTasksVision` SPM (no CocoaPods);
  `hand_landmarker.task` bundled.

### Known limitations

- **`click_2f`** is under-trained (7 samples) and over-fires on `point_2f`. Not
  separable by a hand-crafted geometry rule (index–middle distance + fingertip
  motion overlap). Mitigation shipped: per-class confidence floor
  `perClassActivation = [.click2f: 0.9]`. Robust fix: record more `click_2f` /
  `point_2f` rear clips → re-merge → retrain.
- **No world tracking yet**: objects are device-relative (held in front of the
  phone), not anchored in the room. The 0.5× ultra-wide has no ARKit world
  tracking; world-anchored AR needs the 1× wide camera with
  `ARWorldTrackingConfiguration` (different FOV → re-validate recognition).
- **Simulator has no camera** — hand detection / gestures only run on a device.

### Next steps (in progress)

1. **ARKit world-anchored scene** — float the 3D tasks in space (and set back from
   the camera so they don't block the view) via `ARWorldTrackingConfiguration`.
2. **Hand occlusion** — render the real hand in front of the 3D objects using
   Apple frameworks (people/hand occlusion or a Vision hand mask).
3. **Recognition quality** — analyze the local iPhone clips to retune thresholds
   and gather more `click_2f` / `point_2f`; re-merge and retrain.

---

## 9. Build & run

```bash
brew install xcodegen
cd ios_demo/GestureAR
xcodegen generate
open GestureAR.xcodeproj   # set a signing team, select your iPhone, Run
```

The trained `.mlpackage`, `hand_landmarker.task`, and MediaPipe (`SwiftTasksVision`)
are all wired in `project.yml`; the first build downloads the MediaPipe binaries
(large). Grant the camera prompt on first launch. Refreshing the model: re-export
`rear_c1t_tcn_mv.pkl` and copy `GestureClassifier.mlpackage` into `ios_demo/GestureAR/`.
