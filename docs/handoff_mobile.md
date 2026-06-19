# Project Handoff — continuing on macOS (M1) and the mobile path

This file is the portable project memory. A chat session does not travel between
machines, so read this (plus `docs/final_research_claim.md` and
`docs/evaluation_protocol.md`) to pick up the work on the MacBook. It captures
what the project is, what is done and validated, what to run, and the staged plan
for the rear phone-camera variant.

## What the project is

Gesture-based mid-air interaction for AR. The defended contribution is a
**reproducible continuous gesture-to-action validation pipeline**, not a new SOTA
recognizer. Recognizer output is treated as an action *proposal* that a validation
layer (confidence / stability / cooldown / release) plus TARC (task-aware
acceptance) must approve before it becomes an AR action. Reference paper: OO-dMVMT
(Cunico et al. 2023) — used as methodological direction only, never as a numeric
baseline. Full framing and the four scope boundaries: `docs/final_research_claim.md`.

## Current state (validated)

- **Recognizer:** two-TCN C6 ensemble on IPN Hand (21 MediaPipe landmarks). The
  deployed models use the OO-dMVMT multi-view feature block (JCD + slow/fast
  motion); `feature_set: dual_view_multiview`. Clip-level acc ≈ 0.92, macro F1 ≈ 0.86.
- **Calibration co-optimised:** ECE is in the candidate-selection objective; the
  deployed safety config reaches ECE ≈ 0.015 (better than the raw ensemble).
- **Online ablation (n=24 paired, pseudo-continuous replay):** false-action cost
  169 → 4.2 (TARC); confident completion (τ=0.5) 0.000 → 0.875; both significant
  (bootstrap CI + exact McNemar, p<0.001). Reports: `reports/final/`.
- **Live controller:** `LiveLandmarkGestureController` in
  `research_pipeline/serve/live_backend.py`. Click = index-middle squeeze; zoom =
  thumb-index pinch open/close state machine (absolute gap, hysteresis); swipe =
  horizontal index-tip motion. Thresholds are class constants, calibrated from real
  webcam video. Live session quality on the user's webcam: ~28 FPS, p95 ≈ 18 ms,
  detection ≈ 0.88.
- **Reproducible live protocol:** `research_pipeline/cli/aggregate_live_sessions.py`
  aggregates logged sessions into the same action-level metrics as the replay.
- **Calibration tool:** `scripts/diagnose_live_video.py <video.mp4>` replays the
  live pipeline on a recorded clip and prints the geometry signals + fired events
  (deterministic, frame-based timestamps). Use it to retune thresholds for a new camera.

Known, accepted limitations: tasks should run in **TARC** mode (it rejects
non-expected gestures, so point/click/zoom geometry overlap does not cause false
actions mid-task); the online stream is pseudo-continuous, not a fully annotated
continuous dataset; the live demo is illustration, the replay ablation is the proof.

## Run it (the two machines)

- **Windows (training box, CUDA):** `RUN.bat` / `STOP.bat`, or
  `scripts/start_ar_demo.ps1`. Training happens here.
- **macOS (M1, this handoff's target):** see `docs/macos_m1_setup.md`. Install
  `requirements/macos-arm64.txt` into `.venv-gesture-ar`, then
  `scripts/start_ar_demo.sh` / `stop_ar_demo.sh`. M1 is for **inference + export**
  (ONNX/CoreML), not training (the loop uses CUDA-or-CPU, not MPS).

## What is NOT in git (must come along separately)

`.gitignore` excludes `artifacts/`, `data/raw/`, `data/processed/`, `node_modules/`,
the venv, and `*.onnx` / `*.mlpackage`. To run on the Mac you need, beyond `git clone`:

- **Trained models** `artifacts/models/*.pkl` — small (~6 MB total). They are now
  force-included in git (see the `.gitignore` exception), so a clone has them. If
  they are missing, copy the four `ipn_c1t_tcn_*{,_mv}.pkl` files manually.
- **MediaPipe model** `models/mediapipe/hand_landmarker.task` — tracked in git.
- **IPN dataset** — NOT needed for inference/export; only for retraining (stays on
  the CUDA box). Manifests under `data/interim/manifests/` are tracked.
- **Chat memory** — does not transfer; this file replaces it.

## Mobile / rear phone-camera plan

Keep two things separate — conflating them leads to the wrong conclusion "no retraining needed":

- **Running the existing desktop demo on the Mac = inference only, no retraining.** The shipped
  IPN-trained models run as-is for the webcam demo and for ONNX/CoreML export.
- **The rear phone-camera *recognizer* will almost certainly need retraining.** IPN Hand is a
  frontal fixed webcam; the rear phone camera is a different capture geometry (hand at arm's
  length, smaller and variable scale, a moving handheld background, different viewpoint and
  lighting). Expect a real domain gap, so plan to retrain rather than ship the IPN model. Do NOT
  try to train on a handful of locally recorded clips — too little data and too tedious. Use a
  suitable public dataset and apply the same methodology as on the PC.

Because the recognizer is landmark-based (MediaPipe 21), the whole methodology ports unchanged —
only the training dataset and the capture front-end change:
`MediaPipe-21 landmarks -> preprocess_dual_view (incl. multi-view JCD + slow/fast motion) -> TCN
ensemble -> calibrated fusion (ECE in the objective) -> validation/TARC controller`.
Background: `docs/phone_ar_transfer.md`, `docs/capture_protocol.md`, `docs/phone_rear_gesture_resolution.md`.

Staged plan:

1. **Capture.** On-device MediaPipe-21 capture (Android/iOS) or recorded rear-camera clips that
   produce the same landmark tensor (the `research_pipeline/data` schema).
2. **Measure the domain shift.** Run the IPN-trained models on a small rear-camera validation set
   with the existing offline + online evaluators. This number decides everything and is itself a
   result; a large gap is expected.
3. **Pick a suitable dataset and retrain — on the CUDA box or a cloud GPU, NOT the M1** (the M1
   training loop runs on CPU only, which is too slow). Dataset selection criteria: *dynamic* hand
   gestures (not only static poses); a viewpoint/scale close to arm's-length rear capture; RGB
   video so MediaPipe-21 landmarks are extractable; enough samples for the 7-class action
   vocabulary; a permissive license. Candidates to evaluate before committing (verify viewpoint and
   license each): Jester / 20BN (large, dynamic), EgoGesture (egocentric dynamic), HaGRID (large but
   mostly static — useful for the pose-like classes), plus a small targeted rear-camera set for
   validation and optional fine-tuning. Reuse the exact training methodology and configs
   (`configs/train/ipn_c1t_tcn_*_mv.yaml`): TCN ensemble, multi-view features, class-balanced/focal
   training, then C6 calibrated fusion with ECE in the selection objective. Map the new dataset's
   labels to the 7-class action vocabulary (`research_pipeline/labels.py`).
4. **Recalibrate the live controller** thresholds for the phone camera with
   `scripts/diagnose_live_video.py` on rear-camera clips (the geometry constants in
   `LiveLandmarkGestureController` are camera-specific), and re-run the C6 ECE calibration selection.
5. **Export + port.** `export_onnx` (Android), `export_coreml` (iOS), `export_mobile_bundle`; port
   the lightweight validation/TARC/stabilizer logic.

The contribution and the whole methodology (validation pipeline, calibration, the metrics) carry
over unchanged; only the recognizer's *training data* changes. Measuring the domain shift and
retraining on a rear-camera-appropriate public dataset is the expected path, not an optional
afterthought. The M1 is for capture-side development, threshold calibration, export, and on-device
testing; the heavy training stays on the GPU machine.

## Mobile implementation progress (M1 session)

The landmark pipeline, dataset ingest, local rear capture, and the iPhone export/contract layer
are now built and unit-tested on the M1 (118 unit tests green; `.venv-gesture-ar` from
`requirements/macos-arm64.txt`, including coremltools). What remains is the dataset-scale retraining
and the Swift on-device implementation. Training itself runs fine on the M1 (the loop auto-selects
CUDA -> MPS -> CPU; the compact ~650K-param TCN is ~10x faster on MPS than CPU, ~25 min for a full
run); the reason to keep full Jester on the GPU box is the *data* side — the ~22 GB download plus
extracting MediaPipe landmarks from ~148K clips. A reduced subset is feasible end-to-end on the M1.
The contribution and methodology are unchanged; only the recognizer's training data and the
capture/runtime front-end change. Status by stage:

**Dataset ingest (stage 3 groundwork).** Label adapters map each source to the 7-class vocabulary
and build canonical manifests, reusing the existing feature/TCN/multiview pipeline:

- Jester (primary, dynamic): `labels.remap_jester_label` + `cli/build_jester_manifest`. Covers
  5/7 classes (no_gesture, swipe_left/right, zoom_in/out). point_2f and click_2f are NOT in Jester.
- HaGRID (static poses): `labels.remap_hagrid_label` + `cli/build_hagrid_manifest`. Supplies
  point_2f (two_up/peace; HaGRID's one-finger "point" is deliberately unmapped). Static images are
  turned into a replicated-pose clip by a new static path in `cli/extract_landmarks`
  (`_is_static_image` / `_replicate_frame_to_clip`), because a single-image cv2.VideoCapture only
  yields one frame.
- Merge + balance: `cli/merge_datasets` (`data/merge`) caps per-class / per-source, reports
  coverage, warns on missing classes, preserves domain tags.
- Training config: `configs/train/jester_c1t_tcn_mv.yaml` (same TCN + multiview + balanced/focal
  methodology; GPU only). click_2f has no public source and must come from the local set.

**Local rear capture (stages 1 + 4).** `cli/ingest_local_videos` + `cli/extract_landmarks` process
local iPhone rear clips (back-of-hand, 0.5x). A 50-clip set was captured and extracted; clip
durations/fps are baked into the manifest so the window spans the whole clip. `cli/filter_by_coverage`
(`data/coverage`) drops low-detection clips: click_2f extracts cleanly (~98% frame detection),
while fast handheld swipes leave the frame / motion-blur and are better sourced from Jester. The
IPN webcam models do not transfer to the rear back-of-hand domain, so retraining is required (as
already planned) — the local set's essential role is click_2f plus orientation fine-tuning.

**iPhone export + on-device contracts (stage 5).** Core ML export is now a real conversion (was a
contract stub):

- `cli/export_coreml` -> `models/coreml_export`: converts a TCN artifact to a Core ML .mlpackage
  (mlprogram / FP16 / iOS15) and verifies torch<->Core ML numerical parity. The deployed mv models
  take a **326-dim** feature window (not 74); the converter reads the real input_dim. The traced
  graph is `jit.freeze`d before conversion, otherwise identical BatchNorm `num_batches_tracked`
  constants get deduplicated and trip a coremltools lowering assertion for some trained weights
  (notably MPS-trained models) — important for the GPU-model -> Mac -> Core ML export path.
- `cli/export_preprocessing_contract` -> `models/preprocessing_contract`: emits the exact 326-dim
  feature layout (pose 63 + motion 11 + JCD 210 + slow 21 + fast 21) and landmarks->features golden
  vectors for Swift preprocessing parity.
- `cli/export_validation_contract` -> `interaction/contract`: emits the acceptance-policy
  (ContextAwarePolicy: confidence/stability/cooldown/release) spec and golden decision traces.
- `cli/export_mobile_bundle` now uses the accurate 326-dim preprocessing contract.

Generated mobile artifacts live under `artifacts/` (gitignored): `export/GestureClassifier.mlpackage`,
`mobile/preprocessing/{feature_contract,golden_samples}.json`,
`mobile/validation/{validation_contract,golden_traces}.json`, `mobile/gesture_mobile_bundle/`.

**Rear model (trained on an RTX laptop).** `artifacts/models/rear_c1t_tcn_mv.pkl` — TCN + mv on
the merged Jester (~4k/dynamic class) + HaGRID (point_2f) + local (click_2f) set. Held-out val
accuracy ~0.82; point_2f, zoom_in/out, swipe_left/right and no_gesture all work (a real contrast
with the IPN zero-shot collapse to no_gesture). **click_2f is under-supported — only 7 samples —
and confuses with point_2f** (e.g. a real point_2f clip is read as click_2f). That is the main
quality gap: record more click_2f / point_2f rear clips, re-merge and retrain to fix. Exported to
`artifacts/export/GestureClassifier.mlpackage` (Core ML / FP16, torch<->Core ML parity verified);
the iOS golden resources were regenerated from this model.

**iOS app — now built and verified.** The Xcode app in `ios_demo/GestureAR/` is assembled and
runs: MediaPipe Hands wired (via the `SwiftTasksVision` SPM, no CocoaPods), the rear-trained
`.mlpackage` embedded, three demo tasks (Object Control / AR Scrolling / Sorting Objects), a live
MediaPipe hand-skeleton overlay, and a minimal non-fullscreen UI with a task selector. The 3D layer
uses **SceneKit** (a transparent `SCNView`), not a non-AR RealityKit `ARView`, because the latter
renders an opaque background that hides the camera. Verified in the Simulator (selector + all three
3D scenes render; camera shows through on device). The verified Swift core (preprocessing 326-dim /
policy / AR-interaction-state) remains golden/unit-tested via `swift test`.

**Full mobile architecture, datasets, training and app layout:** `docs/mobile_app_handoff.md`.

**Next on the app:** ARKit world-anchored scene (objects floating in space, set back from the
camera) via `ARWorldTrackingConfiguration`; real-hand occlusion over the 3D objects; and recognition
quality (more `click_2f`/`point_2f` rear clips → re-merge → retrain).

## Bootstrapping a fresh Claude session on the Mac

Point it at this file plus `docs/final_research_claim.md` and
`docs/evaluation_protocol.md`. Ask it to read them before changing code. Tell it:
the contribution is the validation pipeline (not SOTA); chat replies in Russian, but
all in-repo content stays English; run the live demo via `scripts/start_ar_demo.sh`;
verify changes with `python -m pytest tests/unit`.
