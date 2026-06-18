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

## Bootstrapping a fresh Claude session on the Mac

Point it at this file plus `docs/final_research_claim.md` and
`docs/evaluation_protocol.md`. Ask it to read them before changing code. Tell it:
the contribution is the validation pipeline (not SOTA); chat replies in Russian, but
all in-repo content stays English; run the live demo via `scripts/start_ar_demo.sh`;
verify changes with `python -m pytest tests/unit`.
