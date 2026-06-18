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

The recognizer is landmark-based (MediaPipe 21), so it is portable; the same feature
pipeline (`preprocess_dual_view`, including the multi-view block) and the
validation/TARC controller move to the device unchanged — only the capture front-end
changes. Background: `docs/phone_ar_transfer.md`, `docs/capture_protocol.md`,
`docs/phone_rear_gesture_resolution.md`.

Staged plan (do them in order; do not assume transfer without measuring):

1. **Capture.** Build an on-device MediaPipe-21 capture (Android/iOS) — or record
   rear-camera clips — that produces the same landmark tensor format.
2. **Measure the domain shift first.** Collect a small rear-camera validation set
   and run the existing offline + online evaluations on it. Quantify the gap
   honestly (IPN is a frontal fixed webcam; the rear camera is arm's-length, smaller
   variable hand scale, moving handheld background). This measurement is itself a result.
3. **Recalibrate.** Re-run C6 calibration selection (including ECE) and retune the
   live controller thresholds with `scripts/diagnose_live_video.py` on rear-camera clips.
4. **Export + port.** `export_onnx` (Android), `export_coreml` (iOS),
   `export_mobile_bundle`. Port the lightweight validation/TARC/stabilizer logic.
5. **Optional fine-tune** on a rear-camera set if the gap is large — on the CUDA box.

## Bootstrapping a fresh Claude session on the Mac

Point it at this file plus `docs/final_research_claim.md` and
`docs/evaluation_protocol.md`. Ask it to read them before changing code. Tell it:
the contribution is the validation pipeline (not SOTA); chat replies in Russian, but
all in-repo content stays English; run the live demo via `scripts/start_ar_demo.sh`;
verify changes with `python -m pytest tests/unit`.
