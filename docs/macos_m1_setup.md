# macOS (Apple Silicon / M1) Setup

This is the deployment path for the MacBook Air M1, from which the rear phone-camera
variant will be developed. The recognizer is landmark-based (MediaPipe 21 keypoints),
so it is portable; this guide gets the desktop/webcam pipeline running natively on
arm64 and prepares the on-device export path.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3).
- Python 3.11+ arm64. Recommended: `brew install python@3.11`.
- Node.js 18+ for the frontend: `brew install node`.
- Git, and the project checked out locally.

Check the interpreter is arm64 (not Rosetta x86): `python3.11 -c "import platform; print(platform.machine())"` should print `arm64`.

## Python environment

```bash
cd <project root>
python3.11 -m venv .venv-gesture-ar
source .venv-gesture-ar/bin/activate
pip install --upgrade pip
pip install -r requirements/macos-arm64.txt
```

Notes:
- `torch` resolves to the native arm64 wheel with the Metal (MPS) backend. Do not add a
  CUDA `--index-url` on macOS.
- `onnxruntime` (CPU/CoreML) replaces the Windows `onnxruntime-gpu`.
- `coremltools` is included for the CoreML export used by the Apple on-device path.

## Run the demo

```bash
chmod +x scripts/start_ar_demo.sh scripts/stop_ar_demo.sh   # first time only
./scripts/start_ar_demo.sh        # starts backend + frontend, opens the browser
./scripts/stop_ar_demo.sh         # stops both
```

The backend serves on `http://127.0.0.1:8000`, the frontend on `http://127.0.0.1:5173`.

**Camera permission:** the first webcam run will be blocked until you grant camera access
to the terminal (or your IDE) under System Settings -> Privacy & Security -> Camera, then
restart the backend.

## What runs vs what does not on M1

- **Inference (live demo, replay benchmarks, ONNX/CoreML export): yes.** These are the M1
  use cases. The trained model artifacts ship in `artifacts/models/` (including the
  multi-view C6 members), and the MediaPipe landmarker is in `models/mediapipe/`.
- **Re-training the TCN: works but on CPU only** (the training loop selects `cuda` when
  available and otherwise CPU; it does not auto-use MPS). It is slow on M1 and is not needed
  for the phone path - reuse the shipped artifacts. Do heavy training on the CUDA machine.

## On-device / phone export path

For the rear phone-camera variant, export the recognizer for an on-device runtime:

```bash
python -m research_pipeline.cli.export_onnx --help          # ONNX (Android / cross-platform)
python -m research_pipeline.cli.export_coreml --help        # CoreML (iOS / Apple)
python -m research_pipeline.cli.export_mobile_bundle --help  # packaged mobile bundle
```

The phone capture stays on MediaPipe 21 landmarks, so the same feature pipeline
(`preprocess_dual_view`, including the multi-view block) and the validation/TARC controller
port directly; only the capture front-end changes. The expected first milestone is measuring
the rear-camera domain shift (see `docs/phone_ar_transfer.md` and `docs/capture_protocol.md`)
before any retraining.

## Verify the install

```bash
source .venv-gesture-ar/bin/activate
python -m pytest tests/unit -q
```
