# Hand Gestures Research Pipeline

Windows-first research scaffold for **Context-Aware Temporal Landmark Gesture Recognition for AR Interaction**.

The project follows `deep-research-report.md`:

- public-data-first IPN Hand workflow;
- canonical `manifest JSONL + NPZ shards` data contract;
- MediaPipe-style `[T,21,3]` hand landmark tensors;
- C0/C1/C1-T/C2/C2+Local experiment slots;
- recognition and interaction replay benchmarks;
- Windows training mainline with separated ONNX/Core ML export stages.

## One-Click Live Launch

On Windows, use the launcher in the repository root:

```powershell
.\START_PROJECT.bat
```

It creates/updates `.venv-gesture-ar`, installs backend camera dependencies, installs frontend dependencies, starts the FastAPI live backend on `http://127.0.0.1:8000`, starts the React/Three.js interface on `http://127.0.0.1:5173`, waits for both services, and opens the UI.

Close the `Gesture AR Backend` and `Gesture AR Frontend` console windows to stop the project.

## Quick Smoke

```powershell
python -m venv .venv-gesture-ar
.\.venv-gesture-ar\Scripts\Activate.ps1
pip install -r requirements\windows-train.txt -r requirements\dev.txt
python -m research_pipeline.cli.smoke_public
python -m research_pipeline.cli.smoke_demo
python -m research_pipeline.cli.smoke_export
python -m pytest -q
```

## Main CLI Contract

```powershell
python -m research_pipeline.cli.build_ipn_manifest --root <IPN_ROOT> --output data/interim/manifests/ipn_all.jsonl
python -m research_pipeline.cli.remap_ipn_subset --input data/interim/manifests/ipn_all.jsonl --output data/interim/manifests/ipn_subset.jsonl
python -m research_pipeline.cli.extract_landmarks --manifest data/interim/manifests/ipn_subset.jsonl --output-dir data/processed/public_landmarks
python -m research_pipeline.cli.train --config configs/train/c1t_tcn_public.yaml
python -m research_pipeline.cli.calibrate_context --config configs/interaction/c2.yaml
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/recognition.yaml
python -m research_pipeline.cli.benchmark_interaction --config configs/eval/interaction.yaml
python -m demo.webcam_app.main --config configs/demo/webcam.yaml
python -m research_pipeline.cli.export_onnx --config configs/export/onnx.yaml
python -m research_pipeline.cli.export_coreml --config configs/export/coreml.yaml
```

## Repository Contents

The GitHub repository intentionally keeps source materials rather than local build products:

- research code, configs, tests and CLI tools;
- React/Three.js live AR interface in `demo/ar_interaction_app`;
- MediaPipe hand landmarker task in `models/mediapipe`;
- public-dataset manifests in `data/interim/manifests`;
- reference gesture clips and notation in `data/reference_gestures` and `data/interaction_gesture_examples`;
- research and usage documentation in `docs`, `START_HERE.md`, and `PROJECT_OVERVIEW.md`.

The following paths are local/generated and ignored: `.venv-gesture-ar`, `node_modules`, `artifacts`, `data/raw`, `data/processed`, `demo/ar_interaction_app/dist`, and `docs/generated`.

Use `START_HERE.md` for the quickest live launch. Use `requirements/windows-train.txt` when reproducing full extraction/training/evaluation runs.

See `docs/setup_training_summary.md` for exact commands and current metrics.

For the shortest launch path, use [`START_HERE.md`](START_HERE.md).

## AR Interaction UI

Backend:

```powershell
.\.venv-gesture-ar\Scripts\Activate.ps1
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd demo/ar_interaction_app
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. If that port is already occupied, run `npm run dev -- --port 5174`.

In the UI:

- `Live` shows the camera-backed AR task surface.
- `Guide` shows visual gesture cards and execution cues for the live demo gestures.
- `Results` combines experiment charts and compact comparison tables.
- `Start Task` opens the real local webcam through the backend and uses the live frame as the AR scene background.
- `Direct Control` maps the current live landmark-controller gesture directly to the AR action, including fingertip cursor placement from MediaPipe landmarks.
- `TARC Controller` applies the task-aware risk-calibrated AR policy, using the selected task step as action context and passing the expected gesture into the live controller.
- Live gestures use a fixation cycle: `preparing -> locked -> cooldown`; the overlay lock bar shows when a gesture is actually accepted.
- `Robust C6` is the default recognizer: validated TCN + augmented TCN + calibrated geometry fusion.
- `Target FPS`, `Preview px`, and `JPEG` tune camera smoothness and preview quality.
- `Gesture Test Pad` in `Advanced Controls` runs UI-only gesture checks without relying on webcam recognition.
- Current live-model transfer analysis is in `docs/live_model_assessment.md`.
- Live session traces are saved in `artifacts/live_sessions` and summarized with `python -m research_pipeline.cli.summarize_live_session`.
- Task-level live reports are generated with `python -m research_pipeline.cli.report_live_tasks` and use `configs/interaction/ar_task_scenarios.yaml` for ground-truth action windows.
- Autonomous task-level AR research benchmark is generated with `python -m research_pipeline.cli.benchmark_c4_tasks --config configs/eval/c4_task_benchmark.yaml` and `python -m research_pipeline.cli.generate_c4_task_assets`.
- Task-level failure analysis is generated with `python -m research_pipeline.cli.analyze_c4_task_failures`.
- A thesis-ready experiment chapter draft is generated with `python -m research_pipeline.cli.build_experiment_chapter`.
- The strengthened recognition experiment is reproduced with `python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_augmented.yaml` and `python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition.yaml`.
