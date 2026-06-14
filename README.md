# Hand Gestures Research Pipeline

Windows-first research scaffold for **Context-Aware Temporal Landmark Gesture Recognition for AR Interaction**.

The project follows `deep-research-report.md`:

- public-data-first IPN Hand workflow;
- canonical `manifest JSONL + NPZ shards` data contract;
- MediaPipe-style `[T,21,3]` hand landmark tensors;
- C0/C1/C1-T/C2/C2+Local experiment slots;
- recognition and interaction replay benchmarks;
- Windows training mainline with separated ONNX/Core ML export stages.

## Quick Smoke

```powershell
.\.venv311\Scripts\Activate.ps1
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

## Current Prepared Artifacts

- Python 3.11 environment: `.venv311`
- IPN Hand annotations: `data/raw/ipn_hand/annotations`
- IPN Hand videos: `data/raw/ipn_hand/videos/videos` plus original `.tgz` archives
- Thesis subset manifest: `data/interim/manifests/ipn_subset.jsonl`
- Initial extracted landmark subset:
  - `data/interim/manifests/ipn_train_initial_landmarks.jsonl`
  - `data/interim/manifests/ipn_test_initial_landmarks.jsonl`
- Full extracted landmark benchmark:
  - `data/interim/manifests/ipn_train_full_landmarks.jsonl`
  - `data/interim/manifests/ipn_test_full_landmarks.jsonl`
- Trained initial models:
  - `artifacts/models/ipn_c1_rf_initial.pkl`
  - `artifacts/models/ipn_c1t_initial.pkl`
  - `artifacts/models/ipn_c1t_tcn_initial.pkl`
- Trained full models:
  - `artifacts/models/ipn_c1_rf_full.pkl`
  - `artifacts/models/ipn_c1t_tcn_full.pkl`
- ONNX export:
  - `artifacts/export/ipn_c1t_tcn_full.onnx`
  - `artifacts/export/ipn_c1t_tcn_full.onnx.data`
- Validated TCN control branch:
  - `artifacts/models/ipn_c1t_tcn_full_validated.pkl`
  - `artifacts/export/ipn_c1t_tcn_full_validated.onnx`
  - `artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json`
- C6 strengthened recognition branch:
  - `artifacts/models/ipn_c1t_tcn_augmented.pkl`
  - `artifacts/reports/ipn_c1t_tcn_augmented_recognition.json`
  - `artifacts/reports/c6_augmented_robustness.json`
  - `artifacts/reports/c6_ensemble_calibrated_recognition.json`
  - `docs/c6_recognition_upgrade.md`
- Phone AR portability/domain reports:
  - `data/interim/manifests/local_phone_plan.jsonl`
  - `artifacts/reports/domain_readiness.json`
  - `artifacts/reports/recognition_risk_analysis.json`
  - `artifacts/mobile/gesture_mobile_bundle`
- C4 AR interaction-risk research:
  - `artifacts/reports/c4_action_safe_research.json`
  - `artifacts/reports/c4_task_benchmark.json`
  - `artifacts/reports/c4_task_tables/*.csv`
  - `artifacts/figures/c4_task_*.png`
- AR interaction UI:
  - `demo/ar_interaction_app`

See `docs/setup_training_summary.md` for exact commands and current metrics.

For the shortest launch path, use [`START_HERE.md`](START_HERE.md).

## AR Interaction UI

Backend:

```powershell
.\.venv311\Scripts\Activate.ps1
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

- `Demo` shows the AR task surface.
- `Results` shows experiment metrics.
- `Dataset File` streams saved IPN test landmarks through the backend.
- `Camera Stream` opens the real local webcam through the backend and uses the live frame as the AR scene background.
- `Direct Control` maps the current recognized gesture directly to the AR action, including fingertip cursor placement from MediaPipe landmarks.
- `C2 Gate` applies confidence/stability filtering before emitting AR actions.
- `TARC Controller` applies the task-aware risk-calibrated AR policy, using the selected task step as action context.
- `Robust C6` is the default recognizer: validated TCN + augmented TCN + calibrated geometry fusion.
- `Guide` shows visual gesture cards and execution cues for the live demo gestures.
- `Target FPS`, `Preview px`, and `JPEG` tune camera smoothness and preview quality.
- `Start Camera` / `Start Dataset` connects the selected input source.
- `Run Test` runs a UI-only gesture sequence for quick visual checks.
- Live session traces are saved in `artifacts/live_sessions` and summarized with `python -m research_pipeline.cli.summarize_live_session`.
- Task-level live reports are generated with `python -m research_pipeline.cli.report_live_tasks` and use `configs/interaction/ar_task_scenarios.yaml` for ground-truth action windows.
- Autonomous task-level AR research benchmark is generated with `python -m research_pipeline.cli.benchmark_c4_tasks --config configs/eval/c4_task_benchmark.yaml` and `python -m research_pipeline.cli.generate_c4_task_assets`.
- Task-level failure analysis is generated with `python -m research_pipeline.cli.analyze_c4_task_failures`.
- A thesis-ready experiment chapter draft is generated with `python -m research_pipeline.cli.build_experiment_chapter`.
- The strengthened recognition experiment is reproduced with `python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_augmented.yaml` and `python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition.yaml`.
