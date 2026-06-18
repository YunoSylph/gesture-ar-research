# Gesture-Based Mid-Air Interaction for AR

Research and demo code for a master's project on **continuous gesture-to-action validation for augmented reality interaction**.

The project does not claim to be a new SOTA hand gesture recognizer. Its main claim is narrower and more defensible: a raw gesture classifier should not be mapped directly to AR commands. Instead, classifier output is treated as an action proposal and passed through confidence, stability, cooldown, gesture-contract, and task-aware risk validation before becoming an AR action.

## What This Repository Contains

- IPN Hand based landmark-recognition pipeline.
- MediaPipe-style 21 hand landmark preprocessing.
- Temporal TCN / C6 recognition models and calibration code.
- OO-dMVMT-inspired multi-view feature block for joint distances and motion.
- Pseudo-continuous online replay evaluator.
- GestureValidationLayer and TARC task-aware action policy.
- FastAPI live webcam backend.
- React + Three.js AR-style interface with live task scenarios.
- Reports, tables, plots, and tests for the current research state.

Large local assets are intentionally **not** committed: the full IPN Hand dataset, extracted tensors, model artifacts, live session logs, and generated build products stay local.

## Scientific Framing

Reference direction: Cunico et al. 2023, **OO-dMVMT: A Deep Multi-View Multi-Task Classification Framework for Real-Time 3D Hand Gesture Classification and Segmentation**.

This project uses that paper as motivation for moving beyond pre-segmented clip classification toward online recognition, segmentation-like metrics, latency, false positives, and action validation. The project does not reproduce OO-dMVMT's exact architecture or compare against its reported numbers.

The defensible thesis claim is:

> A task-aware continuous gesture-to-action validation pipeline reduces false AR actions and action switching compared with direct classifier-to-action mapping on identical replay sequences.

## Current Results

### Offline Recognition

Real IPN Hand test split, pre-segmented clip-level evaluation:

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced Acc |
| --- | ---: | ---: | ---: | ---: |
| C1-T TCN, validated dual-view | 0.9071 | 0.8502 | 0.9109 | 0.8966 |
| C1-T TCN, augmented dual-view | 0.9090 | 0.8565 | - | - |
| C1-T TCN, validated + multi-view | 0.9197 | 0.8623 | 0.9203 | 0.8724 |

The multi-view block improves several weaker motion/distance-dependent classes, especially `swipe_left`, `zoom_out`, and `click_2f`, while `swipe_right` and `zoom_in` regress slightly. The report keeps those regressions visible.

Calibration on the deployed multi-view C6 fusion:

| Method | Accuracy | Macro F1 | ECE | Brier |
| --- | ---: | ---: | ---: | ---: |
| Raw ensemble | 0.9255 | 0.8731 | 0.0207 | 0.1186 |
| C5 macro objective | 0.9206 | 0.8669 | 0.0261 | 0.1229 |
| C5 safety objective | 0.9274 | 0.8778 | 0.0146 | 0.1157 |

See `reports/final/recognition_summary.md`.

### Online Gesture-To-Action Replay

The online benchmark uses pseudo-continuous replay: real extracted IPN landmark clips are concatenated with idle/no-gesture gaps and evaluated as a stream. This is real-clip replay, not the original uncut IPN timeline.

Headline action-level results on paired replay sequences:

| Method | Mean false-action cost | Confident completion | Graded completion |
| --- | ---: | ---: | ---: |
| Direct C6 baseline | 169.22 | 0.000 | 0.058 |
| C6 + validation | 37.51 | 0.000 | 0.198 |
| C6 + validation + cooldown | 6.98 | 0.667 | 0.578 |
| C6 + validation + TARC | 4.23 | 0.875 | 0.669 |

Compared with direct C6, the full validation/TARC pipeline reduces mean false-action cost by about `-164.99` per paired sequence-task, with `p < 0.001` in the current paired comparison. The same run raises graded task completion from `0.058` to `0.669`.

See:

- `reports/final/online_summary.md`
- `reports/online_gesture/method_comparison.md`
- `reports/online_gesture/summary.md`

## Live AR Demo

Windows one-file launch:

```powershell
.\START_PROJECT.bat
```

The launcher creates or updates `.venv-gesture-ar`, installs backend/frontend dependencies, starts:

- backend: `http://127.0.0.1:8000`
- frontend: `http://127.0.0.1:5173`

Then it opens the UI. Use `STOP.bat` to stop the local services.

In the UI:

- `Live` shows the camera-backed AR interaction surface.
- `Guide` shows gesture hints and execution cues.
- `Results` shows compact experiment summaries.
- `Start Task` starts the live webcam session.
- `TARC Controller` is the intended validated mode.
- `Direct Control` is a baseline and is expected to be less stable.

The live interface is a demonstration and diagnostic tool. The primary scientific proof is the reproducible replay benchmark, not a single webcam session.

## Reproducing The Research Runs

Install Python dependencies:

```powershell
python -m venv .venv-gesture-ar
.\.venv-gesture-ar\Scripts\Activate.ps1
pip install -e ".[train,vision,serve,dev]"
```

Run tests:

```powershell
python -m pytest -q
```

Run the online comparison:

```powershell
python -m research_pipeline.cli.benchmark_online_gesture --config configs/eval/online_gesture.yaml --output-dir reports/online_gesture
```

Run the calibrated C6 recognition comparison:

```powershell
python -m research_pipeline.cli.run_c5_calibrated_recognition --config configs/eval/c6_ensemble_calibrated_recognition_mv.yaml
```

Train the multi-view TCN variant:

```powershell
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_full_validated_mv.yaml
```

## Data And Artifact Policy

The GitHub repository stores source code, configs, tests, documentation, and compact reports. It does not store:

- `IPN_Hand/`
- `data/raw/`
- `data/processed/`
- `artifacts/`
- `.venv-gesture-ar/`
- `node_modules/`
- local live-session recordings or generated gesture-guide videos

To reproduce full numerical results from a clean clone, place the IPN Hand data locally, regenerate manifests/tensors, and train or restore the model artifacts into `artifacts/models/`.

## Known Limitations

- Offline metrics are clip-level; online metrics are pseudo-continuous and intentionally lower.
- The replay evaluator does not reconstruct the original full continuous IPN videos.
- The live webcam demo is sensitive to camera quality, lighting, hand pose, and browser/backend load.
- The project does not make clinical rehabilitation claims.
- Phone rear-camera AR is a plausible next application layer, but it requires a separate domain-shift and calibration stage.

## Repository Map

- `research_pipeline/` - Python research, evaluation, serving, and interaction code.
- `configs/` - train/eval/interaction configuration.
- `demo/ar_interaction_app/` - React + Three.js AR-style web UI.
- `docs/` - research framing, protocol, gesture contract, and implementation notes.
- `reports/` - current result summaries and comparison tables.
- `tests/` - unit tests for evaluation, validation, calibration, and live contracts.
- `START_HERE.md` - quick usage guide.
