# Setup and Initial Training Summary

Date: 2026-05-13  
Machine: Windows, NVIDIA GeForce RTX 5070 Ti Laptop GPU

## Environment

Created a Python 3.11 environment at:

```powershell
.venv311
```

Installed:

- PyTorch `2.11.0+cu128`
- torchvision / torchaudio
- MediaPipe `0.10.35`
- OpenCV `4.13`
- scikit-learn `1.8`
- ONNX `1.21`
- ONNX Runtime GPU `1.26`
- onnxscript
- gdown, pandas, matplotlib, pytest, tqdm, rich

Verified:

- `torch.cuda.is_available() == True`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- ONNX Runtime can use CUDA when `torch` is imported before creating the ORT session.

## Dataset

Downloaded official IPN Hand assets:

- annotations: `data/raw/ipn_hand/annotations`
- videos: `data/raw/ipn_hand/videos/videos`
- archives: five `.tgz` files, total about 4.94 GB
- extracted videos: 200 `.avi` files

Built thesis subset manifest:

```powershell
python -m research_pipeline.cli.build_ipn_manifest `
  --annotations-dir data/raw/ipn_hand/annotations `
  --video-root data/raw/ipn_hand/videos/videos `
  --output data/interim/manifests/ipn_subset.jsonl
```

Result: `3438` target-class segments.

## Initial Public Benchmark Subset

Created deterministic stratified subset:

```powershell
python -m research_pipeline.cli.sample_manifest --input data/interim/manifests/ipn_subset.jsonl --output data/interim/manifests/ipn_train_initial.jsonl --split train --per-class 25 --seed 13
python -m research_pipeline.cli.sample_manifest --input data/interim/manifests/ipn_subset.jsonl --output data/interim/manifests/ipn_test_initial.jsonl --split test --per-class 10 --seed 13
```

Extracted MediaPipe landmarks:

```powershell
python -m research_pipeline.cli.extract_landmarks --manifest data/interim/manifests/ipn_train_initial.jsonl --output-dir data/processed/public_landmarks_initial/train --output-manifest data/interim/manifests/ipn_train_initial_landmarks.jsonl --backend mediapipe --target-length 32 --model-asset models/mediapipe/hand_landmarker.task
python -m research_pipeline.cli.extract_landmarks --manifest data/interim/manifests/ipn_test_initial.jsonl --output-dir data/processed/public_landmarks_initial/test --output-manifest data/interim/manifests/ipn_test_initial_landmarks.jsonl --backend mediapipe --target-length 32 --model-asset models/mediapipe/hand_landmarker.task
```

Result:

- train: 175 clips
- test: 70 clips
- target length: 32 frames per clip

## Initial Metrics

Recognition benchmark on `ipn_test_initial_landmarks.jsonl`:

| Variant | Accuracy | Macro F1 | Weighted F1 | p95 latency |
|---|---:|---:|---:|---:|
| C0 rule | 0.2000 | 0.1358 | 0.1358 | 0.272 ms |
| C1 random forest | 0.7286 | 0.7214 | 0.7214 | 31.558 ms |
| C1-T temporal prototype | 0.6857 | 0.6859 | 0.6859 | 0.293 ms |
| C1-T compact TCN | 0.7857 | 0.7705 | 0.7705 | 5.087 ms |

Reports:

- `artifacts/reports/ipn_c0_initial_recognition.json`
- `artifacts/reports/ipn_c1_rf_initial_recognition.json`
- `artifacts/reports/ipn_c1t_initial_recognition.json`
- `artifacts/reports/ipn_c1t_tcn_initial_recognition.json`

## Full Public Benchmark

Full extraction completed for all target-class IPN segments:

- train: 2405 clips
- test: 1033 clips
- total: 3438 clips
- tensor format: `landmarks [32,21,3] + mask + confidence`

Full training commands:

```powershell
python -m research_pipeline.cli.train --config configs/train/ipn_c1_rf_full.yaml
python -m research_pipeline.cli.train --config configs/train/ipn_c1t_tcn_full.yaml
```

Full recognition metrics on official test split:

| Variant | Accuracy | Macro F1 | Weighted F1 | p95 latency |
|---|---:|---:|---:|---:|
| C0 rule | 0.1820 | 0.0874 | 0.2158 | 0.237 ms |
| C1 random forest | 0.8955 | 0.7987 | 0.8930 | 56.409 ms |
| C1-T compact TCN | 0.9061 | 0.8504 | 0.9093 | 5.228 ms |
| C1-T compact TCN validated | 0.9071 | 0.8502 | 0.9109 | 4.638 ms |

Full artifacts:

- `artifacts/models/ipn_c1_rf_full.pkl`
- `artifacts/models/ipn_c1t_tcn_full.pkl`
- `artifacts/export/ipn_c1t_tcn_full.onnx`
- `artifacts/models/ipn_c1t_tcn_full_validated.pkl`
- `artifacts/export/ipn_c1t_tcn_full_validated.onnx`
- `artifacts/reports/ipn_c0_full_recognition.json`
- `artifacts/reports/ipn_c1_rf_full_recognition.json`
- `artifacts/reports/ipn_c1t_tcn_full_recognition.json`
- `artifacts/reports/ipn_c1t_tcn_full_validated_recognition.json`
- `artifacts/reports/recognition_risk_analysis.json`
- `artifacts/reports/domain_readiness.json`
- `artifacts/mobile/gesture_mobile_bundle`

## How To Use

Activate environment:

```powershell
.\.venv311\Scripts\Activate.ps1
```

Run tests:

```powershell
python -m pytest -q
```

Run full TCN benchmark:

```powershell
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_full.yaml
```

Run validated TCN benchmark:

```powershell
python -m research_pipeline.cli.benchmark_recognition --config configs/eval/ipn_c1t_tcn_full_validated.yaml
```

Export full TCN to ONNX:

```powershell
python -m research_pipeline.cli.export_onnx --config configs/export/ipn_c1t_tcn_full_onnx.yaml
```

Build phone AR readiness reports:

```powershell
python -m research_pipeline.cli.analyze_recognition_risk
python -m research_pipeline.cli.report_domain_readiness --manifests data/interim/manifests/ipn_train_full_landmarks.jsonl data/interim/manifests/ipn_test_full_landmarks.jsonl data/interim/manifests/local_phone_plan.jsonl
python -m research_pipeline.cli.export_mobile_bundle
python -m research_pipeline.cli.report_project_status
```

Run interaction replay smoke:

```powershell
python -m research_pipeline.cli.smoke_demo
```

Run live backend and AR UI:

```powershell
python -m research_pipeline.cli.serve_live --host 127.0.0.1 --port 8000
cd demo/ar_interaction_app
npm run dev
```

## Next Steps

1. Record or import the 35 planned `phone_rear_ar` clips into `data/raw/local_phone/videos`.
2. Extract local landmarks and run zero-shot public-model evaluation on the phone domain.
3. Calibrate C2 thresholds on local validation clips before any fine-tuning.
4. Run Direct vs C2 task sessions with active gestures, not idle camera-only traces.
5. Move Core ML conversion to macOS/Linux and integrate with the iOS RealityKit demo.
