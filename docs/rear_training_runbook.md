# Rear gesture model — GPU training runbook (RTX laptop)

End-to-end recipe to train the rear-camera recognizer on a CUDA laptop (e.g. RTX
5070 Ti) from the tooling in this repo. Datasets are downloaded on the GPU laptop.
The trained `.pkl` is brought back to the Mac for the Core ML export.

## 0. Files to copy from the Mac

Copy the whole `gesture-ar-research/` folder. You can SKIP (to save space):

- `.venv-gesture-ar/` — rebuild on the laptop (step 1)
- `local_vids/` — raw `.MOV` clips; already extracted into tensors
- `artifacts/` — regenerated on the laptop

Keep especially the already-extracted **local rear set** (it is the only source of
`click_2f` and the orientation fine-tune samples), preserving the `data/` layout so
the manifest's relative tensor paths resolve:

- `data/processed/local_phone/*.npz`
- `data/interim/manifests/local_phone_capture_landmarks_filtered.jsonl`

Everything else needed is tracked code: `research_pipeline/`, `configs/`,
`requirements/`, `models/mediapipe/hand_landmarker.task`, `pyproject.toml`.

## 1. Environment (Python 3.11)

The RTX 5070 Ti is Blackwell (compute capability sm_120). Install a **CUDA 12.8**
PyTorch build FIRST — an older build raises `no kernel image is available` /
`sm_120 not supported`:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/windows-train.txt          # torch is already satisfied
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True RTX 5070 Ti`. If False / sm_120 error, the torch build predates
Blackwell — reinstall from cu128 (or nightly: `--index-url
https://download.pytorch.org/whl/nightly/cu128`). Run all commands below from the
repo root (no `pip install -e .` needed; `pythonpath=["."]`).

## 2. Datasets (download on the laptop)

- **Jester** (primary, dynamic — swipe/zoom/no_gesture): register and accept the
  research license at <https://www.qualcomm.com/developer/software/jester-dataset>.
  Mirrors if needed: Kaggle `20bn-jester`, HuggingFace. You need the frame folders
  (`<root>/<video_id>/<frame>.jpg`) plus `jester-v1-train.csv` and
  `jester-v1-validation.csv`.
- **HaGRIDv2** (static poses — `point_2f`): <https://github.com/hukenovs/hagrid>.
  Download only the needed classes + annotations (each class is a separate zip, so
  you avoid the full 1.5 TB):
  - classes: `two_up`, `two_up_inverted`, `peace`, `peace_inverted`, `no_gesture`
  - annotations (per-gesture JSON, by stage): the repo's
    `python download.py --dataset --annotations`, or the annotations archive
    `https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip`
- **Local rear set** — already copied in step 0 (provides `click_2f`).

## 3. Build manifests

```bash
python -m research_pipeline.cli.build_jester_manifest \
  --annotations-dir <JESTER_ANN_DIR> --frames-root <JESTER_FRAMES_ROOT> \
  --output data/interim/manifests/jester.jsonl

python -m research_pipeline.cli.build_hagrid_manifest \
  --annotations-dir <HAGRID_ANN_DIR>/train --images-root <HAGRID_IMAGES_ROOT> \
  --output data/interim/manifests/hagrid.jsonl
```

## 4. Extract landmarks (MediaPipe — CPU-bound, the slow part)

```bash
python -m research_pipeline.cli.extract_landmarks \
  --manifest data/interim/manifests/jester.jsonl --split train --backend mediapipe \
  --output-dir data/processed/jester \
  --output-manifest data/interim/manifests/jester_train_landmarks.jsonl --resume

python -m research_pipeline.cli.extract_landmarks \
  --manifest data/interim/manifests/hagrid.jsonl --backend mediapipe \
  --output-dir data/processed/hagrid \
  --output-manifest data/interim/manifests/hagrid_landmarks.jsonl --resume
```

Extraction runs on MediaPipe (CPU), not the GPU — it is the heaviest step
(~148K Jester clips = many hours). Start smaller with `--limit N`, and `--resume`
restarts where it stopped. HaGRID static images go through the replicated-pose
path automatically.

## 5. Merge + balance into one 7-class manifest

```bash
python -m research_pipeline.cli.merge_datasets \
  --inputs data/interim/manifests/jester_train_landmarks.jsonl \
           data/interim/manifests/hagrid_landmarks.jsonl \
           data/interim/manifests/local_phone_capture_landmarks_filtered.jsonl \
  --output data/interim/manifests/rear_merged_landmarks.jsonl \
  --max-per-class-per-source 4000 \
  --report artifacts/reports/rear_merge.json
```

Check the report: all 7 classes should be present, `missing_targets` empty
(`click_2f` from local, `point_2f` from hagrid+local, the rest from jester).

## 6. Train (uses CUDA automatically)

```bash
python -m research_pipeline.cli.train --config configs/train/rear_c1t_tcn_mv.yaml
```

Output: `artifacts/models/rear_c1t_tcn_mv.pkl`. The compact TCN trains in minutes
on the RTX. Adjust `epochs` / `max-per-class-per-source` to taste.

## 7. Back on the Mac — Core ML export

Copy `artifacts/models/rear_c1t_tcn_mv.pkl` to the Mac, then:

```bash
python -m research_pipeline.cli.export_coreml \
  --model-path artifacts/models/rear_c1t_tcn_mv.pkl \
  --sample-manifest data/interim/manifests/local_phone_capture_landmarks_filtered.jsonl
python -m research_pipeline.cli.export_preprocessing_contract \
  --manifest data/interim/manifests/local_phone_capture_landmarks_filtered.jsonl \
  --model-path artifacts/models/rear_c1t_tcn_mv.pkl
```

This yields `GestureClassifier.mlpackage` (+ parity) and the 326-dim feature
contract/golden vectors for the Swift on-device port.
