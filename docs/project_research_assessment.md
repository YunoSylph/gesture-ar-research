# Project Research Assessment

## Current State

The project is no longer just a gesture classifier demo. It now contains a full Windows-first research and prototype pipeline:

- public IPN Hand dataset ingestion, landmark extraction and manifest contract;
- trained C0, C1, C1-T and validated TCN baselines;
- C3 hybrid recognition with geometry-aware safety prior;
- C4 action-safe interaction layer with calibrated thresholds, temporal stability and action risk costs;
- C4 task-level AR benchmark over 13 interface scenarios;
- web AR-style interface with live webcam, task selection, Three.js overlay, result tables and methodology charts;
- live `C4 Task` interaction mode that applies task-aware action safety during the demonstration;
- mobile export artifacts and an iOS contract skeleton.

## Research Significance

The strongest current research framing is not "we improved gesture classification accuracy". The recognition-level gain from C3 is small: C1-T macro F1 is about `0.850`, while C3 is about `0.852`. This is useful engineering polish but weak as the central thesis claim.

The stronger contribution is interaction safety for AR:

```text
Can a gesture-driven AR system reduce unintended high-cost actions while preserving enough task completion quality?
```

This framing is more defensible because AR failures are not symmetric. A false `select_confirm`, `zoom`, or navigation event can be more damaging than a missed passive frame. The project now explicitly models this through weighted action costs and task-level evaluation.

## Strong Components

- The data pipeline is reproducible and follows a clear manifest/tensor contract.
- The baseline ladder is academically readable: C0, C1, C1-T, C3 and C4.
- The C4 method has a concrete methodological idea: classifier outputs are treated as action proposals, not direct commands.
- The task-level benchmark is a meaningful upgrade because it evaluates scenario completion, unintended actions and false action cost.
- `C4 task-aware` is the most thesis-ready concept: it combines recognition, risk-aware control and task context.
- The web interface now demonstrates the research system and not only a standalone cube interaction.

## Weak Components

- The system still lacks real phone-rear validation data, so claims about rear-camera AR transfer must remain limited.
- C3 recognition improvement is too small to carry the thesis by itself.
- Task-level replay is synthetic: it uses public clips and scripted task windows, not full human-in-the-loop trials.
- The live webcam demo depends heavily on camera visibility, lighting and MediaPipe stability.
- There is no final on-device iOS/RealityKit latency validation yet.
- The task-aware controller currently uses known task-step context; this is realistic for guided AR workflows, but should be clearly described as scenario-aware interaction, not general open-world gesture control.

## Recommended Development Format

The project should be developed as a master's thesis around a multi-layer AR interaction architecture:

1. Recognition baseline: prove that the temporal TCN is a competent public-data recognizer.
2. Error/risk analysis: show why accuracy alone is insufficient for AR control.
3. C3 hybrid recognizer: present it as robustness-oriented support, not the main result.
4. C4 action-safe controller: present calibrated confidence, stability, cooldown and abstention as the core method.
5. C4 task-aware controller: present task context as a new combinative method that improves the safety/completion trade-off.
6. Live AR prototype: demonstrate the system with real webcam input and representative AR tasks.
7. Final validation extension: add phone-rear or live user sessions when local recordings become available.

## Best Next Improvements

- Add a small local validation protocol with repeated webcam/phone sessions for each AR task.
- Add per-task confusion and failure reports: which tasks fail because of missed recall, which fail because of false actions.
- Add a thesis-ready experiment chapter export: tables, figures, metrics and interpretation in one generated report.
- If an iPhone/Mac is available, validate ONNX/Core ML or RealityKit deployment latency.

## Implemented After Assessment

- Live backend now exposes `c4_task_aware` as an interaction mode.
- The web interface shows this mode as `C4 Task`.
- `artifacts/reports/c4_task_failure_analysis.md` summarizes weak scenarios and method trade-offs.

## Current Thesis-Ready Claim

The most solid current claim is:

```text
The project proposes a risk-aware, task-contextual gesture interaction pipeline for AR. While classifier-level gains are modest, the C4/C4 task-aware interaction layer substantially reduces weighted false action cost and unintended AR commands compared with direct gesture-to-action control.
```

This is a credible master's-level direction because it combines machine learning, interaction policy design, AR task modeling, reproducible evaluation and a working interface.
