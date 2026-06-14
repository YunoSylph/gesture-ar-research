# Final Research Claim

Last updated: 2026-06-15

The project should be defended as a reproducible continuous gesture-to-action validation pipeline for AR-style mid-air interaction. It is not a new state-of-the-art gesture recognizer and should not be presented as one.

## Core Claim

The main contribution is a validation layer that treats recognizer output as an action proposal rather than an immediate command. A proposal is accepted only after confidence, score margin, temporal stability, cooldown, release, task expectation, and action-risk checks are satisfied.

In the project terminology:

```text
landmark sequence -> recognizer -> GestureValidationLayer -> TARC -> task replay/live AR action
```

This claim follows the online-recognition direction of OO-dMVMT: live AR interaction must be evaluated with continuous-stream metrics, segmentation-like timing, decision latency, false positives, and action reliability rather than only clip-level classification accuracy.

## Defensible Thesis Claims

- Offline IPN Hand recognition quality is a useful lower-level benchmark, but it is insufficient as proof of reliable live AR interaction.
- Direct classifier-to-action mapping is unsafe for AR tasks because noisy labels, release frames, and transitions can trigger unintended actions.
- A confidence/stability/cooldown validation layer reduces the number of accepted false actions relative to direct classifier-to-action mapping under the same replay sequence.
- TARC adds task-aware acceptance: only ready or locked proposals are eligible for task execution, and unexpected high-risk actions can be rejected.
- False-action cost, accepted/rejected action counts, action switching, false positives per minute, and decision latency are the project-level metrics that best match AR interaction reliability.
- The live webcam demo is an engineering demonstration and failure-analysis surface. The primary scientific proof should use reproducible public-data replay and ablation tables.

## Claims Not To Make

- Do not claim state-of-the-art gesture recognition.
- Do not claim direct numerical comparison with OO-dMVMT unless its full protocol is reproduced.
- Do not claim clinical rehabilitation validation.
- Do not claim robust deployment on iOS, RealityKit, Quest, HoloLens, phone rear-camera AR, or arbitrary real-world cameras.
- Do not claim that pre-segmented IPN Hand clip metrics prove continuous live-stream segmentation quality.
- Do not claim that the current CSS gesture guide is a ground-truth gesture source. The source of truth is `docs/gesture_contract.md`.

## Rehabilitation Positioning

Hand rehabilitation can be considered as a potential application where exercise attempts are treated as gesture events and repetitions are counted only when confidence, stability and movement-consistency conditions are satisfied.

This is an application motivation only. The current project does not include clinical validation, patient studies, medical-device claims, or therapeutic efficacy evaluation.

## Current Evidence Status

The current online comparison demonstrates the evaluation machinery and the expected action-validation effect on pseudo-continuous replay. In the latest available smoke run, the effective predictor was the rule-based fallback because C6 model artifacts and processed tensors were unavailable in the checked workspace. Therefore the current numeric table is useful for validating the pipeline behavior, not for publishing final recognition quality.

The final thesis-grade evidence still requires running the same online evaluator with restored processed landmark tensors and C6 artifacts.
