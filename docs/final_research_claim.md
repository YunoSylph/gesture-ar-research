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

Last updated: 2026-06-16.

The evidence now runs on real data, not a fallback. Real MediaPipe-21 landmark tensors are extracted
from the official IPN Hand videos (2405 train / 1033 test clips). The recognizer is the trained
two-TCN C6 ensemble with the OO-dMVMT multi-view feature block (Joint Collection Distances plus
slow/fast motion), calibrated score fusion, and the validation/TARC layer. The headline ablation
(false-action-cost reduction, n = 24 paired sequences, bootstrap CI + exact McNemar) is in
`reports/final/online_summary.md`; offline recognition, confidence calibration (ECE/MCE/Brier), and
the multi-view comparison are in `reports/final/recognition_summary.md`.

## Scope and Validity of Evidence

This is the desktop/webcam + public-data replay proof. Four boundaries are stated explicitly and
must not be blurred when the work is presented:

1. **Pseudo-continuous, not a fully annotated continuous dataset.** The online stream is real
   gesture clips and real no_gesture idle gaps concatenated in task order; it is not the original
   uncut IPN continuous timeline with frame-level onset/offset ground truth. The replay is valid
   evidence for the *relative* effect of the validation/TARC layer on identical sequences across
   methods (a paired, controlled comparison), but not for *absolute* continuous-stream segmentation
   quality. A truly continuous annotated set is future work.
2. **The live webcam demo is illustration, not proof.** It is an engineering demonstration and a
   failure-analysis surface. Every scientific claim rests on the reproducible replay/evaluation
   pipeline and its ablation tables, never on visual impressions from the live demo.
3. **Task completion is reported with graded metrics, not the floor-effect binary.** TARC
   demonstrably reduces false AR actions (the defended claim) *and* raises task completion on the
   same paired sequences: the graded `task_completion` score rises 0.058 -> 0.669 and the
   confident-completion rate (completion score >= tau = 0.5) rises 0.000 -> 0.875, both significant
   (paired bootstrap, n = 24, p < 0.001). The strict binary success (every step in order with zero
   false-action cost) stays low (<= 0.083) and is kept only for completeness -- it is a deliberately
   conservative bar, not the headline. What is *not* yet claimed: perfectly clean end-to-end runs,
   and validation of task completion on live sessions rather than replay. Tightening task logic,
   timing, gesture locking, and live calibration to lift the strict bar remains future work.
4. **OO-dMVMT alignment is methodological, not architectural.** The project follows OO-dMVMT's
   evaluation direction (continuous-stream recognition/segmentation framing, decision latency, false
   positives) and borrows its multi-view feature idea. It does not reproduce the multi-view
   multi-task architecture or its training protocol, so no numeric comparison with OO-dMVMT results
   is made or implied.
