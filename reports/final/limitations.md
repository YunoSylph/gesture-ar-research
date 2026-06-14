# Limitations

Last updated: 2026-06-15

## Data And Evaluation Limitations

- The current workspace did not contain raw IPN videos, processed landmark tensors, or configured C6 model artifacts.
- The latest online comparison used synthetic fallback landmarks and the rule-based fallback predictor.
- Pseudo-continuous replay is useful for reproducible validation, but it is not the same as a fully annotated continuous dataset.
- The evaluator should not be compared numerically against OO-dMVMT unless the OO-dMVMT protocol and data setting are reproduced.
- Current task success is 0.0 in the fallback run, so the project should not claim complete task-level interaction reliability yet.

## Model And Controller Limitations

- A clip-level TCN/C6 recognizer can perform well offline and still fail in continuous webcam interaction.
- The live path combines recognizer output, landmark heuristics, validation, and TARC. Failures must be attributed carefully rather than blamed on a single component.
- The landmark controller is now executable in the active Python environment, but its current replay numbers still use synthetic fallback landmarks and should not be treated as final live-control evidence.
- Gesture cards and UI hints are helpful only if they match the gesture contract and the actual controller logic.

## Live Demo Limitations

- Webcam performance depends on camera quality, lighting, pose, browser/backend load, and user behavior.
- Live sessions usually lack frame-level ground truth, so they are better for qualitative failure analysis than final numerical proof.
- The current AR UI is an AR-style web demonstration, not a validated phone rear-camera AR, iOS, Quest, or HoloLens deployment.

## Research Boundaries

- The project is not a SOTA gesture-recognition paper.
- The project is not a clinical rehabilitation system.
- No medical or therapeutic claims should be made.
- Hand rehabilitation can be considered as a potential application where exercise attempts are treated as gesture events and repetitions are counted only when confidence, stability and movement-consistency conditions are satisfied.

## Next Required Work

- Restore or regenerate real IPN landmark tensors.
- Restore or retrain C6 artifacts only after data availability is verified.
- Run the online evaluator with real C6 outputs.
- Add action-level plots for false-action cost, accepted/rejected actions, and action switching.
- Improve task scenarios until at least one validation mode achieves non-zero task success without increasing false-action cost.
- Record optional local calibration clips only after the public-data replay protocol is stable.
