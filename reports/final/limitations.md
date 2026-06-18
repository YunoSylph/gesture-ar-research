# Limitations

Last updated: 2026-06-18

## Data And Evaluation Limitations

- The workspace can reproduce the current results when local IPN Hand clips, processed landmarks, and model artifacts are present, but these large assets are intentionally not committed to GitHub.
- IPN Hand is clip/segment oriented in this workspace. The evaluator therefore builds an explicitly marked pseudo-continuous replay by concatenating real gesture clips with real `no_gesture` idle gaps.
- Pseudo-continuous replay is useful for reproducible validation, but it is not the same as a fully annotated original continuous stream or a controlled user study.
- The evaluator should not be compared numerically against OO-dMVMT unless the OO-dMVMT protocol and data setting are reproduced.
- The raw event logs are useful for auditability, but the headline evidence should come from compact summaries and paired comparisons.

## Model And Controller Limitations

- Clip-level TCN/C6 recognition can perform well offline and still produce unstable live commands in continuous webcam interaction.
- The online frame-level recognition metrics are intentionally lower than the clip-level metrics because the stream contains idle gaps, transitions, and window-boundary mismatch.
- The landmark controller remains an engineering baseline. Its current replay metrics are weaker than the C6 validation/TARC path and should not be framed as the central research result.
- The stricter global release-gate ablation is not deployed because it blocks legitimate next steps when the sliding window does not emit a clean `no_gesture` segment.

## Live Demo Limitations

- Webcam performance depends on camera quality, lighting, pose, browser/backend load, and user behavior.
- Live sessions usually lack frame-level ground truth, so they are better for qualitative failure analysis than final numerical proof.
- The current AR UI is an AR-style web demonstration, not a validated phone rear-camera AR, iOS, Quest, or HoloLens deployment.

## Research Boundaries

- The project is not a SOTA gesture-recognition paper.
- The project is not a clinical rehabilitation system.
- No medical or therapeutic claims should be made.
- Hand rehabilitation can be considered only as a potential application where exercise attempts are treated as gesture events and repetitions are counted when confidence, stability, and movement-consistency conditions are satisfied.

## Next Required Work

- Keep the public-data replay protocol stable and reproducible from a clean machine.
- Add a lightweight artifact restoration/training guide for users who clone the GitHub repository without local model files.
- Improve live task timing and user guidance without overstating webcam-demo reliability.
- Add optional local calibration sessions only after the public-data replay evidence remains stable.
- Treat phone rear-camera AR as a separate application/domain-shift stage, not as proof that the webcam-trained pipeline transfers automatically.
