# Live Model Assessment

Date: 2026-06-14

## Input Video Diagnostic

User webcam file:

```text
C:\Users\Maksim Iuzhakov\Desktop\Hand Gestures Project\20260614_15_02_10_615.mp4
```

Measured properties:

- Resolution: 1920x1080.
- Frame rate: 29.14 FPS.
- Duration: 18.43 s.
- Sampled frames: 268, every second frame.
- MediaPipe hand detection rate: 0.8209.
- Mean hand confidence: 0.8022.
- Mean normalized pointer jump: 0.0521, with high outliers up to 0.5147.
- Mean Laplacian blur variance: 15.80, which indicates a soft/motion-blurred webcam image.

## Raw Model Behavior On This Video

Raw `c6_ensemble` predictions were unstable for continuous live control:

| Label | Share |
| --- | ---: |
| no_gesture | 0.4963 |
| click_2f | 0.1604 |
| swipe_left | 0.1381 |
| zoom_in | 0.1194 |
| zoom_out | 0.0597 |
| swipe_right | 0.0261 |

The critical issue is that `point_2f` was not produced by the raw robust model on this recording. That makes direct model-to-action control unsuitable for a real-time AR cursor.

## Revised Live Controller Behavior

The live system now uses a landmark-first controller for webcam AR control:

- visible hand becomes `point_2f`;
- `click_2f` requires open hand -> short index-to-thumb pinch or two-finger tap -> lock-hold -> cooldown;
- swipes require wide horizontal displacement above jitter;
- zoom requires clear relative hand-scale change;
- TARC passes the expected gesture for the current task step into the controller, suppressing unrelated command candidates;
- locked commands are held for several frames so the downstream task policy can accept them reliably;
- the neural model remains available for research logging and offline comparison.

On the same sampled video, the revised live controller produced:

| Label | Share |
| --- | ---: |
| point_2f | 0.7015 |
| no_gesture | 0.1157 |
| zoom_in | 0.0597 |
| swipe_right | 0.0448 |
| swipe_left | 0.0299 |
| zoom_out | 0.0299 |
| click_2f | 0.0187 |

This is a healthier live-control distribution: continuous cursor control dominates while commands become explicit lock-held events. In the actual guided task mode, expected-gesture focus further suppresses unrelated commands.

## Research Interpretation

The IPN-trained temporal recognizer is still useful for the thesis as an offline recognition benchmark, but it is not enough for live AR interaction. IPN clips are pre-segmented gesture samples; the application receives a continuous webcam stream with transitions, pauses, partial gestures, motion blur, and user-specific hand scale. Treating each sliding window as a clean gesture segment produces accidental commands.

The stronger research framing is therefore:

```text
landmarks -> robust recognizer for benchmark evidence
landmarks -> live geometric event controller for real-time AR control
task step -> expected gesture focus and lock-hold fixation
recognizer + controller -> task-aware risk calibration
```

This creates a defensible combined method: learned recognition is measured on public data, while live interaction is stabilized by a continuous-control layer that respects AR task context.

## Remaining Scientific Work

- Add a small local calibration protocol with a few short webcam recordings per gesture.
- Evaluate direct classifier control vs landmark event controller vs task-aware controller on recorded live sessions.
- Add a continuous-control metric: false command rate per minute, click precision, pointer jitter, task completion time.
- Consider a second dataset only if it provides continuous or egocentric hand-control sequences. A second pre-segmented dataset alone will not solve the live-control mismatch.
