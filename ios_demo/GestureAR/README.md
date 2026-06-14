# GestureAR iOS Portability Layer

This folder is the iPhone/iPad handoff point for the Windows-first research pipeline.

The intended runtime pipeline is:

```text
ARKit rear camera frame
-> hand landmarks from MediaPipe/Vision
-> LandmarkPreprocessor.swift, shape [1,32,74]
-> Core ML gesture classifier
-> ContextPolicy.swift
-> RealityKit object action
```

The phone demo should use the rear world camera as the common source for both AR tracking and gesture recognition. A separate front-camera stream is treated as a different experiment, not as the main thesis demo path.

## Prepared Swift Contract

- `Sources/GestureLabels.swift`: final seven-class gesture dictionary and action mapping.
- `Sources/LandmarkPreprocessor.swift`: Swift equivalent of the Python dual-view preprocessing contract.
- `Sources/ContextPolicy.swift`: Swift equivalent of the C2 confidence/stability/cooldown gate.

## Generated Bundle

From the project root:

```powershell
python -m research_pipeline.cli.export_mobile_bundle
```

This writes:

```text
artifacts/mobile/gesture_mobile_bundle/
```

The bundle contains labels, preprocessing shape, C2 policy, ONNX metadata and the Core ML conversion contract.

Core ML conversion is intentionally separated from the Windows training environment. Run the conversion stage later on macOS or Linux, then add the resulting `.mlpackage` to an Xcode RealityKit project.
