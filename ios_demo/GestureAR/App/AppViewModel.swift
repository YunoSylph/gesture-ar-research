// Central view model: owns the camera capture, MediaPipe hand tracking, and the
// CoreML gesture recognizer. Publishes hand/skeleton state for the UI and emits
// discrete gesture actions for the 3D scene to react to.
import AVFoundation
import Combine
import CoreGraphics
import CoreML
import Foundation
import GestureARCore
import simd
#if canImport(MediaPipeTasksVision)
import MediaPipeTasksVision
#endif

final class AppViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    // UI-facing state
    @Published var handDetected = false
    @Published var skeleton = HandSkeleton.empty
    @Published var lastGesture: String = ""
    @Published var lastAction: String? = nil
    @Published var cameraAuthorized = true
    @Published var statusText = "Starting camera…"
    @Published var bufferSize: CGSize = .zero

    // Discrete gesture actions for the 3D scene (fires once per accepted gesture).
    let gestureEvents = PassthroughSubject<ARGestureAction, Never>()

    // Capture
    let captureSession = AVCaptureSession()
    private let cameraQueue = DispatchQueue(label: "gesturear.camera")
    private let processingQueue = DispatchQueue(label: "gesturear.processing", qos: .userInteractive)
    private var lastTimestampMs = 0
    private var configured = false

    // Recognition
    private var recognizer: GestureARCore.GestureRecognizer?
    #if canImport(MediaPipeTasksVision)
    private var handLandmarker: HandLandmarker?
    #endif

    deinit { captureSession.stopRunning() }

    // MARK: - Lifecycle

    func onAppear() {
        // Test hook: skip the camera/permission path so the 3D scene can be
        // screenshotted on the simulator (which has no camera). Production-inert.
        if ProcessInfo.processInfo.environment["GAR_NOCAM"] != nil {
            cameraAuthorized = true
            statusText = ""
            return
        }
        requestAccessAndConfigure()
    }

    func startCamera() {
        cameraQueue.async { [weak self] in
            guard let self, self.configured, !self.captureSession.isRunning else { return }
            self.captureSession.startRunning()
        }
    }

    func stopCamera() {
        cameraQueue.async { [weak self] in
            guard let self, self.captureSession.isRunning else { return }
            self.captureSession.stopRunning()
        }
    }

    // MARK: - Setup

    private func requestAccessAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configureAndStart()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.cameraAuthorized = granted
                    if granted { self?.configureAndStart() }
                    else { self?.statusText = "Camera access denied" }
                }
            }
        default:
            cameraAuthorized = false
            statusText = "Camera access denied — enable it in Settings"
        }
    }

    private func configureAndStart() {
        cameraQueue.async { [weak self] in
            guard let self, !self.configured else { self?.startCamera(); return }
            self.configureSession()
            self.setupHandLandmarker()
            self.setupRecognizer()
            self.configured = true
            self.captureSession.startRunning()
            DispatchQueue.main.async { self.statusText = "" }
        }
    }

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        // Rear 0.5x ultra-wide (matches training domain); fall back to wide.
        let device = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        guard let device,
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "No rear camera available" }
            return
        }
        captureSession.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: cameraQueue)
        if captureSession.canAddOutput(output) { captureSession.addOutput(output) }

        // Deliver upright (portrait) frames so MediaPipe and the preview agree.
        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
        captureSession.commitConfiguration()
    }

    private func setupHandLandmarker() {
        #if canImport(MediaPipeTasksVision)
        guard let modelPath = Bundle.main.path(forResource: "hand_landmarker", ofType: "task") else {
            print("[GestureAR] hand_landmarker.task missing"); return
        }
        let options = HandLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .video
        options.numHands = 1
        options.minHandDetectionConfidence = 0.4
        options.minHandPresenceConfidence = 0.4
        options.minTrackingConfidence = 0.4
        handLandmarker = try? HandLandmarker(options: options)
        if handLandmarker == nil { print("[GestureAR] HandLandmarker failed to init") }
        #endif
    }

    private func setupRecognizer() {
        guard let url = Bundle.main.url(forResource: "GestureClassifier", withExtension: "mlmodelc") else {
            print("[GestureAR] GestureClassifier.mlmodelc missing"); return
        }
        do {
            let model = try GestureARCore.GestureModel(contentsOf: url)
            recognizer = GestureARCore.GestureRecognizer(model: model)
        } catch {
            print("[GestureAR] recognizer init failed: \(error)")
        }
    }

    // MARK: - Capture delegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        var ts = Int(CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds * 1000)
        ts = max(ts, lastTimestampMs + 1)
        lastTimestampMs = ts
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processFrame(pixelBuffer, timestampMs: ts)
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer, timestampMs: Int) {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        if bufferSize.width != w || bufferSize.height != h {
            DispatchQueue.main.async { self.bufferSize = CGSize(width: w, height: h) }
        }

        #if canImport(MediaPipeTasksVision)
        guard let handLandmarker,
              let image = try? MPImage(pixelBuffer: pixelBuffer, orientation: .up),
              let result = try? handLandmarker.detect(videoFrame: image, timestampInMilliseconds: timestampMs)
        else { return }

        guard let hand = result.landmarks.first, hand.count == 21 else {
            DispatchQueue.main.async {
                self.handDetected = false
                self.skeleton = .empty
            }
            // Keep feeding the recognizer an "absent" frame so its window decays.
            feedRecognizer(joints: nil, confidence: 0, timestampMs: timestampMs)
            return
        }

        let joints = hand.map { SIMD2<Float>(Float($0.x), Float($0.y)) }
        let confidence = Float(result.handedness.first?.first?.score ?? 1.0)
        let skel = HandSkeleton(joints: joints, confidence: confidence, timestamp: timestampMs)
        DispatchQueue.main.async {
            self.handDetected = true
            self.skeleton = skel
        }
        feedRecognizer(joints: joints, confidence: confidence, timestampMs: timestampMs)
        #endif
    }

    private func feedRecognizer(joints: [SIMD2<Float>]?, confidence: Float, timestampMs: Int) {
        guard let recognizer else { return }
        let points: [SIMD3<Float>]
        let valid: Bool
        if let joints {
            points = joints.map { SIMD3<Float>($0.x, $0.y, 0) }
            valid = true
        } else {
            points = Array(repeating: .zero, count: 21)
            valid = false
        }
        let frame = LandmarkFrame(points: points, confidence: confidence, valid: valid)
        guard let event = try? recognizer.process(frame: frame, timestampMs: timestampMs) else { return }

        DispatchQueue.main.async {
            self.lastGesture = event.gesture.key
            self.lastAction = self.label(for: event.action)
            self.gestureEvents.send(event.action)
            let stamp = timestampMs
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                // Only clear if no newer gesture arrived.
                if self.skeleton.timestamp <= stamp + 1500 { self.lastAction = nil }
            }
        }
    }

    private func label(for action: ARGestureAction) -> String {
        switch action {
        case .pointerHover: return "Point"
        case .selectConfirm: return "Click"
        case .navigatePrevious: return "Swipe ←"
        case .navigateNext: return "Swipe →"
        case .zoomIn: return "Zoom in"
        case .zoomOut: return "Zoom out"
        }
    }
}
