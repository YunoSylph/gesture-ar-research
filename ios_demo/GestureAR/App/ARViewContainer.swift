// Hosts the layered AR experience:
//   layer 0: camera preview (AVCaptureVideoPreviewLayer)
//   layer 1: SCNView with the task's 3D scene (transparent background)
//   layer 2: hand-skeleton overlay (CAShapeLayer)
import AVFoundation
import Combine
import SceneKit
import SwiftUI
import UIKit

struct ARViewContainer: UIViewControllerRepresentable {
    @ObservedObject var viewModel: AppViewModel
    let task: ARTask

    func makeUIViewController(context: Context) -> ARViewController {
        ARViewController(viewModel: viewModel, task: task)
    }
    func updateUIViewController(_ controller: ARViewController, context: Context) {}
}

final class ARViewController: UIViewController {
    private let viewModel: AppViewModel
    private let task: ARTask

    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var sceneView: SCNView!
    private var skeletonView: SkeletonOverlayView!
    private var sceneController: SceneController!
    private var cancellables = Set<AnyCancellable>()

    init(viewModel: AppViewModel, task: ARTask) {
        self.viewModel = viewModel
        self.task = task
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) not supported") }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        // Layer 0 — camera preview
        previewLayer = AVCaptureVideoPreviewLayer(session: viewModel.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)

        // Layer 1 — SceneKit 3D content (transparent so the camera shows through)
        sceneController = SceneController(task: task)
        sceneView = SCNView(frame: view.bounds)
        sceneView.scene = sceneController.scene
        sceneView.backgroundColor = .clear
        sceneView.isOpaque = false
        sceneView.antialiasingMode = .multisampling2X
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.rendersContinuously = true
        view.addSubview(sceneView)

        // Layer 2 — hand skeleton overlay
        skeletonView = SkeletonOverlayView(frame: view.bounds)
        skeletonView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(skeletonView)

        bind()
    }

    private func bind() {
        viewModel.$skeleton
            .receive(on: DispatchQueue.main)
            .sink { [weak self] skeleton in
                self?.skeletonView.update(skeleton: skeleton, bufferSize: self?.viewModel.bufferSize ?? .zero)
            }
            .store(in: &cancellables)

        viewModel.gestureEvents
            .receive(on: DispatchQueue.main)
            .sink { [weak self] action in
                self?.sceneController.apply(action)
            }
            .store(in: &cancellables)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        sceneView.frame = view.bounds
        skeletonView.frame = view.bounds
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        viewModel.startCamera()
    }
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        viewModel.stopCamera()
    }
}

// MARK: - Skeleton overlay

final class SkeletonOverlayView: UIView {
    private let boneLayer = CAShapeLayer()
    private let jointLayer = CAShapeLayer()
    private var skeleton = HandSkeleton.empty
    private var bufferSize: CGSize = .zero

    private static let bones: [(Int, Int)] = [
        (0, 1), (1, 2), (2, 3), (3, 4),            // thumb
        (0, 5), (5, 6), (6, 7), (7, 8),            // index
        (5, 9), (9, 10), (10, 11), (11, 12),       // middle
        (9, 13), (13, 14), (14, 15), (15, 16),     // ring
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), // pinky + palm
    ]

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false

        boneLayer.strokeColor = UIColor(red: 0.30, green: 0.85, blue: 1.0, alpha: 0.95).cgColor
        boneLayer.fillColor = UIColor.clear.cgColor
        boneLayer.lineWidth = 4
        boneLayer.lineCap = .round
        boneLayer.lineJoin = .round
        boneLayer.shadowColor = UIColor.cyan.cgColor
        boneLayer.shadowRadius = 4
        boneLayer.shadowOpacity = 0.7
        boneLayer.shadowOffset = .zero
        layer.addSublayer(boneLayer)

        jointLayer.fillColor = UIColor(red: 0.55, green: 1.0, blue: 0.65, alpha: 1.0).cgColor
        jointLayer.strokeColor = UIColor.white.cgColor
        jointLayer.lineWidth = 1
        layer.addSublayer(jointLayer)
    }
    required init?(coder: NSCoder) { fatalError() }

    func update(skeleton: HandSkeleton, bufferSize: CGSize) {
        self.skeleton = skeleton
        self.bufferSize = bufferSize
        redraw()
    }

    private func point(_ j: SIMD2<Float>) -> CGPoint {
        // Map MediaPipe normalized (top-left origin) coords through the same
        // .resizeAspectFill transform the preview layer uses.
        let viewW = bounds.width, viewH = bounds.height
        let bufW = bufferSize.width > 0 ? bufferSize.width : viewW
        let bufH = bufferSize.height > 0 ? bufferSize.height : viewH
        let scale = max(viewW / bufW, viewH / bufH)
        let dispW = bufW * scale, dispH = bufH * scale
        let offX = (viewW - dispW) / 2, offY = (viewH - dispH) / 2
        return CGPoint(x: offX + CGFloat(j.x) * dispW, y: offY + CGFloat(j.y) * dispH)
    }

    private func redraw() {
        guard skeleton.isValid else {
            boneLayer.path = nil
            jointLayer.path = nil
            return
        }
        let pts = skeleton.joints.map { point($0) }

        let bonePath = UIBezierPath()
        for (a, b) in Self.bones where a < pts.count && b < pts.count {
            bonePath.move(to: pts[a])
            bonePath.addLine(to: pts[b])
        }
        boneLayer.path = bonePath.cgPath

        let jointPath = UIBezierPath()
        for (i, p) in pts.enumerated() {
            let r: CGFloat = (i == 0) ? 7 : 5
            jointPath.append(UIBezierPath(arcCenter: p, radius: r, startAngle: 0, endAngle: .pi * 2, clockwise: true))
        }
        jointLayer.path = jointPath.cgPath
    }
}
