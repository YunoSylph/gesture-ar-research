// SceneKit 3D scene for each task. SceneKit (unlike a non-AR RealityKit ARView)
// has reliable transparent backgrounds, so the camera preview shows through behind
// the lit, solid-looking 3D objects.
import GestureARCore
import SceneKit
import UIKit

final class SceneController {
    let scene = SCNScene()
    private let task: ARTask

    // Object Control
    private var controlObject: SCNNode!
    private var controlScale: CGFloat = 1.0

    // Scrolling / Sorting
    private var itemsRoot: SCNNode!
    private var items: [SCNNode] = []
    private var focusIndex = 0
    private var holding = false
    private let spacingV: CGFloat = 2.2
    private let spacingH: CGFloat = 1.7

    private let palette: [UIColor] = [
        UIColor(red: 0.95, green: 0.30, blue: 0.35, alpha: 1),
        UIColor(red: 0.98, green: 0.65, blue: 0.20, alpha: 1),
        UIColor(red: 0.30, green: 0.80, blue: 0.55, alpha: 1),
        UIColor(red: 0.25, green: 0.55, blue: 0.95, alpha: 1),
        UIColor(red: 0.65, green: 0.40, blue: 0.95, alpha: 1),
    ]

    init(task: ARTask) {
        self.task = task
        scene.background.contents = nil   // transparent
        setupCameraAndLights()
        switch task {
        case .objectControl: buildObjectControl()
        case .arScrolling: buildScrolling()
        case .sortingObjects: buildSorting()
        }
    }

    // MARK: - Common

    private func setupCameraAndLights() {
        let camera = SCNCamera()
        camera.fieldOfView = 55
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 8)
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNLight()
        ambient.type = .ambient
        ambient.intensity = 450
        ambient.color = UIColor(white: 1, alpha: 1)
        let ambientNode = SCNNode(); ambientNode.light = ambient
        scene.rootNode.addChildNode(ambientNode)

        let key = SCNLight()
        key.type = .directional
        key.intensity = 900
        key.castsShadow = false
        let keyNode = SCNNode(); keyNode.light = key
        keyNode.eulerAngles = SCNVector3(-Float.pi / 4, Float.pi / 6, 0)
        scene.rootNode.addChildNode(keyNode)
    }

    private func makeMaterial(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = color
        m.roughness.contents = 0.45
        m.metalness.contents = 0.0
        m.emission.contents = UIColor.black
        return m
    }

    private func highlight(_ node: SCNNode, color: UIColor, on: Bool) {
        guard let m = node.geometry?.firstMaterial else { return }
        m.emission.contents = on ? color.withAlphaComponent(0.55) : UIColor.black
    }

    // MARK: - Object Control

    private func buildObjectControl() {
        let box = SCNBox(width: 2.4, height: 2.4, length: 2.4, chamferRadius: 0.25)
        box.firstMaterial = makeMaterial(palette[3])
        controlObject = SCNNode(geometry: box)
        scene.rootNode.addChildNode(controlObject)

        // Gentle idle bob so the object feels alive without fighting gesture rotations.
        let bob = SCNAction.sequence([
            .moveBy(x: 0, y: 0.18, z: 0, duration: 1.6),
            .moveBy(x: 0, y: -0.18, z: 0, duration: 1.6),
        ])
        bob.timingMode = .easeInEaseOut
        controlObject.runAction(.repeatForever(bob))
    }

    private var controlColorIndex = 3

    // MARK: - Scrolling (vertical stack of cards)

    private func buildScrolling() {
        itemsRoot = SCNNode()
        scene.rootNode.addChildNode(itemsRoot)
        for i in 0..<5 {
            let card = SCNBox(width: 4.0, height: 1.7, length: 0.18, chamferRadius: 0.12)
            card.firstMaterial = makeMaterial(palette[i % palette.count])
            let node = SCNNode(geometry: card)
            node.position = SCNVector3(0, Float(-CGFloat(i) * spacingV), 0)
            itemsRoot.addChildNode(node)
            items.append(node)
        }
        focusIndex = 0
        applyFocusLayout(animated: false)
    }

    // MARK: - Sorting (horizontal row of objects)

    private func buildSorting() {
        itemsRoot = SCNNode()
        scene.rootNode.addChildNode(itemsRoot)
        let n = 5
        for i in 0..<n {
            let box = SCNBox(width: 1.3, height: 1.3, length: 1.3, chamferRadius: 0.18)
            box.firstMaterial = makeMaterial(palette[i % palette.count])
            let node = SCNNode(geometry: box)
            node.position = SCNVector3(Float((CGFloat(i) - CGFloat(n - 1) / 2) * spacingH), 0, 0)
            itemsRoot.addChildNode(node)
            items.append(node)
        }
        focusIndex = 0
        applyFocusLayout(animated: false)
    }

    // MARK: - Focus layout (scrolling + sorting share highlight logic)

    private func applyFocusLayout(animated: Bool) {
        guard !items.isEmpty else { return }
        let dur = animated ? 0.28 : 0.0
        let mid = CGFloat(items.count - 1) / 2

        // Slide the whole group so the focused item sits at the center.
        SCNTransaction.begin(); SCNTransaction.animationDuration = dur
        if task == .arScrolling {
            itemsRoot.position = SCNVector3(0, Float(CGFloat(focusIndex) * spacingV), 0)
        } else if task == .sortingObjects {
            itemsRoot.position = SCNVector3(Float((mid - CGFloat(focusIndex)) * spacingH), 0, 0)
        }
        SCNTransaction.commit()

        for (i, node) in items.enumerated() {
            let focused = (i == focusIndex)
            let color = (node.geometry?.firstMaterial?.diffuse.contents as? UIColor) ?? .white
            let targetScale: CGFloat = focused ? 1.18 : 0.86
            let lift: Float = (task == .sortingObjects && focused) ? (holding ? 1.4 : 0.7) : 0

            SCNTransaction.begin(); SCNTransaction.animationDuration = dur
            node.scale = SCNVector3(targetScale, targetScale, targetScale)
            if task == .sortingObjects {
                let baseX = Float((CGFloat(i) - mid) * spacingH)
                node.position = SCNVector3(baseX, lift, focused ? 0.6 : 0)
            }
            SCNTransaction.commit()

            highlight(node, color: color, on: focused)
        }
    }

    // MARK: - Gesture entry point

    func apply(_ action: ARGestureAction) {
        switch task {
        case .objectControl: applyControl(action)
        case .arScrolling: applyScrolling(action)
        case .sortingObjects: applySorting(action)
        }
    }

    private func applyControl(_ action: ARGestureAction) {
        switch action {
        case .navigatePrevious:
            controlObject.runAction(.rotateBy(x: 0, y: -.pi / 4, z: 0, duration: 0.3))
        case .navigateNext:
            controlObject.runAction(.rotateBy(x: 0, y: .pi / 4, z: 0, duration: 0.3))
        case .zoomIn:
            controlScale = min(2.2, controlScale * 1.2)
            controlObject.runAction(.scale(to: controlScale, duration: 0.25))
        case .zoomOut:
            controlScale = max(0.4, controlScale * 0.82)
            controlObject.runAction(.scale(to: controlScale, duration: 0.25))
        case .selectConfirm:
            controlColorIndex = (controlColorIndex + 1) % palette.count
            controlObject.geometry?.firstMaterial?.diffuse.contents = palette[controlColorIndex]
            pulse(controlObject)
        case .pointerHover:
            controlObject.runAction(.sequence([
                .rotateBy(x: 0.12, y: 0, z: 0, duration: 0.1),
                .rotateBy(x: -0.12, y: 0, z: 0, duration: 0.1),
            ]))
        }
    }

    private func applyScrolling(_ action: ARGestureAction) {
        switch action {
        case .navigatePrevious:
            focusIndex = max(0, focusIndex - 1); applyFocusLayout(animated: true)
        case .navigateNext:
            focusIndex = min(items.count - 1, focusIndex + 1); applyFocusLayout(animated: true)
        case .selectConfirm:
            pulse(items[focusIndex])
        default:
            break
        }
    }

    private func applySorting(_ action: ARGestureAction) {
        switch action {
        case .selectConfirm:
            holding.toggle()
            applyFocusLayout(animated: true)
        case .navigatePrevious:
            if holding, focusIndex > 0 { swapFocused(with: focusIndex - 1) }
            else { focusIndex = max(0, focusIndex - 1); applyFocusLayout(animated: true) }
        case .navigateNext:
            if holding, focusIndex < items.count - 1 { swapFocused(with: focusIndex + 1) }
            else { focusIndex = min(items.count - 1, focusIndex + 1); applyFocusLayout(animated: true) }
        default:
            break
        }
    }

    private func swapFocused(with other: Int) {
        items.swapAt(focusIndex, other)
        focusIndex = other
        applyFocusLayout(animated: true)
    }

    private func pulse(_ node: SCNNode) {
        let base = node.scale
        let up = SCNVector3(base.x * 1.15, base.y * 1.15, base.z * 1.15)
        node.runAction(.sequence([
            .customAction(duration: 0.12) { n, _ in n.scale = up },
            .customAction(duration: 0.12) { n, _ in n.scale = base },
        ]))
    }
}
