import Foundation

/// Pure, testable model of the AR scene's interaction state. The RealityKit
/// controller renders this; gestures (via the acceptance policy) drive it. Keeping
/// the state logic here lets it be verified with `swift test`, independent of any
/// iOS-only AR framework.
///
/// Scene model: a ring/carousel of world-anchored objects. One is focused (front
/// of the arc); navigation rotates the ring, zoom scales the focused object,
/// select toggles its activated state, point shows a reticle on it.
public struct ARInteractionState: Equatable {
    public let objectCount: Int
    public private(set) var focusedIndex: Int
    public private(set) var focusedScale: Float
    public private(set) var activatedIndex: Int?
    public private(set) var isPointing: Bool

    public var minScale: Float
    public var maxScale: Float
    public var zoomStep: Float

    public init(objectCount: Int, minScale: Float = 0.5, maxScale: Float = 2.5, zoomStep: Float = 1.15) {
        precondition(objectCount > 0, "AR scene needs at least one object")
        self.objectCount = objectCount
        self.focusedIndex = 0
        self.focusedScale = 1.0
        self.activatedIndex = nil
        self.isPointing = false
        self.minScale = minScale
        self.maxScale = maxScale
        self.zoomStep = zoomStep
    }

    /// Apply an accepted AR action and return whether the state changed.
    @discardableResult
    public mutating func apply(_ action: ARGestureAction) -> Bool {
        let previous = self
        isPointing = (action == .pointerHover)
        switch action {
        case .navigateNext:
            focusedIndex = (focusedIndex + 1) % objectCount
            focusedScale = 1.0
        case .navigatePrevious:
            focusedIndex = (focusedIndex - 1 + objectCount) % objectCount
            focusedScale = 1.0
        case .zoomIn:
            focusedScale = min(focusedScale * zoomStep, maxScale)
        case .zoomOut:
            focusedScale = max(focusedScale / zoomStep, minScale)
        case .selectConfirm:
            activatedIndex = (activatedIndex == focusedIndex) ? nil : focusedIndex
        case .pointerHover:
            break
        }
        return self != previous
    }

    public var isFocusedActivated: Bool { activatedIndex == focusedIndex }
}
