// Task definitions for the three AR interaction modes + hand skeleton data.
import CoreGraphics
import simd

enum ARTask: Int, CaseIterable {
    case objectControl = 0
    case arScrolling = 1
    case sortingObjects = 2

    var title: String {
        switch self {
        case .objectControl: return "Object Control"
        case .arScrolling: return "AR Scrolling"
        case .sortingObjects: return "Sorting Objects"
        }
    }

    var subtitle: String {
        switch self {
        case .objectControl: return "Rotate and scale a 3D object with swipes and zoom gestures"
        case .arScrolling: return "Scroll through a stack of cards in 3D space"
        case .sortingObjects: return "Move and swap objects to arrange them"
        }
    }

    var hint: String {
        switch self {
        case .objectControl: return "Swipe ← → to rotate · Zoom in/out to scale · Click to recolor"
        case .arScrolling: return "Swipe ← → to move focus · Click to open card"
        case .sortingObjects: return "Swipe ← → to move · Click to pick up & swap"
        }
    }

    var symbol: String {
        switch self {
        case .objectControl: return "cube.transparent"
        case .arScrolling: return "rectangle.stack"
        case .sortingObjects: return "square.grid.3x1.below.line.grid.1x2"
        }
    }
}

// Hand skeleton snapshot from MediaPipe (21 joints, normalized image space).
struct HandSkeleton {
    let joints: [SIMD2<Float>]   // 21 normalized (x, y), origin top-left, y down
    let confidence: Float
    let timestamp: Int

    static let empty = HandSkeleton(joints: [], confidence: 0, timestamp: 0)

    var isValid: Bool { joints.count == 21 }
}
