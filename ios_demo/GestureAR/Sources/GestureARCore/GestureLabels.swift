import Foundation

public enum GestureLabel: Int, CaseIterable, Codable {
    case noGesture = 0
    case point2f = 1
    case click2f = 2
    case swipeLeft = 3
    case swipeRight = 4
    case zoomIn = 5
    case zoomOut = 6

    public var key: String {
        switch self {
        case .noGesture: return "no_gesture"
        case .point2f: return "point_2f"
        case .click2f: return "click_2f"
        case .swipeLeft: return "swipe_left"
        case .swipeRight: return "swipe_right"
        case .zoomIn: return "zoom_in"
        case .zoomOut: return "zoom_out"
        }
    }

    public init?(key: String) {
        guard let match = GestureLabel.allCases.first(where: { $0.key == key }) else { return nil }
        self = match
    }

    public var action: ARGestureAction? {
        switch self {
        case .noGesture: return nil
        case .point2f: return .pointerHover
        case .click2f: return .selectConfirm
        case .swipeLeft: return .navigatePrevious
        case .swipeRight: return .navigateNext
        case .zoomIn: return .zoomIn
        case .zoomOut: return .zoomOut
        }
    }
}

public enum ARGestureAction: String, Codable {
    case pointerHover = "pointer_hover"
    case selectConfirm = "select_confirm"
    case navigatePrevious = "navigate_previous"
    case navigateNext = "navigate_next"
    case zoomIn = "zoom_in"
    case zoomOut = "zoom_out"
}

public func mirroredTrainingLabel(_ label: GestureLabel) -> GestureLabel {
    switch label {
    case .swipeLeft: return .swipeRight
    case .swipeRight: return .swipeLeft
    default: return label
    }
}
