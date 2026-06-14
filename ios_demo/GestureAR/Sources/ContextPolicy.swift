import Foundation

struct GesturePrediction {
    var label: GestureLabel
    var confidence: Float
}

struct GestureInteractionEvent {
    var timestampMs: Int
    var gesture: GestureLabel
    var action: ARGestureAction
    var confidence: Float
    var state: String
}

struct ContextPolicyConfig {
    var activationThreshold: Float = 0.62
    var stableFrames: Int = 2
    var cooldownMs: Int = 250
    var noGestureResetFrames: Int = 3
}

final class ContextAwareGesturePolicy {
    private let config: ContextPolicyConfig
    private var state = "idle"
    private var candidate: GestureLabel?
    private var candidateCount = 0
    private var lastActionMs = Int.min / 2
    private var noGestureCount = 0

    init(config: ContextPolicyConfig = ContextPolicyConfig()) {
        self.config = config
    }

    func reset() {
        state = "idle"
        candidate = nil
        candidateCount = 0
        lastActionMs = Int.min / 2
        noGestureCount = 0
    }

    func update(_ prediction: GesturePrediction, timestampMs: Int) -> GestureInteractionEvent? {
        if prediction.label == .noGesture || prediction.confidence < config.activationThreshold {
            noGestureCount += 1
            if noGestureCount >= config.noGestureResetFrames {
                state = "idle"
                candidate = nil
                candidateCount = 0
            }
            return nil
        }

        noGestureCount = 0
        if prediction.label == candidate {
            candidateCount += 1
        } else {
            candidate = prediction.label
            candidateCount = 1
            state = "tracking"
        }

        guard candidateCount >= config.stableFrames else { return nil }
        guard timestampMs - lastActionMs >= config.cooldownMs else { return nil }
        guard let action = prediction.label.action else { return nil }

        lastActionMs = timestampMs
        state = "cooldown"
        return GestureInteractionEvent(
            timestampMs: timestampMs,
            gesture: prediction.label,
            action: action,
            confidence: prediction.confidence,
            state: state
        )
    }
}
