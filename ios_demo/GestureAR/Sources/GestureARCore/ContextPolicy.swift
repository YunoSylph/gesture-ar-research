import Foundation

public struct GesturePrediction {
    public var label: GestureLabel
    public var confidence: Float
    public init(label: GestureLabel, confidence: Float) {
        self.label = label
        self.confidence = confidence
    }
}

public struct GestureInteractionEvent: Equatable {
    public var timestampMs: Int
    public var gesture: GestureLabel
    public var action: ARGestureAction
    public var confidence: Float
    public var state: String
}

public struct ContextPolicyConfig {
    public var activationThreshold: Float = 0.62
    public var stableFrames: Int = 2
    public var cooldownMs: Int = 250
    public var noGestureResetFrames: Int = 3
    /// Optional per-gesture confidence floors (override the global threshold). Use a
    /// higher floor for high-risk / under-trained classes — e.g. click_2f, which is
    /// under-supported (7 local samples) and over-fires on point_2f.
    public var perClassActivation: [GestureLabel: Float] = [:]
    public init() {}

    func threshold(for label: GestureLabel) -> Float {
        max(activationThreshold, perClassActivation[label] ?? 0)
    }
}

/// Swift port of research_pipeline.interaction.fsm.ContextAwarePolicy.
/// Verified against artifacts/mobile/validation/golden_traces.json.
public final class ContextAwareGesturePolicy {
    private let config: ContextPolicyConfig
    private var state = "idle"
    private var candidate: GestureLabel?
    private var candidateCount = 0
    private var lastActionMs = Int.min / 2
    private var noGestureCount = 0

    public init(config: ContextPolicyConfig = ContextPolicyConfig()) {
        self.config = config
    }

    public func reset() {
        state = "idle"
        candidate = nil
        candidateCount = 0
        lastActionMs = Int.min / 2
        noGestureCount = 0
    }

    public func update(_ prediction: GesturePrediction, timestampMs: Int) -> GestureInteractionEvent? {
        if prediction.label == .noGesture || prediction.confidence < config.threshold(for: prediction.label) {
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
