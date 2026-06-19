import Foundation

/// End-to-end on-device recognizer: maintains a sliding landmark window, runs the
/// Core ML classifier on the preprocessed features, and gates the result through
/// the acceptance policy. Feed one captured frame per camera frame; a non-nil
/// return value is an accepted AR action.
public final class GestureRecognizer {
    private let model: GestureModel
    private let policy: ContextAwareGesturePolicy
    private let windowLength: Int
    private var window: [LandmarkFrame] = []

    public init(
        model: GestureModel,
        policy: ContextAwareGesturePolicy = ContextAwareGesturePolicy(),
        windowLength: Int = LandmarkPreprocessor.targetLength
    ) {
        self.model = model
        self.policy = policy
        self.windowLength = windowLength
        window.reserveCapacity(windowLength)
    }

    public func process(frame: LandmarkFrame, timestampMs: Int) throws -> GestureInteractionEvent? {
        window.append(frame)
        if window.count > windowLength {
            window.removeFirst(window.count - windowLength)
        }
        guard window.count == windowLength else { return nil }

        let features = LandmarkPreprocessor.preprocess(window).features
        let prediction = try model.predict(features: features)
        return policy.update(prediction, timestampMs: timestampMs)
    }

    public func reset() {
        window.removeAll(keepingCapacity: true)
        policy.reset()
    }
}
