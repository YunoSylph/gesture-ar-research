import XCTest
@testable import GestureARCore

final class ContextPolicyTests: XCTestCase {
    func testPerClassThresholdGatesClickOnly() {
        var config = ContextPolicyConfig()
        config.stableFrames = 1
        config.perClassActivation = [.click2f: 0.9]
        let policy = ContextAwareGesturePolicy(config: config)

        // click below its raised floor (0.8 < 0.9) is rejected
        XCTAssertNil(policy.update(GesturePrediction(label: .click2f, confidence: 0.8), timestampMs: 0))
        // click above its floor fires
        XCTAssertNotNil(policy.update(GesturePrediction(label: .click2f, confidence: 0.95), timestampMs: 1000))

        // other classes keep the global floor (0.7 >= 0.62) and fire
        let other = ContextAwareGesturePolicy(config: config)
        XCTAssertNotNil(other.update(GesturePrediction(label: .swipeRight, confidence: 0.7), timestampMs: 0))
    }
}
