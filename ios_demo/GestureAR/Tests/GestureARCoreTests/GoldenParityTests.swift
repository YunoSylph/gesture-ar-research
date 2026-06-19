import XCTest
import simd
@testable import GestureARCore

// Verifies the Swift on-device port reproduces the Python reference, using the
// golden artifacts emitted by export_preprocessing_contract / export_validation_contract.

private struct GoldenSamplesFile: Decodable { let samples: [GoldenSample] }
private struct GoldenSample: Decodable {
    let sample_id: String
    let target_label: String
    let input: GoldenInput
    let expected_features: [[Float]]
}
private struct GoldenInput: Decodable {
    let landmarks: [[[Float]]]
    let sequence_mask: [Bool]
    let frame_confidence: [Float]
}

private struct GoldenTracesFile: Decodable { let scenarios: [GoldenScenario] }
private struct GoldenScenario: Decodable {
    let name: String
    let frames: [GoldenFrame]
    let events: [GoldenEvent]
}
private struct GoldenFrame: Decodable {
    let timestampMs: Int
    let label: String
    let confidence: Float
    init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        timestampMs = try c.decode(Int.self)
        label = try c.decode(String.self)
        confidence = try c.decode(Float.self)
    }
}
private struct GoldenEvent: Decodable {
    let timestamp_ms: Int
    let gesture: String
    let action: String
    let state: String
}

final class GoldenParityTests: XCTestCase {
    private func loadResource(_ name: String) throws -> Data {
        guard let url = Bundle.module.url(forResource: name, withExtension: "json") else {
            throw XCTSkip("Missing golden resource \(name).json")
        }
        return try Data(contentsOf: url)
    }

    func testPreprocessingMatchesGoldenFeatures() throws {
        let file = try JSONDecoder().decode(GoldenSamplesFile.self, from: loadResource("golden_samples"))
        XCTAssertFalse(file.samples.isEmpty)
        for sample in file.samples {
            let frames = (0..<sample.input.landmarks.count).map { t -> LandmarkFrame in
                let pts = sample.input.landmarks[t].map { SIMD3<Float>($0[0], $0[1], $0[2]) }
                return LandmarkFrame(points: pts, confidence: sample.input.frame_confidence[t], valid: sample.input.sequence_mask[t])
            }
            let out = LandmarkPreprocessor.preprocess(frames)
            let expected = sample.expected_features.flatMap { $0 }
            XCTAssertEqual(out.shape.dim, 326)
            XCTAssertEqual(out.features.count, expected.count, "\(sample.sample_id): feature count")
            var maxDiff: Float = 0
            for i in 0..<expected.count { maxDiff = max(maxDiff, abs(out.features[i] - expected[i])) }
            XCTAssertLessThan(maxDiff, 1e-4, "\(sample.target_label): preprocessing parity (max diff \(maxDiff))")
        }
    }

    func testPolicyMatchesGoldenTraces() throws {
        let file = try JSONDecoder().decode(GoldenTracesFile.self, from: loadResource("golden_traces"))
        XCTAssertFalse(file.scenarios.isEmpty)
        for scenario in file.scenarios {
            let policy = ContextAwareGesturePolicy()
            var events: [GestureInteractionEvent] = []
            for frame in scenario.frames {
                guard let label = GestureLabel(key: frame.label) else {
                    return XCTFail("unknown label \(frame.label)")
                }
                if let event = policy.update(GesturePrediction(label: label, confidence: frame.confidence), timestampMs: frame.timestampMs) {
                    events.append(event)
                }
            }
            XCTAssertEqual(events.count, scenario.events.count, "\(scenario.name): event count")
            for (got, want) in zip(events, scenario.events) {
                XCTAssertEqual(got.timestampMs, want.timestamp_ms, "\(scenario.name): timestamp")
                XCTAssertEqual(got.gesture.key, want.gesture, "\(scenario.name): gesture")
                XCTAssertEqual(got.action.rawValue, want.action, "\(scenario.name): action")
                XCTAssertEqual(got.state, want.state, "\(scenario.name): state")
            }
        }
    }
}
