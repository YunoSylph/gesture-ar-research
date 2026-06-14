import Foundation
import simd

struct LandmarkFrame {
    var points: [SIMD3<Float>]
    var confidence: Float
}

struct PreprocessedGestureWindow {
    var features: [Float]
    var shape: (time: Int, dim: Int)
}

enum LandmarkPreprocessor {
    static let targetLength = 32
    static let featureDim = 74
    private static let wrist = 0
    private static let indexMCP = 5
    private static let middleMCP = 9
    private static let pinkyMCP = 17

    static func preprocess(_ frames: [LandmarkFrame], targetLength: Int = Self.targetLength) -> PreprocessedGestureWindow {
        let sampled = resample(frames, targetLength: targetLength)
        var output: [Float] = []
        output.reserveCapacity(targetLength * featureDim)
        var previousCentroid = SIMD2<Float>(repeating: 0)
        var previousWrist = SIMD2<Float>(repeating: 0)
        var previousHandSize: Float = 0

        for (index, frame) in sampled.enumerated() {
            let points = frame.points
            let wristPoint = points[wrist]
            let scale = max(palmScale(points), 0.000001)
            for point in points {
                let normalized = (point - wristPoint) / scale
                output.append(normalized.x)
                output.append(normalized.y)
                output.append(normalized.z)
            }

            let centroid = points.reduce(SIMD2<Float>(repeating: 0)) { partial, point in
                partial + SIMD2<Float>(point.x, point.y)
            } / Float(points.count)
            let wristXY = SIMD2<Float>(wristPoint.x, wristPoint.y)
            let handSize = scale
            let centroidDelta = index == 0 ? SIMD2<Float>(repeating: 0) : centroid - previousCentroid
            let wristDelta = index == 0 ? SIMD2<Float>(repeating: 0) : wristXY - previousWrist
            let handSizeDelta = index == 0 ? Float(0) : handSize - previousHandSize

            output.append(contentsOf: [
                centroid.x, centroid.y,
                wristXY.x, wristXY.y,
                centroidDelta.x, centroidDelta.y,
                wristDelta.x, wristDelta.y,
                handSize, handSizeDelta,
                frame.confidence
            ])

            previousCentroid = centroid
            previousWrist = wristXY
            previousHandSize = handSize
        }
        return PreprocessedGestureWindow(features: output, shape: (targetLength, featureDim))
    }

    private static func resample(_ frames: [LandmarkFrame], targetLength: Int) -> [LandmarkFrame] {
        guard let first = frames.first else {
            let zeroFrame = LandmarkFrame(points: Array(repeating: SIMD3<Float>(repeating: 0), count: 21), confidence: 0)
            return Array(repeating: zeroFrame, count: targetLength)
        }
        guard frames.count > 1 else {
            return Array(repeating: first, count: targetLength)
        }
        return (0..<targetLength).map { outputIndex in
            let position = Float(outputIndex) * Float(frames.count - 1) / Float(targetLength - 1)
            return frames[Int(position.rounded())]
        }
    }

    private static func palmScale(_ points: [SIMD3<Float>]) -> Float {
        let width = simd_distance(
            SIMD2<Float>(points[indexMCP].x, points[indexMCP].y),
            SIMD2<Float>(points[pinkyMCP].x, points[pinkyMCP].y)
        )
        let length = simd_distance(
            SIMD2<Float>(points[middleMCP].x, points[middleMCP].y),
            SIMD2<Float>(points[wrist].x, points[wrist].y)
        )
        return max(width, length)
    }
}
