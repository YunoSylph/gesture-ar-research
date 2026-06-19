import Foundation
import simd

public struct LandmarkFrame {
    public var points: [SIMD3<Float>]   // 21 MediaPipe landmarks, image-normalized (x, y, z)
    public var confidence: Float
    public var valid: Bool

    public init(points: [SIMD3<Float>], confidence: Float, valid: Bool = true) {
        self.points = points
        self.confidence = confidence
        self.valid = valid
    }
}

public struct PreprocessedGestureWindow {
    public var features: [Float]   // row-major [time * dim]
    public var shape: (time: Int, dim: Int)
}

/// Swift port of research_pipeline.features.preprocess_dual_view(include_multiview=True,
/// multiview_coords=2). Produces the exact 326-dim per-frame feature vector the mv
/// Core ML model expects. Verified against
/// artifacts/mobile/preprocessing/{feature_contract,golden_samples}.json.
///
/// Layout per frame: pose(63) + motion(11) + jcd(210) + slow_motion(21) + fast_motion(21).
public enum LandmarkPreprocessor {
    public static let targetLength = 32
    public static let landmarkCount = 21
    public static let poseDim = landmarkCount * 3                          // 63
    public static let motionDim = 11
    public static let jcdDim = landmarkCount * (landmarkCount - 1) / 2     // 210
    public static let featureDim = poseDim + motionDim + jcdDim + landmarkCount + landmarkCount // 326

    private static let wrist = 0
    private static let indexMCP = 5
    private static let middleMCP = 9
    private static let pinkyMCP = 17
    private static let eps: Float = 1e-6

    public static func preprocess(_ frames: [LandmarkFrame], targetLength: Int = Self.targetLength) -> PreprocessedGestureWindow {
        let sampled = resample(frames, targetLength: targetLength)
        var output = [Float]()
        output.reserveCapacity(targetLength * featureDim)

        var prevCentroid = SIMD2<Float>(repeating: 0)
        var prevWrist = SIMD2<Float>(repeating: 0)
        var prevHandSize: Float = 0

        for index in 0..<sampled.count {
            let frame = sampled[index]
            let pts = frame.points
            let wristPoint = pts[wrist]
            let scale = max(palmScale(pts), eps)
            var row = [Float]()
            row.reserveCapacity(featureDim)

            // pose (63): wrist-centered, palm-scaled, all 3 coords
            for point in pts {
                let n = (point - wristPoint) / scale
                row.append(n.x); row.append(n.y); row.append(n.z)
            }

            // motion (11): global image-plane motion (raw xy, not palm-scaled)
            let centroid = pts.reduce(SIMD2<Float>(repeating: 0)) { $0 + SIMD2<Float>($1.x, $1.y) } / Float(pts.count)
            let wristXY = SIMD2<Float>(wristPoint.x, wristPoint.y)
            let handSize = scale
            let centroidDelta = index == 0 ? SIMD2<Float>(repeating: 0) : centroid - prevCentroid
            let wristDelta = index == 0 ? SIMD2<Float>(repeating: 0) : wristXY - prevWrist
            let handSizeDelta = index == 0 ? Float(0) : handSize - prevHandSize
            row.append(contentsOf: [
                centroid.x, centroid.y,
                wristXY.x, wristXY.y,
                centroidDelta.x, centroidDelta.y,
                wristDelta.x, wristDelta.y,
                handSize, handSizeDelta,
                frame.confidence,
            ])

            // jcd (210): upper-triangle pairwise xy distances / palm_scale
            for i in 0..<landmarkCount {
                let pi = SIMD2<Float>(pts[i].x, pts[i].y)
                for j in (i + 1)..<landmarkCount {
                    let pj = SIMD2<Float>(pts[j].x, pts[j].y)
                    row.append(simd_distance(pi, pj) / scale)
                }
            }

            // slow_motion (21): per-joint |p_t - p_{t-1}|.xy / palm_scale, zero on first frame
            for j in 0..<landmarkCount {
                if index >= 1 {
                    let a = SIMD2<Float>(pts[j].x, pts[j].y)
                    let b = SIMD2<Float>(sampled[index - 1].points[j].x, sampled[index - 1].points[j].y)
                    row.append(simd_distance(a, b) / scale)
                } else {
                    row.append(0)
                }
            }
            // fast_motion (21): per-joint |p_t - p_{t-2}|.xy / palm_scale, zero on first two frames
            for j in 0..<landmarkCount {
                if index >= 2 {
                    let a = SIMD2<Float>(pts[j].x, pts[j].y)
                    let b = SIMD2<Float>(sampled[index - 2].points[j].x, sampled[index - 2].points[j].y)
                    row.append(simd_distance(a, b) / scale)
                } else {
                    row.append(0)
                }
            }

            // frames without a detected hand are zeroed (matches features[~mask] = 0)
            if !frame.valid {
                for k in 0..<row.count { row[k] = 0 }
            }
            output.append(contentsOf: row)

            prevCentroid = centroid
            prevWrist = wristXY
            prevHandSize = handSize
        }
        return PreprocessedGestureWindow(features: output, shape: (targetLength, featureDim))
    }

    static func resample(_ frames: [LandmarkFrame], targetLength: Int) -> [LandmarkFrame] {
        guard let first = frames.first else {
            let zero = LandmarkFrame(points: Array(repeating: SIMD3<Float>(repeating: 0), count: landmarkCount), confidence: 0, valid: false)
            return Array(repeating: zero, count: targetLength)
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
