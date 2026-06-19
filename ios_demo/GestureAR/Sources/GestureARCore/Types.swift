// Core data types for camera input.
import Foundation
import AVFoundation

public struct MediaPipeFrame {
    public let buffer: CVPixelBuffer
    public let hasHand: Bool
    public let timestamp: Int

    public init(buffer: CVPixelBuffer, hasHand: Bool, timestamp: Int) {
        self.buffer = buffer
        self.hasHand = hasHand
        self.timestamp = timestamp
    }
}
