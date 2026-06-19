// swift-tools-version: 5.9
import PackageDescription

// Pure-logic core of the GestureAR iOS app: the landmark preprocessing, the
// acceptance policy, and the label/action vocabulary. It depends only on
// Foundation + simd, so it builds and runs `swift test` on macOS — the golden
// parity tests verify it matches the Python reference exactly. The CoreML model
// wrapper, camera/MediaPipe capture, and RealityKit AR view are added in the
// Xcode app target (they need iOS-only frameworks) and are documented in README.
let package = Package(
    name: "GestureAR",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "GestureARCore", targets: ["GestureARCore"]),
    ],
    targets: [
        .target(name: "GestureARCore"),
        .testTarget(
            name: "GestureARCoreTests",
            dependencies: ["GestureARCore"],
            resources: [.process("Resources")]
        ),
    ]
)
