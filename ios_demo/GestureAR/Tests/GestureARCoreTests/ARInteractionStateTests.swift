import XCTest
@testable import GestureARCore

final class ARInteractionStateTests: XCTestCase {
    func testInitialState() {
        let s = ARInteractionState(objectCount: 4)
        XCTAssertEqual(s.focusedIndex, 0)
        XCTAssertEqual(s.focusedScale, 1.0)
        XCTAssertNil(s.activatedIndex)
        XCTAssertFalse(s.isPointing)
    }

    func testNavigationWrapsBothWays() {
        var s = ARInteractionState(objectCount: 3)
        s.apply(.navigateNext); XCTAssertEqual(s.focusedIndex, 1)
        s.apply(.navigateNext); s.apply(.navigateNext); XCTAssertEqual(s.focusedIndex, 0)  // wrap
        s.apply(.navigatePrevious); XCTAssertEqual(s.focusedIndex, 2)                       // wrap back
    }

    func testZoomClampsWithinBounds() {
        var s = ARInteractionState(objectCount: 2, minScale: 0.5, maxScale: 2.5, zoomStep: 1.15)
        for _ in 0..<50 { s.apply(.zoomIn) }
        XCTAssertEqual(s.focusedScale, 2.5, accuracy: 1e-5)
        for _ in 0..<100 { s.apply(.zoomOut) }
        XCTAssertEqual(s.focusedScale, 0.5, accuracy: 1e-5)
    }

    func testNavigationResetsScale() {
        var s = ARInteractionState(objectCount: 3)
        s.apply(.zoomIn)
        XCTAssertGreaterThan(s.focusedScale, 1.0)
        s.apply(.navigateNext)
        XCTAssertEqual(s.focusedScale, 1.0)
    }

    func testSelectToggles() {
        var s = ARInteractionState(objectCount: 3)
        s.apply(.selectConfirm); XCTAssertEqual(s.activatedIndex, 0); XCTAssertTrue(s.isFocusedActivated)
        s.apply(.selectConfirm); XCTAssertNil(s.activatedIndex)
    }

    func testPointerHoverSetsFlagWithoutMoving() {
        var s = ARInteractionState(objectCount: 3)
        s.apply(.navigateNext)
        let changed = s.apply(.pointerHover)
        XCTAssertTrue(s.isPointing)
        XCTAssertEqual(s.focusedIndex, 1)
        XCTAssertTrue(changed)  // isPointing flipped false->true
    }
}
