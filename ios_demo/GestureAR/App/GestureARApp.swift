// Main iOS app: task selector + layered AR experience (camera, 3D scene, skeleton).
import GestureARCore
import SwiftUI

@main
struct GestureARApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    // Optional launch hook for UI verification: SIMCTL_CHILD_GAR_TASK=0|1|2 boots
    // straight into a task. No effect in normal use (env var absent).
    @State private var selectedTask: ARTask? = {
        if let raw = ProcessInfo.processInfo.environment["GAR_TASK"],
           let i = Int(raw), let t = ARTask(rawValue: i) { return t }
        return nil
    }()

    var body: some View {
        if let task = selectedTask {
            ARTaskView(task: task, onBack: { selectedTask = nil })
                .id(task)   // fresh view model per task entry
        } else {
            TaskSelector(selectedTask: $selectedTask)
        }
    }
}

// MARK: - Task selector

struct TaskSelector: View {
    @Binding var selectedTask: ARTask?

    var body: some View {
        VStack(spacing: 0) {
            Text("AR Gesture Control")
                .font(.system(size: 30, weight: .bold))
                .padding(.top, 48)

            Text("Rear 0.5× camera · MediaPipe hand tracking")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.top, 6)

            Spacer()

            VStack(spacing: 14) {
                ForEach(ARTask.allCases, id: \.self) { task in
                    Button { selectedTask = task } label: {
                        HStack(spacing: 14) {
                            Image(systemName: task.symbol)
                                .font(.title2)
                                .frame(width: 38)
                                .foregroundStyle(.tint)
                            VStack(alignment: .leading, spacing: 4) {
                                Text(task.title).font(.headline).foregroundStyle(.primary)
                                Text(task.subtitle).font(.caption).foregroundStyle(.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                                    .multilineTextAlignment(.leading)
                            }
                            Spacer()
                            Image(systemName: "chevron.right").foregroundStyle(.tertiary)
                        }
                        .padding(16)
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(16)
                    }
                }
            }
            .padding(.horizontal, 20)

            Spacer()

            Text("Hold the phone steady and raise your hand into view")
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
                .padding(.bottom, 36)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

// MARK: - AR task view

struct ARTaskView: View {
    let task: ARTask
    let onBack: () -> Void
    @StateObject private var viewModel = AppViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            ARViewContainer(viewModel: viewModel, task: task)
                .ignoresSafeArea()

            // Camera-unavailable / status overlay
            if !viewModel.cameraAuthorized || !viewModel.statusText.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: viewModel.cameraAuthorized ? "camera" : "camera.fill")
                        .font(.largeTitle).foregroundStyle(.white.opacity(0.8))
                    Text(viewModel.cameraAuthorized ? viewModel.statusText : "Camera access is off")
                        .font(.callout).foregroundStyle(.white.opacity(0.9))
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 40)
            }

            VStack(spacing: 0) {
                topBar
                Spacer()
                bottomBar
            }
        }
        .statusBarHidden(false)
        .onAppear { viewModel.onAppear() }
        .onDisappear { viewModel.stopCamera() }
        .onChange(of: scenePhase) { phase in
            if phase == .active { viewModel.startCamera() }
            else if phase == .background { viewModel.stopCamera() }
        }
    }

    private var topBar: some View {
        HStack(spacing: 12) {
            Button(action: onBack) {
                Image(systemName: "chevron.left").font(.headline).foregroundStyle(.white)
            }
            Text(task.title).font(.headline).foregroundStyle(.white)
            Spacer()
            HStack(spacing: 6) {
                Circle()
                    .fill(viewModel.handDetected ? Color.green : Color.white.opacity(0.4))
                    .frame(width: 9, height: 9)
                Text(viewModel.handDetected ? "Hand" : "No hand")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.95))
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(12)
    }

    private var bottomBar: some View {
        VStack(spacing: 10) {
            if let action = viewModel.lastAction {
                Text(action)
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16).padding(.vertical, 9)
                    .background(.tint, in: Capsule())
                    .transition(.scale.combined(with: .opacity))
                    .id(action + String(viewModel.skeleton.timestamp))
            }
            Text(task.hint)
                .font(.caption).foregroundStyle(.white.opacity(0.85))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 12)
        .padding(.bottom, 14)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.lastAction)
    }
}
