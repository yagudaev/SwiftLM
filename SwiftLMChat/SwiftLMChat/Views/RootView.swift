// RootView.swift — Adaptive root layout: tab bar on iOS, sidebar on macOS
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @StateObject private var viewModel = ChatViewModel()

    // iOS: tab selection
    @State private var selectedTab: Tab = .chat

    // macOS sheets
    @State private var showModelPicker = false
    @State private var showSettings = false

    enum Tab { case chat, models, settings }

    var body: some View {
        Group {
            #if os(macOS)
            macOSLayout
                .sheet(isPresented: $showModelPicker) {
                    ModelPickerView(onSelect: { modelId in
                        showModelPicker = false
                        Task { await engine.load(modelId: modelId) }
                    })
                    .environmentObject(engine)
                }
                .sheet(isPresented: $showSettings) {
                    SettingsView(viewModel: viewModel)
                }
                .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
                    showModelPicker = true
                }
                .onAppear {
                    viewModel.engine = engine
                    if case .idle = engine.state { showModelPicker = true }
                }
                .onChange(of: engine.state) { _, state in
                    if case .idle = state { showModelPicker = true }
                }
            #else
            iOSTabView
                .onAppear { viewModel.engine = engine }
            #endif
        }
    }

    // MARK: — iOS Tab View

    #if os(iOS)
    private var iOSTabView: some View {
        TabView(selection: $selectedTab) {
            // ── Chat Tab ────────────────────────────────────────────────
            NavigationStack {
                ChatView(viewModel: viewModel)
                    .environmentObject(engine)
            }
            .tabItem {
                Label("Chat", systemImage: selectedTab == .chat
                      ? "bubble.left.and.bubble.right.fill"
                      : "bubble.left.and.bubble.right")
            }
            .tag(Tab.chat)

            // ── Models Tab ──────────────────────────────────────────────
            NavigationStack {
                ModelsView(viewModel: viewModel)
                    .environmentObject(engine)
            }
            .tabItem {
                Label("Models", systemImage: selectedTab == .models ? "cpu.fill" : "cpu")
            }
            .tag(Tab.models)
            .badge(engine.downloadManager.activeDownloads.isEmpty ? 0 : engine.downloadManager.activeDownloads.count)

            // ── Settings Tab ────────────────────────────────────────────
            NavigationStack {
                SettingsView(viewModel: viewModel, isTab: true)
            }
            .tabItem {
                Label("Settings", systemImage: selectedTab == .settings ? "gearshape.fill" : "gearshape")
            }
            .tag(Tab.settings)
        }
        // Navigate to Models tab when a model load is requested from chat
        .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
            selectedTab = .models
        }
    }
    #endif

    // MARK: — macOS Split View

    #if os(macOS)
    private var macOSLayout: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                modelStatusView
                Divider()
                List {
                    Label("New Chat", systemImage: "plus.bubble")
                        .onTapGesture { viewModel.newConversation() }
                }
                .listStyle(.sidebar)
            }
            .frame(minWidth: 200)
        } detail: {
            ChatView(viewModel: viewModel, showSettings: $showSettings, showModelPicker: $showModelPicker)
        }
        .navigationTitle("")
    }

    private var modelStatusView: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "cpu").foregroundStyle(.secondary)
                Text("SwiftLM").font(.headline)
                Spacer()
            }
            engineStateView
        }
        .padding()
    }

    @ViewBuilder
    private var engineStateView: some View {
        switch engine.state {
        case .idle:
            Button("Load Model") { showModelPicker = true }
                .buttonStyle(.borderedProminent).controlSize(.small)
        case .loading:
            Label("Loading…", systemImage: "arrow.2.circlepath")
                .font(.caption).foregroundStyle(.secondary)
        case .downloading(let progress, let speed):
            VStack(alignment: .leading, spacing: 2) {
                ProgressView(value: progress)
                Text("\(Int(progress * 100))% · \(speed)")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        case .ready(let modelId):
            Label(modelId.components(separatedBy: "/").last ?? modelId, systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green).lineLimit(1)
        case .generating:
            Label("Generating…", systemImage: "ellipsis.bubble")
                .font(.caption).foregroundStyle(.blue)
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.red).lineLimit(2)
        }
    }
    #endif
}
