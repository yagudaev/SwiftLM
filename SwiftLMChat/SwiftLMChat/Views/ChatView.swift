// ChatView.swift — Main chat interface (iOS + macOS)
import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @EnvironmentObject private var engine: InferenceEngine

    // macOS-only sheet control (iOS: these are tabs now)
    var showSettings: Binding<Bool>? = nil
    var showModelPicker: Binding<Bool>? = nil

    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // ── Message list or empty state ─────────────────────────────
            ScrollViewReader { proxy in
                ScrollView {
                    if viewModel.messages.isEmpty && viewModel.streamingText.isEmpty {
                        emptyStateView
                            .frame(maxWidth: .infinity)
                            .padding(.top, 60)
                    } else {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                            if !viewModel.streamingText.isEmpty || viewModel.thinkingText != nil {
                                StreamingBubble(
                                    text: viewModel.streamingText,
                                    thinkingText: viewModel.thinkingText
                                )
                                .id("streaming")
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding()
                        .padding(.bottom, 4)
                    }
                }
                // Swipe the message list down to dismiss keyboard —
                // this reveals the tab bar so user can navigate while typing.
                .scrollDismissesKeyboard(.interactively)
                // Tap anywhere in the message area to dismiss the keyboard
                .onTapGesture { inputFocused = false }
                .onChange(of: viewModel.streamingText) { _, _ in
                    withAnimation(.easeOut(duration: 0.1)) {
                        proxy.scrollTo("bottom")
                    }
                }
            }

            // ── Engine state banner (loading / error) ────────────────────
            engineBanner

            Divider()

            // ── Input bar ────────────────────────────────────────────────
            inputBar
        }
        .navigationTitle("SwiftLM Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { iOSToolbar }
        #else
        .toolbar { macOSToolbar }
        #endif
    }

    // MARK: — Empty State

    @ViewBuilder
    private var emptyStateView: some View {
        switch engine.state {
        case .downloading(let progress, let speed):
            VStack(spacing: 16) {
                ZStack {
                    Circle()
                        .stroke(Color.blue.opacity(0.15), lineWidth: 6)
                        .frame(width: 72, height: 72)
                    Circle()
                        .trim(from: 0, to: progress)
                        .stroke(Color.blue, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 72, height: 72)
                        .animation(.linear(duration: 0.3), value: progress)
                    Text("\(Int(progress * 100))%")
                        .font(.caption.monospacedDigit().weight(.semibold))
                }
                VStack(spacing: 4) {
                    Text("Downloading model…")
                        .font(.headline)
                    Text(speed.isEmpty ? "Preparing…" : speed)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("You'll be able to chat once the download completes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
        case .loading:
            VStack(spacing: 12) {
                ProgressView().controlSize(.large)
                Text("Loading model into Metal GPU…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        case .idle:
            VStack(spacing: 12) {
                Image(systemName: "cpu")
                    .font(.system(size: 48))
                    .foregroundStyle(.tertiary)
                Text("No model loaded")
                    .font(.headline)
                Text("Go to the Models tab to download and select a model.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
        case .error(let msg):
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 48))
                    .foregroundStyle(.red)
                Text("Load failed")
                    .font(.headline)
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
        case .ready, .generating:
            VStack(spacing: 12) {
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 48))
                    .foregroundStyle(.tertiary)
                Text("Start a conversation")
                    .font(.headline)
                Text("Type a message below to begin.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: — Engine State Banner

    @ViewBuilder
    private var engineBanner: some View {
        switch engine.state {
        case .idle:
            HStack(spacing: 8) {
                Image(systemName: "cpu").foregroundStyle(.secondary)
                Text("No model loaded — tap Models to get started")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
        case .loading:
            HStack(spacing: 8) {
                ProgressView().controlSize(.mini)
                Text("Loading model…")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
        case .downloading(let progress, let speed):
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Downloading…")
                        .font(.caption.weight(.medium))
                    Spacer()
                    Text("\(Int(progress * 100))% · \(speed)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: progress).tint(.blue)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
        case .error(let msg):
            HStack(spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                Text(msg).font(.caption).foregroundStyle(.red).lineLimit(2)
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
        case .ready, .generating:
            EmptyView()
        }
    }

    // MARK: — Input Bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(.systemGray).opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .lineLimit(1...8)
                .focused($inputFocused)
                .onSubmit {
                    #if os(macOS)
                    sendMessage()
                    #endif
                }
                .disabled(!engine.state.canSend)

            if viewModel.isGenerating {
                Button(action: viewModel.stopGeneration) {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(canSend ? .blue : .gray)
                }
                .buttonStyle(.plain)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial)
    }

    private var canSend: Bool {
        !inputText.trimmingCharacters(in: .whitespaces).isEmpty && engine.state.canSend
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !viewModel.isGenerating else { return }
        inputText = ""
        Task { await viewModel.send(text) }
    }

    // MARK: — Toolbars

    #if os(iOS)
    @ToolbarContentBuilder
    private var iOSToolbar: some ToolbarContent {
        // Model status pill (center)
        ToolbarItem(placement: .principal) {
            modelStatusPill
        }
        // Dismiss keyboard when focused — lets user reach the tab bar
        ToolbarItem(placement: .topBarLeading) {
            if inputFocused {
                Button {
                    inputFocused = false
                } label: {
                    Image(systemName: "keyboard.chevron.compact.down")
                }
                .transition(.opacity)
            }
        }
        // New conversation
        ToolbarItem(placement: .topBarTrailing) {
            Button {
                viewModel.newConversation()
            } label: {
                Image(systemName: "square.and.pencil")
            }
        }
    }

    private var modelStatusPill: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(engine.state.statusColor)
                .frame(width: 7, height: 7)
            Text(engine.state.shortLabel)
                .font(.caption.weight(.medium))
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(.regularMaterial, in: Capsule())
    }
    #endif

    #if os(macOS)
    @ToolbarContentBuilder
    private var macOSToolbar: some ToolbarContent {
        ToolbarItem {
            Button { viewModel.newConversation() } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
        }
        ToolbarItem {
            Button { showSettings?.wrappedValue = true } label: {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
    }
    #endif
}

// MARK: — ModelState Extensions

extension ModelState {
    var canSend: Bool {
        if case .ready = self { return true }
        return false
    }

    var statusColor: Color {
        switch self {
        case .idle:                        return .gray
        case .loading, .downloading:       return .orange
        case .ready:                       return .green
        case .generating:                  return .blue
        case .error:                       return .red
        }
    }

    var shortLabel: String {
        switch self {
        case .idle:                          return "No model"
        case .loading:                       return "Loading…"
        case .downloading(let p, _):         return "\(Int(p * 100))% downloading"
        case .ready(let modelId):            return modelId.components(separatedBy: "/").last ?? modelId
        case .generating:                    return "Generating…"
        case .error:                         return "Error"
        }
    }
}
