// SettingsView.swift — Inference + appearance settings (iOS tab or macOS sheet)
import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    /// When true, the view is embedded as a tab (no Done button needed on iOS)
    var isTab: Bool = false

    var body: some View {
        Form {
            // ── Generation ────────────────────────────────────────────────
            Section {
                // Temperature
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Label("Temperature", systemImage: "thermometer.medium")
                        Spacer()
                        Text(String(format: "%.2f", viewModel.config.temperature))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .font(.callout)
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.temperature) },
                        set: { viewModel.config.temperature = Float($0) }
                    ), in: 0...2, step: 0.05)
                    .tint(.orange)
                    Text("Higher = more creative, lower = more focused")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)

                // Max Tokens
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Label("Max Tokens", systemImage: "text.word.spacing")
                        Spacer()
                        Text("\(viewModel.config.maxTokens)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .font(.callout)
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.maxTokens) },
                        set: { viewModel.config.maxTokens = Int($0) }
                    ), in: 128...8192, step: 128)
                    .tint(.blue)
                }
                .padding(.vertical, 2)

                // Top P
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Label("Top P", systemImage: "chart.bar.xaxis")
                        Spacer()
                        Text(String(format: "%.2f", viewModel.config.topP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .font(.callout)
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.topP) },
                        set: { viewModel.config.topP = Float($0) }
                    ), in: 0...1, step: 0.05)
                    .tint(.purple)
                }
                .padding(.vertical, 2)
            } header: {
                Label("Generation", systemImage: "slider.horizontal.3")
            }

            // ── Advanced ──────────────────────────────────────────────────
            Section {
                Toggle(isOn: Binding(
                    get: { viewModel.config.enableThinking },
                    set: { viewModel.config.enableThinking = $0 }
                )) {
                    VStack(alignment: .leading, spacing: 2) {
                        Label("Thinking Mode", systemImage: "brain.head.profile")
                        Text("Step-by-step reasoning for Qwen3, DeepSeek-R1, and compatible models")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } header: {
                Label("Advanced", systemImage: "gearshape.2")
            }

            // ── System Prompt ─────────────────────────────────────────────
            Section {
                TextEditor(text: $viewModel.systemPrompt)
                    .frame(minHeight: 88)
                    .font(.callout)
            } header: {
                Label("System Prompt", systemImage: "text.bubble")
            } footer: {
                Text("Injected as the system message before every conversation.")
                    .font(.caption)
            }

            // ── Reset ─────────────────────────────────────────────────────
            Section {
                Button(role: .destructive) {
                    viewModel.config = .default
                    viewModel.systemPrompt = ""
                } label: {
                    HStack {
                        Spacer()
                        Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
                        Spacer()
                    }
                }
            }

            // ── About ─────────────────────────────────────────────────────
            Section {
                LabeledContent("SwiftLM Chat", value: "1.0")
                LabeledContent("Engine", value: "MLX Swift")
                LabeledContent("Backend", value: "Metal GPU")
                LabeledContent("Platform") {
                    #if os(iOS)
                    Text("iOS / iPadOS")
                        .foregroundStyle(.secondary)
                    #else
                    Text("macOS")
                        .foregroundStyle(.secondary)
                    #endif
                }
            } header: {
                Label("About", systemImage: "info.circle")
            }
        }
        .navigationTitle("Settings")
        #if os(iOS)
        .navigationBarTitleDisplayMode(isTab ? .large : .inline)
        #endif
        .toolbar {
            // Show Done button only when presented as a sheet (macOS or modal push)
            if !isTab {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(width: 420, height: 600)
        #endif
    }
}
