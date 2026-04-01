// ModelsView.swift — Unified iOS-first Models tab
// Combines: active model, downloads in progress, downloaded list, catalog, HF search
import SwiftUI

struct ModelsView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @ObservedObject var viewModel: ChatViewModel

    @State private var showHFSearch = false
    @State private var showManagement = false
    @State private var device = DeviceProfile.current

    private var dm: ModelDownloadManager { engine.downloadManager }

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0) {
                // ── 1. Active model hero card ─────────────────────────────
                activeModelCard
                    .padding(.horizontal)
                    .padding(.top, 16)
                    .padding(.bottom, 12)

                // ── 2. Active downloads (live) ────────────────────────────
                if !dm.activeDownloads.isEmpty {
                    sectionHeader("Downloading")
                    ForEach(Array(dm.activeDownloads.keys), id: \.self) { modelId in
                        if let progress = dm.activeDownloads[modelId] {
                            DownloadProgressCard(modelId: modelId, progress: progress)
                                .padding(.horizontal)
                                .padding(.bottom, 8)
                        }
                    }
                }

                // ── 3. Downloaded models ──────────────────────────────────
                if !dm.downloadedModels.isEmpty {
                    sectionHeader("Downloaded (\(dm.downloadedModels.count))")
                        .padding(.top, 4)
                    ForEach(dm.downloadedModels) { downloaded in
                        let entry = ModelCatalog.all.first(where: { $0.id == downloaded.id })
                        let isActive: Bool = {
                            if case .ready(let id) = engine.state { return id == downloaded.id }
                            return false
                        }()
                        DownloadedModelRow(
                            downloaded: downloaded,
                            entry: entry,
                            isActive: isActive,
                            onLoad: { Task { await engine.load(modelId: downloaded.id) } },
                            onDelete: { try? dm.delete(downloaded.id) }
                        )
                        .padding(.horizontal)
                        if downloaded.id != dm.downloadedModels.last?.id {
                            Divider().padding(.leading, 72)
                        }
                    }
                    .padding(.bottom, 8)
                }

                // ── 4. Catalog — recommended ──────────────────────────────
                let recommended = dm.modelsForDevice()
                    .filter { !dm.isDownloaded($0.id) }
                if !recommended.isEmpty {
                    sectionHeader("Recommended for your \(String(format: "%.0f GB", device.physicalRAMGB)) device")
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(recommended) { model in
                                CatalogCard(
                                    model: model,
                                    fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                                    onTap: { handleSelect(model.id) }
                                )
                            }
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 4)
                    }
                    .padding(.bottom, 8)
                }

                // ── 5. Catalog — all models ───────────────────────────────
                let others = ModelCatalog.all
                    .filter { model in
                        !dm.isDownloaded(model.id) &&
                        !recommended.contains(where: { $0.id == model.id })
                    }
                if !others.isEmpty {
                    sectionHeader("All Models")
                    ForEach(others) { model in
                        CatalogListRow(
                            model: model,
                            fitStatus: ModelCatalog.fitStatus(for: model, on: device),
                            onTap: { handleSelect(model.id) }
                        )
                        .padding(.horizontal)
                        if model.id != others.last?.id {
                            Divider().padding(.leading, 56)
                        }
                    }
                    .padding(.bottom, 8)
                }

                // ── 6. HuggingFace search ─────────────────────────────────
                Button {
                    showHFSearch = true
                } label: {
                    HStack {
                        Image(systemName: "magnifyingglass")
                        Text("Search HuggingFace MLX models")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .padding(.horizontal)
                .padding(.vertical, 8)

                Spacer(minLength: 32)
            }
        }
        .background(Color(.systemGroupedBackground))
        .navigationTitle("Models")
        .navigationBarTitleDisplayMode(.large)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showManagement = true
                } label: {
                    Image(systemName: "externaldrive.badge.minus")
                }
            }
        }
        .sheet(isPresented: $showHFSearch) {
            HFSearchSheet(onSelect: handleSelect)
                .environmentObject(engine)
        }
        .sheet(isPresented: $showManagement) {
            ModelManagementView()
                .environmentObject(engine)
        }
    }

    // MARK: — Helpers

    private func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(.footnote.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
            .padding(.horizontal)
            .padding(.top, 20)
            .padding(.bottom, 6)
    }

    private func handleSelect(_ modelId: String) {
        showHFSearch = false
        // Don't clear the conversation — user can still read previous messages
        // while the new model downloads. newConversation() is available from the Chat tab.
        Task { await engine.load(modelId: modelId) }
    }
}

// MARK: — Active Model Hero Card

private struct ActiveModelHeroCard: View {
    let modelId: String
    let entry: ModelEntry?
    let state: ModelState

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            // Gradient background
            LinearGradient(
                colors: [Color.teal.opacity(0.8), Color.blue.opacity(0.9)],
                startPoint: .topLeading, endPoint: .bottomTrailing
            )
            .clipShape(RoundedRectangle(cornerRadius: 16))

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Active Model")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white.opacity(0.8))
                    Spacer()
                    stateBadge
                }

                Text(entry?.displayName ?? modelId.components(separatedBy: "/").last ?? modelId)
                    .font(.title2.weight(.bold))
                    .foregroundStyle(.white)
                    .lineLimit(2)

                HStack(spacing: 12) {
                    if let entry {
                        Label(String(format: "%.1f GB RAM", entry.ramRequiredGB), systemImage: "memorychip")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.85))
                        if entry.isMoE {
                            Label("MoE", systemImage: "square.grid.3x3.fill")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.85))
                        }
                        Label(entry.quantization, systemImage: "slider.horizontal.3")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
            }
            .padding(16)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 130)
        .shadow(color: .teal.opacity(0.3), radius: 8, y: 4)
    }

    @ViewBuilder
    private var stateBadge: some View {
        switch state {
        case .ready(_):
            Label("Ready", systemImage: "checkmark.circle.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.white.opacity(0.2), in: Capsule())
        case .generating:
            Label("Generating", systemImage: "waveform")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.white.opacity(0.2), in: Capsule())
        default:
            EmptyView()
        }
    }
}

// MARK: — Active Model Card (handles all ModelState cases)

private struct ActiveModelCardView: View {
    @EnvironmentObject private var engine: InferenceEngine

    var body: some View {
        Group {
            switch engine.state {
            case .ready(let id):
                let entry = ModelCatalog.all.first(where: { $0.id == id })
                ActiveModelHeroCard(modelId: id, entry: entry, state: engine.state)
            case .generating:
                ActiveModelHeroCard(
                    modelId: engine.loadedModelId ?? "Model",
                    entry: engine.loadedModelId.flatMap { id in ModelCatalog.all.first(where: { $0.id == id }) },
                    state: engine.state
                )
            case .loading:
                loadingCard
            case .downloading(let progress, let speed):
                downloadingCard(progress: progress, speed: speed)
            case .idle, .error:
                idleCard
            }
        }
    }

    private var loadingCard: some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.regular)
            VStack(alignment: .leading, spacing: 2) {
                Text("Loading model…")
                    .font(.subheadline.weight(.semibold))
                Text("Initializing Metal GPU")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func downloadingCard(progress: Double, speed: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "arrow.down.circle.fill")
                    .foregroundStyle(.blue)
                Text("Downloading model…")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: progress).tint(.blue)
            Text(speed).font(.caption).foregroundStyle(.secondary)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var idleCard: some View {
        HStack(spacing: 12) {
            Image(systemName: "cpu")
                .font(.title2)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text("No model loaded")
                    .font(.subheadline.weight(.semibold))
                Text("Select a model below to start chatting")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

// Workaround: top-level computed property for activeModelCard
extension ModelsView {
    @ViewBuilder
    var activeModelCard: some View {
        ActiveModelCardView()
            .environmentObject(engine)
    }
}

// MARK: — Download Progress Card

private struct DownloadProgressCard: View {
    let modelId: String
    let progress: ModelDownloadProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ZStack {
                    Circle()
                        .stroke(Color.blue.opacity(0.15), lineWidth: 3)
                    Circle()
                        .trim(from: 0, to: progress.fractionCompleted)
                        .stroke(Color.blue, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .animation(.linear(duration: 0.3), value: progress.fractionCompleted)
                }
                .frame(width: 32, height: 32)

                VStack(alignment: .leading, spacing: 1) {
                    Text(modelId.components(separatedBy: "/").last ?? modelId)
                        .font(.subheadline.weight(.semibold))
                        .lineLimit(1)
                    HStack(spacing: 6) {
                        Text("\(Int(progress.fractionCompleted * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.blue)
                        if let speed = progress.speedMBps {
                            Text("·")
                                .foregroundStyle(.secondary)
                            Text(String(format: "%.1f MB/s", speed))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer()
            }
            ProgressView(value: progress.fractionCompleted)
                .tint(.blue)
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: — Downloaded Model Row

private struct DownloadedModelRow: View {
    let downloaded: DownloadedModel
    let entry: ModelEntry?
    let isActive: Bool
    let onLoad: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Button {
            if !isActive { onLoad() }
        } label: {
            HStack(spacing: 12) {
                // Icon
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(isActive ? Color.teal : Color.secondary.opacity(0.15))
                        .frame(width: 44, height: 44)
                    Image(systemName: entry?.isMoE == true ? "square.grid.3x3.fill" : "brain")
                        .font(.body)
                        .foregroundStyle(isActive ? .white : .primary)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(entry?.displayName ?? downloaded.id.components(separatedBy: "/").last ?? downloaded.id)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.primary)
                        if isActive {
                            Text("IN USE")
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(Color.teal.opacity(0.15), in: Capsule())
                                .foregroundStyle(.teal)
                        }
                    }
                    HStack(spacing: 6) {
                        Text(downloaded.displaySize)
                            .font(.caption).foregroundStyle(.secondary)
                        if let entry {
                            Text("·").foregroundStyle(.tertiary).font(.caption)
                            Text(entry.quantization).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                Spacer()

                if isActive {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.teal)
                        .font(.title3)
                } else {
                    Image(systemName: "arrow.right.circle")
                        .foregroundStyle(.secondary)
                        .font(.title3)
                }
            }
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }
}

// MARK: — Catalog Card (horizontal scroll)

private struct CatalogCard: View {
    let model: ModelEntry
    let fitStatus: ModelCatalog.FitStatus
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: model.isMoE ? "square.grid.3x3.fill" : "brain")
                        .font(.title3)
                        .foregroundStyle(fitColor)
                    Spacer()
                    fitBadge
                }

                Spacer()

                Text(model.displayName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)

                Text(String(format: "~%.0f GB RAM", model.ramRequiredGB))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Label("Download", systemImage: "arrow.down.circle")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(fitColor)
                    .padding(.top, 2)
            }
            .padding(14)
            .frame(width: 150, height: 160)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    private var fitColor: Color {
        switch fitStatus {
        case .fits: return .blue
        case .tight: return .orange
        case .requiresFlash: return .indigo
        case .tooLarge: return .red
        }
    }

    @ViewBuilder
    private var fitBadge: some View {
        switch fitStatus {
        case .fits:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
        case .tight:
            Image(systemName: "exclamationmark.circle").foregroundStyle(.orange).font(.caption)
        case .requiresFlash:
            Image(systemName: "externaldrive.badge.wifi").foregroundStyle(.indigo).font(.caption)
        case .tooLarge:
            Image(systemName: "xmark.circle").foregroundStyle(.red).font(.caption)
        }
    }
}

// MARK: — Catalog List Row (vertical list)

private struct CatalogListRow: View {
    let model: ModelEntry
    let fitStatus: ModelCatalog.FitStatus
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(fitColor.opacity(0.12))
                        .frame(width: 40, height: 40)
                    Image(systemName: model.isMoE ? "square.grid.3x3.fill" : "brain")
                        .font(.callout)
                        .foregroundStyle(fitColor)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(model.displayName)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.primary)
                        if let badge = model.badge {
                            Text(badge)
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 5).padding(.vertical, 2)
                                .background(Color.blue.opacity(0.12), in: Capsule())
                                .foregroundStyle(.blue)
                        }
                    }
                    Text("\(model.parameterSize) · \(model.quantization) · ~\(String(format: "%.0f GB", model.ramRequiredGB)) RAM")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                fitIcon
            }
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var fitColor: Color {
        switch fitStatus {
        case .fits: return .blue
        case .tight: return .orange
        case .requiresFlash: return .indigo
        case .tooLarge: return .red
        }
    }

    @ViewBuilder
    private var fitIcon: some View {
        switch fitStatus {
        case .fits:
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(.blue).font(.title3)
        case .tight:
            Image(systemName: "arrow.down.circle")
                .foregroundStyle(.orange).font(.title3)
        case .requiresFlash:
            Image(systemName: "externaldrive.badge.wifi")
                .foregroundStyle(.indigo).font(.title3)
        case .tooLarge:
            Image(systemName: "xmark.circle")
                .foregroundStyle(.red).font(.title3)
        }
    }
}

// MARK: — HF Search Sheet (extracted from ModelPickerView)

private struct HFSearchSheet: View {
    @EnvironmentObject private var engine: InferenceEngine
    @Environment(\.dismiss) private var dismiss
    let onSelect: (String) -> Void

    var body: some View {
        NavigationStack {
            HFSearchTab(onSelect: { id in
                onSelect(id)
                dismiss()
            })
            .navigationTitle("Search HuggingFace")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
