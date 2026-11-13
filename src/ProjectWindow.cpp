#include "ProjectWindow.h"

#include "SpectrogramView.h"
#include "Dsp.h"
#include "PeakAnalysis.h"
#include "OplFit.h"
#include "SineFit.h"
#include "WavWriter.h"
#include "ChipVgmWriter.h"
#include "MidiWriter.h"
#include "chips/MidiTarget.h"
#include "AppSettings.h"

#include <wx/sizer.h>
#include <wx/filename.h>
#include <wx/progdlg.h>
#include <wx/spinctrl.h>
#include <wx/stattext.h>
#include <wx/button.h>
#include <wx/radiobox.h>
#include <wx/choice.h>
#include <wx/sound.h>
#include <wx/stdpaths.h>
#include <wx/filefn.h>
#include <wx/filedlg.h>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <thread>

namespace
{

// The Chip tab's dropdown. OPL4's entry is refined by the separate OPL4
// mode dropdown (see kOpl4Modes); its id here is just a placeholder.
struct ChipChoice
{
    const char* label;
    chips::ChipId id;
};
constexpr ChipChoice kChipChoices[] = {
    {"OPL3 (YMF262)", chips::ChipId::Opl3},
    {"OPL2 (YM3812)", chips::ChipId::Opl2},
    {"OPL4 (YMF278B)", chips::ChipId::Opl4Fm},
    {"OPM (YM2151)", chips::ChipId::Opm},
    {"OPN (YM2203)", chips::ChipId::Opn},
    {"OPNA (YM2608)", chips::ChipId::Opna},
    {"OPN2 (YM2612)", chips::ChipId::Opn2},
    {"HuC6280", chips::ChipId::HuC6280},
    {"VSU (Virtual Boy)", chips::ChipId::Vsu},
    {"MultiPCM (YMW258)", chips::ChipId::MultiPcm},
    {"MIDI (GM Ocarina)", chips::ChipId::Midi},
};
constexpr int kOpl4ChoiceIndex = 2;

struct Opl4Mode
{
    const char* label;
    chips::ChipId id;
};
constexpr Opl4Mode kOpl4Modes[] = {
    {"FM (18)", chips::ChipId::Opl4Fm},
    {"PCM (24)", chips::ChipId::Opl4Pcm},
    {"FM + PCM (42)", chips::ChipId::Opl4Both},
};

} // namespace

ProjectWindow::ProjectWindow(wxMDIParentFrame* parent)
    : wxMDIChildFrame(parent, wxID_ANY, "Project")
{
    m_opl3PlaybackTimer.SetOwner(this);
    Bind(wxEVT_TIMER, &ProjectWindow::OnOpl3PlaybackTimer, this, m_opl3PlaybackTimer.GetId());
    SetupUi();
}

ProjectWindow::~ProjectWindow()
{
    delete m_opl3PreviewSound;
    if (!m_opl3PreviewWavPath.empty() && wxFileExists(m_opl3PreviewWavPath))
        wxRemoveFile(m_opl3PreviewWavPath);
}

bool ProjectWindow::LoadImportedFile(const std::string& path)
{
    wxProgressDialog progressDlg("Importing Audio", "Starting import...", 100, this,
                                  wxPD_APP_MODAL | wxPD_AUTO_HIDE | wxPD_ELAPSED_TIME);

    std::string err;
    bool ok = m_project.ImportFromFile(path, err,
        [&progressDlg](int percent, const std::string& stage) {
            progressDlg.Update(std::clamp(percent, 0, 100), stage);
        });
    if (!ok)
    {
        wxMessageBox(err, "Import failed", wxICON_ERROR);
        return false;
    }

    RefreshViews();
    SetTitle(wxFileName(path).GetName());
    return true;
}

const AudioProject& ProjectWindow::GetProject() const
{
    return m_project;
}

AudioProject& ProjectWindow::GetProject()
{
    return m_project;
}

void ProjectWindow::SetupUi()
{
    m_notebook = new wxNotebook(this, wxID_ANY);

    // --- "Chip" tab: page 0, selected by default (see RefreshViews) ---
    wxPanel* opl3Page = new wxPanel(m_notebook);
    wxBoxSizer* opl3PageSizer = new wxBoxSizer(wxVERTICAL);

    m_opl3ControlsPanel = new wxPanel(opl3Page);
    wxBoxSizer* opl3Sizer = new wxBoxSizer(wxHORIZONTAL);

    opl3Sizer->Add(new wxStaticText(m_opl3ControlsPanel, wxID_ANY, "Chip:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    wxArrayString chipLabels;
    for (const ChipChoice& c : kChipChoices)
        chipLabels.Add(c.label);
    m_chipChoice = new wxChoice(m_opl3ControlsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, chipLabels);
    opl3Sizer->Add(m_chipChoice, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    wxArrayString opl4Labels;
    for (const Opl4Mode& m : kOpl4Modes)
        opl4Labels.Add(m.label);
    m_opl4ModeChoice = new wxChoice(m_opl3ControlsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, opl4Labels);
    opl3Sizer->Add(m_opl4ModeChoice, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    m_midiBendLabel = new wxStaticText(m_opl3ControlsPanel, wxID_ANY, "Bend range (semitones):");
    opl3Sizer->Add(m_midiBendLabel, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_midiBendRangeSpin = new wxSpinCtrl(m_opl3ControlsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition,
                                         wxDefaultSize, wxSP_ARROW_KEYS, chips::MidiTarget::kMinBendRange,
                                         chips::MidiTarget::kMaxBendRange, chips::MidiTarget::kDefaultBendRange);
    opl3Sizer->Add(m_midiBendRangeSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    const settings::AppSettings& saved = settings::Get();
    m_chipChoice->SetSelection(std::clamp(saved.targetChip, 0, static_cast<int>(std::size(kChipChoices)) - 1));
    m_opl4ModeChoice->SetSelection(std::clamp(saved.opl4Mode, 0, static_cast<int>(std::size(kOpl4Modes)) - 1));
    m_midiBendRangeSpin->SetValue(std::clamp(saved.midiBendRange, chips::MidiTarget::kMinBendRange,
                                             chips::MidiTarget::kMaxBendRange));

    opl3Sizer->Add(new wxStaticText(m_opl3ControlsPanel, wxID_ANY, "Frame rate (Hz):"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_opl3FrameRateSpin = new wxSpinCtrl(m_opl3ControlsPanel, wxID_ANY, wxEmptyString,
                                          wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                          static_cast<int>(analysis::kMinFrameRateHz),
                                          static_cast<int>(analysis::kMaxFrameRateHz),
                                          static_cast<int>(oplfit::kDefaultInteractiveFrameRateHz));
    opl3Sizer->Add(m_opl3FrameRateSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    opl3Sizer->Add(new wxStaticText(m_opl3ControlsPanel, wxID_ANY, "Max channels:"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_opl3MaxChannelsSpin = new wxSpinCtrl(m_opl3ControlsPanel, wxID_ANY, wxEmptyString,
                                            wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                            oplfit::kMinMaxChannels, oplfit::kMaxMaxChannels,
                                            oplfit::kDefaultMaxChannels);
    opl3Sizer->Add(m_opl3MaxChannelsSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    // Three fitting workflows live side by side in SineFit.h for direct
    // comparison -- see the doc comments on FitChannelsInTurn,
    // FitChannelsPerFrame and FitChannelsByChains for what each trades
    // away. Per-frame stays the default (previous behaviour); "Chains" is
    // the newest and generally the closest fit, at fewer re-attacks.
    const wxString algorithmChoices[] = {"Trend-based", "Per-frame", "Chains"};
    m_opl3AlgorithmRadio = new wxRadioBox(m_opl3ControlsPanel, wxID_ANY, "Fitting",
                                           wxDefaultPosition, wxDefaultSize, 3, algorithmChoices,
                                           3, wxRA_SPECIFY_COLS);
    m_opl3AlgorithmRadio->SetSelection(1); // Per-frame
    opl3Sizer->Add(m_opl3AlgorithmRadio, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    // Fits up to Max channels channels against the whole recording, using
    // whichever workflow the radio box above selects (see
    // sinefit::FitChannelsInTurn / sinefit::FitChannelsPerFrame).
    wxButton* startButton = new wxButton(m_opl3ControlsPanel, wxID_ANY, "Make");
    opl3Sizer->Add(startButton, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    m_opl3PlayButton = new wxButton(m_opl3ControlsPanel, wxID_ANY, "Play");
    opl3Sizer->Add(m_opl3PlayButton, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    m_exportButton = new wxButton(m_opl3ControlsPanel, wxID_ANY, "Export VGM...");
    opl3Sizer->Add(m_exportButton, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    const wxString visibilityChoices[] = {"Hidden", "Visible", "Difference"};
    m_opl3VisibilityRadio = new wxRadioBox(m_opl3ControlsPanel, wxID_ANY, "Overlay",
                                            wxDefaultPosition, wxDefaultSize, 3, visibilityChoices,
                                            3, wxRA_SPECIFY_COLS);
    opl3Sizer->Add(m_opl3VisibilityRadio, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    m_opl3ControlsPanel->SetSizer(opl3Sizer);
    m_chipChoice->Bind(wxEVT_CHOICE, &ProjectWindow::OnChipChanged, this);
    m_opl4ModeChoice->Bind(wxEVT_CHOICE, &ProjectWindow::OnChipChanged, this);
    m_opl3VisibilityRadio->Bind(wxEVT_RADIOBOX, &ProjectWindow::OnOpl3VisibilityChanged, this);
    startButton->Bind(wxEVT_BUTTON, &ProjectWindow::OnOpl3Start, this);
    m_opl3PlayButton->Bind(wxEVT_BUTTON, &ProjectWindow::OnOpl3PlayToggle, this);
    m_exportButton->Bind(wxEVT_BUTTON, &ProjectWindow::OnOpl3ExportVgm, this);
    m_midiBendRangeSpin->Bind(wxEVT_SPINCTRL, &ProjectWindow::OnMidiBendRangeChanged, this);

    m_opl3View = new SpectrogramView(opl3Page);
    m_opl3View->SetAnalysisMode(SpectrogramView::AnalysisMode::Opl3);

    opl3PageSizer->Add(m_opl3ControlsPanel, 0, wxEXPAND);
    opl3PageSizer->Add(m_opl3View, 1, wxEXPAND);
    opl3Page->SetSizer(opl3PageSizer);
    m_notebook->AddPage(opl3Page, "Chip");
    ApplyChipSelection();

    // --- "Peak Trends" tab: page 1 ---
    wxPanel* peakPage = new wxPanel(m_notebook);
    wxBoxSizer* peakPageSizer = new wxBoxSizer(wxVERTICAL);

    m_peakControlsPanel = new wxPanel(peakPage);
    wxBoxSizer* peakSizer = new wxBoxSizer(wxHORIZONTAL);
    peakSizer->Add(new wxStaticText(m_peakControlsPanel, wxID_ANY, "Frame rate (Hz):"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_frameRateSpin = new wxSpinCtrl(m_peakControlsPanel, wxID_ANY, wxEmptyString,
                                      wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                      static_cast<int>(analysis::kMinFrameRateHz),
                                      static_cast<int>(analysis::kMaxFrameRateHz),
                                      static_cast<int>(analysis::kDefaultFrameRateHz));
    peakSizer->Add(m_frameRateSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    peakSizer->Add(new wxStaticText(m_peakControlsPanel, wxID_ANY, "Analysis rate (Hz):"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_analysisRateSpin = new wxSpinCtrl(m_peakControlsPanel, wxID_ANY, wxEmptyString,
                                         wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                         static_cast<int>(analysis::kMinAnalysisRateHz),
                                         static_cast<int>(analysis::kMaxAnalysisRateHz),
                                         static_cast<int>(analysis::kDefaultAnalysisRateHz));
    peakSizer->Add(m_analysisRateSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    peakSizer->Add(new wxStaticText(m_peakControlsPanel, wxID_ANY, "Max peaks:"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_maxPeaksSpin = new wxSpinCtrl(m_peakControlsPanel, wxID_ANY, wxEmptyString,
                                     wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                     analysis::kMinPeaksPerStep, analysis::kMaxPeaksPerStep,
                                     analysis::kDefaultPeaksPerStep);
    peakSizer->Add(m_maxPeaksSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    peakSizer->Add(new wxStaticText(m_peakControlsPanel, wxID_ANY, "Peak window (+/- bins):"),
                    0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    m_peakWindowSpin = new wxSpinCtrl(m_peakControlsPanel, wxID_ANY, wxEmptyString,
                                       wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS,
                                       analysis::kMinPeakWindowHalfWidth, analysis::kMaxPeakWindowHalfWidth,
                                       analysis::kDefaultPeakWindowHalfWidth);
    peakSizer->Add(m_peakWindowSpin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);

    m_peakControlsPanel->SetSizer(peakSizer);
    m_frameRateSpin->Bind(wxEVT_SPINCTRL, &ProjectWindow::OnPeakControlChanged, this);
    m_analysisRateSpin->Bind(wxEVT_SPINCTRL, &ProjectWindow::OnPeakControlChanged, this);
    m_maxPeaksSpin->Bind(wxEVT_SPINCTRL, &ProjectWindow::OnPeakControlChanged, this);
    m_peakWindowSpin->Bind(wxEVT_SPINCTRL, &ProjectWindow::OnPeakControlChanged, this);

    m_peakTrendsView = new SpectrogramView(peakPage);
    m_peakTrendsView->SetAnalysisMode(SpectrogramView::AnalysisMode::PeakTrends);

    peakPageSizer->Add(m_peakControlsPanel, 0, wxEXPAND);
    peakPageSizer->Add(m_peakTrendsView, 1, wxEXPAND);
    peakPage->SetSizer(peakPageSizer);
    m_notebook->AddPage(peakPage, "Peak Trends");

    m_notebook->Bind(wxEVT_NOTEBOOK_PAGE_CHANGING, &ProjectWindow::OnPageChanging, this);
    m_notebook->Bind(wxEVT_NOTEBOOK_PAGE_CHANGED, &ProjectWindow::OnPageChanged, this);

    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    mainSizer->Add(m_notebook, 1, wxEXPAND | wxALL, 4);
    SetSizer(mainSizer);
}

void ProjectWindow::RefreshViews()
{
    m_sharedZoom = 1.0;
    m_sharedScrollSeconds = 0.0;
    m_peaksComputedAtFrameRate = -1.0; // new samples -- any previously computed peaks are stale
    m_peaksComputedAtAnalysisRate = -1.0;
    m_peaksComputedAtMaxPeaks = -1;
    m_peaksComputedAtWindowHalfWidth = -1;
    m_chipFit = sinefit::ChipFit{};
    m_opl3CumulativeRendered.clear();

    const double secondsPerColumn = static_cast<double>(dsp::kHopSize) / std::max(1, m_project.SampleRate());
    m_opl3View->SetImageData(m_project.SpectrogramWidth(), m_project.SpectrogramHeight(),
                              m_project.SpectrogramRGB(), secondsPerColumn, m_project.SampleRate());
    // SetImageData resets the vertical range to full-scale for every view
    // (new data invalidates any previous framing) -- re-apply the chip's
    // own default now that there's real data to look at.
    ApplyChipVRange();
    m_opl3View->SetZoomAndScroll(m_sharedZoom, m_sharedScrollSeconds);
    m_peakTrendsView->SetImageData(m_project.SpectrogramWidth(), m_project.SpectrogramHeight(),
                                    m_project.SpectrogramRGB(), secondsPerColumn, m_project.SampleRate());
    m_peakTrendsView->SetZoomAndScroll(m_sharedZoom, m_sharedScrollSeconds);

    m_notebook->SetSelection(0); // Chip -- the default tab for a freshly-imported file
}

chips::ChipId ProjectWindow::SelectedChipId() const
{
    const int chip = std::clamp(m_chipChoice->GetSelection(), 0, static_cast<int>(std::size(kChipChoices)) - 1);
    if (chip == kOpl4ChoiceIndex)
    {
        const int mode = std::clamp(m_opl4ModeChoice->GetSelection(), 0, static_cast<int>(std::size(kOpl4Modes)) - 1);
        return kOpl4Modes[mode].id;
    }
    return kChipChoices[chip].id;
}

std::shared_ptr<const chips::ChipTarget> ProjectWindow::SelectedTarget() const
{
    const chips::ChipId id = SelectedChipId();
    if (id == chips::ChipId::Midi)
        return chips::CreateMidiTarget(m_midiBendRangeSpin->GetValue());
    return chips::CreateTarget(id);
}

void ProjectWindow::OnMidiBendRangeChanged(wxSpinEvent& WXUNUSED(evt))
{
    const int range = m_midiBendRangeSpin->GetValue();
    settings::Get().midiBendRange = range;
    settings::Save();

    // The bend range only affects how a fit is written out (pitch codes are
    // absolute notes), so an existing MIDI fit picks up the new range
    // without being refitted.
    if (m_chipFit.target && m_chipFit.target->Id() == chips::ChipId::Midi)
        m_chipFit.target = chips::CreateMidiTarget(range);
}

void ProjectWindow::ApplyChipSelection()
{
    const bool isMidi = SelectedChipId() == chips::ChipId::Midi;
    m_opl4ModeChoice->Show(m_chipChoice->GetSelection() == kOpl4ChoiceIndex);
    m_midiBendLabel->Show(isMidi);
    m_midiBendRangeSpin->Show(isMidi);
    m_exportButton->SetLabel(isMidi ? "Export MIDI..." : "Export VGM...");
    const auto target = SelectedTarget();
    const int voices = target ? target->VoiceCount() : 1;
    m_opl3MaxChannelsSpin->SetRange(1, voices);
    // Every real voice by default -- except MIDI, which leaves channel 10
    // (the drum channel on most GM synths) out unless the user asks for it.
    m_opl3MaxChannelsSpin->SetValue(isMidi ? std::min(chips::MidiTarget::kDefaultVoices, voices) : voices);
    m_opl3ControlsPanel->Layout();
    ApplyChipVRange();
}

void ProjectWindow::ApplyChipVRange()
{
    double topHz = 8192.0;
    if (const auto target = SelectedTarget())
        for (const chips::VoiceKind& kind : target->Kinds())
            topHz = std::max(topHz, kind.maxHz * 1.25);
    const double nyquist = std::max(1, m_project.SampleRate()) / 2.0;
    if (m_project.SampleRate() > 0)
        topHz = std::min(topHz, nyquist);
    m_opl3View->SetVRangeHz(0.0, topHz);
}

void ProjectWindow::OnChipChanged(wxCommandEvent& WXUNUSED(evt))
{
    settings::AppSettings& s = settings::Get();
    s.targetChip = m_chipChoice->GetSelection();
    s.opl4Mode = m_opl4ModeChoice->GetSelection();
    settings::Save();
    ApplyChipSelection();
}

void ProjectWindow::OnPageChanging(wxNotebookEvent& evt)
{
    // Capture the outgoing tab's current (horizontal, time-axis) zoom/
    // scroll before it's hidden, so the tab we're switching to can be
    // brought into sync with it.
    switch (evt.GetOldSelection())
    {
    case 0:
        m_sharedZoom = m_opl3View->GetZoom();
        m_sharedScrollSeconds = m_opl3View->GetScrollSeconds();
        break;
    case 1:
        m_sharedZoom = m_peakTrendsView->GetZoom();
        m_sharedScrollSeconds = m_peakTrendsView->GetScrollSeconds();
        break;
    default:
        break;
    }
    evt.Skip();
}

void ProjectWindow::OnPageChanged(wxNotebookEvent& evt)
{
    switch (evt.GetSelection())
    {
    case 0:
        m_opl3View->SetZoomAndScroll(m_sharedZoom, m_sharedScrollSeconds);
        break;
    case 1:
        m_peakTrendsView->SetZoomAndScroll(m_sharedZoom, m_sharedScrollSeconds);
        EnsurePeaksComputed();
        break;
    default:
        break;
    }
    evt.Skip();
}

void ProjectWindow::OnPeakControlChanged(wxSpinEvent& WXUNUSED(evt))
{
    EnsurePeaksComputed();
}

void ProjectWindow::OnOpl3VisibilityChanged(wxCommandEvent& WXUNUSED(evt))
{
    const int sel = m_opl3VisibilityRadio->GetSelection();
    const auto visibility = static_cast<SpectrogramView::Opl3OverlayVisibility>(
        std::clamp(sel, 0, 2));
    m_opl3View->SetOpl3OverlayVisibility(visibility);
}

void ProjectWindow::OnOpl3Start(wxCommandEvent& WXUNUSED(evt))
{
    if (m_project.Samples().empty())
        return;

    const double frameRateHz = static_cast<double>(m_opl3FrameRateSpin->GetValue());
    const int maxChannels = m_opl3MaxChannelsSpin->GetValue();

    // Discards any previous fit and starts fresh, using whichever of
    // SineFit.h's three side-by-side workflows the radio box selects (see
    // their doc comments for what actually differs between them).
    const int algorithm = m_opl3AlgorithmRadio->GetSelection();
    const auto target = SelectedTarget();
    if (!target)
        return;
    wxProgressDialog progressDlg("Analyzing", "Fitting " + target->Name() + " channels...", 100, this,
                                  wxPD_APP_MODAL | wxPD_AUTO_HIDE | wxPD_ELAPSED_TIME);
    auto progressCallback = [&progressDlg](int percent) { progressDlg.Update(std::clamp(percent, 0, 100)); };
    const auto started = std::chrono::steady_clock::now();
    sinefit::ChipFit fit = (algorithm == 2)
        ? sinefit::FitChannelsByChains(target, m_project.Samples(), m_project.SampleRate(), frameRateHz, maxChannels,
                                        progressCallback)
        : (algorithm == 1)
        ? sinefit::FitChannelsPerFrame(target, m_project.Samples(), m_project.SampleRate(), frameRateHz, maxChannels,
                                        progressCallback)
        : sinefit::FitChannelsInTurn(target, m_project.Samples(), m_project.SampleRate(), frameRateHz, maxChannels,
                                      progressCallback);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();

    m_opl3CumulativeRendered = std::move(fit.mixRendered);
    m_chipFit = std::move(fit);

    // The fit's own quality readout -- the honest way to compare the three
    // workflows against each other on the same recording.
    if (wxFrame* frame = dynamic_cast<wxFrame*>(wxGetTopLevelParent(this)))
    {
        const sinefit::FitScore& score = m_chipFit.score;
        frame->SetStatusText(wxString::Format("%s: %zu voices, similarity %.3f, %.2f dB avg spectral error, "
                                               "%d key-ons, %d pitch jumps (%lld ms)",
                                               m_chipFit.target->Name(), m_chipFit.tracks.size(), score.similarity,
                                               score.meanAbsDbError, score.keyOnCount, score.pitchJumpCount,
                                               static_cast<long long>(elapsedMs)));
    }

    // Reuses the same progress dialog for this second phase (rebuilding
    // the comparison overlay) rather than letting it close and the app
    // go silently unresponsive for however long that takes -- it used to
    // run as a plain, un-instrumented call right here.
    progressDlg.Update(0, "Building comparison overlay...");
    RefreshOpl3ComparisonOverlay(
        [&progressDlg](int percent) { progressDlg.Update(std::clamp(percent, 0, 100), "Building comparison overlay..."); });

    // Switch straight to Visible so the freshly-computed fit is actually
    // seen without an extra manual step -- there's nothing to look at
    // yet on a Hidden overlay right after Start finishes.
    m_opl3VisibilityRadio->SetSelection(1);
    m_opl3View->SetOpl3OverlayVisibility(SpectrogramView::Opl3OverlayVisibility::Visible);
}

void ProjectWindow::OnOpl3ExportVgm(wxCommandEvent& WXUNUSED(evt))
{
    if (m_chipFit.tracks.empty())
    {
        wxMessageBox("Press 'Make' first to build an approximation to export.", "Nothing to export",
                     wxICON_INFORMATION);
        return;
    }

    // Exports what the current fit was made for, even if the dropdown has
    // since been changed.
    const bool isMidi = m_chipFit.target->Format() == chips::OutputFormat::Midi;
    const wxString kind = isMidi ? "MIDI" : "VGM";
    settings::AppSettings& s = settings::Get();
    const wxString defaultFileName = wxFileName(m_project.SourcePath()).GetName() + (isMidi ? ".mid" : ".vgm");
    wxFileDialog dlg(this, "Export " + kind, s.lastExportDir, defaultFileName,
                      isMidi ? "Standard MIDI files (*.mid)|*.mid|All files (*.*)|*.*"
                             : "VGM files (*.vgm)|*.vgm|All files (*.*)|*.*",
                      wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
    if (dlg.ShowModal() != wxID_OK)
        return;

    s.lastExportDir = dlg.GetDirectory();
    settings::Save();

    std::string err;
    const std::string path = dlg.GetPath().ToStdString();
    const bool ok = isMidi ? midi::WriteMidiFile(path, m_chipFit, err) : vgm::WriteChipVgmFile(path, m_chipFit, err);
    if (!ok)
    {
        wxMessageBox(err, "Export failed", wxICON_ERROR);
        return;
    }

    if (wxFrame* frame = dynamic_cast<wxFrame*>(wxGetTopLevelParent(this)))
        frame->SetStatusText("Exported " + m_chipFit.target->Name() + " " + kind + " to " + dlg.GetPath());
}

void ProjectWindow::RefreshOpl3ComparisonOverlay(const dsp::ProgressCallback& progress)
{
    // Precompute every frame's original-vs-cumulative-mix spectrum pair
    // up front (done once here rather than repeatedly during painting) so
    // Visible/Difference can draw the whole recording's comparison
    // directly. Frame rate/count follow the first channel, since that's
    // this recording's own grid.
    std::vector<std::vector<double>> originalDbfsPerFrame;
    std::vector<std::vector<double>> renderedDbfsPerFrame;
    double frameRateHz = 0.0;
    if (!m_chipFit.tracks.empty() && m_chipFit.frameRateHz > 0.0)
    {
        frameRateHz = m_chipFit.frameRateHz;
        const double frameDurationSeconds = 1.0 / frameRateHz;
        const size_t frameCount = m_chipFit.tracks.front().frames.size();
        originalDbfsPerFrame.assign(frameCount, {});
        renderedDbfsPerFrame.assign(frameCount, {});

        // A several-minute recording at a typical 100Hz OPL3 frame rate
        // is tens of thousands of frames, two full FFTs each -- this used
        // to run single-threaded with no progress reporting at all right
        // after Start's own progress dialog closed, which looked exactly
        // like the app had frozen. Each frame's pair of spectra is
        // completely independent of every other frame's (same read-only-
        // buffer, disjoint-output-index precondition PrecomputeSpectra/
        // ComputeStepPeaks/GenerateSpectrogram already rely on), so this
        // is safe to spread across hardware threads the same way, with
        // each worker reusing its own scratch buffers across its whole
        // frame range instead of every frame allocating fresh ones.
        auto computeRange = [&](size_t startFrame, size_t endFrame) {
            std::vector<float> origFrameScratch, renderedFrameScratch;
            std::vector<std::complex<double>> origComplexScratch, renderedComplexScratch;
            for (size_t f = startFrame; f < endFrame; ++f)
            {
                const double frameMidSeconds = (static_cast<double>(f) + 0.5) * frameDurationSeconds;
                dsp::ComputeSpectrumAtTimeInto(m_project.Samples(), m_project.SampleRate(), frameMidSeconds,
                                                origFrameScratch, origComplexScratch, originalDbfsPerFrame[f]);
                dsp::ComputeSpectrumAtTimeInto(m_opl3CumulativeRendered, m_project.SampleRate(), frameMidSeconds,
                                                renderedFrameScratch, renderedComplexScratch,
                                                renderedDbfsPerFrame[f]);
            }
        };

        const unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
        const size_t numThreads = std::min<size_t>(hwThreads, frameCount);
        if (numThreads <= 1)
        {
            computeRange(0, frameCount);
        }
        else
        {
            std::vector<std::thread> workers;
            workers.reserve(numThreads);
            const size_t chunk = (frameCount + numThreads - 1) / numThreads;
            for (size_t t = 0; t < numThreads; ++t)
            {
                const size_t start = t * chunk;
                const size_t end = std::min(frameCount, start + chunk);
                if (start >= end)
                    break;
                workers.emplace_back(computeRange, start, end);
            }
            for (std::thread& worker : workers)
                worker.join();
        }
    }
    if (progress) progress(100);

    m_opl3View->SetOpl3AllFramesComparison(std::move(originalDbfsPerFrame), std::move(renderedDbfsPerFrame),
                                            frameRateHz);
}

void ProjectWindow::OnOpl3PlayToggle(wxCommandEvent& WXUNUSED(evt))
{
    if (m_opl3IsPlaying)
    {
        StopOpl3Playback();
        return;
    }

    if (m_chipFit.tracks.empty())
    {
        wxMessageBox("Press 'Make' first.", "Nothing to play", wxICON_INFORMATION);
        return;
    }
    PlayOpl3Preview(m_opl3CumulativeRendered);
}

void ProjectWindow::PlayOpl3Preview(const std::vector<float>& samples)
{
    if (samples.empty())
        return;

    // Stop and release any previous preview before overwriting the temp
    // file it was loaded from -- wxSound reads the file fully into memory
    // at construction, so it's safe to overwrite once the old instance is
    // gone (its destructor also stops playback).
    m_opl3PlaybackTimer.Stop();
    delete m_opl3PreviewSound;
    m_opl3PreviewSound = nullptr;

    if (m_opl3PreviewWavPath.empty())
        m_opl3PreviewWavPath = (wxStandardPaths::Get().GetTempDir() + "/wav2vgm_chip_preview.wav").ToStdString();

    std::string err;
    if (!wavwriter::WriteMonoWavFile(m_opl3PreviewWavPath, samples, m_project.SampleRate(), err))
    {
        wxMessageBox(err, "Playback failed", wxICON_ERROR);
        return;
    }

    m_opl3PreviewSound = new wxSound(m_opl3PreviewWavPath);
    if (!m_opl3PreviewSound->IsOk())
    {
        wxMessageBox("Unable to load rendered preview audio.", "Playback failed", wxICON_ERROR);
        delete m_opl3PreviewSound;
        m_opl3PreviewSound = nullptr;
        return;
    }
    m_opl3PreviewSound->Play(wxSOUND_ASYNC);

    m_opl3IsPlaying = true;
    if (m_opl3PlayButton)
        m_opl3PlayButton->SetLabel("Stop");

    // wxSound has no completion event of its own -- this stands in for
    // one, armed for the rendered audio's own known duration, so the
    // button reverts on its own once playback would naturally have
    // finished (an early manual Stop cancels this timer first instead).
    const double durationSeconds = static_cast<double>(samples.size()) / std::max(1, m_project.SampleRate());
    m_opl3PlaybackTimer.StartOnce(std::max(1, static_cast<int>(std::llround(durationSeconds * 1000.0))));
}

void ProjectWindow::StopOpl3Playback()
{
    m_opl3PlaybackTimer.Stop();
    wxSound::Stop(); // stops whatever's currently playing, not just m_opl3PreviewSound specifically
    m_opl3IsPlaying = false;
    if (m_opl3PlayButton)
        m_opl3PlayButton->SetLabel("Play");
}

void ProjectWindow::OnOpl3PlaybackTimer(wxTimerEvent& WXUNUSED(evt))
{
    m_opl3IsPlaying = false;
    if (m_opl3PlayButton)
        m_opl3PlayButton->SetLabel("Play");
}

void ProjectWindow::EnsurePeaksComputed()
{
    const double frameRateHz = static_cast<double>(m_frameRateSpin->GetValue());
    const double analysisRateHz = static_cast<double>(m_analysisRateSpin->GetValue());
    const int maxPeaks = m_maxPeaksSpin->GetValue();
    const int windowHalfWidth = m_peakWindowSpin->GetValue();
    if (frameRateHz == m_peaksComputedAtFrameRate && analysisRateHz == m_peaksComputedAtAnalysisRate
        && maxPeaks == m_peaksComputedAtMaxPeaks && windowHalfWidth == m_peaksComputedAtWindowHalfWidth)
        return;

    wxProgressDialog progressDlg("Analyzing", "Finding spectral peaks...", 100, this,
                                  wxPD_APP_MODAL | wxPD_AUTO_HIDE | wxPD_ELAPSED_TIME);
    analysis::PeakTrendsResult peakTrends = analysis::ComputePeakTrends(
        m_project.Samples(), m_project.SampleRate(), frameRateHz, analysisRateHz, maxPeaks, windowHalfWidth,
        [&progressDlg](int percent) { progressDlg.Update(std::clamp(percent, 0, 100)); });

    m_peakTrendsView->SetPeakTrends(std::move(peakTrends));
    m_peaksComputedAtFrameRate = frameRateHz;
    m_peaksComputedAtAnalysisRate = analysisRateHz;
    m_peaksComputedAtMaxPeaks = maxPeaks;
    m_peaksComputedAtWindowHalfWidth = windowHalfWidth;
}
