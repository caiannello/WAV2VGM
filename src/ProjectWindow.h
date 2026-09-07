#pragma once

#include "AudioProject.h"
#include "SpectrogramView.h"
#include "OplFit.h"
#include <wx/wx.h>
#include <wx/notebook.h>
#include <string>

// One tab per analysis mode -- "OPL3" (selected by default) and "Peak
// Trends" -- each holding its own SpectrogramView locked to that mode for
// its whole life, plus that mode's own controls row. Selecting a tab IS
// selecting the analysis mode; there's no separate menu/toggle for it.
class ProjectWindow : public wxMDIChildFrame
{
public:
    explicit ProjectWindow(wxMDIParentFrame* parent = nullptr);
    ~ProjectWindow() override;

    bool LoadImportedFile(const std::string& path);

    const AudioProject& GetProject() const;
    AudioProject& GetProject();

private:
    void SetupUi();
    void RefreshViews();
    void OnPageChanging(wxNotebookEvent& evt);
    void OnPageChanged(wxNotebookEvent& evt);
    void OnPeakControlChanged(wxSpinEvent& evt);
    void OnOpl3VisibilityChanged(wxCommandEvent& evt);
    void OnOpl3Start(wxCommandEvent& evt);
    void OnOpl3PlayToggle(wxCommandEvent& evt);
    void OnOpl3ExportVgm(wxCommandEvent& evt);
    // Fires once playback's own known duration has elapsed -- the signal
    // that it finished naturally (StopOpl3Playback, used for an early
    // manual Stop, always stops this timer first, so it never fires for
    // that case) and the Play button should revert from "Stop".
    void OnOpl3PlaybackTimer(wxTimerEvent& evt);

    // Precomputes every frame's original-vs-cumulative-mix dBFS spectrum
    // pair and pushes it to the OPL3 view, so Visible/Difference reflect
    // m_opl3Channels/m_opl3CumulativeRendered as they currently stand.
    // Called once Start finishes -- computed in parallel across hardware
    // threads (a long recording is many thousands of frames, two FFTs
    // each), so it can share OnOpl3Start's own progress dialog instead of
    // running as a silent, unresponsive-looking freeze after it closes.
    void RefreshOpl3ComparisonOverlay(const dsp::ProgressCallback& progress = nullptr);

    // (Re)computes peaks at the current frame/analysis rate, max-peaks-per-
    // step, and peak window if they're out of date, showing a progress
    // dialog since this walks the full sample buffer. No-op if peaks are
    // already current for all of them.
    void EnsurePeaksComputed();

    // Renders `samples` to a temp WAV file and plays it via wxSound,
    // replacing any previously-playing OPL3 preview -- also flips the
    // Play button to "Stop" and arms m_opl3PlaybackTimer for the
    // rendered audio's own duration, so it flips back on its own once
    // playback would naturally have finished.
    void PlayOpl3Preview(const std::vector<float>& samples);
    // Stops whatever's currently playing (wxSound::Stop() -- MSW backs
    // wxSound with PlaySound(), which only ever plays one sound at a
    // time anyway, so this isn't specific to m_opl3PreviewSound), cancels
    // the pending completion timer, and reverts the Play button.
    void StopOpl3Playback();

    wxNotebook* m_notebook = nullptr;
    AudioProject m_project;

    // The OPL3 tab's own view (page 0, always AnalysisMode::Opl3) and the
    // Peak Trends tab's own view (page 1, always AnalysisMode::PeakTrends)
    // -- separate instances, not one view whose mode gets toggled, since a
    // wxNotebook page needs its own child window regardless.
    SpectrogramView* m_opl3View = nullptr;
    SpectrogramView* m_peakTrendsView = nullptr;

    // Horizontal (time) zoom/scroll shared across the two tabs, so
    // switching tabs keeps looking at the same time range. Captured from
    // the outgoing tab and pushed into the incoming tab on notebook page
    // change. Vertical (Hz) zoom/scroll deliberately isn't shared -- each
    // view keeps its own, since OPL3's own default range (see SetupUi)
    // isn't necessarily what Peak Trends wants to default to too.
    double m_sharedZoom = 1.0;
    double m_sharedScrollSeconds = 0.0;

    // Peak Trends tab's own controls, embedded directly in that tab's page.
    wxPanel* m_peakControlsPanel = nullptr;
    class wxSpinCtrl* m_frameRateSpin = nullptr;
    class wxSpinCtrl* m_analysisRateSpin = nullptr;
    class wxSpinCtrl* m_maxPeaksSpin = nullptr;
    class wxSpinCtrl* m_peakWindowSpin = nullptr;
    double m_peaksComputedAtFrameRate = -1.0; // < 0 means "never computed"
    double m_peaksComputedAtAnalysisRate = -1.0;
    int m_peaksComputedAtMaxPeaks = -1;
    int m_peaksComputedAtWindowHalfWidth = -1;

    // OPL3 tab's own controls, embedded directly in that tab's page. The
    // user sets frame rate/max channels, then presses Start, which fits
    // up to that many channels in turn against the whole recording (see
    // oplfit::FitChannelsInTurn/FitChannelsPerFrame) -- nothing runs
    // automatically just from having the tab open, since the whole point
    // is to let the user configure things first.
    wxPanel* m_opl3ControlsPanel = nullptr;
    class wxSpinCtrl* m_opl3FrameRateSpin = nullptr;
    class wxSpinCtrl* m_opl3MaxChannelsSpin = nullptr;
    class wxRadioBox* m_opl3VisibilityRadio = nullptr;

    // Picks which of oplfit.h's two side-by-side fitting workflows Start
    // runs -- FitChannelsInTurn (persistent per-trend channel identity)
    // or FitChannelsPerFrame (no identity tracking at all, modeled on an
    // earlier Python tool that empirically sounds denser on real speech)
    // -- kept as a direct A/B choice rather than picking a winner.
    class wxRadioBox* m_opl3AlgorithmRadio = nullptr;

    // The Play/Stop button itself (relabeled in place by
    // PlayOpl3Preview/StopOpl3Playback/OnOpl3PlaybackTimer) and the
    // playback state those three keep in sync with it. wxSound has no
    // completion event of its own, so m_opl3PlaybackTimer -- armed for
    // the rendered audio's own known duration -- stands in for one.
    class wxButton* m_opl3PlayButton = nullptr;
    wxTimer m_opl3PlaybackTimer;
    bool m_opl3IsPlaying = false;

    // The most recently computed fit, kept around for Play/Export VGM
    // without redoing the work. m_opl3CumulativeRendered is
    // m_opl3Channels' combined render (see oplfit::MultiChannelSineFit).
    std::vector<oplfit::AllFramesFit> m_opl3Channels;
    std::vector<float> m_opl3CumulativeRendered;

    // wxSound loads its file's audio fully into memory on construction, so
    // once constructed the temp file backing it is free to be overwritten
    // for the next preview -- this member just keeps that buffered audio
    // (and, on MSW, an internal PlaySound() handle) alive while it plays.
    class wxSound* m_opl3PreviewSound = nullptr;
    std::string m_opl3PreviewWavPath;
};
