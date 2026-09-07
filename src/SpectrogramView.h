#pragma once

#include "PeakAnalysis.h"

#include <wx/wx.h>
#include <wx/scrolwin.h>
#include <vector>

class SpectrogramView : public wxScrolledWindow
{
public:
    // Only one mode is active at a time; None means analysis is off. More
    // modes will be added later (each will need its own overlay-drawing
    // branch in DrawAnalysisOverlay).
    enum class AnalysisMode { None, PeakTrends, Opl3 };

    // What the Opl3 overlay draws (see SetOpl3AllFramesComparison): Hidden
    // shows the plain original spectrogram, Visible draws the fitted
    // channels' own rendered spectrum in its place, and Difference draws
    // the per-bin gap between the two.
    enum class Opl3OverlayVisibility { Hidden, Visible, Difference };

    explicit SpectrogramView(wxWindow* parent = nullptr);

    void SetImageData(int width, int height, const std::vector<unsigned char>& rgb,
                       double secondsPerColumn, int sampleRate);
    void SetZoom(double zoom);
    double GetZoom() const;

    void SetAnalysisMode(AnalysisMode mode);
    AnalysisMode GetAnalysisMode() const { return m_analysisMode; }

    // Replaces the data drawn by the PeakTrends overlay. Owned/computed
    // externally (ProjectWindow, which has access to the raw samples the
    // analysis runs over); this view only ever renders it.
    void SetPeakTrends(analysis::PeakTrendsResult peakTrends);

    void SetOpl3OverlayVisibility(Opl3OverlayVisibility visibility);

    // Replaces the comparison data drawn by Visible/Difference across the
    // whole recording: one dBFS spectrum per frame (same layout as
    // dsp::ComputeSpectrumAtTime) for the original recording and for the
    // fitted channels' combined render, both centered the same way so
    // they're directly comparable bin-for-bin -- frames[i] covers
    // [i/frameRateHz, (i+1)/frameRateHz). Pass empty vectors to clear it.
    void SetOpl3AllFramesComparison(std::vector<std::vector<double>> originalDbfsPerFrame,
                                     std::vector<std::vector<double>> renderedDbfsPerFrame, double frameRateHz);

    // Current left edge of the visible viewport, in seconds. 0 if no data.
    double GetScrollSeconds() const;
    // Sets zoom (clamped, same as SetZoom) then scrolls so the given time
    // position is at the left edge of the viewport. Used to keep this view
    // in sync with another time-series view sharing the same time axis.
    void SetZoomAndScroll(double zoom, double scrollSeconds);

    // Sets the vertical (Hz) viewport directly to [bottomHz, topHz] --
    // e.g. defaulting to a more relevant range on entering a particular
    // analysis mode -- via the same m_vZoom/m_vCenterHz state the mouse-
    // wheel/drag vertical zoom-and-pan already use, so it's just a
    // starting point, not a restriction: the user can still zoom/scroll
    // out past it same as ever.
    void SetVRangeHz(double bottomHz, double topHz);

private:
    void OnDraw(wxDC& dc) override;
    // Scrolling uses wx's native ScrollWindow() pixel-shift blit (fast,
    // partial-repaint) rather than a forced full Refresh() per scroll step.
    // SetTargetRect() in UpdateVirtualSize() keeps that blit confined to the
    // content area so it can never shift the vertical axis column. HUD text
    // drawn atop the content (e.g. "Zoom: x.xxx") can still show transient
    // ghosting from the blit; accepted for now.
    //
    // OnDraw itself draws several layers in sequence (cached bitmap,
    // analysis overlay, HUD text, ruler, axis) straight onto the screen DC
    // by default, which can be caught mid-sequence (e.g. the overlay drawn
    // but the axis not yet repainted on top of it) whenever a repaint --
    // RebuildCache in particular isn't free -- takes long enough for the OS
    // to display it before it's done. OnPaint below replaces
    // wxScrolledWindow's default handler with a buffered one so every
    // OnDraw() call is composited off-screen and blitted atomically.
    void OnPaint(wxPaintEvent& evt);
    void OnMouseWheel(wxMouseEvent& evt);
    void OnSize(wxSizeEvent& evt);
    void OnLeftDown(wxMouseEvent& evt);
    void OnLeftUp(wxMouseEvent& evt);
    void OnMotion(wxMouseEvent& evt);
    void OnLeaveWindow(wxMouseEvent& evt);

    // `refresh=false` skips the trailing Refresh() -- for a caller about
    // to correct the scroll position (Scroll()) right afterward, so the
    // stale pre-correction offset never actually gets painted; see
    // ZoomAt/SetZoomAndScroll, which refresh once themselves instead.
    void UpdateVirtualSize(bool refresh = true);
    void SetZoomImpl(double zoom, bool refresh);
    void RebuildCache(int offsetX, int visibleWidth, int visibleHeight);
    // Converts a mouse position (client coords) to the (time, Hz) point it
    // sits over, using the exact same coordinate mapping as the analysis
    // overlay, and shows it in the top-level frame's status bar. Clears the
    // status text when the mouse is over the axis/ruler labels or outside
    // the spectrogram entirely, since those aren't part of that mapping.
    void UpdateHoverStatus(const wxPoint& pos);
    // The seconds value under an absolute (unscrolled, "virtual") x
    // coordinate -- the shared inverse of the various TimeToX lambdas
    // used for drawing/hit-testing (UpdateHoverStatus and the analysis
    // overlay).
    double PixelXToSeconds(int clientX) const;

    void SetVZoom(double zoom);
    void GetVRange(double& topHz, double& bottomHz) const;
    void ClampVCenterHz();
    // Wheel-zoom helpers: rescale by `factor` while keeping whatever
    // value (Hz / time) is under the mouse fixed at that same screen
    // position -- the cursor "pins" the point it's over instead of the
    // view zooming around its own center or left edge.
    void ZoomVAt(double factor, int mouseYClient);
    void ZoomAt(double factor, int mouseXClient);

    // Draws whatever the current analysis mode calls for, in the same
    // absolute content-coordinate space as the spectrogram bitmap itself,
    // so it scrolls/zooms in lockstep with it (see .cpp for the coordinate
    // system this relies on).
    void DrawAnalysisOverlay(wxDC& dc, int offsetX, int contentWidth, int contentHeight,
                              double unitsPerPixel, double unitsPerSecond, double topHz, double bottomHz);

    AnalysisMode m_analysisMode = AnalysisMode::None;
    analysis::PeakTrendsResult m_peakTrends;

    Opl3OverlayVisibility m_opl3Visibility = Opl3OverlayVisibility::Hidden;

    // Whole-recording comparison data (see SetOpl3AllFramesComparison) --
    // index-aligned with m_opl3AllFramesRateHz's frame grid.
    std::vector<std::vector<double>> m_opl3AllOriginalDbfs;
    std::vector<std::vector<double>> m_opl3AllRenderedDbfs;
    double m_opl3AllFramesRateHz = 0.0;

    std::vector<unsigned char> m_rgb; // source document: m_imgW * m_imgH * 3
    int m_imgW = 0;
    int m_imgH = 0;
    double m_secondsPerColumn = 0.0;
    int m_sampleRate = 44100;
    double m_zoom = 1.0;

    // The virtual width UpdateVirtualSize() *requested* via SetVirtualSize().
    // wxScrolledWindow silently clamps the virtual size up to never be
    // smaller than the window's own client size, so at low zoom levels
    // GetVirtualSize() can return the full client width (axis column
    // included) instead of the narrower content-only width we asked for.
    // Everything that derives units-per-pixel from the virtual width reads
    // this member instead of calling GetVirtualSize().
    int m_virtualContentWidth = 1;

    // Vertical (Hz) zoom+scroll, independent of the shared horizontal time
    // axis -- Hz has no relationship to the waveform's counts axis, so this
    // state lives entirely in this view.
    double m_vZoom = 1.0;
    double m_vCenterHz = 0.0;

    wxBitmap m_cache;
    bool m_cacheDirty = true;
    int m_cacheOffsetX = -1;
    int m_cacheWidth = -1;
    int m_cacheHeight = -1;
    double m_cacheVZoom = -1.0;
    double m_cacheVCenterHz = -1.0;

    bool m_panning = false;
    wxPoint m_panAnchorMouse;
    int m_panAnchorScrollX = 0;

    bool m_vPanning = false;
    wxPoint m_vPanAnchorMouse;
    double m_vPanAnchorCenterHz = 0.0;

    // Whether the mouse is currently over the vertical axis strip (or
    // dragging it) -- drives both the ns-resize cursor and the axis's own
    // hover highlight (see VerticalAxis.h), the discoverability cue for
    // "this is wheel-zoomable/draggable".
    bool m_vAxisHovered = false;
    // Same idea as m_vAxisHovered, for the horizontal time ruler strip
    // (see TimeRuler.h) -- sizewe cursor instead of sizens.
    bool m_hAxisHovered = false;

    wxDECLARE_EVENT_TABLE();
};
