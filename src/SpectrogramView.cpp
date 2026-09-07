#include "SpectrogramView.h"
#include "TimeRuler.h"
#include "VerticalAxis.h"
#include "Dsp.h"

#include <wx/dcclient.h>
#include <wx/dcbuffer.h>

#include <algorithm>
#include <cmath>
#include <cstring>

wxBEGIN_EVENT_TABLE(SpectrogramView, wxScrolledWindow)
    EVT_PAINT(SpectrogramView::OnPaint)
    EVT_MOUSEWHEEL(SpectrogramView::OnMouseWheel)
    EVT_SIZE(SpectrogramView::OnSize)
    EVT_LEFT_DOWN(SpectrogramView::OnLeftDown)
    EVT_LEFT_UP(SpectrogramView::OnLeftUp)
    EVT_MOTION(SpectrogramView::OnMotion)
    EVT_LEAVE_WINDOW(SpectrogramView::OnLeaveWindow)
wxEND_EVENT_TABLE()

SpectrogramView::SpectrogramView(wxWindow* parent)
    : wxScrolledWindow(parent, wxID_ANY)
{
    SetMinSize(wxSize(320, 200));
    SetBackgroundColour(wxColour(18, 18, 18));
    SetBackgroundStyle(wxBG_STYLE_PAINT); // required for wxAutoBufferedPaintDC; see OnPaint
    SetScrollRate(1, 1);
}

void SpectrogramView::SetImageData(int width, int height, const std::vector<unsigned char>& rgb,
                                    double secondsPerColumn, int sampleRate)
{
    m_imgW = width;
    m_imgH = height;
    m_rgb = rgb;
    m_secondsPerColumn = secondsPerColumn;
    m_sampleRate = sampleRate > 0 ? sampleRate : 44100;
    m_vZoom = 1.0;
    m_vCenterHz = m_sampleRate / 4.0; // Nyquist/2: full range, centered -- callers that want a
                                       // different default (e.g. Opl3 mode's own SetVRangeHz call)
                                       // re-apply it after this, since new image data invalidates
                                       // any previous vertical framing either way
    // Analysis mode is NOT reset here -- each SpectrogramView instance is
    // now permanently dedicated to one mode for its whole life (see
    // ProjectWindow, which owns one view per analysis-mode tab), set once
    // at construction and never toggled, unlike when a single shared view
    // used to switch modes and genuinely needed resetting on new data.
    m_peakTrends = analysis::PeakTrendsResult{};
    m_opl3AllOriginalDbfs.clear();
    m_opl3AllRenderedDbfs.clear();
    m_opl3AllFramesRateHz = 0.0;
    m_cacheDirty = true;
    UpdateVirtualSize();
}

void SpectrogramView::SetZoom(double zoom)
{
    SetZoomImpl(zoom, /*refresh=*/true);
}

void SpectrogramView::SetZoomImpl(double zoom, bool refresh)
{
    m_zoom = std::clamp(zoom, 1.0, 64.0);
    m_cacheDirty = true;
    UpdateVirtualSize(refresh);
}

double SpectrogramView::GetZoom() const { return m_zoom; }

void SpectrogramView::SetAnalysisMode(AnalysisMode mode)
{
    m_analysisMode = mode;
    Refresh();
}

void SpectrogramView::SetPeakTrends(analysis::PeakTrendsResult peakTrends)
{
    m_peakTrends = std::move(peakTrends);
    Refresh();
}

void SpectrogramView::SetOpl3OverlayVisibility(Opl3OverlayVisibility visibility)
{
    m_opl3Visibility = visibility;
    Refresh();
}

void SpectrogramView::SetOpl3AllFramesComparison(std::vector<std::vector<double>> originalDbfsPerFrame,
                                                  std::vector<std::vector<double>> renderedDbfsPerFrame,
                                                  double frameRateHz)
{
    m_opl3AllOriginalDbfs = std::move(originalDbfsPerFrame);
    m_opl3AllRenderedDbfs = std::move(renderedDbfsPerFrame);
    m_opl3AllFramesRateHz = frameRateHz;
    Refresh();
}

void SpectrogramView::SetVZoom(double zoom)
{
    m_vZoom = std::clamp(zoom, 1.0, dsp::kFftSize / 4.0);
    ClampVCenterHz();
    Refresh();
}

void SpectrogramView::SetVRangeHz(double bottomHz, double topHz)
{
    const double half = std::max((topHz - bottomHz) / 2.0, 1.0);
    m_vCenterHz = (topHz + bottomHz) / 2.0;
    const double nyquist = m_sampleRate / 2.0;
    m_vZoom = std::clamp(nyquist / (2.0 * half), 1.0, dsp::kFftSize / 4.0);
    ClampVCenterHz();
    Refresh();
}

void SpectrogramView::GetVRange(double& topHz, double& bottomHz) const
{
    const double nyquist = m_sampleRate / 2.0;
    const double half = nyquist / (2.0 * m_vZoom);
    topHz = m_vCenterHz + half;
    bottomHz = m_vCenterHz - half;
}

void SpectrogramView::ClampVCenterHz()
{
    const double nyquist = m_sampleRate / 2.0;
    const double half = nyquist / (2.0 * m_vZoom);
    m_vCenterHz = std::clamp(m_vCenterHz, half, nyquist - half);
}

void SpectrogramView::ZoomVAt(double factor, int mouseYClient)
{
    const int contentHeight = std::max(1, GetClientSize().GetHeight() - timeruler::kHeight);
    double topHz = 0.0, bottomHz = 0.0;
    GetVRange(topHz, bottomHz);
    const double f = std::clamp(static_cast<double>(mouseYClient) / contentHeight, 0.0, 1.0);
    const double hzUnderCursor = topHz - f * (topHz - bottomHz);

    m_vZoom = std::clamp(m_vZoom * factor, 1.0, dsp::kFftSize / 4.0);
    const double nyquist = m_sampleRate / 2.0;
    const double half = nyquist / (2.0 * m_vZoom);
    // Solving topHz' - f*(topHz'-bottomHz') = hzUnderCursor for the new
    // center, where topHz'/bottomHz' = center' +/- half.
    m_vCenterHz = hzUnderCursor - half * (1.0 - 2.0 * f);
    ClampVCenterHz();
    Refresh();
}

void SpectrogramView::ZoomAt(double factor, int mouseXClient)
{
    if (m_imgW <= 0 || m_secondsPerColumn <= 0.0)
    {
        SetZoom(m_zoom * factor);
        return;
    }

    const double secondsUnderCursor = PixelXToSeconds(mouseXClient);

    // refresh=false: painting here, before Scroll() below corrects the
    // offset for the new zoom, would flash the stale pre-correction
    // position -- the "jitter" this shape of zoom-then-rescroll update is
    // prone to. One Refresh() at the end instead.
    SetZoomImpl(m_zoom * factor, /*refresh=*/false); // recomputes m_virtualContentWidth for the new zoom

    const double unitsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);
    // Solving (offsetX + mouseXClient) * unitsPerPixel * secondsPerColumn
    // == secondsUnderCursor for the new raw scroll offset (device pixels,
    // matching SetScrollRate(1, 1)).
    const double targetOffsetX = (secondsUnderCursor / m_secondsPerColumn) / unitsPerPixel - mouseXClient;
    Scroll(static_cast<int>(std::lround(targetOffsetX)), 0);
    Refresh();
}

double SpectrogramView::GetScrollSeconds() const
{
    if (m_imgW <= 0 || m_secondsPerColumn <= 0.0)
        return 0.0;
    const double unitsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);
    int viewStartX = 0, viewStartY = 0;
    GetViewStart(&viewStartX, &viewStartY);
    int rateX = 1, rateY = 1;
    GetScrollPixelsPerUnit(&rateX, &rateY);
    const int offsetX = viewStartX * rateX;
    return offsetX * unitsPerPixel * m_secondsPerColumn;
}

double SpectrogramView::PixelXToSeconds(int clientX) const
{
    if (m_imgW <= 0 || m_secondsPerColumn <= 0.0)
        return 0.0;
    int viewStartX = 0, viewStartY = 0;
    GetViewStart(&viewStartX, &viewStartY);
    int rateX = 1, rateY = 1;
    GetScrollPixelsPerUnit(&rateX, &rateY);
    const int offsetX = viewStartX * rateX;
    const double unitsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);
    return (offsetX + clientX) * unitsPerPixel * m_secondsPerColumn;
}

void SpectrogramView::SetZoomAndScroll(double zoom, double scrollSeconds)
{
    // No intermediate refresh -- painting the still-stale pre-Scroll()
    // offset against the new zoom is exactly the visible "wrong position,
    // then snaps" jitter this two-step (rezoom, then correct scroll) shape
    // is prone to; see ZoomAt for the same fix.
    SetZoomImpl(zoom, /*refresh=*/false);

    if (m_imgW > 0 && m_secondsPerColumn > 0.0)
    {
        const double unitsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);
        const int targetX = static_cast<int>((scrollSeconds / m_secondsPerColumn) / unitsPerPixel);
        Scroll(targetX, 0);
    }
    Refresh(); // single repaint reflecting the final zoom+scroll state
}

void SpectrogramView::UpdateVirtualSize(bool refresh)
{
    const wxSize client = GetClientSize();
    const int clientW = std::max(1, client.GetWidth() - verticalaxis::kWidth);
    const int clientH = std::max(1, client.GetHeight());
    const double totalUnits = static_cast<double>(std::max(1, m_imgW));
    const double baseUnitsPerPixel = totalUnits / clientW;
    const double unitsPerPixel = std::max(1e-6, baseUnitsPerPixel / m_zoom);
    const int virtualWidth = std::max(clientW, static_cast<int>(totalUnits / unitsPerPixel));
    m_virtualContentWidth = virtualWidth;

    SetVirtualSize(virtualWidth, clientH);

    // Restrict the native scroll blit to the content area, excluding the
    // vertical axis column. That column never moves horizontally, so the
    // blit-scroll optimization must not touch it -- otherwise its pixels
    // get shifted along with the content and end up briefly overlapping it
    // during a scroll before the next full repaint catches up.
    SetTargetRect(wxRect(0, 0, clientW, clientH));

    if (refresh)
        Refresh();
}

void SpectrogramView::RebuildCache(int offsetX, int visibleWidth, int visibleHeight)
{
    m_cache = wxBitmap();
    if (m_imgW <= 0 || m_imgH <= 0 || m_rgb.empty() || visibleWidth <= 0 || visibleHeight <= 0)
        return;

    const double colsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);

    const double hzPerBin = static_cast<double>(m_sampleRate) / dsp::kFftSize;
    double topHz = 0.0, bottomHz = 0.0;
    GetVRange(topHz, bottomHz);
    // Row decreases as Hz increases (row height-1 = 0 Hz, row 0 = Nyquist).
    const double topRow = (m_imgH - 1) - (topHz / hzPerBin);
    const double bottomRow = (m_imgH - 1) - (bottomHz / hzPerBin);
    const double rowSpan = bottomRow - topRow;

    std::vector<unsigned char> out(static_cast<size_t>(visibleWidth) * visibleHeight * 3);

    for (int y = 0; y < visibleHeight; ++y)
    {
        int srcY0 = static_cast<int>(std::floor(topRow + rowSpan * y / visibleHeight));
        int srcY1 = static_cast<int>(std::ceil(topRow + rowSpan * (y + 1) / visibleHeight));
        srcY0 = std::clamp(srcY0, 0, m_imgH - 1);
        srcY1 = std::clamp(srcY1, srcY0 + 1, m_imgH);

        for (int x = 0; x < visibleWidth; ++x)
        {
            int virtualX = offsetX + x;
            int srcCol = std::clamp(static_cast<int>(virtualX * colsPerPixel), 0, m_imgW - 1);

            long rs = 0, gs = 0, bs = 0;
            int count = 0;
            for (int sy = srcY0; sy < srcY1; ++sy)
            {
                size_t idx = (static_cast<size_t>(sy) * m_imgW + srcCol) * 3;
                rs += m_rgb[idx + 0];
                gs += m_rgb[idx + 1];
                bs += m_rgb[idx + 2];
                ++count;
            }
            size_t outIdx = (static_cast<size_t>(y) * visibleWidth + x) * 3;
            out[outIdx + 0] = static_cast<unsigned char>(rs / std::max(1, count));
            out[outIdx + 1] = static_cast<unsigned char>(gs / std::max(1, count));
            out[outIdx + 2] = static_cast<unsigned char>(bs / std::max(1, count));
        }
    }

    wxImage img(visibleWidth, visibleHeight, out.data(), true);
    m_cache = wxBitmap(img);
}

void SpectrogramView::DrawAnalysisOverlay(wxDC& dc, int offsetX, int contentWidth, int contentHeight,
                                           double unitsPerPixel, double unitsPerSecond,
                                           double topHz, double bottomHz)
{
    // Drawn in absolute content coordinates (no offsetX addition), the same
    // space the spectrogram bitmap itself occupies -- OnDraw's DC has
    // already been origin-shifted for the current scroll position, so this
    // scrolls and zooms in lockstep with the underlying spectrogram
    // automatically. Visibility bounds checks below still need offsetX,
    // though: TimeToX() returns that same absolute coordinate, not one
    // relative to the current viewport, so "is this on screen" has to
    // compare against [offsetX, offsetX + contentWidth), not [0, contentWidth).
    auto TimeToX = [&](double seconds) {
        return static_cast<int>((seconds * unitsPerSecond) / unitsPerPixel);
    };
    auto HzToY = [&](double hz) {
        return static_cast<int>(contentHeight * (topHz - hz) / (topHz - bottomHz));
    };

    if (m_analysisMode != AnalysisMode::None && contentWidth > 0 && contentHeight > 0)
    {
        // A plain wxDC (GDI) can't alpha-blend a translucent brush against
        // whatever's already drawn -- it just paints opaque. So the whole
        // overlay (dimming wash + markers) is composited by hand into a
        // wxImage with a real per-pixel alpha channel, then blitted in one
        // shot; wx uses AlphaBlend() for that automatically since the
        // resulting wxBitmap has alpha.
        wxImage layer(contentWidth, contentHeight);
        layer.InitAlpha();
        unsigned char* rgb = layer.GetData();
        unsigned char* alpha = layer.GetAlpha();
        const size_t pixelCount = static_cast<size_t>(contentWidth) * contentHeight;
        std::memset(rgb, 0, pixelCount * 3);
        if (m_analysisMode == AnalysisMode::PeakTrends)
            std::memset(alpha, 96, pixelCount); // semi-transparent black wash, to emphasise trend lines against a busy background
        else
            std::memset(alpha, 0, pixelCount);  // fully transparent by default -- Opl3's own comparison strip
                                                 // already covers the informative area directly in real color
                                                 // (see drawComparisonStrip below), so a uniform dimming wash
                                                 // underneath it -- or over nothing at all, when Hidden or
                                                 // before a fit has run -- served no purpose and just darkened
                                                 // the whole spectrogram for no reason

        auto setPixel = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
            if (x < 0 || x >= contentWidth || y < 0 || y >= contentHeight)
                return;
            const size_t idx = static_cast<size_t>(y) * contentWidth + x;
            rgb[idx * 3 + 0] = r;
            rgb[idx * 3 + 1] = g;
            rgb[idx * 3 + 2] = b;
            alpha[idx] = 255; // markers are pure solid color, fully opaque
        };
        auto drawLine = [&](int x0, int y0, int x1, int y1, unsigned char r, unsigned char g, unsigned char b) {
            const int dx = std::abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
            const int dy = -std::abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
            int err = dx + dy;
            while (true)
            {
                setPixel(x0, y0, r, g, b);
                if (x0 == x1 && y0 == y1) break;
                const int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        };

        auto toLocal = [&](const analysis::SpectralPeak& peak, int& x, int& y) {
            if (peak.freqHz < bottomHz || peak.freqHz > topHz)
                return false;
            x = TimeToX(peak.timeSeconds) - offsetX;
            if (x < 0 || x >= contentWidth)
                return false;
            y = HzToY(peak.freqHz);
            return true;
        };

        // Trend line segments, drawn from framePoints, not the full
        // analysis-rate points list -- playback will linearly interpolate
        // frequency/volume frame to frame, so the line should show exactly
        // that (straight segments between frame-rate samples), not a
        // denser polyline through every analysis step.
        auto drawTrendLines = [&](const analysis::PeakTrendsResult& result,
                                   unsigned char r, unsigned char g, unsigned char b) {
            for (const analysis::PeakTrend& trend : result.trends)
            {
                int xPrev = 0, yPrev = 0;
                bool havePrev = false;
                for (const analysis::SpectralPeak& peak : trend.framePoints)
                {
                    int x = 0, y = 0;
                    const bool visible = toLocal(peak, x, y);
                    if (visible && havePrev)
                        drawLine(xPrev, yPrev, x, y, r, g, b);
                    havePrev = visible;
                    xPrev = x;
                    yPrev = y;
                }
            }
        };

        if (m_analysisMode == AnalysisMode::PeakTrends)
        {
            drawTrendLines(m_peakTrends, 0, 255, 255); // pure cyan
        }
        else if (m_analysisMode == AnalysisMode::Opl3)
        {
            // Draws one frame's comparison strip (Visible: the fitted
            // render's own spectrum; Difference: the per-bin gap against
            // the original) across [xLeft, xRight).
            auto drawComparisonStrip = [&](int xLeft, int xRight, const std::vector<double>& originalDbfs,
                                            const std::vector<double>& renderedDbfs) {
                const double hzPerBin = static_cast<double>(m_sampleRate) / dsp::kFftSize;
                const size_t binCount = std::min(originalDbfs.size(), renderedDbfs.size());

                for (int y = 0; y < contentHeight; ++y)
                {
                    const double hz = topHz - (static_cast<double>(y) / contentHeight) * (topHz - bottomHz);
                    if (hz < 0.0)
                        continue;
                    const size_t bin = static_cast<size_t>(std::llround(hz / hzPerBin));
                    if (bin >= binCount)
                        continue;

                    unsigned char r = 0, g = 0, b = 0;
                    if (m_opl3Visibility == Opl3OverlayVisibility::Visible)
                    {
                        const dsp::RGB c = dsp::DbfsToColor(renderedDbfs[bin]);
                        r = c.r; g = c.g; b = c.b;
                    }
                    else // Difference
                    {
                        // Original louder than rendered -> red (the fit
                        // is missing energy there); rendered louder ->
                        // blue (the fit is adding energy that isn't in
                        // the original); near zero -> dark/black.
                        const double deltaDb = std::clamp(originalDbfs[bin] - renderedDbfs[bin], -40.0, 40.0);
                        const unsigned char intensity =
                            static_cast<unsigned char>(std::clamp(std::abs(deltaDb) / 40.0, 0.0, 1.0) * 255.0);
                        if (deltaDb >= 0.0) { r = intensity; g = 0; b = 0; }
                        else { r = 0; g = 0; b = intensity; }
                    }

                    for (int x = std::max(0, xLeft); x < std::min(contentWidth, xRight); ++x)
                        setPixel(x, y, r, g, b);
                }
            };

            if (m_opl3Visibility != Opl3OverlayVisibility::Hidden && !m_opl3AllOriginalDbfs.empty()
                && m_opl3AllFramesRateHz > 0.0)
            {
                const double frameInterval = 1.0 / m_opl3AllFramesRateHz;
                const size_t frameCount = std::min(m_opl3AllOriginalDbfs.size(), m_opl3AllRenderedDbfs.size());
                for (size_t f = 0; f < frameCount; ++f)
                {
                    const int xLeft = TimeToX(static_cast<double>(f) * frameInterval) - offsetX;
                    const int xRight = TimeToX(static_cast<double>(f + 1) * frameInterval) - offsetX;
                    if (xRight < 0 || xLeft >= contentWidth)
                        continue;
                    drawComparisonStrip(xLeft, xRight, m_opl3AllOriginalDbfs[f], m_opl3AllRenderedDbfs[f]);
                }
            }
        }

        dc.DrawBitmap(wxBitmap(layer), offsetX, 0, true);
    }
}

void SpectrogramView::OnPaint(wxPaintEvent&)
{
    wxAutoBufferedPaintDC dc(this);
    DoPrepareDC(dc);
    OnDraw(dc);
}

void SpectrogramView::OnDraw(wxDC& dc)
{
    const wxSize client = GetClientSize();
    dc.SetBackground(wxBrush(wxColour(18, 18, 18)));
    dc.Clear();

    if (m_imgW <= 0 || m_imgH <= 0 || m_rgb.empty())
    {
        dc.SetTextForeground(*wxWHITE);
        dc.DrawText("No spectrogram data", 10, client.GetHeight() / 2 - 10);
        return;
    }

    int viewStartX = 0, viewStartY = 0;
    GetViewStart(&viewStartX, &viewStartY);
    int rateX = 1, rateY = 1;
    GetScrollPixelsPerUnit(&rateX, &rateY);
    const int offsetX = viewStartX * rateX;
    const int visibleWidth = client.GetWidth();
    const int visibleHeight = client.GetHeight();
    const int contentWidth = std::max(1, visibleWidth - verticalaxis::kWidth);
    const int contentHeight = std::max(1, visibleHeight - timeruler::kHeight);

    const bool needRebuild = m_cacheDirty
        || offsetX != m_cacheOffsetX
        || contentWidth != m_cacheWidth
        || contentHeight != m_cacheHeight
        || m_vZoom != m_cacheVZoom
        || m_vCenterHz != m_cacheVCenterHz;

    if (needRebuild)
    {
        RebuildCache(offsetX, contentWidth, contentHeight);
        m_cacheOffsetX = offsetX;
        m_cacheWidth = contentWidth;
        m_cacheHeight = contentHeight;
        m_cacheVZoom = m_vZoom;
        m_cacheVCenterHz = m_vCenterHz;
        m_cacheDirty = false;
    }

    if (m_cache.IsOk())
        dc.DrawBitmap(m_cache, offsetX, 0, false);

    const double unitsPerPixel = static_cast<double>(m_imgW) / std::max(1, m_virtualContentWidth);
    const double unitsPerSecond = (m_secondsPerColumn > 0.0) ? (1.0 / m_secondsPerColumn) : 1.0;
    double topHz = 0.0, bottomHz = 0.0;
    GetVRange(topHz, bottomHz);

    if (m_analysisMode != AnalysisMode::None)
        DrawAnalysisOverlay(dc, offsetX, contentWidth, contentHeight, unitsPerPixel, unitsPerSecond, topHz, bottomHz);

    dc.SetTextForeground(wxColour(200, 200, 200));
    dc.DrawText(wxString::Format("Zoom: %.2fx", m_zoom), offsetX + 8, 8);

    timeruler::Draw(dc, offsetX, contentWidth, contentHeight, unitsPerPixel,
                     unitsPerSecond, m_imgW * m_secondsPerColumn, m_hAxisHovered);

    verticalaxis::Draw(dc, offsetX + contentWidth, 0, contentHeight, topHz, bottomHz, m_vAxisHovered);

    // Blank corner where the horizontal ruler and vertical axis strips
    // meet -- hit-tested as part of the vertical axis (see OnMotion), so
    // its hover state matches that strip's.
    dc.SetBrush(wxBrush(m_vAxisHovered ? wxColour(48, 48, 48) : wxColour(30, 30, 30)));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(offsetX + contentWidth, contentHeight, verticalaxis::kWidth, timeruler::kHeight);
}

void SpectrogramView::OnMouseWheel(wxMouseEvent& evt)
{
    const int clientW = GetClientSize().GetWidth();
    const bool overAxis = evt.GetPosition().x >= (clientW - verticalaxis::kWidth);

    int rot = evt.GetWheelRotation();
    double delta = rot / 120.0;
    const double factor = (delta > 0 ? 1.25 : 0.8);
    if (overAxis)
        ZoomVAt(factor, evt.GetPosition().y);
    else
        ZoomAt(factor, evt.GetPosition().x);
}

void SpectrogramView::OnSize(wxSizeEvent& evt)
{
    UpdateVirtualSize();
    evt.Skip();
}

void SpectrogramView::OnLeftDown(wxMouseEvent& evt)
{
    const int clientW = GetClientSize().GetWidth();
    const bool overAxis = evt.GetPosition().x >= (clientW - verticalaxis::kWidth);

    if (overAxis)
    {
        m_vPanning = true;
        CaptureMouse();
        m_vPanAnchorMouse = evt.GetPosition();
        m_vPanAnchorCenterHz = m_vCenterHz;
        return;
    }

    m_panning = true;
    CaptureMouse();
    m_panAnchorMouse = evt.GetPosition();
    int vx = 0, vy = 0;
    GetViewStart(&vx, &vy);
    m_panAnchorScrollX = vx;
}

void SpectrogramView::OnMotion(wxMouseEvent& evt)
{
    const wxSize client = GetClientSize();
    const int contentHeight = std::max(1, client.GetHeight() - timeruler::kHeight);
    const bool overAxis = m_vPanning || evt.GetPosition().x >= (client.GetWidth() - verticalaxis::kWidth);
    // The bottom-right corner where the two strips meet counts as the
    // vertical axis (matches overAxis's own x-only check, which already
    // claims it), not the ruler.
    const bool overRuler = !overAxis && evt.GetPosition().y >= contentHeight;

    if (overAxis != m_vAxisHovered || overRuler != m_hAxisHovered)
    {
        m_vAxisHovered = overAxis;
        m_hAxisHovered = overRuler;
        SetCursor(overAxis ? wxCursor(wxCURSOR_SIZENS) : overRuler ? wxCursor(wxCURSOR_SIZEWE) : wxNullCursor);
        Refresh();
    }

    if (m_vPanning && evt.Dragging() && evt.LeftIsDown())
    {
        const int dy = evt.GetPosition().y - m_vPanAnchorMouse.y;
        const int contentHeight = std::max(1, GetClientSize().GetHeight() - timeruler::kHeight);
        double top = 0.0, bottom = 0.0;
        GetVRange(top, bottom); // current zoom is unchanged during a pure drag
        const double valuePerPixel = (top - bottom) / contentHeight;
        m_vCenterHz = m_vPanAnchorCenterHz + dy * valuePerPixel;
        ClampVCenterHz();
        Refresh(); // needRebuild auto-detects the m_vCenterHz change; see OnDraw
    }
    else if (m_panning && evt.Dragging() && evt.LeftIsDown())
    {
        int dx = evt.GetPosition().x - m_panAnchorMouse.x;
        Scroll(m_panAnchorScrollX - dx, 0); // native partial-repaint blit; see SetTargetRect in UpdateVirtualSize
    }

    UpdateHoverStatus(evt.GetPosition());
}

void SpectrogramView::OnLeftUp(wxMouseEvent&)
{
    if (m_panning || m_vPanning)
    {
        m_panning = false;
        m_vPanning = false;
        if (HasCapture()) ReleaseMouse();
    }
}

void SpectrogramView::OnLeaveWindow(wxMouseEvent&)
{
    if (wxFrame* frame = dynamic_cast<wxFrame*>(wxGetTopLevelParent(this)))
        frame->SetStatusText(wxEmptyString);

    if ((m_vAxisHovered && !m_vPanning) || m_hAxisHovered)
    {
        m_vAxisHovered = m_vPanning; // preserved only while an axis drag is still in progress
        m_hAxisHovered = false;
        SetCursor(m_vAxisHovered ? wxCursor(wxCURSOR_SIZENS) : wxNullCursor);
        Refresh();
    }
}

void SpectrogramView::UpdateHoverStatus(const wxPoint& pos)
{
    wxFrame* frame = dynamic_cast<wxFrame*>(wxGetTopLevelParent(this));
    if (!frame)
        return;

    const wxSize client = GetClientSize();
    const int contentWidth = std::max(1, client.GetWidth() - verticalaxis::kWidth);
    const int contentHeight = std::max(1, client.GetHeight() - timeruler::kHeight);

    if (m_imgW <= 0 || m_secondsPerColumn <= 0.0
        || pos.x < 0 || pos.x >= contentWidth || pos.y < 0 || pos.y >= contentHeight)
    {
        frame->SetStatusText(wxEmptyString);
        return;
    }

    const double seconds = PixelXToSeconds(pos.x);

    double topHz = 0.0, bottomHz = 0.0;
    GetVRange(topHz, bottomHz);
    const double hz = topHz - (static_cast<double>(pos.y) / contentHeight) * (topHz - bottomHz);

    frame->SetStatusText(wxString::Format("Time: %.3f s   Frequency: %.1f Hz", seconds, hz));
}
