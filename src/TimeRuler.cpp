#include "TimeRuler.h"

#include <algorithm>
#include <cmath>

namespace timeruler
{

namespace
{

constexpr double kLadder[] = {
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 30, 60, 300, 600, 1800, 3600
};

double ChooseMajorInterval(double pixelsPerSecond)
{
    for (double candidate : kLadder)
    {
        if (candidate * pixelsPerSecond >= kMinLabelSpacingPx)
            return candidate;
    }
    double interval = 3600.0;
    while (interval * pixelsPerSecond < kMinLabelSpacingPx)
        interval *= 2.0;
    return interval;
}

wxString FormatLabel(double seconds, double majorInterval, bool useHours)
{
    if (seconds < 0.0) seconds = 0.0;

    int decimals = (majorInterval >= 1.0)
        ? 1
        : static_cast<int>(std::ceil(-std::log10(majorInterval)));
    decimals = std::max(0, decimals);

    // Round to the displayed precision first so minute/hour carries (e.g.
    // 119.96s at 1 decimal) land on "2:00.0" rather than a stray "1:60.0".
    const double scale = std::pow(10.0, decimals);
    seconds = std::round(seconds * scale) / scale;

    const int width = decimals > 0 ? decimals + 3 : 2;

    if (useHours)
    {
        long long totalWhole = static_cast<long long>(std::floor(seconds));
        long long hours = totalWhole / 3600;
        long long minutes = (totalWhole % 3600) / 60;
        double secs = seconds - static_cast<double>(hours * 3600 + minutes * 60);
        return wxString::Format("%lld:%02lld:%0*.*f", hours, minutes, width, decimals, secs);
    }

    long long totalMinutes = static_cast<long long>(std::floor(seconds / 60.0));
    double secs = seconds - static_cast<double>(totalMinutes) * 60.0;
    return wxString::Format("%lld:%0*.*f", totalMinutes, width, decimals, secs);
}

} // namespace

void Draw(wxDC& dc, int offsetX, int visibleWidth, int top,
          double unitsPerPixel, double unitsPerSecond, double totalSeconds, bool hovered)
{
    if (unitsPerPixel <= 0.0 || unitsPerSecond <= 0.0 || visibleWidth <= 0)
        return;

    const double secondsPerPixel = unitsPerPixel / unitsPerSecond;
    const double pixelsPerSecond = 1.0 / secondsPerPixel;

    dc.SetBrush(wxBrush(hovered ? wxColour(48, 48, 48) : wxColour(30, 30, 30)));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(offsetX, top, visibleWidth, kHeight);
    dc.SetPen(wxPen(hovered ? wxColour(120, 120, 120) : wxColour(70, 70, 70)));
    dc.DrawLine(offsetX, top, offsetX + visibleWidth, top);

    const double majorInterval = ChooseMajorInterval(pixelsPerSecond);
    const double minorInterval = majorInterval / 5.0;
    const bool drawMinor = (minorInterval * pixelsPerSecond) >= 4.0;
    const bool useHours = totalSeconds >= 3600.0;

    const double startSeconds = (offsetX * unitsPerPixel) / unitsPerSecond;
    const double endSeconds = ((offsetX + visibleWidth) * unitsPerPixel) / unitsPerSecond;
    double firstTick = std::ceil(startSeconds / majorInterval - 1e-9) * majorInterval;
    firstTick = std::max(0.0, firstTick);

    dc.SetTextForeground(wxColour(200, 200, 200));

    for (double t = firstTick; t <= endSeconds + majorInterval; t += majorInterval)
    {
        int px = static_cast<int>(t * unitsPerSecond / unitsPerPixel);
        if (px < offsetX - 1) continue;
        if (px > offsetX + visibleWidth + 1) break;

        dc.SetPen(wxPen(wxColour(210, 210, 210)));
        dc.DrawLine(px, top, px, top + 7);
        dc.DrawText(FormatLabel(t, majorInterval, useHours), px + 2, top + 9);

        if (drawMinor)
        {
            for (int k = 1; k <= 4; ++k)
            {
                double mt = t + minorInterval * k;
                if (mt >= t + majorInterval) break;
                int mpx = static_cast<int>(mt * unitsPerSecond / unitsPerPixel);
                if (mpx < offsetX || mpx > offsetX + visibleWidth) continue;
                dc.SetPen(wxPen(wxColour(90, 90, 90)));
                dc.DrawLine(mpx, top, mpx, top + 4);
            }
        }
    }
}

} // namespace timeruler
