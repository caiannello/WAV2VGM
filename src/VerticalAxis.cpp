#include "VerticalAxis.h"

#include <algorithm>
#include <cmath>

namespace verticalaxis
{

namespace
{

constexpr double kLadder[] = {
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000
};

double ChooseMajorInterval(double pixelsPerUnit)
{
    for (double candidate : kLadder)
    {
        if (candidate * pixelsPerUnit >= kMinLabelSpacingPx)
            return candidate;
    }
    double interval = 100000.0;
    while (interval * pixelsPerUnit < kMinLabelSpacingPx)
        interval *= 2.0;
    return interval;
}

wxString FormatLabel(double value)
{
    return wxString::Format("%lld", static_cast<long long>(std::llround(value)));
}

} // namespace

void Draw(wxDC& dc, int left, int top, int height, double topValue, double bottomValue, bool hovered)
{
    if (height <= 0 || topValue <= bottomValue)
        return;

    const double pixelsPerUnit = height / (topValue - bottomValue);

    dc.SetBrush(wxBrush(hovered ? wxColour(48, 48, 48) : wxColour(30, 30, 30)));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(left, top, kWidth, height);
    dc.SetPen(wxPen(hovered ? wxColour(120, 120, 120) : wxColour(70, 70, 70)));
    dc.DrawLine(left, top, left, top + height);

    const double majorInterval = ChooseMajorInterval(pixelsPerUnit);
    const double minorInterval = majorInterval / 5.0;
    const bool drawMinor = (minorInterval * pixelsPerUnit) >= 4.0;

    const double firstTick = std::ceil(bottomValue / majorInterval - 1e-9) * majorInterval;

    dc.SetTextForeground(wxColour(200, 200, 200));

    for (double t = firstTick; t <= topValue + majorInterval; t += majorInterval)
    {
        int y = top + static_cast<int>(height * (topValue - t) / (topValue - bottomValue));
        if (y < top - 1) continue;
        if (y > top + height + 1) break;

        dc.SetPen(wxPen(wxColour(210, 210, 210)));
        dc.DrawLine(left, y, left + 7, y);

        wxString label = FormatLabel(t);
        wxSize extent = dc.GetTextExtent(label);
        dc.DrawText(label, left + 9, y - extent.GetHeight() / 2);

        if (drawMinor)
        {
            for (int k = 1; k <= 4; ++k)
            {
                double mt = t + minorInterval * k;
                if (mt >= t + majorInterval) break;
                int my = top + static_cast<int>(height * (topValue - mt) / (topValue - bottomValue));
                if (my < top || my > top + height) continue;
                dc.SetPen(wxPen(wxColour(90, 90, 90)));
                dc.DrawLine(left, my, left + 4, my);
            }
        }
    }
}

} // namespace verticalaxis
