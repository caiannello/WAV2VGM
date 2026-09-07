#pragma once

#include <wx/dc.h>

// Shared vertical value-axis drawing, used by SpectrogramView (Hz).
// Mirrors TimeRuler's "nice interval" tick approach, generalized to a
// plain linear value range instead of time.
namespace verticalaxis
{

constexpr int kWidth = 56;             // px reserved on the right of the view
constexpr int kMinLabelSpacingPx = 30; // stacked single-line labels need less clearance than TimeRuler's 80px

// Draws an axis strip of width kWidth at x=`left`, spanning device rows
// [top, top+height). `topValue` is the value at the top of the strip,
// `bottomValue` at the bottom (topValue must be > bottomValue). Values may
// be negative (e.g. waveform counts). `hovered` lightens the strip's
// background -- the same "this is a control" cue as a button's hover
// state -- since the axis being draggable/wheel-zoomable isn't otherwise
// obvious.
void Draw(wxDC& dc, int left, int top, int height, double topValue, double bottomValue, bool hovered = false);

} // namespace verticalaxis
