#pragma once

#include <wx/dc.h>

// Shared time-axis ruler drawing, used by SpectrogramView, kept separate
// so the ruler stays pixel-identical and in sync with the view's own
// scroll/zoom state.
namespace timeruler
{

constexpr int kHeight = 22;            // px reserved at the bottom of the view
constexpr int kMinLabelSpacingPx = 80; // minimum px between adjacent major-tick labels

// Draws a ruler strip of height kHeight starting at y=`top`, spanning the
// visible horizontal range [offsetX, offsetX+visibleWidth) of the caller's
// already-scrolled/prepared wxDC (the same offsetX/device-pixel space the
// caller already uses to draw its own content).
//
// `unitsPerPixel`: device pixels -> the view's native horizontal unit
//   (samples for the waveform, STFT columns for the spectrogram).
// `unitsPerSecond`: that same native unit -> seconds.
// `totalSeconds`: duration of the whole file (not just the visible
//   viewport) -- used only to choose the M:SS.S vs H:MM:SS label format.
// `hovered` lightens the strip's background -- see VerticalAxis.h's
// identical parameter, the same hover cue applied to the other axis.
void Draw(wxDC& dc, int offsetX, int visibleWidth, int top,
          double unitsPerPixel, double unitsPerSecond, double totalSeconds, bool hovered = false);

} // namespace timeruler
