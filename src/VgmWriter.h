#pragma once

#include "OplFit.h"

#include <string>
#include <vector>

// Writes an OPL3 channel approximation (as built up by the OPL3 analysis
// mode's Start button -- see oplfit::FitChannelsInTurn or
// oplfit::FitChannelsPerFrame, whichever the mode's fitting radio box
// selected) out to a standard VGM file, playable on real YMF262 hardware
// or in any VGM player (e.g. Winamp's in_vgm plugin).
namespace vgm
{

// `channels[i]` is written to real OPL3 channel i (same convention as
// oplfit::RenderAllFramesMix). All channels should share the same frame
// rate; shorter ones just hold their last register state for any
// remaining frames. Only register writes that actually change a value
// are emitted -- the standard VGM size optimization -- computed here at
// export time rather than by OplPatch.h's SetupChannel/ApplyFrame
// themselves, which always (re)write every relevant register so a live
// chip stays correct via dbopl's own change-detection.
bool WriteVgmFile(const std::string& path, const std::vector<oplfit::AllFramesFit>& channels,
                   std::string& errorMessage);

} // namespace vgm
