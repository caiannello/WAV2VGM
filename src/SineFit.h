#pragma once

#include "Dsp.h"
#include "chips/ChipTarget.h"

#include <memory>
#include <vector>

// Chip-neutral versions of oplfit's two pure-sine fitting workflows
// (FitChannelsInTurn / FitChannelsPerFrame -- see OplFit.h for the full
// reasoning behind each), driven by a chips::ChipTarget instead of
// OPL3's own registers. Frequencies and levels are quantized through the
// target, each voice kind's loudness is calibrated by rendering it, and
// heterogeneous targets (OPL4 FM + PCM) only ever place a partial on a
// voice whose own range can play it. For the OPL3 target the results
// match oplfit's exactly.
namespace sinefit
{

struct VoiceTrack
{
    int voice = 0; // the target voice this track drives
    std::vector<chips::VoiceFrame> frames; // frames[i] covers [i/frameRateHz, (i+1)/frameRateHz)
};

struct ChipFit
{
    std::shared_ptr<const chips::ChipTarget> target;
    double frameRateHz = 0.0;
    std::vector<VoiceTrack> tracks;
    std::vector<float> mixRendered; // every track's solo render, summed and clamped
};

// Trend-based workflow: see oplfit::FitChannelsInTurn. maxChannels is
// clamped to [1, target->VoiceCount()].
ChipFit FitChannelsInTurn(const std::shared_ptr<const chips::ChipTarget>& target, const std::vector<float>& samples,
                          int sampleRate, double frameRateHz, int maxChannels,
                          const dsp::ProgressCallback& progress = nullptr);

// Per-frame workflow: see oplfit::FitChannelsPerFrame. Uses the target's
// first maxChannels voices (clamped to [1, target->VoiceCount()]) and
// always returns exactly that many tracks.
ChipFit FitChannelsPerFrame(const std::shared_ptr<const chips::ChipTarget>& target, const std::vector<float>& samples,
                            int sampleRate, double frameRateHz, int maxChannels,
                            const dsp::ProgressCallback& progress = nullptr);

// Peak dBFS of one voice of `kind` playing a steady ~1 kHz tone at its
// loudest level, measured the same way the recording itself is analyzed
// -- the anchor that maps target dB onto the voice's level table.
double MeasureKindPeakDb(const chips::ChipTarget& target, int kind, int sampleRate);

} // namespace sinefit
