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

// How faithful a fit is, so the workflows can be compared without relying
// on listening alone.
struct FitScore
{
    // Magnitude-weighted mean |original - rendered| per spectrum bin, in
    // dB, sampled across the recording (bins are weighted by how much
    // signal either side has there, so silent bins don't dominate).
    double meanAbsDbError = 0.0;
    // exp(-meanAbsDbError / 12), a convenient 0..1 summary: 0 dB error is
    // 1.0, ~3 dB is 0.78, ~6 dB is 0.61.
    double similarity = 0.0;
    // Number of silent -> sounding transitions across all voices. Each one
    // is a re-attack (phase reset, envelope restart) on real hardware, so
    // lower is smoother for the same fidelity.
    int keyOnCount = 0;
    // Number of times a voice that keeps sounding jumps by more than a
    // semitone from one frame to the next. This is what a "channel swapped
    // to a different partial" reassignment sounds like, and unlike key-ons
    // it can't be hidden by simply never keying off.
    int pitchJumpCount = 0;
};

struct ChipFit
{
    std::shared_ptr<const chips::ChipTarget> target;
    double frameRateHz = 0.0;
    std::vector<VoiceTrack> tracks;
    std::vector<float> mixRendered; // every track's solo render, summed and clamped
    FitScore score;
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

// Chain-based workflow: the per-frame density of FitChannelsPerFrame with
// the persistent voice identity of FitChannelsInTurn.
//
// It extracts one continuous pitch path ("chain") at a time from the
// residual spectrum, assigns it to whichever voice is free for its whole
// lifetime, models that chain's leakage and subtracts it from the
// residual, and repeats. Because a chain is found by dynamic programming
// over every frame, it can be as short as a single frame (a transient is
// never dropped for being too brief) while a sustained partial still comes
// out as one unbroken chain on one voice, so there's no per-frame
// reassignment and no re-attack mid-note. Because each iteration works on
// the residual, a voice is never spent re-covering a partial that's
// already represented, unlike ranking peaks independently per frame.
//
// Finally each frame's levels are solved jointly (see LeakageModel) so
// overlapping partials aren't each set as loud as the sum they appear to
// have in the analysis spectrum.
ChipFit FitChannelsByChains(const std::shared_ptr<const chips::ChipTarget>& target, const std::vector<float>& samples,
                            int sampleRate, double frameRateHz, int maxChannels,
                            const dsp::ProgressCallback& progress = nullptr);

// Fills in `FitScore` for an already-fitted mix.
FitScore ScoreFit(const chips::ChipTarget& target, const std::vector<float>& samples, int sampleRate,
                  double frameRateHz, const std::vector<VoiceTrack>& tracks, const std::vector<float>& mixRendered);

} // namespace sinefit
