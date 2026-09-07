#pragma once

#include "Dsp.h"

#include <vector>

namespace analysis
{

struct SpectralPeak
{
    double timeSeconds;
    double freqHz;
    double magnitudeDb;
};

constexpr int kMinPeaksPerStep = 1;
constexpr int kMaxPeaksPerStep = 128;
constexpr int kDefaultPeaksPerStep = 128;

// The output frame rate: how often oscillator parameters (frequency,
// volume) may change during eventual synthesis playback, which is also
// what the exported file's size scales with. A PeakTrend's framePoints
// (see below) sit exactly on this grid, one sample per frame -- playback
// linearly interpolates frequency/volume between them, so this is also
// what determines the connecting line's actual endpoints, not just a
// display rate.
constexpr double kMinFrameRateHz = 1.0;
constexpr double kMaxFrameRateHz = 1378.0;
constexpr double kDefaultFrameRateHz = 200.0;

// The rate peaks are actually detected at internally, independent of and
// normally higher than frameRateHz -- frameRateHz is a separate, coarser
// grid that the result gets resampled onto afterwards (see framePoints).
// Denser analysis gives trend-tracking more, closer-spaced samples to
// decide which peak in one step should link to which peak in the next, so
// that resampled result follows real frequency movement more faithfully
// even though the output frame rate is much lower. 1378 Hz is the
// spectrogram's own native rate (sampleRate/kHopSize = 44100/32), i.e. a
// peak search on every spectrogram column -- no reason to ever analyze
// finer than that.
constexpr double kMinAnalysisRateHz = 1.0;
constexpr double kMaxAnalysisRateHz = 1378.0;
constexpr double kDefaultAnalysisRateHz = 1378.0;

// How many bins on each side of a candidate a peak must beat to count as a
// local maximum, and how wide a window its final frequency/amplitude is
// estimated over (see ComputePeakTrends).
constexpr int kMinPeakWindowHalfWidth = 1;
constexpr int kMaxPeakWindowHalfWidth = 8;
constexpr int kDefaultPeakWindowHalfWidth = 2;

// A sequence of >=2 time-ordered peaks believed to be the same underlying
// partial, linked across consecutive (or near-consecutive -- see
// ComputePeakTrends) internal analysis steps. Meant to eventually drive one
// sine oscillator per trend, reused across its whole lifetime so phase
// stays continuous.
struct PeakTrend
{
    // Every analysis-step point that linked into this trend, at whatever
    // resolution analysisRateHz produced. Kept around for display (a dot
    // per point) and debugging while the analysis approach is still being
    // tuned; not what drives playback.
    std::vector<SpectralPeak> points;

    // This trend resampled onto the frame-rate grid (one point per output
    // frame the trend spans, at exactly f/frameRateHz), each frequency/
    // amplitude linearly interpolated from the surrounding points above.
    // This -- not `points` -- is the actual intended synthesis data
    // (oscillator playback linearly interpolates between these) and so is
    // also what the connecting line's endpoints are drawn from.
    std::vector<SpectralPeak> framePoints;
};

struct PeakTrendsResult
{
    std::vector<PeakTrend> trends;
    std::vector<SpectralPeak> isolatedPeaks; // peaks that never linked to any neighbor
};

// Internally analyzes at analysisRateHz (clamped to
// [max(kMinAnalysisRateHz, frameRateHz), kMaxAnalysisRateHz] -- analyzing
// slower than the output frame rate would leave gaps in it) so brief
// frequency drift or a couple of missed detections between analysis steps
// don't break a trend, then links each step's peaks to the neighboring
// step's peaks by mutual nearest neighbor (within tolerance) -- a link is
// only made when each side is the other's closest candidate, checking
// both directions so a peak never gets claimed by a track it isn't
// actually closest to just because that track happened to look first. A
// trend survives up to kMaxTrackGapSteps consecutive steps with no match before
// it's closed. Peaks that never link to anything are still reported (as
// isolatedPeaks), for display. Every resulting trend is then resampled
// onto the frameRateHz grid -- see PeakTrend::framePoints.
//
// A bin only qualifies as a candidate peak if it's strictly louder than
// its peakWindowHalfWidth neighbors on each side (clamped to
// [kMinPeakWindowHalfWidth, kMaxPeakWindowHalfWidth]) -- wider windows
// reject narrow one-bin noise ripples that a real, several-bins-wide
// spectral lobe wouldn't trip. Each surviving candidate's reported
// frequency and amplitude are then a center-of-mass over that same
// window (linear-magnitude-weighted), not just the single tallest bin, so
// the estimate reflects the whole lobe rather than one noisy sample of it.
//
// maxPeaksPerStep caps how many peaks (loudest first, by that
// center-of-mass amplitude) are kept per analysis step, clamped to
// [kMinPeaksPerStep, kMaxPeaksPerStep].
PeakTrendsResult ComputePeakTrends(const std::vector<float>& samples, int sampleRate,
                                    double frameRateHz, double analysisRateHz = kDefaultAnalysisRateHz,
                                    int maxPeaksPerStep = kDefaultPeaksPerStep,
                                    int peakWindowHalfWidth = kDefaultPeakWindowHalfWidth,
                                    const dsp::ProgressCallback& progress = nullptr);

} // namespace analysis
