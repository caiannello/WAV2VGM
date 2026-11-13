#pragma once

#include "PeakAnalysis.h"

#include <vector>

// Chip-independent spectral helpers shared by the OPL3-specific fitter
// (OplFit.cpp) and the chip-neutral one (SineFit.cpp).
namespace fitcommon
{

// Linear-magnitude version of dsp::ComputeSpectrumAtTime.
std::vector<double> ExtractLinearSpectrumAt(const std::vector<float>& samples, int sampleRate, double timeSeconds);

// One linear-magnitude spectrum per frame (frame f centered at
// (f+0.5)/frameRateHz), computed in parallel across hardware threads --
// each frame's FFT is independent of every other frame's.
using SpectrumTable = std::vector<std::vector<double>>;
SpectrumTable PrecomputeSpectra(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                 int totalFrames);

struct SpectralCandidate { double freqHz; double db; double magnitude; };

// Up to `maxCandidates` local-maximum bins in `spectrum`, loudest first,
// at least `minBinDistance` bins apart (scipy.signal.find_peaks's
// `distance`, mirroring the reference Python tool). Bins outside
// [minHz, maxHz] -- what the target chip can physically play -- are
// excluded outright rather than clamped, so unplayable partials don't all
// collapse onto one tone at the range edge.
//
// `refineSubBin` applies parabolic interpolation across the three bins
// around each maximum (on the dB values) to estimate the true peak
// frequency and level between bins. A real partial is rarely bin-centred,
// so this both removes up to ~1.5 dB of level under-read and stops a
// steady tone from being quantized to a neighbouring bin frequency that
// beats against the original. It's opt-in so the existing workflows keep
// producing exactly the same output they always have.
std::vector<SpectralCandidate> FindDistinctPeaks(const std::vector<double>& spectrum, double hzPerBin,
                                                   int minBinDistance, int maxCandidates,
                                                   double minHz, double maxHz, bool refineSubBin = false);

// Magnitude response of the analysis window, for modelling how much of one
// partial's energy spills into a bin that actually belongs to a
// neighbouring partial. Built once per fit (it only depends on
// dsp::kFftSize) and queried by frequency offset.
//
// The analysed spectrum is |sum of each partial's window response|, so two
// partials a bin or two apart each read several dB louder than they really
// are. That's what makes naive per-peak levels overshoot, and what the
// chain workflow's per-frame level solve corrects for.
class LeakageModel
{
public:
    explicit LeakageModel(int fftSize);

    // Relative magnitude response at `deltaHz` from a partial, normalized
    // to 1 at zero offset, linearly interpolated between samples. Returns
    // 0 beyond the range the model covers (Blackman sidelobes are ~-58 dB
    // there and decay from there).
    double Response(double deltaHz, double hzPerBin) const;

    // Widest offset (in bins) the model covers -- what a caller subtracting
    // a partial's footprint needs to iterate over.
    static constexpr int kMaxBins = 8;

private:
    static constexpr int kStepsPerBin = 8;
    std::vector<double> m_response; // index = offset in 1/kStepsPerBin bins
};

double TrendEnergy(const analysis::PeakTrend& trend);

struct TrendLoudestPoint { double freqHz; double magnitudeDb; };
TrendLoudestPoint TrendLoudest(const analysis::PeakTrend& trend);

} // namespace fitcommon
