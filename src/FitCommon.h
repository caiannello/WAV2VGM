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
std::vector<SpectralCandidate> FindDistinctPeaks(const std::vector<double>& spectrum, double hzPerBin,
                                                   int minBinDistance, int maxCandidates,
                                                   double minHz, double maxHz);

double TrendEnergy(const analysis::PeakTrend& trend);

struct TrendLoudestPoint { double freqHz; double magnitudeDb; };
TrendLoudestPoint TrendLoudest(const analysis::PeakTrend& trend);

} // namespace fitcommon
