#include "FitCommon.h"

#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace fitcommon
{

std::vector<double> ExtractLinearSpectrumAt(const std::vector<float>& samples, int sampleRate, double timeSeconds)
{
    const std::vector<double> dbfs = dsp::ComputeSpectrumAtTime(samples, sampleRate, timeSeconds);
    std::vector<double> linear(dbfs.size());
    for (size_t i = 0; i < dbfs.size(); ++i)
        linear[i] = std::pow(10.0, dbfs[i] / 20.0);
    return linear;
}

SpectrumTable PrecomputeSpectra(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                 int totalFrames)
{
    SpectrumTable table(static_cast<size_t>(std::max(0, totalFrames)));
    if (totalFrames <= 0)
        return table;

    const double frameInterval = 1.0 / frameRateHz;
    auto computeRange = [&](int startFrame, int endFrame) {
        for (int f = startFrame; f < endFrame; ++f)
        {
            const double frameMid = (static_cast<double>(f) + 0.5) * frameInterval;
            table[static_cast<size_t>(f)] = ExtractLinearSpectrumAt(samples, sampleRate, frameMid);
        }
    };

    const unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
    const int numThreads = static_cast<int>(std::min<unsigned int>(hwThreads, static_cast<unsigned int>(totalFrames)));
    if (numThreads <= 1)
    {
        computeRange(0, totalFrames);
        return table;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(numThreads));
    const int chunk = (totalFrames + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; ++t)
    {
        const int start = t * chunk;
        const int end = std::min(totalFrames, start + chunk);
        if (start >= end)
            break;
        workers.emplace_back(computeRange, start, end);
    }
    for (std::thread& worker : workers)
        worker.join();

    return table;
}

std::vector<SpectralCandidate> FindDistinctPeaks(const std::vector<double>& spectrum, double hzPerBin,
                                                   int minBinDistance, int maxCandidates,
                                                   double minHz, double maxHz)
{
    std::vector<SpectralCandidate> all;
    for (size_t i = 1; i + 1 < spectrum.size(); ++i)
    {
        const double freqHz = static_cast<double>(i) * hzPerBin;
        if (freqHz > maxHz)
            break; // bins are in ascending frequency order, so nothing further out matters either
        if (freqHz < minHz)
            continue;
        if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1])
            all.push_back(SpectralCandidate{freqHz, 20.0 * std::log10(std::max(spectrum[i], 1e-12)), spectrum[i]});
    }
    std::sort(all.begin(), all.end(),
              [](const SpectralCandidate& a, const SpectralCandidate& b) { return a.magnitude > b.magnitude; });

    std::vector<SpectralCandidate> kept;
    const double minHzDistance = minBinDistance * hzPerBin;
    for (const SpectralCandidate& c : all)
    {
        if (static_cast<int>(kept.size()) >= maxCandidates)
            break;
        bool tooClose = false;
        for (const SpectralCandidate& k : kept)
        {
            if (std::abs(c.freqHz - k.freqHz) < minHzDistance)
            {
                tooClose = true;
                break;
            }
        }
        if (!tooClose)
            kept.push_back(c);
    }
    return kept;
}

double TrendEnergy(const analysis::PeakTrend& trend)
{
    double energy = 0.0;
    for (const analysis::SpectralPeak& p : trend.framePoints)
        energy += std::pow(10.0, p.magnitudeDb / 20.0);
    return energy;
}

TrendLoudestPoint TrendLoudest(const analysis::PeakTrend& trend)
{
    TrendLoudestPoint best{0.0, -1e300};
    for (const analysis::SpectralPeak& p : trend.framePoints)
        if (p.magnitudeDb > best.magnitudeDb)
            best = TrendLoudestPoint{p.freqHz, p.magnitudeDb};
    return best;
}

} // namespace fitcommon
