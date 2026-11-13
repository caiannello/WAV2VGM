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
                                                   double minHz, double maxHz, bool refineSubBin)
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
        {
            double db = 20.0 * std::log10(std::max(spectrum[i], 1e-12));
            double peakHz = freqHz;
            if (refineSubBin)
            {
                // Parabolic fit through the three log-magnitudes (the
                // Blackman mainlobe is close enough to parabolic near its
                // top): the vertex offset is in bins, -0.5..+0.5.
                const double a = 20.0 * std::log10(std::max(spectrum[i - 1], 1e-12));
                const double c = 20.0 * std::log10(std::max(spectrum[i + 1], 1e-12));
                const double denom = a - 2.0 * db + c;
                if (std::abs(denom) > 1e-9)
                {
                    const double offset = std::clamp(0.5 * (a - c) / denom, -0.5, 0.5);
                    peakHz = (static_cast<double>(i) + offset) * hzPerBin;
                    db -= 0.25 * (a - c) * offset; // vertex value, likewise in dB
                }
            }
            all.push_back(SpectralCandidate{peakHz, db, std::pow(10.0, db / 20.0)});
        }
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

LeakageModel::LeakageModel(int fftSize)
{
    if (fftSize <= 0)
        return;
    const std::vector<double> window = dsp::MakeBlackmanWindow(fftSize);
    const int padded = fftSize * kStepsPerBin;
    std::vector<std::complex<double>> buf(static_cast<size_t>(padded), std::complex<double>(0.0, 0.0));
    double sum = 0.0;
    for (int i = 0; i < fftSize; ++i)
    {
        buf[static_cast<size_t>(i)] = window[static_cast<size_t>(i)];
        sum += window[static_cast<size_t>(i)];
    }
    if (sum <= 0.0)
        return;
    dsp::FftForwardRadix2(buf);

    // dsp normalizes a bin-centred full-scale sinusoid to 0 dBFS, which is
    // exactly |DTFT| / sum(window) -- so this is the same scale as the
    // magnitudes the fitters work with.
    m_response.resize(static_cast<size_t>(kMaxBins * kStepsPerBin) + 1);
    for (size_t i = 0; i < m_response.size(); ++i)
        m_response[i] = std::abs(buf[i]) / sum;
}

double LeakageModel::Response(double deltaHz, double hzPerBin) const
{
    if (m_response.empty() || hzPerBin <= 0.0)
        return 0.0;
    const double steps = std::abs(deltaHz) / hzPerBin * kStepsPerBin;
    const double limit = static_cast<double>(m_response.size() - 1);
    if (steps >= limit)
        return 0.0;
    const size_t i = static_cast<size_t>(steps);
    const double frac = steps - static_cast<double>(i);
    const double a = m_response[i];
    const double b = m_response[std::min(i + 1, m_response.size() - 1)];
    return a + (b - a) * frac;
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
