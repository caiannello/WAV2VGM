#include "PeakAnalysis.h"

#include <algorithm>
#include <cmath>
#include <thread>

namespace analysis
{

namespace
{

// A trend survives this many consecutive internal steps with no matching
// peak before it's closed -- "even if the peak isn't detected in a couple
// places in between, ... consider such peaks as connected".
constexpr int kMaxTrackGapSteps = 2;

// Two peaks in adjacent steps are considered the same partial if their
// frequencies differ by no more than this. A handful of FFT bins' worth
// (bin spacing = sampleRate/kFftSize, ~10.8 Hz at 44100 Hz) gives enough
// slack for real drift without bridging distinct partials.
constexpr double kTrackFreqToleranceHz = 60.0;

// One analysis step's up-to-maxPeaksPerStep peaks, loudest first, each
// refined via a linear-magnitude-weighted center-of-mass over
// [bin-halfWidth, bin+halfWidth]. Factored out of the old flat
// ComputePeakTrends so it can run at the finer analysisRateHz.
std::vector<std::vector<SpectralPeak>> ComputeStepPeaks(const std::vector<float>& samples, int sampleRate,
                                                          double analysisRateHz, int maxPeaksPerStep,
                                                          int halfWidth,
                                                          const dsp::ProgressCallback& progress)
{
    const int n = dsp::kFftSize;
    const int total = static_cast<int>(samples.size());
    const double stepSamples = sampleRate / analysisRateHz;
    const int numSteps = static_cast<int>(std::floor(total / stepSamples)) + 1;

    std::vector<std::vector<SpectralPeak>> peaksByStep(static_cast<size_t>(std::max(0, numSteps)));
    if (numSteps <= 0)
        return peaksByStep;

    const std::vector<double> window = dsp::MakeBlackmanWindow(n);
    const double hzPerBin = static_cast<double>(sampleRate) / n;

    // Each step's FFT/peak-picking only touches its own stack-local
    // buffers plus the read-only shared window -- completely independent
    // of every other step, the same precondition PrecomputeSpectra (in
    // OplFit.cpp) relies on for its own parallelization. This is the
    // single most expensive part of trend tracking (up to ~1378 steps per
    // second of input), and unlike the mutual-nearest-neighbor linking
    // that consumes this table afterward, has no sequential dependency
    // between steps at all, so it's safe to spread across hardware
    // threads the same way.
    auto computeStep = [&](int step) {
        const int start = static_cast<int>(std::llround(step * stepSamples));
        if (start >= total)
            return;

        std::vector<float> stepBuf(static_cast<size_t>(n));
        const int available = std::clamp(total - start, 0, n);
        for (int i = 0; i < available; ++i)
            stepBuf[static_cast<size_t>(i)] = samples[static_cast<size_t>(start + i)];
        for (int i = available; i < n; ++i)
            stepBuf[static_cast<size_t>(i)] = 0.0f;

        const std::vector<double> dbfs = dsp::ComputeFrameDbfs(stepBuf.data(), n, window);
        const int binCount = static_cast<int>(dbfs.size());
        const double timeSeconds = static_cast<double>(start) / sampleRate;

        // A candidate must beat every neighbor within +/-halfWidth, not just
        // the immediate one -- a real several-bins-wide spectral lobe still
        // clears this easily, but a single noisy one-bin ripple usually
        // won't, so widening the window is what actually cuts down on
        // spurious peaks.
        std::vector<int> maximaIdx;
        for (int bin = halfWidth; bin < binCount - halfWidth; ++bin)
        {
            bool isPeak = true;
            for (int k = 1; k <= halfWidth; ++k)
            {
                if (dbfs[static_cast<size_t>(bin)] <= dbfs[static_cast<size_t>(bin - k)]
                    || dbfs[static_cast<size_t>(bin)] <= dbfs[static_cast<size_t>(bin + k)])
                {
                    isPeak = false;
                    break;
                }
            }
            if (isPeak)
                maximaIdx.push_back(bin);
        }

        std::vector<SpectralPeak> stepPeaks;
        stepPeaks.reserve(maximaIdx.size());
        for (int bin : maximaIdx)
        {
            // Center of mass over the same window, weighted by linear
            // magnitude (dB is already log-compressed, so it isn't a
            // physically meaningful "mass" to average). The resulting
            // amplitude is the lobe's total magnitude, not just the single
            // tallest bin's -- a narrow noise spike has little of that
            // spread-out mass compared to a genuine partial, which helps
            // it lose out once peaks are ranked and capped below.
            double massSum = 0.0, weightedBin = 0.0;
            for (int k = -halfWidth; k <= halfWidth; ++k)
            {
                const double mag = std::pow(10.0, dbfs[static_cast<size_t>(bin + k)] / 20.0);
                massSum += mag;
                weightedBin += mag * (bin + k);
            }
            const double centroidBin = (massSum > 0.0) ? (weightedBin / massSum) : static_cast<double>(bin);
            const double centroidDb = 20.0 * std::log10(std::max(massSum, 1e-12));
            stepPeaks.push_back(SpectralPeak{timeSeconds, centroidBin * hzPerBin, centroidDb});
        }

        std::sort(stepPeaks.begin(), stepPeaks.end(),
                  [](const SpectralPeak& a, const SpectralPeak& b) { return a.magnitudeDb > b.magnitudeDb; });
        if (static_cast<int>(stepPeaks.size()) > maxPeaksPerStep)
            stepPeaks.resize(static_cast<size_t>(maxPeaksPerStep));

        peaksByStep[static_cast<size_t>(step)] = std::move(stepPeaks);
    };

    const unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
    const int numThreads = static_cast<int>(std::min<unsigned int>(hwThreads, static_cast<unsigned int>(numSteps)));
    if (numThreads <= 1)
    {
        const int reportStep = std::max(1, numSteps / 100);
        for (int step = 0; step < numSteps; ++step)
        {
            if (progress && (step % reportStep) == 0)
                progress((step * 100) / std::max(1, numSteps - 1));
            computeStep(step);
        }
        if (progress) progress(100);
        return peaksByStep;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(numThreads));
    const int chunk = (numSteps + numThreads - 1) / numThreads;
    for (int t = 0; t < numThreads; ++t)
    {
        const int start = t * chunk;
        const int end = std::min(numSteps, start + chunk);
        if (start >= end)
            break;
        workers.emplace_back([&computeStep, start, end]() {
            for (int step = start; step < end; ++step)
                computeStep(step);
        });
    }
    for (std::thread& worker : workers)
        worker.join();

    if (progress) progress(100);
    return peaksByStep;
}

// Resamples a trend's analysis-step points onto the global frame grid
// (f/frameRateHz for integer f), linearly interpolating frequency and
// amplitude between the two points bracketing each frame time -- this is
// what playback will actually do between frames, so the connecting line
// should show exactly that, not a polyline through every analysis step.
// Frame times outside the trend's own span are skipped rather than
// extrapolated, so a trend shorter than one frame interval yields no
// frame points (and so draws no line) at all.
std::vector<SpectralPeak> ResampleToFrameGrid(const std::vector<SpectralPeak>& points, double frameRateHz)
{
    std::vector<SpectralPeak> framePoints;
    if (points.empty())
        return framePoints;

    const double frameInterval = 1.0 / frameRateHz;
    const double tStart = points.front().timeSeconds;
    const double tEnd = points.back().timeSeconds;
    const long long fFirst = static_cast<long long>(std::ceil(tStart / frameInterval - 1e-9));
    const long long fLast = static_cast<long long>(std::floor(tEnd / frameInterval + 1e-9));

    size_t idx = 0; // points[idx] is the last point with timeSeconds <= current frame time
    for (long long f = fFirst; f <= fLast; ++f)
    {
        const double tFrame = static_cast<double>(f) * frameInterval;
        while (idx + 1 < points.size() && points[idx + 1].timeSeconds < tFrame)
            ++idx;

        if (idx + 1 >= points.size() || tFrame <= points[idx].timeSeconds)
        {
            const SpectralPeak& p = points[idx];
            framePoints.push_back(SpectralPeak{tFrame, p.freqHz, p.magnitudeDb});
            continue;
        }

        const SpectralPeak& a = points[idx];
        const SpectralPeak& b = points[idx + 1];
        const double t = (tFrame - a.timeSeconds) / (b.timeSeconds - a.timeSeconds);
        framePoints.push_back(SpectralPeak{
            tFrame,
            a.freqHz + t * (b.freqHz - a.freqHz),
            a.magnitudeDb + t * (b.magnitudeDb - a.magnitudeDb)
        });
    }

    return framePoints;
}

} // namespace

PeakTrendsResult ComputePeakTrends(const std::vector<float>& samples, int sampleRate,
                                    double frameRateHz, double analysisRateHz, int maxPeaksPerStep,
                                    int peakWindowHalfWidth, const dsp::ProgressCallback& progress)
{
    PeakTrendsResult result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;

    maxPeaksPerStep = std::clamp(maxPeaksPerStep, kMinPeaksPerStep, kMaxPeaksPerStep);
    peakWindowHalfWidth = std::clamp(peakWindowHalfWidth, kMinPeakWindowHalfWidth, kMaxPeakWindowHalfWidth);
    // Analyzing slower than the output frame rate would leave gaps in it,
    // so the floor tracks frameRateHz rather than the fixed
    // kMinAnalysisRateHz alone.
    analysisRateHz = std::clamp(analysisRateHz, std::max(kMinAnalysisRateHz, frameRateHz), kMaxAnalysisRateHz);
    const std::vector<std::vector<SpectralPeak>> steps = ComputeStepPeaks(
        samples, sampleRate, analysisRateHz, maxPeaksPerStep, peakWindowHalfWidth, progress);

    struct ActiveTrack
    {
        std::vector<SpectralPeak> points;
        int gap = 0;
    };
    std::vector<ActiveTrack> active;

    auto finalizeTrack = [&result](ActiveTrack&& track) {
        if (track.points.size() >= 2)
            result.trends.push_back(PeakTrend{std::move(track.points), {}});
        else
            result.isolatedPeaks.push_back(track.points.front());
    };

    for (const std::vector<SpectralPeak>& step : steps)
    {
        // Match active tracks to this step's peaks by mutual nearest
        // neighbor: track t only connects to peak p if p is t's closest
        // candidate *and* t is p's closest candidate. A one-directional
        // scan (each track just grabbing whichever unclaimed peak is
        // nearest, in creation order) can hand a track a peak that, from
        // that peak's own point of view, actually belongs to a different,
        // closer track -- order-dependent and wrong whenever two peaks
        // pass close to each other. Requiring agreement in both directions
        // rules that out and is independent of iteration order.
        std::vector<int> forward(active.size(), -1); // forward[t] = nearest peak index for track t
        for (size_t t = 0; t < active.size(); ++t)
        {
            const double lastFreq = active[t].points.back().freqHz;
            int bestIdx = -1;
            double bestDist = kTrackFreqToleranceHz;
            for (size_t p = 0; p < step.size(); ++p)
            {
                const double dist = std::abs(step[p].freqHz - lastFreq);
                if (dist <= bestDist)
                {
                    bestDist = dist;
                    bestIdx = static_cast<int>(p);
                }
            }
            forward[t] = bestIdx;
        }

        std::vector<int> backward(step.size(), -1); // backward[p] = nearest track index for peak p
        for (size_t p = 0; p < step.size(); ++p)
        {
            int bestIdx = -1;
            double bestDist = kTrackFreqToleranceHz;
            for (size_t t = 0; t < active.size(); ++t)
            {
                const double dist = std::abs(step[p].freqHz - active[t].points.back().freqHz);
                if (dist <= bestDist)
                {
                    bestDist = dist;
                    bestIdx = static_cast<int>(t);
                }
            }
            backward[p] = bestIdx;
        }

        std::vector<bool> claimed(step.size(), false);
        for (size_t t = 0; t < active.size(); ++t)
        {
            const int p = forward[t];
            if (p >= 0 && backward[static_cast<size_t>(p)] == static_cast<int>(t))
            {
                active[t].points.push_back(step[static_cast<size_t>(p)]);
                claimed[static_cast<size_t>(p)] = true;
                active[t].gap = 0;
            }
            else
            {
                ++active[t].gap;
            }
        }

        // Close out tracks that have gone too long without a match.
        for (size_t i = 0; i < active.size(); )
        {
            if (active[i].gap > kMaxTrackGapSteps)
            {
                finalizeTrack(std::move(active[i]));
                active.erase(active.begin() + static_cast<long>(i));
            }
            else
            {
                ++i;
            }
        }

        // Unclaimed peaks start new candidate tracks.
        for (size_t i = 0; i < step.size(); ++i)
        {
            if (!claimed[i])
                active.push_back(ActiveTrack{ {step[i]}, 0 });
        }
    }

    for (ActiveTrack& track : active)
        finalizeTrack(std::move(track));

    for (PeakTrend& trend : result.trends)
        trend.framePoints = ResampleToFrameGrid(trend.points, frameRateHz);

    if (progress) progress(100);
    return result;
}

} // namespace analysis
