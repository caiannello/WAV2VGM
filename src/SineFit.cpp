#include "SineFit.h"

#include "FitCommon.h"
#include "PeakAnalysis.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace sinefit
{

using chips::ChipTarget;
using chips::VoiceFrame;
using chips::VoiceKind;

namespace
{

// Per voice kind: where its loudest output lands in dBFS, and the
// quietest level it can still play above silence.
struct KindCalibration
{
    double dbAtMax = 0.0;
    double audibilityFloorDb = 0.0;
};

std::vector<KindCalibration> CalibrateKinds(const ChipTarget& target, int sampleRate)
{
    std::vector<KindCalibration> result;
    for (int k = 0; k < static_cast<int>(target.Kinds().size()); ++k)
    {
        KindCalibration c;
        c.dbAtMax = MeasureKindPeakDb(target, k, sampleRate);
        const VoiceKind& kind = target.Kinds()[static_cast<size_t>(k)];
        c.audibilityFloorDb = c.dbAtMax + (kind.levelDb.empty() ? 0.0 : kind.levelDb.back());
        result.push_back(c);
    }
    return result;
}

bool KindCanPlay(const VoiceKind& kind, const KindCalibration& cal, double freqHz, double db)
{
    return freqHz >= kind.minHz && freqHz <= kind.maxHz && db >= cal.audibilityFloorDb;
}

VoiceFrame MakeFrame(const ChipTarget& target, int voice, const KindCalibration& cal, double freqHz, double db)
{
    VoiceFrame frame;
    target.QuantizePitch(voice, freqHz, frame.pitch);
    frame.level = target.QuantizeLevel(voice, db - cal.dbAtMax);
    frame.keyOn = true;
    return frame;
}

std::vector<float> SumSoloRenders(const ChipTarget& target, const std::vector<VoiceTrack>& tracks, double frameRateHz,
                                  int sampleRate)
{
    // Every target uses its voices as independent, purely additive
    // oscillators, so summing each voice's own render is the chip's real
    // combined output (see the matching note in OplFit.h).
    std::vector<float> mix;
    for (const VoiceTrack& track : tracks)
    {
        const std::vector<float> solo = target.RenderVoice(track.voice, track.frames, frameRateHz, sampleRate);
        if (mix.size() < solo.size())
            mix.resize(solo.size(), 0.0f);
        for (size_t i = 0; i < solo.size(); ++i)
            mix[i] += solo[i];
    }
    for (float& s : mix)
        s = std::clamp(s, -1.0f, 1.0f);
    return mix;
}

} // namespace

double MeasureKindPeakDb(const ChipTarget& target, int kind, int sampleRate)
{
    const int voice = target.FirstVoiceOfKind(kind);
    if (voice < 0 || sampleRate <= 0)
        return -240.0;
    const VoiceKind& vk = target.Kinds()[static_cast<size_t>(kind)];
    double probeHz = 1000.0;
    if (probeHz < vk.minHz || probeHz > vk.maxHz)
        probeHz = std::sqrt(std::max(vk.minHz, 1.0) * vk.maxHz);

    VoiceFrame frame;
    if (!target.QuantizePitch(voice, probeHz, frame.pitch))
        return -240.0;
    frame.level = 0;
    frame.keyOn = true;

    // Same capture as oplfit's own probe: let ~10 ms settle, then one
    // kFftSize-sample Blackman-windowed frame.
    const double probeFrameRate = 100.0;
    const int settleSamples = sampleRate / 100;
    const int n = dsp::kFftSize;
    const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / probeFrameRate)));
    const size_t frameCount = static_cast<size_t>((settleSamples + n) / samplesPerFrame + 2);
    const std::vector<float> rendered =
        target.RenderVoice(voice, std::vector<VoiceFrame>(frameCount, frame), probeFrameRate, sampleRate);
    if (rendered.size() < static_cast<size_t>(settleSamples + n))
        return -240.0;

    static const std::vector<double> window = dsp::MakeBlackmanWindow(n);
    const std::vector<double> dbfs = dsp::ComputeFrameDbfs(rendered.data() + settleSamples, n, window);
    double peak = -240.0;
    for (double db : dbfs)
        peak = std::max(peak, db);
    return peak;
}

ChipFit FitChannelsInTurn(const std::shared_ptr<const ChipTarget>& targetPtr, const std::vector<float>& samples,
                          int sampleRate, double frameRateHz, int maxChannels, const dsp::ProgressCallback& progress)
{
    ChipFit result;
    result.target = targetPtr;
    result.frameRateHz = frameRateHz;
    if (!targetPtr || targetPtr->VoiceCount() <= 0 || samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;
    const ChipTarget& target = *targetPtr;
    maxChannels = std::clamp(maxChannels, 1, target.VoiceCount());

    constexpr double kAnalysisRateOverFrameRate = 4.0;
    const double analysisRateHz = std::clamp(frameRateHz * kAnalysisRateOverFrameRate,
                                              analysis::kMinAnalysisRateHz, analysis::kMaxAnalysisRateHz);
    const analysis::PeakTrendsResult peaks = analysis::ComputePeakTrends(
        samples, sampleRate, frameRateHz, analysisRateHz,
        analysis::kDefaultPeaksPerStep, analysis::kDefaultPeakWindowHalfWidth,
        [&progress](int percent) { if (progress) progress(percent * 20 / 100); });
    if (peaks.trends.empty())
        return result;

    const std::vector<KindCalibration> cal = CalibrateKinds(target, sampleRate);
    const std::vector<VoiceKind>& kinds = target.Kinds();
    double globalFloorDb = 1e300, globalMinHz = 1e300, globalMaxHz = 0.0;
    for (size_t k = 0; k < kinds.size(); ++k)
    {
        globalFloorDb = std::min(globalFloorDb, cal[k].audibilityFloorDb);
        globalMinHz = std::min(globalMinHz, kinds[k].minHz);
        globalMaxHz = std::max(globalMaxHz, kinds[k].maxHz);
    }

    // Trends no voice kind could ever play (too quiet, or loudest point
    // out of every kind's range) are dropped before banding -- see
    // oplfit::FitChannelsInTurn.
    std::vector<size_t> validTrendIdx;
    std::vector<double> energies(peaks.trends.size(), 0.0);
    std::vector<fitcommon::TrendLoudestPoint> loudestOf(peaks.trends.size(), fitcommon::TrendLoudestPoint{0.0, -1e300});
    double minFreqHz = 1e300, maxFreqHz = 0.0;
    for (size_t i = 0; i < peaks.trends.size(); ++i)
    {
        if (peaks.trends[i].framePoints.empty())
            continue;
        const fitcommon::TrendLoudestPoint loudest = fitcommon::TrendLoudest(peaks.trends[i]);
        if (loudest.magnitudeDb < globalFloorDb)
            continue;
        if (loudest.freqHz > globalMaxHz || loudest.freqHz < globalMinHz)
            continue;
        validTrendIdx.push_back(i);
        energies[i] = fitcommon::TrendEnergy(peaks.trends[i]);
        loudestOf[i] = loudest;
        minFreqHz = std::min(minFreqHz, loudest.freqHz);
        maxFreqHz = std::max(maxFreqHz, loudest.freqHz);
    }
    if (validTrendIdx.empty())
        return result;

    // Log-spaced frequency bands, round-robin across bands, loudest
    // first within each -- see oplfit::FitChannelsInTurn.
    struct Candidate { size_t trendIdx; double energy; };
    const int numBands = std::clamp(maxChannels, 1, static_cast<int>(validTrendIdx.size()));
    const double logMinFreq = std::log2(std::max(minFreqHz, 1.0));
    const double logMaxFreq = std::log2(std::max(maxFreqHz, 1.0));
    const double logFreqSpan = std::max(logMaxFreq - logMinFreq, 1e-9);

    std::vector<std::vector<Candidate>> bands(static_cast<size_t>(numBands));
    for (size_t idx : validTrendIdx)
    {
        const double logF = std::log2(std::max(loudestOf[idx].freqHz, 1.0));
        const int band = std::clamp(static_cast<int>((logF - logMinFreq) / logFreqSpan * numBands), 0, numBands - 1);
        bands[static_cast<size_t>(band)].push_back(Candidate{idx, energies[idx]});
    }
    for (std::vector<Candidate>& band : bands)
    {
        std::sort(band.begin(), band.end(), [](const Candidate& a, const Candidate& b) { return a.energy > b.energy; });
        constexpr double kMinRelativeEnergyWithinBand = 0.05;
        if (!band.empty())
        {
            const double bandTop = band.front().energy;
            size_t keep = band.size();
            while (keep > 0 && band[keep - 1].energy < bandTop * kMinRelativeEnergyWithinBand)
                --keep;
            band.resize(keep);
        }
    }

    std::vector<size_t> processingOrder;
    std::vector<size_t> bandCursor(bands.size(), 0);
    for (bool progressed = true; progressed; )
    {
        progressed = false;
        for (size_t b = 0; b < bands.size(); ++b)
        {
            if (bandCursor[b] < bands[b].size())
            {
                processingOrder.push_back(bands[b][bandCursor[b]].trendIdx);
                ++bandCursor[b];
                progressed = true;
            }
        }
    }
    if (progress) progress(20);

    const double frameInterval = 1.0 / frameRateHz;
    const double totalSeconds = static_cast<double>(samples.size()) / sampleRate;
    const int totalFrames = std::max(1, static_cast<int>(std::ceil(totalSeconds / frameInterval)));

    // Greedy voice allocation -- see oplfit::FitChannelsInTurn -- with one
    // addition for mixed-kind targets: a trend only goes onto a voice
    // whose own kind can play its loudest point, and only the frames
    // inside that kind's range are kept.
    struct PendingChannel
    {
        VoiceTrack track;
        std::vector<bool> occupied;
    };
    std::vector<PendingChannel> channels;
    std::vector<bool> voiceOpen(static_cast<size_t>(target.VoiceCount()), false);

    for (size_t rank = 0; rank < processingOrder.size(); ++rank)
    {
        const size_t idx = processingOrder[rank];
        const analysis::PeakTrend& trend = peaks.trends[idx];
        const fitcommon::TrendLoudestPoint& loudest = loudestOf[idx];

        auto pointsForKind = [&](int kindIdx) {
            const VoiceKind& kind = kinds[static_cast<size_t>(kindIdx)];
            std::vector<std::pair<int, const analysis::SpectralPeak*>> points;
            for (const analysis::SpectralPeak& p : trend.framePoints)
            {
                if (p.freqHz > kind.maxHz || p.freqHz < kind.minHz)
                    continue;
                const long long f = std::llround(p.timeSeconds * frameRateHz);
                if (f >= 0 && f < totalFrames)
                    points.emplace_back(static_cast<int>(f), &p);
            }
            return points;
        };
        auto kindAccepts = [&](int kindIdx) {
            return KindCanPlay(kinds[static_cast<size_t>(kindIdx)], cal[static_cast<size_t>(kindIdx)], loudest.freqHz,
                               loudest.magnitudeDb);
        };

        PendingChannel* targetChannel = nullptr;
        std::vector<std::pair<int, const analysis::SpectralPeak*>> points;
        for (PendingChannel& channel : channels)
        {
            const int kindIdx = target.KindIndexOf(channel.track.voice);
            if (!kindAccepts(kindIdx))
                continue;
            std::vector<std::pair<int, const analysis::SpectralPeak*>> candidatePoints = pointsForKind(kindIdx);
            if (candidatePoints.empty())
                continue;
            bool overlaps = false;
            for (const auto& fp : candidatePoints)
            {
                if (channel.occupied[static_cast<size_t>(fp.first)])
                {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps)
            {
                targetChannel = &channel;
                points = std::move(candidatePoints);
                break;
            }
        }
        if (!targetChannel)
        {
            if (static_cast<int>(channels.size()) >= maxChannels)
                continue;
            int newVoice = -1;
            for (int v = 0; v < target.VoiceCount(); ++v)
            {
                if (!voiceOpen[static_cast<size_t>(v)] && kindAccepts(target.KindIndexOf(v)))
                {
                    newVoice = v;
                    break;
                }
            }
            if (newVoice < 0)
                continue;
            points = pointsForKind(target.KindIndexOf(newVoice));
            if (points.empty())
                continue;
            voiceOpen[static_cast<size_t>(newVoice)] = true;
            channels.push_back(PendingChannel{});
            targetChannel = &channels.back();
            targetChannel->track.voice = newVoice;
            targetChannel->track.frames.assign(static_cast<size_t>(totalFrames), target.SilentFrame(newVoice));
            targetChannel->occupied.assign(static_cast<size_t>(totalFrames), false);
        }

        const int voice = targetChannel->track.voice;
        const KindCalibration& kc = cal[static_cast<size_t>(target.KindIndexOf(voice))];
        for (const auto& fp : points)
        {
            const analysis::SpectralPeak& p = *fp.second;
            targetChannel->track.frames[static_cast<size_t>(fp.first)] = MakeFrame(target, voice, kc, p.freqHz, p.magnitudeDb);
            targetChannel->occupied[static_cast<size_t>(fp.first)] = true;
        }

        if (progress) progress(20 + static_cast<int>(rank + 1) * 60 / static_cast<int>(processingOrder.size()));
    }

    for (PendingChannel& channel : channels)
        result.tracks.push_back(std::move(channel.track));
    result.mixRendered = SumSoloRenders(target, result.tracks, frameRateHz, sampleRate);

    if (progress) progress(100);
    return result;
}

ChipFit FitChannelsPerFrame(const std::shared_ptr<const ChipTarget>& targetPtr, const std::vector<float>& samples,
                            int sampleRate, double frameRateHz, int maxChannels, const dsp::ProgressCallback& progress)
{
    ChipFit result;
    result.target = targetPtr;
    result.frameRateHz = frameRateHz;
    if (!targetPtr || targetPtr->VoiceCount() <= 0 || samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;
    const ChipTarget& target = *targetPtr;
    maxChannels = std::clamp(maxChannels, 1, target.VoiceCount());

    const double frameInterval = 1.0 / frameRateHz;
    const double totalSeconds = static_cast<double>(samples.size()) / sampleRate;
    const int totalFrames = std::max(1, static_cast<int>(std::ceil(totalSeconds / frameInterval)));
    const double hzPerBin = static_cast<double>(sampleRate) / dsp::kFftSize;

    const fitcommon::SpectrumTable spectra = fitcommon::PrecomputeSpectra(samples, sampleRate, frameRateHz, totalFrames);
    if (progress) progress(30);

    const std::vector<KindCalibration> cal = CalibrateKinds(target, sampleRate);
    const std::vector<VoiceKind>& kinds = target.Kinds();

    double minHz = 1e300, maxHz = 0.0;
    result.tracks.resize(static_cast<size_t>(maxChannels));
    for (int v = 0; v < maxChannels; ++v)
    {
        VoiceTrack& track = result.tracks[static_cast<size_t>(v)];
        track.voice = v;
        track.frames.assign(static_cast<size_t>(totalFrames), target.SilentFrame(v));
        minHz = std::min(minHz, target.KindOf(v).minHz);
        maxHz = std::max(maxHz, target.KindOf(v).maxHz);
    }

    constexpr int kMinPeakBinDistance = 5;
    const int reportStep = std::max(1, totalFrames / 100);
    std::vector<bool> used(static_cast<size_t>(maxChannels));
    for (int f = 0; f < totalFrames; ++f)
    {
        if (progress && (f % reportStep) == 0) progress(30 + f * 60 / totalFrames);

        const std::vector<double>& spectrum =
            (static_cast<size_t>(f) < spectra.size()) ? spectra[static_cast<size_t>(f)] : std::vector<double>{};
        const std::vector<fitcommon::SpectralCandidate> peaks =
            fitcommon::FindDistinctPeaks(spectrum, hzPerBin, kMinPeakBinDistance, maxChannels, minHz, maxHz);

        // Loudest candidate to the lowest free voice that can play it --
        // for a single-kind target that's simply channel = rank, as in
        // oplfit::FitChannelsPerFrame.
        std::fill(used.begin(), used.end(), false);
        for (const fitcommon::SpectralCandidate& c : peaks)
        {
            for (int v = 0; v < maxChannels; ++v)
            {
                const int kindIdx = target.KindIndexOf(v);
                if (used[static_cast<size_t>(v)]
                    || !KindCanPlay(kinds[static_cast<size_t>(kindIdx)], cal[static_cast<size_t>(kindIdx)], c.freqHz, c.db))
                    continue;
                result.tracks[static_cast<size_t>(v)].frames[static_cast<size_t>(f)] =
                    MakeFrame(target, v, cal[static_cast<size_t>(kindIdx)], c.freqHz, c.db);
                used[static_cast<size_t>(v)] = true;
                break;
            }
        }
    }
    if (progress) progress(90);

    result.mixRendered = SumSoloRenders(target, result.tracks, frameRateHz, sampleRate);
    if (progress) progress(100);
    return result;
}

} // namespace sinefit
