#include "OplFit.h"

#include "OplChip.h"
#include "PeakAnalysis.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <utility>

namespace oplfit
{

namespace
{

// Bessel function of the first kind, via its defining power series --
// converges quickly for the modest modulation indices this model deals
// in (see kBetaScale below), no external math library needed.
double BesselJ(int k, double x)
{
    if (k < 0)
        return ((-k) % 2 == 0 ? 1.0 : -1.0) * BesselJ(-k, x);

    const double halfX = x / 2.0;
    double term = 1.0;
    for (int i = 1; i <= k; ++i)
        term *= halfX / i;
    double sum = term;
    const double negHalfXSq = -halfX * halfX;
    for (int m = 1; m <= 40; ++m)
    {
        term *= negHalfXSq / (m * (m + k));
        sum += term;
        if (std::abs(term) < 1e-12 * std::abs(sum) + 1e-300)
            break;
    }
    return sum;
}

// OPL total level is attenuation in ~0.75 dB steps (0 = loudest, 63 = most
// attenuated) -- standard OPL2/3 hardware behavior.
constexpr double kDbPerTotalLevelStep = 0.75;

double TotalLevelToLinearAmplitude(uint8_t totalLevel)
{
    return std::pow(10.0, -(totalLevel * kDbPerTotalLevelStep) / 20.0);
}

// Linear-magnitude version of dsp::ComputeSpectrumAtTime -- the local
// spectral "shape" a timbre search tries to match, not just the single
// detected peak bin.
std::vector<double> ExtractLinearSpectrumAt(const std::vector<float>& samples, int sampleRate, double timeSeconds)
{
    const std::vector<double> dbfs = dsp::ComputeSpectrumAtTime(samples, sampleRate, timeSeconds);
    std::vector<double> linear(dbfs.size());
    for (size_t i = 0; i < dbfs.size(); ++i)
        linear[i] = std::pow(10.0, dbfs[i] / 20.0);
    return linear;
}

// One linear-magnitude spectrum per frame (frame f centered at
// (f+0.5)/frameRateHz), used by FitAllFrames instead of computing each
// frame's spectrum inline as it goes.
using SpectrumTable = std::vector<std::vector<double>>;

// Computes a SpectrumTable in parallel across available hardware
// threads -- each frame's FFT is completely independent of every other
// frame's (dsp::ComputeSpectrumAtTime and everything it calls only touch
// their own stack-local buffers and a read-only cached window, so this
// is safe), which makes this the one genuinely parallelizable part of
// the whole fitting pipeline: the per-frame frequency/timbre decisions
// downstream of it are inherently sequential (each one depends on the
// previous frame's/channel's choice for continuity/residual targeting),
// but the expensive spectrum extraction they read from doesn't have to
// be computed on that same sequential path.
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

// Renders a short probe tone (channel 0, carrier at total level 0) and
// returns its own linear-magnitude spectrum, the same shape
// ExtractLinearSpectrumAt produces -- used both to calibrate the carrier's
// dB-at-TL-0 reference point and to verify Tier-1 timbre candidates for
// real (Tier 2).
std::vector<double> RenderProbeSpectrum(double freqHz, uint8_t modMultipleIndex, uint8_t modTotalLevel,
                                         uint8_t modWaveform, uint8_t carWaveform, bool algorithmAdditive,
                                         int sampleRate)
{
    OplChip chip(sampleRate);
    chip.WriteReg(1, 0x05, 0x01); // OPL3 mode

    opl::ChannelStaticSetup setup;
    setup.modMultipleIndex = modMultipleIndex;
    setup.modTotalLevel = modTotalLevel;
    setup.modWaveform = modWaveform;
    setup.carMultipleIndex = 1;
    setup.carWaveform = carWaveform;
    setup.algorithmAdditive = algorithmAdditive;
    opl::SetupChannel(chip, 0, setup);

    const OplFnumBlock fb = HzToFnumBlock(freqHz);
    opl::ApplyFrame(chip, 0, opl::FramePatch{fb.fnum, fb.block, 0}); // TL=0: loudest, for a stable reference

    // Discard the first ~10ms so the envelope has settled (see
    // opl3_selftest, which needs the same warm-up before its own capture).
    const int settleSamples = sampleRate / 100;
    std::vector<float> settle(static_cast<size_t>(settleSamples) * 2);
    chip.GenerateStereo(settle.data(), settleSamples);

    const int n = dsp::kFftSize;
    std::vector<float> stereo(static_cast<size_t>(n) * 2);
    chip.GenerateStereo(stereo.data(), n);

    std::vector<float> mono(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        mono[static_cast<size_t>(i)] = 0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);

    static const std::vector<double> window = dsp::MakeBlackmanWindow(n);
    const std::vector<double> dbfs = dsp::ComputeFrameDbfs(mono.data(), n, window);

    std::vector<double> linear(dbfs.size());
    for (size_t i = 0; i < dbfs.size(); ++i)
        linear[i] = std::pow(10.0, dbfs[i] / 20.0);
    return linear;
}

// Tier 1: fast analytical score for one (modulator multiplier, modulator
// level) candidate -- predicts FM sideband positions at fc +/- k*fm and
// their relative Bessel-function weight, and scores how much of that
// predicted energy lands where the real target spectrum actually has
// energy. kBetaScale is a heuristic mapping from modulator output
// amplitude to modulation index (OPL doesn't expose true FM modulation
// index directly); this is only ever used to rank/shortlist candidates,
// never as the final answer -- Tier 2 re-scores the shortlist against a
// real render, which is what actually decides.
double Tier1SidebandScore(double fc, double modMultiplier, uint8_t modTotalLevel,
                           const std::vector<double>& targetLinear, double hzPerBin)
{
    const double fm = fc * modMultiplier;
    if (fm <= 0.0)
        return 0.0;

    constexpr double kBetaScale = 16.0;
    const double beta = TotalLevelToLinearAmplitude(modTotalLevel) * kBetaScale;

    const int numBins = static_cast<int>(targetLinear.size());
    double score = 0.0;
    for (int k = -8; k <= 8; ++k)
    {
        const double freq = fc + k * fm;
        if (freq < 0.0)
            continue;
        const int bin = static_cast<int>(std::llround(freq / hzPerBin));
        if (bin < 0 || bin >= numBins)
            continue;
        score += std::abs(BesselJ(k, beta)) * targetLinear[static_cast<size_t>(bin)];
    }
    return score;
}

// Tier 2: cosine similarity between a candidate's real rendered spectrum
// and the target spectrum -- the actual, non-approximated quality metric.
double CosineSimilarity(const std::vector<double>& a, const std::vector<double>& b)
{
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
}

double TrendEnergy(const analysis::PeakTrend& trend)
{
    double energy = 0.0;
    for (const analysis::SpectralPeak& p : trend.framePoints)
        energy += std::pow(10.0, p.magnitudeDb / 20.0);
    return energy;
}

struct TrendLoudestPoint { double freqHz; double magnitudeDb; };

// A trend's single loudest point -- its representative pitch/level for
// frequency-banding and audibility-floor purposes (see FitChannelsInTurn),
// the same "loudest point" convention FitTrendToChannel already uses to
// pick a representative frame for timbre selection.
TrendLoudestPoint TrendLoudest(const analysis::PeakTrend& trend)
{
    TrendLoudestPoint best{0.0, -1e300};
    for (const analysis::SpectralPeak& p : trend.framePoints)
        if (p.magnitudeDb > best.magnitudeDb)
            best = TrendLoudestPoint{p.freqHz, p.magnitudeDb};
    return best;
}

struct TimbreChoice
{
    uint8_t modMultipleIndex;
    uint8_t modTotalLevel;
    uint8_t modWaveform = 0;
    uint8_t carWaveform = 0;
};

// Two-stage search for one static 2-operator timbre (carrier fixed at
// multiple x1; modulator multiplier/level, and both operators' waveform,
// vary) against a target spectrum. Shared by both the single-frame and
// whole-trend fitting workflows below.
//
// Stage 1 (modulator multiplier/level) has an actual two-tier search:
// Tier 1 (analytical, cheap) ranks a small grid of candidates by the
// Bessel-sideband cost model -- but that model is only a closed form for
// pure sine-on-sine FM, so this stage always renders sine/sine
// candidates; Tier 2 (real emulator) re-scores the top fidelity-scaled
// slice of that shortlist by actually rendering and comparing, which is
// what actually decides.
//
// Stage 2 (modulator + carrier waveform) has no analytical shortcut at
// all -- OPL3's other 7 waveforms per operator (half/abs/quarter sine,
// square, etc.) don't have a simple sideband formula the way sine does,
// so every combination tried is just real-rendered and scored directly
// against the stage-1 modulator settings; how many of the 8x8 grid get
// tried is fidelity-scaled the same way stage 1's shortlist is. A
// non-sine carrier alone already has its own built-in harmonic series
// independent of any FM at all (e.g. a square-wave carrier's odd
// harmonics), and a non-sine modulator produces richer, non-Bessel
// sideband patterns -- either can let a single channel cover more of a
// harmonically rich target (like a
// vowel's formants) than sine-on-sine FM alone.
TimbreChoice SearchTimbre(double freqHz, const std::vector<double>& targetSpectrum, double hzPerBin,
                           double fidelity, int sampleRate)
{
    struct Candidate { uint8_t modMultipleIndex; uint8_t modTotalLevel; double tier1Score; };
    static constexpr uint8_t kMultipleIndices[] = {1, 2, 3, 4, 6, 8};
    static constexpr uint8_t kLevelSteps[] = {0, 8, 16, 24, 32, 40, 48, 56};
    std::vector<Candidate> candidates;
    candidates.reserve(std::size(kMultipleIndices) * std::size(kLevelSteps));
    for (uint8_t mi : kMultipleIndices)
        for (uint8_t tl : kLevelSteps)
        {
            const double score = Tier1SidebandScore(freqHz, opl::kMultipleTable[mi], tl, targetSpectrum, hzPerBin);
            candidates.push_back(Candidate{mi, tl, score});
        }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.tier1Score > b.tier1Score; });

    const int tier2Count = std::clamp(
        static_cast<int>(std::llround(1 + fidelity * (static_cast<double>(candidates.size()) - 1))),
        1, static_cast<int>(candidates.size()));
    TimbreChoice best{candidates.front().modMultipleIndex, candidates.front().modTotalLevel, 0, 0};
    double bestScore = -1.0;
    for (int i = 0; i < tier2Count; ++i)
    {
        const Candidate& c = candidates[static_cast<size_t>(i)];
        const std::vector<double> rendered =
            RenderProbeSpectrum(freqHz, c.modMultipleIndex, c.modTotalLevel, 0, 0, false, sampleRate);
        const double score = CosineSimilarity(rendered, targetSpectrum);
        if (score > bestScore)
        {
            bestScore = score;
            best = TimbreChoice{c.modMultipleIndex, c.modTotalLevel, 0, 0};
        }
    }

    // Stage 2: with the modulator multiplier/level now fixed, sweep
    // modulator/carrier waveform pairs by real render -- no analytical
    // shortcut exists for non-sine FM, so this is Tier-2-only, gated by
    // fidelity same as stage 1 (at fidelity 0 this stage doesn't run at
    // all, keeping the sine/sine baseline; at 1.0 all 63 remaining
    // combinations are tried). The mw==0/cw==0 case is the sine/sine
    // baseline already scored above, so it's excluded from the count and
    // never re-rendered.
    const int waveformCount = std::clamp(static_cast<int>(std::llround(fidelity * 63.0)), 0, 63);
    int waveformTried = 0;
    for (uint8_t mw = 0; mw < 8 && waveformTried < waveformCount; ++mw)
    {
        for (uint8_t cw = 0; cw < 8 && waveformTried < waveformCount; ++cw)
        {
            if (mw == 0 && cw == 0)
                continue;
            ++waveformTried;
            const std::vector<double> rendered =
                RenderProbeSpectrum(freqHz, best.modMultipleIndex, best.modTotalLevel, mw, cw, false, sampleRate);
            const double score = CosineSimilarity(rendered, targetSpectrum);
            if (score > bestScore)
            {
                bestScore = score;
                best.modWaveform = mw;
                best.carWaveform = cw;
            }
        }
    }
    return best;
}

// Calibrates carrier TL -> dBFS via one real probe render (modulator
// silent, carrier alone) so target amplitudes map onto OPL's actual
// loudness curve rather than an assumed one -- carWaveform matters here
// too, since e.g. a square wave's peak/RMS relationship differs from a
// sine's.
double MeasureDbAtTl0(double freqHz, uint8_t modMultipleIndex, uint8_t carWaveform, bool algorithmAdditive,
                       int sampleRate)
{
    const std::vector<double> tl0Spectrum =
        RenderProbeSpectrum(freqHz, modMultipleIndex, 63, 0, carWaveform, algorithmAdditive, sampleRate);
    double dbAtTl0 = -240.0;
    for (double linearMag : tl0Spectrum)
        dbAtTl0 = std::max(dbAtTl0, 20.0 * std::log10(std::max(linearMag, 1e-12)));
    return dbAtTl0;
}

// Fits one OPL3 channel's static timbre and per-frame register timeline
// to a single, already-identified target trend -- the shared back half of
// both FitSingleChannel and FitMultiChannel (see there for how the target
// trend(s) get picked).
FitResult FitTrendToChannel(const analysis::PeakTrend& target, const std::vector<float>& samples, int sampleRate,
                             double frameRateHz, double fidelity, double targetEnergy)
{
    FitResult result;
    if (target.framePoints.empty())
        return result;

    // Representative frame -- the loudest point in the target trend --
    // used to pick one static timbre for the channel's whole lifetime.
    const analysis::SpectralPeak* loudest = &target.framePoints.front();
    for (const analysis::SpectralPeak& p : target.framePoints)
        if (p.magnitudeDb > loudest->magnitudeDb)
            loudest = &p;

    const std::vector<double> targetSpectrum = ExtractLinearSpectrumAt(samples, sampleRate, loudest->timeSeconds);
    const double hzPerBin = static_cast<double>(sampleRate) / dsp::kFftSize;

    const TimbreChoice timbre = SearchTimbre(loudest->freqHz, targetSpectrum, hzPerBin, fidelity, sampleRate);
    const double dbAtTl0 =
        MeasureDbAtTl0(loudest->freqHz, timbre.modMultipleIndex, timbre.carWaveform, false, sampleRate);

    result.setup.modMultipleIndex = timbre.modMultipleIndex;
    result.setup.modTotalLevel = timbre.modTotalLevel;
    result.setup.modWaveform = timbre.modWaveform;
    result.setup.carMultipleIndex = 1;
    result.setup.carWaveform = timbre.carWaveform;
    result.startSeconds = target.framePoints.front().timeSeconds;
    result.frameRateHz = frameRateHz;
    result.targetEnergy = targetEnergy;
    result.frames.reserve(target.framePoints.size());
    for (const analysis::SpectralPeak& p : target.framePoints)
    {
        const OplFnumBlock fb = HzToFnumBlock(p.freqHz);
        const int tl = std::clamp(static_cast<int>(std::llround((dbAtTl0 - p.magnitudeDb) / kDbPerTotalLevelStep)),
                                   0, 63);
        result.frames.push_back(opl::FramePatch{fb.fnum, fb.block, static_cast<uint8_t>(tl)});
    }

    return result;
}

struct SpectralCandidate { double freqHz; double db; double magnitude; };

// Up to `maxCandidates` local-maximum bins in `spectrum` (linear
// magnitude), loudest first -- the same "prominent feature" notion used
// elsewhere, just simple single-bin maxima (no center-of-mass
// refinement) since this only needs to produce a short candidate list to
// choose among, not a precise frequency estimate. Falls back to the
// single global-max bin if the spectrum has no interior local maximum at
// all (e.g. a flat or monotonically-shaped frame).
std::vector<SpectralCandidate> FindTopCandidates(const std::vector<double>& spectrum, double hzPerBin,
                                                   int maxCandidates)
{
    std::vector<SpectralCandidate> found;
    for (size_t i = 1; i + 1 < spectrum.size(); ++i)
    {
        if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1])
            found.push_back(SpectralCandidate{static_cast<double>(i) * hzPerBin,
                                                20.0 * std::log10(std::max(spectrum[i], 1e-12)), spectrum[i]});
    }
    if (found.empty() && !spectrum.empty())
    {
        size_t bestIdx = 0;
        for (size_t i = 1; i < spectrum.size(); ++i)
            if (spectrum[i] > spectrum[bestIdx]) bestIdx = i;
        found.push_back(SpectralCandidate{static_cast<double>(bestIdx) * hzPerBin,
                                            20.0 * std::log10(std::max(spectrum[bestIdx], 1e-12)), spectrum[bestIdx]});
    }
    std::sort(found.begin(), found.end(),
              [](const SpectralCandidate& a, const SpectralCandidate& b) { return a.magnitude > b.magnitude; });
    if (static_cast<int>(found.size()) > maxCandidates)
        found.resize(static_cast<size_t>(maxCandidates));
    return found;
}

// Up to `maxCandidates` local-maximum bins in `spectrum`, loudest first,
// but unlike FindTopCandidates also enforces a minimum spacing between
// kept candidates (scipy.signal.find_peaks's `distance` parameter,
// mirroring the reference Python tool FitChannelsPerFrame is modeled on)
// -- a real partial's spectral mainlobe is several bins wide under this
// project's Blackman-windowed analysis, so without a minimum spacing a
// noisy ripple partway down that same lobe can register as its own
// separate "peak" a few bins from the true one. Candidates are scanned
// loudest-first and a closer, quieter one is simply dropped rather than
// merged, so the single loudest bin in each real lobe always wins.
// Stops scanning once maxCandidates have been kept (the whole point of
// this workflow is picking at most maxChannels per frame), which also
// bounds the cost of the distance check itself on a busy/noisy frame.
//
// Bins above kOplMaxFreqHz are excluded entirely, not just deprioritized
// -- OPL3 physically cannot play anything above that (HzToFnumBlock
// clamps), and every excluded-instead-of-clamped candidate here is a
// channel freed up for something actually reproducible. Without this, a
// recording with a lot of energy above the ceiling (sibilants, cymbals,
// any real content from ~6.2kHz up to Nyquist) would have many distinct,
// genuinely different high partials all collapse onto the exact same
// clamped fnum/block, piling multiple oscillators onto one identical
// tone right at the ceiling -- audible as a loud, artificial whine there
// instead of silence, while starving lower, representable content of
// channels in the process.
std::vector<SpectralCandidate> FindDistinctPeaks(const std::vector<double>& spectrum, double hzPerBin,
                                                   int minBinDistance, int maxCandidates)
{
    std::vector<SpectralCandidate> all;
    for (size_t i = 1; i + 1 < spectrum.size(); ++i)
    {
        const double freqHz = static_cast<double>(i) * hzPerBin;
        if (freqHz > kOplMaxFreqHz)
            break; // bins are in ascending frequency order, so nothing further out matters either
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

// A frame quieter than this is treated as silent -- don't waste a search
// fitting noise, just mute the channel for that frame.
constexpr double kSilenceFloorDb = -55.0;

// How strongly a frame favors reusing the previous frame's frequency, in
// [0,1]: 0 at/under kQuietFloorDb (pick purely by prominence -- any
// discontinuity there is inaudible anyway), 1 at/over kLoudCeilingDb
// (favor continuity strongly -- a switch there would be audible),
// linearly interpolated between.
constexpr double kQuietFloorDb = -50.0;
constexpr double kLoudCeilingDb = -15.0;
double ContinuityWeight(double frameDb)
{
    return std::clamp((frameDb - kQuietFloorDb) / (kLoudCeilingDb - kQuietFloorDb), 0.0, 1.0);
}

// A candidate within this many FFT bins of a frequency another channel
// is already playing at this same frame is treated as a spectral-leakage
// artifact of that already-covered peak, not a genuinely distinct
// feature -- see FilterOccupiedCandidates. A pure tone's energy isn't
// confined to a single bin (the analysis window's mainlobe spreads it
// over several neighbors), so an already-covered peak's own leftover
// leakage can look, bin for bin, like "the next loudest thing" in the
// residual even where the residual subtraction landed close to zero.
constexpr int kOccupiedExclusionBins = 3;

// Drops any candidate within kOccupiedExclusionBins of a frequency in
// `occupiedFreqs` (each entry another already-added channel's frequency
// at this same frame) -- a hard exclusion, independent of and in
// addition to the magnitude-domain residual subtraction already applied
// to the spectrum candidates were found in. Never drops every candidate:
// if all of them would be excluded, returns the original list unfiltered
// rather than forcing a silent frame purely on this heuristic.
std::vector<SpectralCandidate> FilterOccupiedCandidates(const std::vector<SpectralCandidate>& candidates,
                                                          const std::vector<double>& occupiedFreqs,
                                                          double hzPerBin)
{
    if (occupiedFreqs.empty())
        return candidates;
    const double exclusionHz = kOccupiedExclusionBins * hzPerBin;
    std::vector<SpectralCandidate> filtered;
    for (const SpectralCandidate& c : candidates)
    {
        bool tooClose = false;
        for (double occupied : occupiedFreqs)
        {
            if (std::abs(c.freqHz - occupied) <= exclusionHz)
            {
                tooClose = true;
                break;
            }
        }
        if (!tooClose)
            filtered.push_back(c);
    }
    return filtered.empty() ? candidates : filtered;
}

// How far (in octaves) a candidate needs to be from the nearest already-
// occupied frequency before the diversity bonus below is fully saturated
// -- half an octave is comfortably more than the "same note" leakage
// zone FilterOccupiedCandidates already handles, so this is specifically
// about preferring a genuinely different frequency trail over a merely
// nearby one, not re-doing that exclusion.
constexpr double kDiversitySaturationOctaves = 0.5;
constexpr double kDiversityWeight = 0.15;

// Picks this frame's target frequency from its top spectral candidates,
// blending each one's own prominence against how close it is (in
// octaves, so this is pitch-perceptual rather than a raw Hz distance) to
// the previous frame's frequency, plus a smaller bonus for candidates
// further from any already-occupied frequency (see occupiedFreqs) --
// weighted low enough to only break close ties between similarly
// prominent/continuous candidates, not override a clearly better match
// on those, per the "favor the diverse option on a close tie" ask.
// continuityWeight controls the main blend -- 0 is pure prominence
// (equivalent to just picking the loudest bin), 1 strongly prefers
// whichever candidate is closest to prevFreqHz even if it isn't the
// loudest one, i.e. "favoring less prominent features... for better
// continuity", per the original ask.
double ChooseTargetFrequency(const std::vector<SpectralCandidate>& candidates, double prevFreqHz,
                              double continuityWeight, const std::vector<double>& occupiedFreqs)
{
    if (candidates.empty())
        return 0.0;

    const double loudestDb = candidates.front().db;
    double bestScore = -1e300;
    double bestFreq = candidates.front().freqHz;
    for (const SpectralCandidate& c : candidates)
    {
        double score;
        if (continuityWeight <= 0.0 || prevFreqHz <= 0.0)
        {
            const double loudnessScore = std::clamp(1.0 - (loudestDb - c.db) / 24.0, 0.0, 1.0);
            score = loudnessScore;
        }
        else
        {
            const double loudnessScore = std::clamp(1.0 - (loudestDb - c.db) / 24.0, 0.0, 1.0);
            const double octaves = std::abs(std::log2(std::max(c.freqHz, 1.0) / prevFreqHz));
            const double continuityScore = std::clamp(1.0 - octaves, 0.0, 1.0); // ~0 past one octave away
            score = (1.0 - continuityWeight) * loudnessScore + continuityWeight * continuityScore;
        }

        if (!occupiedFreqs.empty())
        {
            double minOctaves = 1e300;
            for (double occupied : occupiedFreqs)
                minOctaves = std::min(minOctaves, std::abs(std::log2(std::max(c.freqHz, 1.0) / std::max(occupied, 1.0))));
            const double diversityScore = std::clamp(minOctaves / kDiversitySaturationOctaves, 0.0, 1.0);
            score += kDiversityWeight * diversityScore;
        }

        if (score > bestScore)
        {
            bestScore = score;
            bestFreq = c.freqHz;
        }
    }
    return bestFreq;
}

// How good the "just keep the previous frame's timbre, retargeted to
// this frame's chosen frequency" continuity check needs to score
// (cosine similarity) to be accepted without paying for a fresh search,
// interpolated by continuityWeight (see ContinuityWeight): quiet frames
// need a near-perfect match, which in practice almost never holds, so
// quiet material searches freely ("that favoritism could be ended during
// quiet parts"); loud/sustained frames accept a much looser match,
// favoring continuity strongly since a switch there would be audible.
constexpr double kContinuityRequiredScoreQuiet = 0.97;
constexpr double kContinuityRequiredScoreLoud = 0.55;

// Deterministic pure-sine channel setup: modulator fully silenced AND
// disconnected from the carrier via additive (non-FM) algorithm, both
// operators' waveform forced to sine. No search needed -- there's
// nothing left to choose, so FitAllFrames uses this directly instead of
// running SearchTimbre at all. See the user-supplied register recipe
// this mirrors: CNT=1, FB=0, modulator TL=63, both
// operators waveform=0.
opl::ChannelStaticSetup PureSineSetup()
{
    opl::ChannelStaticSetup setup;
    setup.modMultipleIndex = 1;
    setup.modTotalLevel = 63;
    setup.modWaveform = 0;
    setup.carMultipleIndex = 1;
    setup.carWaveform = 0;
    setup.feedback = 0;
    setup.algorithmAdditive = true;
    return setup;
}

// The fnum/block a channel parks at for a frame it has nothing real to
// play (always paired with carTotalLevel=63, so its own contribution is
// individually inaudible) -- spread across channelIndex rather than one
// shared constant. A single such "silent" channel really is inaudible in
// isolation (TL=63 alone is only ~47dB down), but every channel that has
// nothing to play this frame reaches this same default, and FitChannelsInTurn/
// FitChannelsPerFrame both sum every channel's own independent solo
// render sample-for-sample -- if they all defaulted to the identical
// frequency, they'd also share the identical phase (same setup, same
// key-on timing), so their individually-quiet residuals would sum
// COHERENTLY rather than average out. Measured directly: 18 channels
// all parked on one shared 440Hz default add up to a clearly audible
// -35dB tone, not the -60dB a single one produces on its own -- exactly
// the "silent" sections of a recording, where most/all channels
// simultaneously have nothing real to contribute, are when this bites.
// Different channels landing on different frequencies here can't
// reinforce that way.
OplFnumBlock SilentDefaultFb(int channelIndex)
{
    return HzToFnumBlock(220.0 + 37.0 * channelIndex);
}

} // namespace

FitResult FitSingleChannel(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                            double fidelity, const dsp::ProgressCallback& progress)
{
    FitResult result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;

    fidelity = std::clamp(fidelity, kMinFidelity, kMaxFidelity);
    if (progress) progress(0);

    // The single most significant feature to approximate: the trend with
    // the greatest total (linear-magnitude) energy, reusing the exact
    // peak-picking/tracking Peak Trends mode already relies on.
    const analysis::PeakTrendsResult peaks = analysis::ComputePeakTrends(
        samples, sampleRate, frameRateHz, analysis::kDefaultAnalysisRateHz,
        analysis::kDefaultPeaksPerStep, analysis::kDefaultPeakWindowHalfWidth,
        [&progress](int percent) { if (progress) progress(percent * 40 / 100); });

    const analysis::PeakTrend* target = nullptr;
    double bestEnergy = -1.0;
    for (const analysis::PeakTrend& trend : peaks.trends)
    {
        const double energy = TrendEnergy(trend);
        if (energy > bestEnergy)
        {
            bestEnergy = energy;
            target = &trend;
        }
    }
    if (!target || target->framePoints.empty())
        return result;
    if (progress) progress(40);

    result = FitTrendToChannel(*target, samples, sampleRate, frameRateHz, fidelity, bestEnergy);
    if (progress) progress(100);
    return result;
}

// Shared implementation behind the public FitAllFrames: identical
// behavior, but takes an optional already-computed SpectrumTable for the
// original recording -- FitChannelsInTurn computes that ONCE and reuses
// it for every channel it fits, instead of every one of up to 18 calls
// separately recomputing the exact same FFTs of the exact same
// recording (previously the single biggest cost of a multi-channel fit).
// nullptr (what the public wrapper passes) means "compute it here",
// still parallelized via PrecomputeSpectra, just not shared afterward.
AllFramesFit FitAllFramesImpl(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                               const dsp::ProgressCallback& progress, const std::vector<float>& previousMixRendered,
                               const std::vector<AllFramesFit>& previousChannels,
                               const SpectrumTable* precomputedOriginalSpectra)
{
    AllFramesFit result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;
    result.frameRateHz = frameRateHz;

    const double frameInterval = 1.0 / frameRateHz;
    const double totalSeconds = static_cast<double>(samples.size()) / sampleRate;
    const int totalFrames = std::max(1, static_cast<int>(std::ceil(totalSeconds / frameInterval)));
    const double hzPerBin = static_cast<double>(sampleRate) / dsp::kFftSize;
    constexpr int kMaxCandidates = 5;

    if (progress) progress(0);

    // The original recording's own per-frame spectrum: reused as-is if
    // the caller already computed it once to share across every channel,
    // otherwise computed here -- either way, in parallel (see
    // PrecomputeSpectra), since this is the expensive part.
    SpectrumTable ownOriginalSpectra;
    if (!precomputedOriginalSpectra)
    {
        ownOriginalSpectra = PrecomputeSpectra(samples, sampleRate, frameRateHz, totalFrames);
        precomputedOriginalSpectra = &ownOriginalSpectra;
    }
    const SpectrumTable& originalSpectra = *precomputedOriginalSpectra;
    if (progress) progress(20);

    // This channel's own residual coverage, if any -- always computed
    // fresh (it's different for every channel, unlike the original
    // recording's own spectrum above), but still parallelized.
    SpectrumTable residualSpectra;
    if (!previousMixRendered.empty())
        residualSpectra = PrecomputeSpectra(previousMixRendered, sampleRate, frameRateHz, totalFrames);
    if (progress) progress(40);

    result.frames.reserve(static_cast<size_t>(totalFrames));

    // Every frame uses this same fixed timbre -- see PureSineSetup --
    // so, unlike the old FM/waveform-search version, there's nothing to
    // (re-)decide frame to frame on that front, and dB-at-TL0 calibration
    // only needs measuring once for the whole channel (it's
    // ~frequency-independent -- real OPL TL steps are a fixed ~0.75dB/
    // unit regardless of pitch) rather than per frame.
    const opl::ChannelStaticSetup setup = PureSineSetup();
    const double dbAtTl0 = MeasureDbAtTl0(1000.0, setup.modMultipleIndex, setup.carWaveform,
                                           setup.algorithmAdditive, sampleRate);

    double prevFreqHz = 0.0;
    bool havePrev = false;
    // Reused across frames (cleared, not reallocated, each iteration) --
    // this is now the only remaining per-frame cost that grows with the
    // number of channels already added, and it's cheap: a handful of
    // simple comparisons per previous channel, not an FFT.
    std::vector<double> occupiedFreqs;

    // Reported at most ~100 times total, not once per frame -- a long
    // recording can have many thousands of frames, and the real caller
    // behind `progress` is typically a wxProgressDialog::Update(), which
    // does real GUI work (repaint, event pump) on every call, not a cheap
    // no-op; calling it every single frame measurably slows the fit down
    // for no benefit once it's updating far faster than eyes can read.
    const int reportStep = std::max(1, totalFrames / 100);

    for (int f = 0; f < totalFrames; ++f)
    {
        if (progress && (f % reportStep) == 0) progress(40 + f * 60 / totalFrames);

        std::vector<double> targetSpectrum =
            (static_cast<size_t>(f) < originalSpectra.size()) ? originalSpectra[static_cast<size_t>(f)]
                                                                : std::vector<double>{};
        if (!residualSpectra.empty() && static_cast<size_t>(f) < residualSpectra.size())
        {
            // Residual target: only what the channels already added
            // haven't covered yet -- see the doc comment above.
            const std::vector<double>& alreadyCovered = residualSpectra[static_cast<size_t>(f)];
            for (size_t i = 0; i < targetSpectrum.size() && i < alreadyCovered.size(); ++i)
                targetSpectrum[i] = std::max(targetSpectrum[i] - alreadyCovered[i], 0.0);
        }
        std::vector<SpectralCandidate> candidates = FindTopCandidates(targetSpectrum, hzPerBin, kMaxCandidates);

        // Frequencies other already-added channels are playing at this
        // same frame -- used both to hard-exclude near-duplicate
        // candidates (spectral-leakage artifacts of an already-covered
        // peak) and to nudge the choice among the rest toward a more
        // diverse frequency trail on a close tie. Silent frames
        // (carTotalLevel == 63) don't count as "occupied".
        occupiedFreqs.clear();
        for (const AllFramesFit& prevChannel : previousChannels)
        {
            if (static_cast<size_t>(f) < prevChannel.frames.size())
            {
                const FrameFit& prevFrame = prevChannel.frames[static_cast<size_t>(f)];
                if (prevFrame.frame.carTotalLevel < 63)
                    occupiedFreqs.push_back(FnumBlockToHz(prevFrame.frame.fnum, prevFrame.frame.block));
            }
        }
        candidates = FilterOccupiedCandidates(candidates, occupiedFreqs, hzPerBin);

        double freqHz = 0.0;
        uint8_t tl = 63; // silent unless a real candidate is found below

        if (!candidates.empty() && candidates.front().db >= kSilenceFloorDb)
        {
            // Frequency selection still favors continuity with the
            // previous frame's pitch (weighted by how loud this frame
            // is) -- unrelated to the timbre simplification above, and
            // still useful for not octave-jumping between adjacent
            // frames' peak picks -- plus a smaller nudge toward whichever
            // candidate is more diverse relative to occupiedFreqs.
            const double continuityWeight = havePrev ? ContinuityWeight(candidates.front().db) : 0.0;
            freqHz = ChooseTargetFrequency(candidates, prevFreqHz, continuityWeight, occupiedFreqs);

            // The chosen candidate's own loudness (not necessarily the
            // frame's loudest) is what the carrier level should target,
            // since that's the specific feature actually being tracked.
            double chosenDb = candidates.front().db;
            for (const SpectralCandidate& c : candidates)
                if (c.freqHz == freqHz) { chosenDb = c.db; break; }

            tl = static_cast<uint8_t>(std::clamp(
                static_cast<int>(std::llround((dbAtTl0 - chosenDb) / kDbPerTotalLevelStep)), 0, 63));

            prevFreqHz = freqHz;
            havePrev = true;
        }
        else
        {
            // Silent frame -- keep whatever frequency was last in use
            // (cosmetic only, since TL=63 mutes it either way) rather
            // than fitting noise.
            freqHz = havePrev ? prevFreqHz : 440.0;
        }

        const OplFnumBlock fb = HzToFnumBlock(freqHz);
        result.frames.push_back(FrameFit{setup, opl::FramePatch{fb.fnum, fb.block, tl}});
    }

    if (progress) progress(100);
    return result;
}

AllFramesFit FitAllFrames(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                           const dsp::ProgressCallback& progress, const std::vector<float>& previousMixRendered,
                           const std::vector<AllFramesFit>& previousChannels)
{
    return FitAllFramesImpl(samples, sampleRate, frameRateHz, progress, previousMixRendered, previousChannels,
                             nullptr);
}

std::vector<float> RenderAllFrames(const AllFramesFit& fit, int sampleRate)
{
    std::vector<float> mono;
    if (fit.frames.empty() || sampleRate <= 0 || fit.frameRateHz <= 0.0)
        return mono;

    OplChip chip(sampleRate);
    chip.WriteReg(1, 0x05, 0x01); // OPL3 mode

    const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / fit.frameRateHz)));
    mono.assign(static_cast<size_t>(samplesPerFrame) * fit.frames.size(), 0.0f);

    std::vector<float> stereo(static_cast<size_t>(samplesPerFrame) * 2);
    size_t writePos = 0;
    for (const FrameFit& frame : fit.frames)
    {
        // SetupChannel never touches key-on, and dbopl treats a register
        // write with an unchanged value as a complete no-op, so frames
        // that keep the previous timbre cause zero disruption to the
        // running oscillator/envelope state -- only genuinely different
        // values actually take effect.
        opl::SetupChannel(chip, 0, frame.setup);
        opl::ApplyFrame(chip, 0, frame.frame);
        chip.GenerateStereo(stereo.data(), samplesPerFrame);
        for (int i = 0; i < samplesPerFrame; ++i)
            mono[writePos + static_cast<size_t>(i)] =
                0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);
        writePos += static_cast<size_t>(samplesPerFrame);
    }
    return mono;
}

std::vector<float> RenderAllFramesMix(const std::vector<AllFramesFit>& fits, int sampleRate)
{
    std::vector<float> mono;
    if (fits.empty() || sampleRate <= 0 || fits.front().frameRateHz <= 0.0)
        return mono;

    double maxEndSeconds = 0.0;
    for (const AllFramesFit& fit : fits)
        if (fit.frameRateHz > 0.0)
            maxEndSeconds = std::max(maxEndSeconds, static_cast<double>(fit.frames.size()) / fit.frameRateHz);
    if (maxEndSeconds <= 0.0)
        return mono;

    OplChip chip(sampleRate);
    chip.WriteReg(1, 0x05, 0x01); // OPL3 mode

    // The render tick follows the first channel's own frame rate; other
    // channels (if fit at a different rate, e.g. the user changed the
    // Frame Rate control between "Add channel" presses) are sampled by
    // absolute time against their own frame array each tick, rather than
    // assumed to line up index-for-index with the first channel's.
    const double tickRateHz = fits.front().frameRateHz;
    const double tickInterval = 1.0 / tickRateHz;
    const size_t totalTicks = static_cast<size_t>(std::llround(maxEndSeconds / tickInterval));
    const int samplesPerTick = std::max(1, static_cast<int>(std::llround(sampleRate / tickRateHz)));

    mono.assign(static_cast<size_t>(samplesPerTick) * totalTicks, 0.0f);
    std::vector<float> stereo(static_cast<size_t>(samplesPerTick) * 2);

    for (size_t t = 0; t < totalTicks; ++t)
    {
        const double tSeconds = static_cast<double>(t) * tickInterval;
        for (size_t ch = 0; ch < fits.size(); ++ch)
        {
            const AllFramesFit& fit = fits[ch];
            if (fit.frameRateHz <= 0.0 || fit.frames.empty())
                continue;
            const size_t frameIdx = static_cast<size_t>(tSeconds * fit.frameRateHz);
            if (frameIdx < fit.frames.size())
            {
                opl::SetupChannel(chip, static_cast<int>(ch), fit.frames[frameIdx].setup);
                opl::ApplyFrame(chip, static_cast<int>(ch), fit.frames[frameIdx].frame);
            }
        }

        chip.GenerateStereo(stereo.data(), samplesPerTick);
        const size_t writeBase = t * static_cast<size_t>(samplesPerTick);
        for (int i = 0; i < samplesPerTick; ++i)
            mono[writeBase + static_cast<size_t>(i)] =
                0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);
    }

    return mono;
}

MultiChannelSineFit FitChannelsInTurn(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                       int maxChannels, const dsp::ProgressCallback& progress)
{
    MultiChannelSineFit result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;
    maxChannels = std::clamp(maxChannels, kMinMaxChannels, kMaxMaxChannels);

    // Track the whole recording's peaks ONCE, giving each underlying
    // partial a persistent identity across frames via the exact same
    // mutual-nearest-neighbor linking Peak Trends mode uses (and draws as
    // its connecting lines) -- reused here so that, for as long as a
    // given trend exists, exactly one channel represents it for its
    // entire lifetime, rather than channel assignment being re-decided
    // independently every single frame (which had no way to stop a
    // channel drifting off one trend onto a louder one mid-flight, or two
    // channels swapping which trend each was following -- both far more
    // likely to produce an audible discontinuity at a frame boundary than
    // an honestly silent gap). This also replaces the old per-channel
    // residual-audio re-analysis: trends are already time/frequency-
    // disjoint by construction (that's why the tracker didn't link them
    // into one trend to begin with), so ranking them once by energy and
    // handing each to its own channel naturally spreads channels across
    // the recording without needing to re-run peak tracking per channel.
    //
    // Unlike Peak Trends mode -- whose own displayed frameRateHz can be
    // set as high as analysis::kMaxFrameRateHz, so it defaults to
    // analyzing at the spectrogram's own native rate to have any headroom
    // above that -- Opl3 mode's frame rate is inherently coarse (its
    // default is two orders of magnitude below Peak Trends', and pushing
    // it much higher isn't the point of a chip-register-timeline export).
    // Analyzing several times denser than the output frame grid is still
    // plenty to track real frequency drift between frames; analyzing a
    // further ~14x denser than that on top (kDefaultAnalysisRateHz, still
    // used as-is by ComputePeakTrends's other, unrelated caller
    // FitSingleChannel/FitMultiChannel) buys negligible extra tracking
    // quality here for a lot of extra FFT work.
    constexpr double kAnalysisRateOverFrameRate = 4.0;
    const double analysisRateHz = std::clamp(frameRateHz * kAnalysisRateOverFrameRate,
                                              analysis::kMinAnalysisRateHz, analysis::kMaxAnalysisRateHz);
    const analysis::PeakTrendsResult peaks = analysis::ComputePeakTrends(
        samples, sampleRate, frameRateHz, analysisRateHz,
        analysis::kDefaultPeaksPerStep, analysis::kDefaultPeakWindowHalfWidth,
        [&progress](int percent) { if (progress) progress(percent * 20 / 100); });
    if (peaks.trends.empty())
        return result;

    // One fixed timbre for every channel -- see PureSineSetup -- so, as
    // with FitAllFramesImpl, dB-at-TL0 calibration is measured once for
    // the whole workflow rather than per channel or per frame. Computed
    // up front (rather than after trend selection, as an earlier version
    // of this function did) because it's also what the audibility floor
    // just below needs.
    const opl::ChannelStaticSetup setup = PureSineSetup();
    const double dbAtTl0 = MeasureDbAtTl0(1000.0, setup.modMultipleIndex, setup.carWaveform,
                                           setup.algorithmAdditive, sampleRate);

    // A trend whose own loudest point is already this far below the
    // channel's loudest achievable output can never be rendered above
    // silence: carrier total level only goes up to 63 steps of
    // kDbPerTotalLevelStep each, so anything quieter than that maps to
    // TL=63 (maximum attenuation) regardless of which channel it lands
    // on -- literally inaudible, not just quiet. Excluding such trends
    // up front (rather than letting round-robin spend a whole channel
    // "representing" one) matters specifically because round-robin
    // guarantees every frequency band a turn: without this, a band whose
    // real content is below OPL3's own dynamic range (e.g. a mostly-
    // silent noise floor well above the input's actual highest partial)
    // could still claim a channel purely for having *a* local maximum,
    // producing a channel assigned to a frequency the input barely has
    // any energy at.
    const double audibilityFloorDb = dbAtTl0 - 63.0 * kDbPerTotalLevelStep;

    std::vector<size_t> validTrendIdx;
    validTrendIdx.reserve(peaks.trends.size());
    std::vector<double> energies(peaks.trends.size(), 0.0);
    std::vector<double> repFreqHz(peaks.trends.size(), 0.0);
    double minFreqHz = 1e300, maxFreqHz = 0.0;
    for (size_t i = 0; i < peaks.trends.size(); ++i)
    {
        if (peaks.trends[i].framePoints.empty())
            continue;
        const TrendLoudestPoint loudest = TrendLoudest(peaks.trends[i]);
        if (loudest.magnitudeDb < audibilityFloorDb)
            continue;
        // A trend whose own loudest point is already past kOplMaxFreqHz
        // is entirely unplayable -- see FindDistinctPeaks's identical
        // reasoning for why excluding it outright (rather than letting
        // HzToFnumBlock clamp it) matters: every such trend would
        // otherwise collapse onto the exact same clamped fnum/block,
        // piling multiple channels onto one identical tone at the
        // ceiling. Excluding it here also keeps it from inflating
        // maxFreqHz below and stretching band resolution up into a
        // range nothing can ever actually play. A trend that only
        // partly crosses the ceiling (its loudest point still valid)
        // survives this check; its individual above-ceiling frames are
        // dropped further down instead, once actually building points.
        if (loudest.freqHz > kOplMaxFreqHz)
            continue;
        validTrendIdx.push_back(i);
        energies[i] = TrendEnergy(peaks.trends[i]);
        repFreqHz[i] = loudest.freqHz;
        minFreqHz = std::min(minFreqHz, loudest.freqHz);
        maxFreqHz = std::max(maxFreqHz, loudest.freqHz);
    }
    if (validTrendIdx.empty())
        return result;

    // Group trends into log-spaced frequency bands spanning whatever
    // range the recording's own significant content actually occupies,
    // then process them in round-robin order across bands (loudest
    // trend within a band first) instead of one flat sort by raw
    // energy. Pure loudness ranking systematically favors low/mid
    // frequencies -- a voice's fundamental and its lower harmonics
    // simply carry far more energy than a sibilant or a high formant
    // ever will -- so a channel budget spent strictly loudest-first
    // sounds like the input was run through a low-pass filter: every
    // channel goes to the low end before anything above it gets a look
    // in. Round-robin instead guarantees every band gets a channel
    // before any band gets a SECOND one, so frequency range gets to
    // compete on equal footing with amplitude for which regions are
    // represented at all, while amplitude still decides the order
    // within any one region. Band count tracks the channel budget
    // itself -- more channels bought means finer frequency resolution
    // is worth asking for.
    struct Candidate { size_t trendIdx; double energy; };
    const int numBands = std::clamp(maxChannels, 1, static_cast<int>(validTrendIdx.size()));
    const double logMinFreq = std::log2(std::max(minFreqHz, 1.0));
    const double logMaxFreq = std::log2(std::max(maxFreqHz, 1.0));
    const double logFreqSpan = std::max(logMaxFreq - logMinFreq, 1e-9);

    std::vector<std::vector<Candidate>> bands(static_cast<size_t>(numBands));
    for (size_t idx : validTrendIdx)
    {
        const double logF = std::log2(std::max(repFreqHz[idx], 1.0));
        const int band = std::clamp(static_cast<int>((logF - logMinFreq) / logFreqSpan * numBands), 0, numBands - 1);
        bands[static_cast<size_t>(band)].push_back(Candidate{idx, energies[idx]});
    }
    for (std::vector<Candidate>& band : bands)
    {
        std::sort(band.begin(), band.end(), [](const Candidate& a, const Candidate& b) { return a.energy > b.energy; });

        // Drop this band's own negligible trends (its own noise floor) --
        // the same 5% relative-energy idea the old flat-sorted version
        // used, just scoped to each band's own loudest trend instead of
        // to the single loudest trend in the whole recording. A global
        // threshold would exclude an entire quiet-but-real high-
        // frequency band just for being quieter than an unrelated low-
        // frequency one -- precisely the low-pass-filter effect being
        // fixed here.
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
    processingOrder.reserve(validTrendIdx.size());
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

    // Greedy voice allocation, not a flat "top maxChannels trends": a
    // long-sustained feature (e.g. a speaker's fundamental pitch) is
    // often broken into several separate PeakTrend objects by pauses/
    // consonants, and each such fragment can individually carry more
    // total energy than a shorter, genuinely different simultaneous
    // feature (a formant, a harmonic, a consonant's noise burst) ever
    // does -- taking the top maxChannels in a flat order let several
    // fragments of the SAME frequency band monopolize the whole channel
    // budget, leaving nothing for anything else. Instead, each trend
    // (processed in the frequency-banded round-robin order built above)
    // is placed onto the first already-open channel whose own material
    // doesn't overlap it in time -- reusing a channel across time-
    // disjoint trends the way a single voice plays many different,
    // sequential notes -- and a fresh channel is opened only when no
    // existing one has room and the budget allows. This keeps the
    // budget spent on genuinely SIMULTANEOUS distinct content instead of
    // redundant fragments of the same one, while still giving every
    // trend exactly one fixed channel for its entire lifetime.
    struct PendingChannel
    {
        AllFramesFit fit;
        std::vector<bool> occupied; // occupied[f]: some trend already placed a real note at frame f
    };
    std::vector<PendingChannel> channels;

    for (size_t rank = 0; rank < processingOrder.size(); ++rank)
    {
        const size_t idx = processingOrder[rank];
        const analysis::PeakTrend& trend = peaks.trends[idx];

        // framePoints' timeSeconds sits exactly on f/frameRateHz (see
        // PeakAnalysis.cpp's ResampleToFrameGrid), so recovering the
        // integer frame index is an exact round-trip modulo float error.
        std::vector<std::pair<int, const analysis::SpectralPeak*>> points;
        points.reserve(trend.framePoints.size());
        for (const analysis::SpectralPeak& p : trend.framePoints)
        {
            // A trend that dips above kOplMaxFreqHz for only part of its
            // life (its loudest point, checked above, was still valid)
            // simply goes silent for those specific frames rather than
            // clamping to the ceiling -- see FindDistinctPeaks's doc
            // comment for why clamping instead would be audibly wrong.
            if (p.freqHz > kOplMaxFreqHz)
                continue;
            const long long f = std::llround(p.timeSeconds * frameRateHz);
            if (f >= 0 && f < totalFrames)
                points.emplace_back(static_cast<int>(f), &p);
        }
        if (points.empty())
            continue;

        PendingChannel* target = nullptr;
        for (PendingChannel& channel : channels)
        {
            bool overlaps = false;
            for (const auto& fp : points)
            {
                if (channel.occupied[static_cast<size_t>(fp.first)])
                {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps)
            {
                target = &channel;
                break;
            }
        }
        if (!target)
        {
            if (static_cast<int>(channels.size()) >= maxChannels)
                continue; // no room anywhere -- only maxChannels real oscillators exist
            const OplFnumBlock silentFb = SilentDefaultFb(static_cast<int>(channels.size()));
            channels.push_back(PendingChannel{});
            target = &channels.back();
            target->fit.frameRateHz = frameRateHz;
            // keyOn=false: genuinely silent (not just carTotalLevel=63,
            // which alone only reaches ~-60dB -- see FramePatch's doc
            // comment) until this channel's first real assignment below.
            target->fit.frames.assign(
                static_cast<size_t>(totalFrames),
                FrameFit{setup, opl::FramePatch{silentFb.fnum, silentFb.block, 63, false}});
            target->occupied.assign(static_cast<size_t>(totalFrames), false);
        }

        for (const auto& fp : points)
        {
            const int f = fp.first;
            const analysis::SpectralPeak& p = *fp.second;
            const OplFnumBlock fb = HzToFnumBlock(p.freqHz);
            const int tl = std::clamp(
                static_cast<int>(std::llround((dbAtTl0 - p.magnitudeDb) / kDbPerTotalLevelStep)), 0, 63);
            target->fit.frames[static_cast<size_t>(f)].frame =
                opl::FramePatch{fb.fnum, fb.block, static_cast<uint8_t>(tl), true};
            target->occupied[static_cast<size_t>(f)] = true;
        }

        if (progress) progress(20 + static_cast<int>(rank + 1) * 60 / static_cast<int>(processingOrder.size()));
    }

    // Sum each channel's own solo render into the running mix rather than
    // re-simulating every channel together via RenderAllFramesMix --
    // exact, not an approximation, for this fully-additive-mode timbre
    // (see CheckFitChannelsInTurn). Individual channels are already
    // clamped to [-1,1] by GenerateStereo; the final clamp below
    // approximates (doesn't exactly reproduce) how a real chip's shared
    // internal accumulator would clip the true combined signal once, but
    // in/near-clipping territory either way, so this is close enough.
    for (PendingChannel& channel : channels)
    {
        const std::vector<float> solo = RenderAllFrames(channel.fit, sampleRate);
        if (result.mixRendered.size() < solo.size())
            result.mixRendered.resize(solo.size(), 0.0f);
        for (size_t i = 0; i < solo.size(); ++i)
            result.mixRendered[i] += solo[i];

        result.channels.push_back(std::move(channel.fit));
    }

    for (float& s : result.mixRendered)
        s = std::clamp(s, -1.0f, 1.0f);

    if (progress) progress(100);
    return result;
}

MultiChannelSineFit FitChannelsPerFrame(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                         int maxChannels, const dsp::ProgressCallback& progress)
{
    MultiChannelSineFit result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;
    maxChannels = std::clamp(maxChannels, kMinMaxChannels, kMaxMaxChannels);

    const double frameInterval = 1.0 / frameRateHz;
    const double totalSeconds = static_cast<double>(samples.size()) / sampleRate;
    const int totalFrames = std::max(1, static_cast<int>(std::ceil(totalSeconds / frameInterval)));
    const double hzPerBin = static_cast<double>(sampleRate) / dsp::kFftSize;

    const SpectrumTable spectra = PrecomputeSpectra(samples, sampleRate, frameRateHz, totalFrames);
    if (progress) progress(30);

    // One fixed timbre for every channel -- see PureSineSetup -- exactly
    // as in FitChannelsInTurn.
    const opl::ChannelStaticSetup setup = PureSineSetup();
    const double dbAtTl0 = MeasureDbAtTl0(1000.0, setup.modMultipleIndex, setup.carWaveform,
                                           setup.algorithmAdditive, sampleRate);
    // Same reasoning as FitChannelsInTurn's own audibility floor: a
    // candidate this far below the channel's loudest achievable output
    // maps to TL=63 (maximum attenuation) regardless of which channel it
    // lands on, so it's simply skipped rather than wastefully assigned.
    const double audibilityFloorDb = dbAtTl0 - 63.0 * kDbPerTotalLevelStep;

    // Exactly maxChannels channels, always -- even ones that end up
    // entirely silent -- mirroring the reference tool's own fixed-size
    // per-frame channel list. Each starts at its own SilentDefaultFb
    // (not one shared constant -- see its doc comment for why that
    // coherently sums into a clearly audible tone during genuinely quiet
    // stretches, when most/all channels have nothing real to play).
    result.channels.assign(static_cast<size_t>(maxChannels), AllFramesFit{});
    for (size_t ch = 0; ch < result.channels.size(); ++ch)
    {
        const OplFnumBlock silentFb = SilentDefaultFb(static_cast<int>(ch));
        AllFramesFit& channel = result.channels[ch];
        channel.frameRateHz = frameRateHz;
        // keyOn=false: genuinely silent (not just carTotalLevel=63, which
        // alone only reaches ~-60dB -- see FramePatch's doc comment)
        // until this channel gets a real assignment for a given frame,
        // below.
        channel.frames.assign(static_cast<size_t>(totalFrames),
                               FrameFit{setup, opl::FramePatch{silentFb.fnum, silentFb.block, 63, false}});
    }

    // scipy.signal.find_peaks's own default minimum peak spacing, in
    // bins, in the reference Python tool this workflow mirrors.
    constexpr int kMinPeakBinDistance = 5;
    const int reportStep = std::max(1, totalFrames / 100);
    for (int f = 0; f < totalFrames; ++f)
    {
        if (progress && (f % reportStep) == 0) progress(30 + f * 60 / totalFrames);

        const std::vector<double>& spectrum =
            (static_cast<size_t>(f) < spectra.size()) ? spectra[static_cast<size_t>(f)] : std::vector<double>{};
        const std::vector<SpectralCandidate> peaks =
            FindDistinctPeaks(spectrum, hzPerBin, kMinPeakBinDistance, maxChannels);

        // No persistent identity at all: this frame's loudest candidate
        // goes to channel 0, its 2nd loudest to channel 1, and so on,
        // completely independently of what those same channel indices
        // played the frame before -- see the doc comment on this
        // function's declaration for why that's the point, not an
        // oversight.
        for (int ch = 0; ch < static_cast<int>(peaks.size()); ++ch)
        {
            const SpectralCandidate& c = peaks[static_cast<size_t>(ch)];
            if (c.db < audibilityFloorDb)
                break; // sorted loudest-first, so every candidate after this is quieter still
            const OplFnumBlock fb = HzToFnumBlock(c.freqHz);
            const int tl = std::clamp(
                static_cast<int>(std::llround((dbAtTl0 - c.db) / kDbPerTotalLevelStep)), 0, 63);
            result.channels[static_cast<size_t>(ch)].frames[static_cast<size_t>(f)].frame =
                opl::FramePatch{fb.fnum, fb.block, static_cast<uint8_t>(tl), true};
        }
    }
    if (progress) progress(90);

    // Sum each channel's own solo render -- exact for this fully-
    // additive-mode timbre, same reasoning as FitChannelsInTurn.
    for (AllFramesFit& channel : result.channels)
    {
        const std::vector<float> solo = RenderAllFrames(channel, sampleRate);
        if (result.mixRendered.size() < solo.size())
            result.mixRendered.resize(solo.size(), 0.0f);
        for (size_t i = 0; i < solo.size(); ++i)
            result.mixRendered[i] += solo[i];
    }
    for (float& s : result.mixRendered)
        s = std::clamp(s, -1.0f, 1.0f);

    if (progress) progress(100);
    return result;
}

std::vector<float> RenderChannel(const FitResult& fit, int sampleRate)
{
    std::vector<float> mono;
    if (fit.frames.empty() || sampleRate <= 0 || fit.frameRateHz <= 0.0)
        return mono;

    OplChip chip(sampleRate);
    chip.WriteReg(1, 0x05, 0x01); // OPL3 mode
    opl::SetupChannel(chip, 0, fit.setup);

    const int startSample = std::max(0, static_cast<int>(std::llround(fit.startSeconds * sampleRate)));
    const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / fit.frameRateHz)));
    const size_t totalSamples =
        static_cast<size_t>(startSample) + static_cast<size_t>(samplesPerFrame) * fit.frames.size();
    mono.assign(totalSamples, 0.0f);

    std::vector<float> stereo(static_cast<size_t>(samplesPerFrame) * 2);
    size_t writePos = static_cast<size_t>(startSample);
    for (const opl::FramePatch& frame : fit.frames)
    {
        opl::ApplyFrame(chip, 0, frame);
        chip.GenerateStereo(stereo.data(), samplesPerFrame);
        for (int i = 0; i < samplesPerFrame; ++i)
            mono[writePos + static_cast<size_t>(i)] =
                0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);
        writePos += static_cast<size_t>(samplesPerFrame);
    }
    return mono;
}

MultiChannelResult FitMultiChannel(const std::vector<float>& samples, int sampleRate, double frameRateHz,
                                    double fidelity, int maxChannels, const dsp::ProgressCallback& progress)
{
    MultiChannelResult result;
    if (samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return result;

    fidelity = std::clamp(fidelity, kMinFidelity, kMaxFidelity);
    maxChannels = std::clamp(maxChannels, kMinMaxChannels, kMaxMaxChannels);

    // Peak-track the whole recording once (not once per channel -- an
    // earlier version of this loop re-ran ComputePeakTrends against an
    // audio-domain "residual" it rebuilt after every channel by
    // subtracting that channel's own rendered waveform sample-by-sample.
    // That's only valid if the synthesized waveform is phase-aligned with
    // the original at every sample; OPL's oscillator starts at an
    // arbitrary phase on key-on, unrelated to the original recording's
    // phase at that instant, so the subtraction didn't reliably cancel
    // the target's energy -- every channel kept re-discovering roughly
    // the same loud stretch instead of moving on. Trends are already
    // time/frequency-disjoint by construction (that's why the tracker
    // couldn't link them into one trend to begin with), so handing each
    // channel a different pre-existing trend sidesteps the whole
    // phase problem and naturally spreads channels across the recording.
    const analysis::PeakTrendsResult peaks = analysis::ComputePeakTrends(
        samples, sampleRate, frameRateHz, analysis::kDefaultAnalysisRateHz,
        analysis::kDefaultPeaksPerStep, analysis::kDefaultPeakWindowHalfWidth,
        [&progress](int percent) { if (progress) progress(percent * 20 / 100); });
    if (peaks.trends.empty())
        return result;

    std::vector<size_t> order(peaks.trends.size());
    std::vector<double> energies(peaks.trends.size());
    for (size_t i = 0; i < peaks.trends.size(); ++i)
    {
        order[i] = i;
        energies[i] = TrendEnergy(peaks.trends[i]);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return energies[a] > energies[b]; });
    if (progress) progress(20);

    // A trend carrying less than this fraction of the first (most
    // significant) channel's own target energy isn't worth a whole
    // channel -- "ending early if few features remain to be
    // approximated", per the original ask.
    constexpr double kMinRelativeEnergyToContinue = 0.05;
    const int channelCount = std::min(maxChannels, static_cast<int>(order.size()));
    double firstChannelEnergy = -1.0;

    for (int ch = 0; ch < channelCount; ++ch)
    {
        const size_t idx = order[static_cast<size_t>(ch)];
        const double energy = energies[idx];
        if (ch == 0)
            firstChannelEnergy = energy;
        else if (energy < firstChannelEnergy * kMinRelativeEnergyToContinue)
            break;

        FitResult fit = FitTrendToChannel(peaks.trends[idx], samples, sampleRate, frameRateHz, fidelity, energy);
        if (fit.frames.empty())
            continue;
        result.channels.push_back(std::move(fit));

        if (progress) progress(20 + (ch + 1) * 80 / channelCount);
    }

    if (progress) progress(100);
    return result;
}

std::vector<float> RenderMix(const std::vector<FitResult>& channels, int sampleRate)
{
    std::vector<float> mono;
    if (channels.empty() || sampleRate <= 0)
        return mono;

    const double frameRateHz = channels.front().frameRateHz;
    if (frameRateHz <= 0.0)
        return mono;
    const double frameInterval = 1.0 / frameRateHz;

    double maxEndSeconds = 0.0;
    for (const FitResult& fit : channels)
        maxEndSeconds = std::max(maxEndSeconds, fit.startSeconds + static_cast<double>(fit.frames.size()) * frameInterval);

    OplChip chip(sampleRate);
    chip.WriteReg(1, 0x05, 0x01); // OPL3 mode
    for (size_t ch = 0; ch < channels.size(); ++ch)
        opl::SetupChannel(chip, static_cast<int>(ch), channels[ch].setup);

    const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / frameRateHz)));
    const size_t totalFrames = static_cast<size_t>(std::llround(maxEndSeconds / frameInterval));
    mono.assign(static_cast<size_t>(totalFrames) * static_cast<size_t>(samplesPerFrame), 0.0f);

    // Tracks whether channel ch has already been silenced once past the
    // end of its own frame span, so it doesn't hold its last note forever
    // (and so it isn't rewritten every tick after that, once is enough).
    std::vector<bool> silencedPastEnd(channels.size(), false);
    std::vector<float> stereo(static_cast<size_t>(samplesPerFrame) * 2);

    for (size_t f = 0; f < totalFrames; ++f)
    {
        const double tFrame = static_cast<double>(f) * frameInterval;
        for (size_t ch = 0; ch < channels.size(); ++ch)
        {
            const FitResult& fit = channels[ch];
            const long long localIdx =
                static_cast<long long>(std::llround((tFrame - fit.startSeconds) / frameInterval));
            if (localIdx >= 0 && localIdx < static_cast<long long>(fit.frames.size()))
            {
                opl::ApplyFrame(chip, static_cast<int>(ch), fit.frames[static_cast<size_t>(localIdx)]);
                silencedPastEnd[ch] = false;
            }
            else if (localIdx >= static_cast<long long>(fit.frames.size()) && !fit.frames.empty() && !silencedPastEnd[ch])
            {
                opl::FramePatch last = fit.frames.back();
                last.carTotalLevel = 63; // max attenuation -- effectively silent
                opl::ApplyFrame(chip, static_cast<int>(ch), last);
                silencedPastEnd[ch] = true;
            }
        }

        chip.GenerateStereo(stereo.data(), samplesPerFrame);
        const size_t writeBase = f * static_cast<size_t>(samplesPerFrame);
        for (int i = 0; i < samplesPerFrame; ++i)
            mono[writeBase + static_cast<size_t>(i)] =
                0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);
    }

    return mono;
}

} // namespace oplfit
