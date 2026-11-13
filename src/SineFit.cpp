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

// ---------------------------------------------------------------------------
// Chain workflow internals
// ---------------------------------------------------------------------------

// A candidate partial position for one frame. `magnitude` is the original
// (unsubtracted) linear magnitude there, kept for the per-frame level solve
// once the chain-candidate table's own spectra have been freed.
struct ChainNode
{
    double freqHz = 0.0;
    int bin = 0;
    double magnitude = 0.0;
};

struct ChainPoint
{
    int frame = 0;
    int node = 0;
};

struct Chain
{
    std::vector<ChainPoint> points;
    double score = 0.0;
    double repHz = 0.0; // magnitude-weighted mean pitch, for voice selection
};

// At most this many candidate positions per frame are tracked. More would
// let a busy frame's noise floor crowd out real partials while costing DP
// time quadratically; 12 comfortably covers the polyphony a chip voice
// budget can actually use.
constexpr int kCandidatesPerFrame = 12;
constexpr int kMinPeakBinDistance = 5;

// A gap of up to this many frames between two detections of the same
// partial is filled by holding the previous pitch/level rather than going
// silent -- a detection dropout is an analysis artifact, and a 20 ms hole
// in an otherwise sustained tone is far more audible than the held note.
constexpr int kMaxHoldFrames = 2;

// Node value: how far above the audibility floor a partial is, in units of
// 10 dB, capped so one very loud frame can't outweigh a long partial.
constexpr double kNodeValuePer10Db = 1.0;
constexpr double kMaxNodeValue = 4.0;
// A node below this (i.e. under ~5 dB above the floor) is treated as
// absent, and every frame a chain covers costs this much again -- so a
// very long near-inaudible partial can't outrank a short loud one purely
// on length.
constexpr double kNodeFloorValue = 0.5;
// Pitch change cost per octave travelled between adjacent frames, in the
// same units. A semitone costs ~0.33 and a full octave ~4 (about one loud
// frame), so smooth drift is free and octave jumps aren't.
constexpr double kContinuityCostPerOctave = 4.0;
constexpr double kMinChainScore = 0.5;

Chain FindBestChain(const fitcommon::SpectrumTable& residual,
                    const std::vector<std::vector<ChainNode>>& nodes, double floorDb,
                    const std::vector<std::vector<char>>* excluded)
{
    Chain best;
    const int frames = static_cast<int>(nodes.size());
    if (frames <= 0)
        return best;

    const size_t stride = static_cast<size_t>(kCandidatesPerFrame);
    std::vector<double> score(static_cast<size_t>(frames) * stride, 0.0);
    std::vector<int> prev(static_cast<size_t>(frames) * stride, -1);
    double bestScore = 0.0;
    int bestFrame = -1;
    int bestNode = -1;

    for (int f = 0; f < frames; ++f)
    {
        const std::vector<ChainNode>& frameNodes = nodes[static_cast<size_t>(f)];
        if (frameNodes.empty())
            continue;
        const std::vector<double>& res = residual[static_cast<size_t>(f)];

        for (size_t c = 0; c < frameNodes.size(); ++c)
        {
            if (excluded && !excluded->at(static_cast<size_t>(f)).empty() && excluded->at(static_cast<size_t>(f))[c])
                continue;
            const size_t pos = static_cast<size_t>(f) * stride + c;
            const ChainNode& node = frameNodes[c];
            const double magnitude = (static_cast<size_t>(node.bin) < res.size())
                ? res[static_cast<size_t>(node.bin)] : 0.0;
            const double db = 20.0 * std::log10(std::max(magnitude, 1e-12));
            const double value = std::clamp((db - floorDb) / 10.0 * kNodeValuePer10Db, 0.0, kMaxNodeValue);
            if (value < kNodeFloorValue)
                continue; // not usable this frame (already covered, or inaudible)

            // Either this chain starts here, or it continues from the best
            // continuation of any candidate in the previous frame.
            double bestPrevScore = 0.0;
            int bestPrevNode = -1;
            if (f > 0)
            {
                const std::vector<ChainNode>& prevNodes = nodes[static_cast<size_t>(f - 1)];
                for (size_t p = 0; p < prevNodes.size(); ++p)
                {
                    const double prevScore = score[static_cast<size_t>(f - 1) * stride + p];
                    if (prevScore <= 0.0)
                        continue;
                    const double octaves = std::abs(std::log2(node.freqHz / prevNodes[p].freqHz));
                    const double continued = prevScore - kContinuityCostPerOctave * octaves;
                    if (continued > bestPrevScore)
                    {
                        bestPrevScore = continued;
                        bestPrevNode = static_cast<int>(p);
                    }
                }
            }

            score[pos] = bestPrevScore + value - kNodeFloorValue;
            prev[pos] = bestPrevNode;
            if (score[pos] > bestScore)
            {
                bestScore = score[pos];
                bestFrame = f;
                bestNode = static_cast<int>(c);
            }
        }
    }
    if (bestFrame < 0 || bestScore < kMinChainScore)
        return best;

    double weightSum = 0.0;
    double logPitchSum = 0.0;
    int f = bestFrame;
    int c = bestNode;
    while (f >= 0 && c >= 0)
    {
        best.points.push_back(ChainPoint{f, c});
        const ChainNode& node = nodes[static_cast<size_t>(f)][static_cast<size_t>(c)];
        weightSum += node.magnitude;
        logPitchSum += node.magnitude * std::log(node.freqHz);
        const int next = prev[static_cast<size_t>(f) * stride + static_cast<size_t>(c)];
        --f;
        c = next;
    }
    std::reverse(best.points.begin(), best.points.end());
    best.score = bestScore;
    best.repHz = (weightSum > 0.0) ? std::exp(logPitchSum / weightSum)
                                   : nodes[static_cast<size_t>(bestFrame)][static_cast<size_t>(bestNode)].freqHz;
    return best;
}

// Fills short detection dropouts by repeating the preceding point.
std::vector<ChainPoint> ExpandChain(const Chain& chain)
{
    std::vector<ChainPoint> out;
    out.reserve(chain.points.size() * 2);
    for (size_t i = 0; i < chain.points.size(); ++i)
    {
        if (i > 0)
        {
            const int gap = chain.points[i].frame - chain.points[i - 1].frame;
            if (gap > 1 && gap - 1 <= kMaxHoldFrames)
                for (int f = chain.points[i - 1].frame + 1; f < chain.points[i].frame; ++f)
                    out.push_back(ChainPoint{f, chain.points[i - 1].node});
        }
        out.push_back(chain.points[i]);
    }
    return out;
}

// Solves the small (at most a few dozen unknowns) non-negative least
// squares problem  min ||M a - t||^2, a >= 0  by Gaussian elimination plus
// an active set: solve, pin the most negative component to zero, re-solve.
bool SolveNonNegative(const std::vector<double>& m, const std::vector<double>& t, int k, std::vector<double>& out)
{
    std::vector<char> pinned(static_cast<size_t>(k), 0);
    out.assign(static_cast<size_t>(k), 0.0);

    for (int pass = 0; pass <= k; ++pass)
    {
        std::vector<int> freeIdx;
        for (int i = 0; i < k; ++i)
            if (!pinned[static_cast<size_t>(i)])
                freeIdx.push_back(i);
        const int n = static_cast<int>(freeIdx.size());
        if (n == 0)
            return true;

        std::vector<double> a(static_cast<size_t>(n) * static_cast<size_t>(n));
        std::vector<double> b(static_cast<size_t>(n));
        for (int r = 0; r < n; ++r)
        {
            b[static_cast<size_t>(r)] = t[static_cast<size_t>(freeIdx[static_cast<size_t>(r)])];
            for (int c = 0; c < n; ++c)
                a[static_cast<size_t>(r) * n + c] =
                    m[static_cast<size_t>(freeIdx[static_cast<size_t>(r)]) * k + freeIdx[static_cast<size_t>(c)]];
        }

        // Gaussian elimination with partial pivoting.
        std::vector<double> x(static_cast<size_t>(n), 0.0);
        bool singular = false;
        for (int col = 0; col < n && !singular; ++col)
        {
            int pivot = col;
            for (int r = col + 1; r < n; ++r)
                if (std::abs(a[static_cast<size_t>(r) * n + col]) > std::abs(a[static_cast<size_t>(pivot) * n + col]))
                    pivot = r;
            if (std::abs(a[static_cast<size_t>(pivot) * n + col]) < 1e-12)
            {
                singular = true;
                break;
            }
            if (pivot != col)
            {
                for (int c = 0; c < n; ++c)
                    std::swap(a[static_cast<size_t>(pivot) * n + c], a[static_cast<size_t>(col) * n + c]);
                std::swap(b[static_cast<size_t>(pivot)], b[static_cast<size_t>(col)]);
            }
            const double d = a[static_cast<size_t>(col) * n + col];
            for (int r = col + 1; r < n; ++r)
            {
                const double factor = a[static_cast<size_t>(r) * n + col] / d;
                if (factor == 0.0)
                    continue;
                for (int c = col; c < n; ++c)
                    a[static_cast<size_t>(r) * n + c] -= factor * a[static_cast<size_t>(col) * n + c];
                b[static_cast<size_t>(r)] -= factor * b[static_cast<size_t>(col)];
            }
        }
        if (singular)
            return false;

        for (int r = n - 1; r >= 0; --r)
        {
            double s = b[static_cast<size_t>(r)];
            for (int c = r + 1; c < n; ++c)
                s -= a[static_cast<size_t>(r) * n + c] * x[static_cast<size_t>(c)];
            x[static_cast<size_t>(r)] = s / a[static_cast<size_t>(r) * n + r];
        }

        int worst = -1;
        double worstValue = 0.0;
        for (int i = 0; i < n; ++i)
        {
            out[static_cast<size_t>(freeIdx[static_cast<size_t>(i)])] = x[static_cast<size_t>(i)];
            if (x[static_cast<size_t>(i)] < worstValue)
            {
                worstValue = x[static_cast<size_t>(i)];
                worst = freeIdx[static_cast<size_t>(i)];
            }
        }
        for (int i = 0; i < k; ++i)
            if (pinned[static_cast<size_t>(i)])
                out[static_cast<size_t>(i)] = 0.0;
        if (worst < 0)
            return true;
        pinned[static_cast<size_t>(worst)] = 1;
    }
    return true;
}

// Joint per-frame level solve. Each sounding voice's own amplitude is found
// from the original magnitudes at the other sounding voices' bins, using
// the window's leakage model as the coupling -- so two partials a couple of
// bins apart no longer each get set as loud as their combined appearance in
// the spectrum.
void RefineLevels(const ChipTarget& target, const std::vector<KindCalibration>& cal,
                  const fitcommon::LeakageModel& leakage, double hzPerBin, std::vector<VoiceTrack>& tracks,
                  const std::vector<std::vector<double>>& targetMagnitude)
{
    const int voices = static_cast<int>(tracks.size());
    size_t frames = 0;
    for (const VoiceTrack& track : tracks)
        frames = std::max(frames, track.frames.size());

    std::vector<int> active;
    std::vector<double> freq;
    std::vector<double> mag;
    for (size_t f = 0; f < frames; ++f)
    {
        active.clear();
        freq.clear();
        mag.clear();
        for (int v = 0; v < voices; ++v)
        {
            if (f >= tracks[static_cast<size_t>(v)].frames.size())
                continue;
            const VoiceFrame& frame = tracks[static_cast<size_t>(v)].frames[f];
            if (!frame.keyOn)
                continue;
            active.push_back(v);
            freq.push_back(target.PitchToHz(v, frame.pitch));
            mag.push_back(targetMagnitude[static_cast<size_t>(v)][f]);
        }
        const int k = static_cast<int>(active.size());
        if (k == 0)
            continue;
        if (k == 1)
        {
            const int v = active[0];
            const double db = 20.0 * std::log10(std::max(mag[0], 1e-9));
            tracks[static_cast<size_t>(v)].frames[f].level =
                target.QuantizeLevel(v, db - cal[static_cast<size_t>(target.KindIndexOf(v))].dbAtMax);
            continue;
        }

        std::vector<double> m(static_cast<size_t>(k) * static_cast<size_t>(k));
        std::vector<double> t(static_cast<size_t>(k));
        for (int r = 0; r < k; ++r)
        {
            // Off-diagonal coupling is scaled down from the full window
            // response because the coherent-addition model (partials adding
            // linearly) is only the worst case: unrelated partials add more
            // like powers, so a partial correction of about half the
            // leakage keeps dense material from being pulled too quiet.
            constexpr double kLeakageCoupling = 0.5;
            t[static_cast<size_t>(r)] = mag[static_cast<size_t>(r)];
            for (int c = 0; c < k; ++c)
                m[static_cast<size_t>(r) * k + c] = (r == c)
                    ? 1.0
                    : kLeakageCoupling * leakage.Response(
                          std::abs(freq[static_cast<size_t>(r)] - freq[static_cast<size_t>(c)]), hzPerBin);
            m[static_cast<size_t>(r) * k + r] += 1e-6; // keep nearly-coincident partials solvable
        }
        std::vector<double> amplitudes;
        if (!SolveNonNegative(m, t, k, amplitudes))
            continue;
        for (int i = 0; i < k; ++i)
        {
            const int v = active[static_cast<size_t>(i)];
            const double db = 20.0 * std::log10(std::max(amplitudes[static_cast<size_t>(i)], 1e-9));
            tracks[static_cast<size_t>(v)].frames[f].level =
                target.QuantizeLevel(v, db - cal[static_cast<size_t>(target.KindIndexOf(v))].dbAtMax);
        }
    }
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

FitScore ScoreFit(const ChipTarget& target, const std::vector<float>& samples, int sampleRate, double frameRateHz,
                  const std::vector<VoiceTrack>& tracks, const std::vector<float>& mixRendered)
{
    FitScore score;
    if (tracks.empty() || mixRendered.empty() || samples.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return score;

    size_t frames = 0;
    for (const VoiceTrack& track : tracks)
    {
        frames = std::max(frames, track.frames.size());
        bool sounding = false;
        double prevHz = 0.0;
        for (size_t f = 0; f < track.frames.size(); ++f)
        {
            const VoiceFrame& frame = track.frames[f];
            if (frame.keyOn && !sounding)
                ++score.keyOnCount;
            if (frame.keyOn)
            {
                const double hz = target.PitchToHz(track.voice, frame.pitch);
                // Same voice, still sounding: a step of more than a semitone
                // is a reassignment to different content rather than tracking.
                if (sounding && prevHz > 0.0 && std::abs(std::log2(hz / prevHz)) > 1.0 / 12.0)
                    ++score.pitchJumpCount;
                prevHz = hz;
            }
            sounding = frame.keyOn;
        }
    }

    // A thousand or so frames is plenty to characterise a whole recording,
    // and keeps this cheap next to the fit itself.
    const size_t stride = std::max<size_t>(1, frames / 1000);
    double weightedError = 0.0;
    double weightSum = 0.0;
    for (size_t f = 0; f < frames; f += stride)
    {
        const double timeSeconds = (static_cast<double>(f) + 0.5) / frameRateHz;
        const std::vector<double> original = dsp::ComputeSpectrumAtTime(samples, sampleRate, timeSeconds);
        const std::vector<double> rendered = dsp::ComputeSpectrumAtTime(mixRendered, sampleRate, timeSeconds);
        const size_t bins = std::min(original.size(), rendered.size());
        for (size_t k = 1; k < bins; ++k)
        {
            // Weight by whichever side has signal, so a spurious tone costs
            // as much as a missing one, while bins that are silent on both
            // sides (the vast majority) don't drown the average.
            const double weight = std::max(std::pow(10.0, original[k] / 20.0), std::pow(10.0, rendered[k] / 20.0));
            if (weight < 1e-5)
                continue;
            weightedError += weight * std::abs(original[k] - rendered[k]);
            weightSum += weight;
        }
    }
    if (weightSum > 0.0)
    {
        score.meanAbsDbError = weightedError / weightSum;
        score.similarity = std::exp(-score.meanAbsDbError / 12.0);
    }
    return score;
}

ChipFit FitChannelsByChains(const std::shared_ptr<const ChipTarget>& targetPtr, const std::vector<float>& samples,
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

    if (progress) progress(0);
    fitcommon::SpectrumTable spectra = fitcommon::PrecomputeSpectra(samples, sampleRate, frameRateHz, totalFrames);
    if (progress) progress(20);

    double minHz = 1e300, maxHz = 0.0;
    for (const VoiceKind& kind : target.Kinds())
    {
        minHz = std::min(minHz, kind.minHz);
        maxHz = std::max(maxHz, kind.maxHz);
    }

    // Candidate positions are fixed once, from the original recording:
    // they're just the places a partial could live. Which of them are still
    // available is decided per iteration by reading the residual at those
    // positions, so an iteration costs no FFTs at all.
    std::vector<std::vector<ChainNode>> nodes(static_cast<size_t>(totalFrames));
    for (int f = 0; f < totalFrames; ++f)
    {
        const std::vector<double>& spectrum = spectra[static_cast<size_t>(f)];
        if (spectrum.empty())
            continue;
        const std::vector<fitcommon::SpectralCandidate> peaks = fitcommon::FindDistinctPeaks(
            spectrum, hzPerBin, kMinPeakBinDistance, kCandidatesPerFrame, minHz, maxHz, true);
        for (const fitcommon::SpectralCandidate& peak : peaks)
        {
            ChainNode node;
            node.freqHz = peak.freqHz;
            node.bin = std::clamp(static_cast<int>(std::llround(peak.freqHz / hzPerBin)), 1,
                                  static_cast<int>(spectrum.size()) - 2);
            node.magnitude = peak.magnitude;
            nodes[static_cast<size_t>(f)].push_back(node);
        }
    }
    if (progress) progress(30);

    // From here on the residual *is* the original table -- moved, not copied
    // (a long recording's table is hundreds of megabytes).
    fitcommon::SpectrumTable residual = std::move(spectra);

    const std::vector<KindCalibration> cal = CalibrateKinds(target, sampleRate);
    double floorDb = 1e300;
    for (const KindCalibration& c : cal)
        floorDb = std::min(floorDb, c.audibilityFloorDb);

    const fitcommon::LeakageModel leakage(dsp::kFftSize);

    // Voices are opened on demand rather than taking the first maxChannels
    // indices: on a mixed-kind target (OPL4's FM + PCM) the budget should be
    // spendable on whichever kind can actually play the content, so a
    // high-frequency chain opens a PCM voice while a low one takes FM --
    // rather than the first N voices all being FM and the rest of the range
    // unreachable.
    std::vector<VoiceTrack> tracks;
    std::vector<int> trackVoice;
    std::vector<std::vector<double>> trackMagnitude;
    std::vector<std::vector<char>> occupied;
    std::vector<char> voiceOpen(static_cast<size_t>(target.VoiceCount()), 0);

    const auto canPlay = [&target](int voice, double hz) {
        const VoiceKind& kind = target.KindOf(voice);
        return hz >= kind.minHz && hz <= kind.maxHz;
    };
    const auto isFree = [&occupied](int slot, const std::vector<ChainPoint>& points) {
        for (const ChainPoint& point : points)
            if (occupied[static_cast<size_t>(slot)][static_cast<size_t>(point.frame)])
                return false;
        return true;
    };

    // Chain candidates that could never be placed are skipped rather than
    // ending the whole fit -- one unplaceable (or kind-incompatible) chain
    // shouldn't hide every quieter one behind it.
    std::vector<std::vector<char>> excluded(static_cast<size_t>(totalFrames));
    for (int f = 0; f < totalFrames; ++f)
        excluded[static_cast<size_t>(f)].assign(nodes[static_cast<size_t>(f)].size(), 0);

    // Stops long before this unless the material is extremely dense; each
    // iteration claims the loudest remaining chain, so the residual drains.
    const int maxIterations = std::max(16, maxChannels * 6);
    int rejects = 0;
    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        const Chain chain = FindBestChain(residual, nodes, floorDb, &excluded);
        if (chain.points.empty())
            break;
        const std::vector<ChainPoint> points = ExpandChain(chain);

        // A voice can host any number of chains as long as they don't
        // overlap in time -- exactly like one physical voice playing
        // successive notes -- so prefer an already-open voice of a kind
        // that can play this chain, and only open a new one when the budget
        // allows and no open voice is free.
        int slot = -1;
        for (size_t s = 0; s < tracks.size(); ++s)
        {
            if (!canPlay(trackVoice[s], chain.repHz))
                continue;
            if (isFree(static_cast<int>(s), points))
            {
                slot = static_cast<int>(s);
                break;
            }
        }
        if (slot < 0 && static_cast<int>(tracks.size()) < maxChannels)
        {
            for (int v = 0; v < target.VoiceCount(); ++v)
            {
                if (voiceOpen[static_cast<size_t>(v)] || !canPlay(v, chain.repHz))
                    continue;
                voiceOpen[static_cast<size_t>(v)] = 1;
                slot = static_cast<int>(tracks.size());
                tracks.push_back(VoiceTrack{});
                tracks.back().voice = v;
                tracks.back().frames.assign(static_cast<size_t>(totalFrames), target.SilentFrame(v));
                trackVoice.push_back(v);
                trackMagnitude.emplace_back(static_cast<size_t>(totalFrames), 0.0);
                occupied.emplace_back(static_cast<size_t>(totalFrames), 0);
                break;
            }
        }
        if (slot < 0)
        {
            // Nothing can ever host this chain (voices never free up, and
            // kinds don't change), so retire it and look for the next one.
            for (const ChainPoint& point : chain.points)
                excluded[static_cast<size_t>(point.frame)][static_cast<size_t>(point.node)] = 1;
            if (++rejects > 4 * maxIterations)
                break;
            continue;
        }

        const int voice = trackVoice[static_cast<size_t>(slot)];
        const int kindIdx = target.KindIndexOf(voice);
        const KindCalibration& kc = cal[static_cast<size_t>(kindIdx)];
        const VoiceKind& kind = target.Kinds()[static_cast<size_t>(kindIdx)];

        for (const ChainPoint& point : points)
        {
            const ChainNode& node = nodes[static_cast<size_t>(point.frame)][static_cast<size_t>(point.node)];
            if (node.freqHz < kind.minHz || node.freqHz > kind.maxHz)
                continue;
            const double db = 20.0 * std::log10(std::max(node.magnitude, 1e-12));
            if (db < kc.audibilityFloorDb)
                continue;
            tracks[static_cast<size_t>(slot)].frames[static_cast<size_t>(point.frame)] =
                MakeFrame(target, voice, kc, node.freqHz, db);
            trackMagnitude[static_cast<size_t>(slot)][static_cast<size_t>(point.frame)] = node.magnitude;
            occupied[static_cast<size_t>(slot)][static_cast<size_t>(point.frame)] = 1;

            // Remove this partial's modelled footprint (mainlobe and nearby
            // sidelobes) from the residual, so no later chain re-covers it.
            std::vector<double>& res = residual[static_cast<size_t>(point.frame)];
            constexpr int kLeakageRadiusBins = fitcommon::LeakageModel::kMaxBins;
            const int radius = kLeakageRadiusBins;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int bin = node.bin + offset;
                if (bin <= 0 || bin >= static_cast<int>(res.size()))
                    continue;
                const double deltaHz = static_cast<double>(bin) * hzPerBin - node.freqHz;
                const double contribution = node.magnitude * leakage.Response(deltaHz, hzPerBin);
                res[static_cast<size_t>(bin)] = std::max(0.0, res[static_cast<size_t>(bin)] - contribution);
            }
        }

        if (progress)
            progress(30 + 40 * std::min(iteration + 1, maxIterations) / maxIterations);
    }

    // Dense fill. Chains capture content that behaves like a partial; what's
    // left is diffuse -- noise, fricatives, transients -- which no chain can
    // track and which would otherwise leave voices idle while the residual
    // still has audible energy. So each frame's loudest remaining peaks go
    // onto whatever voices are free there, exactly like the per-frame
    // workflow, but only where the chains didn't already place something.
    // This is what keeps the density of the per-frame workflow while the
    // coherent content keeps its identity; the tradeoff is that these fill
    // notes have no continuity, which is why they're second-class.
    {
        std::vector<char> filledThisFrame(static_cast<size_t>(target.VoiceCount()), 0);
        std::vector<double> soundingHz;
        // Each open voice's most recent pitch, so a fill note can prefer a
        // voice that was already near it -- it then reads as that voice
        // carrying on (a small retune) rather than being reassigned, which
        // is the one thing fill notes can't otherwise avoid.
        std::vector<double> lastHz;

        for (int f = 0; f < totalFrames; ++f)
        {
            const std::vector<double>& res = residual[static_cast<size_t>(f)];
            if (res.empty())
                continue;
            lastHz.resize(tracks.size(), 0.0);
            if (f > 0)
                for (size_t s = 0; s < tracks.size(); ++s)
                    if (tracks[s].frames[static_cast<size_t>(f - 1)].keyOn)
                        lastHz[s] = target.PitchToHz(trackVoice[s], tracks[s].frames[static_cast<size_t>(f - 1)].pitch);

            // Peaks are re-detected from the residual itself rather than
            // reusing the chain-candidate table: that table is deliberately
            // capped small (it drives an O(candidates^2) DP), while here the
            // budget is however many voices are actually free this frame.
            const int openFree = static_cast<int>(std::count_if(
                occupied.begin(), occupied.end(), [f](const std::vector<char>& o) { return !o[static_cast<size_t>(f)]; }));
            const int budget = openFree + (maxChannels - static_cast<int>(tracks.size()));
            if (budget <= 0)
                continue;
            // Ask for more candidates than there are free voices: the
            // loudest residual peaks are usually the already-covered ones
            // (subtraction leaves a little behind), and those get rejected
            // by the closeness check below -- with an exact budget they'd
            // consume it and no fill note would ever be placed.
            const std::vector<fitcommon::SpectralCandidate> peaks = fitcommon::FindDistinctPeaks(
                res, hzPerBin, kMinPeakBinDistance, budget + 8, minHz, maxHz, true);
            if (peaks.empty())
                continue;

            // What this frame already has, so a fill note doesn't double an
            // already-represented partial (residual subtraction leaves a
            // little behind right at the peak).
            soundingHz.clear();
            std::fill(filledThisFrame.begin(), filledThisFrame.end(), 0);
            for (size_t s = 0; s < tracks.size(); ++s)
            {
                const VoiceFrame& frame = tracks[s].frames[static_cast<size_t>(f)];
                if (frame.keyOn)
                    soundingHz.push_back(target.PitchToHz(trackVoice[s], frame.pitch));
            }

            const double exclusionHz = kMinPeakBinDistance * hzPerBin;
            for (const fitcommon::SpectralCandidate& peak : peaks)
            {
                if (peak.db < floorDb)
                    break; // loudest first, so everything after is quieter still
                bool tooClose = false;
                for (double hz : soundingHz)
                    if (std::abs(hz - peak.freqHz) < exclusionHz)
                    {
                        tooClose = true;
                        break;
                    }
                if (tooClose)
                    continue;

                // Voice choice, in order of preference: continue a voice that
                // was already sounding and is close in pitch (no key-on, small
                // retune); otherwise open a fresh voice while the budget
                // allows; otherwise reuse whichever free voice there is.
                // A voice carrying on across a huge interval would be a
                // reassignment in disguise, so distance is capped.
                int continuation = -1;
                double bestCost = 1e300;
                int anyFree = -1;
                for (size_t s = 0; s < tracks.size(); ++s)
                {
                    const int v = trackVoice[s];
                    if (filledThisFrame[static_cast<size_t>(v)] || occupied[s][static_cast<size_t>(f)])
                        continue;
                    if (!canPlay(v, peak.freqHz))
                        continue;
                    const bool soundingBefore = f > 0 && tracks[s].frames[static_cast<size_t>(f - 1)].keyOn;
                    if (soundingBefore && lastHz[s] > 0.0)
                    {
                        const double cost = std::abs(std::log2(peak.freqHz / lastHz[s]));
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            continuation = static_cast<int>(s);
                        }
                    }
                    else if (anyFree < 0)
                    {
                        anyFree = static_cast<int>(s);
                    }
                }
                constexpr double kMaxFillRetuneOctaves = 0.5;
                int slot = -1;
                if (continuation >= 0 && bestCost <= kMaxFillRetuneOctaves)
                    slot = continuation;
                if (slot < 0 && static_cast<int>(tracks.size()) < maxChannels)
                {
                    for (int v = 0; v < target.VoiceCount(); ++v)
                    {
                        if (voiceOpen[static_cast<size_t>(v)] || !canPlay(v, peak.freqHz))
                            continue;
                        voiceOpen[static_cast<size_t>(v)] = 1;
                        slot = static_cast<int>(tracks.size());
                        tracks.push_back(VoiceTrack{});
                        tracks.back().voice = v;
                        tracks.back().frames.assign(static_cast<size_t>(totalFrames), target.SilentFrame(v));
                        trackVoice.push_back(v);
                        trackMagnitude.emplace_back(static_cast<size_t>(totalFrames), 0.0);
                        occupied.emplace_back(static_cast<size_t>(totalFrames), 0);
                        break;
                    }
                }
                if (slot < 0)
                    slot = (continuation >= 0) ? continuation : anyFree;
                if (slot < 0)
                    continue;

                const int voice = trackVoice[static_cast<size_t>(slot)];
                const int kindIdx = target.KindIndexOf(voice);
                tracks[static_cast<size_t>(slot)].frames[static_cast<size_t>(f)] =
                    MakeFrame(target, voice, cal[static_cast<size_t>(kindIdx)], peak.freqHz, peak.db);
                trackMagnitude[static_cast<size_t>(slot)][static_cast<size_t>(f)] = peak.magnitude;
                filledThisFrame[static_cast<size_t>(voice)] = 1;
                soundingHz.push_back(peak.freqHz);
            }
        }
    }
    if (progress) progress(72);

    // Overlapping partials each read louder than they are, so correct the
    // frame's levels jointly now that which voice plays what is settled.
    RefineLevels(target, cal, leakage, hzPerBin, tracks, trackMagnitude);
    if (progress) progress(75);

    // Fully silent voices add nothing but register traffic.
    for (VoiceTrack& track : tracks)
    {
        const bool used = std::any_of(track.frames.begin(), track.frames.end(),
                                      [](const VoiceFrame& frame) { return frame.keyOn; });
        if (used)
            result.tracks.push_back(std::move(track));
    }
    if (result.tracks.empty())
        return result;

    result.mixRendered = SumSoloRenders(target, result.tracks, frameRateHz, sampleRate);
    if (progress) progress(85);
    result.score = ScoreFit(target, samples, sampleRate, frameRateHz, result.tracks, result.mixRendered);
    if (progress) progress(100);
    return result;
}

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
    result.score = ScoreFit(target, samples, sampleRate, frameRateHz, result.tracks, result.mixRendered);

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
    result.score = ScoreFit(target, samples, sampleRate, frameRateHz, result.tracks, result.mixRendered);
    if (progress) progress(100);
    return result;
}

} // namespace sinefit
