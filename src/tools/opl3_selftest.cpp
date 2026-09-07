#include "../OplChip.h"
#include "../OplFit.h"
#include "../VgmWriter.h"
#include "../Dsp.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace
{

// kPi is a GNU/POSIX extension, not standard C++ -- MSVC's <cmath> doesn't
// define it, so this test defines its own constant (matching Dsp.cpp's kPi)
// instead of relying on it.
constexpr double kPi = 3.14159265358979323846;

// Channel 0's register offsets in the classic OPL2/OPL3 map: operator 1
// (modulator) at base 0x00, operator 2 (carrier) at base 0x03. Only
// channel 0 is touched by this test.
constexpr uint8_t kOp1Base = 0x00;
constexpr uint8_t kOp2Base = 0x03;

bool CheckSingleToneFundamental()
{
    const int sampleRate = 44100;
    OplChip chip(sampleRate);

    // Enable OPL3 ("new") mode -- required for stereo GenerateBlock3
    // output; also harmless/expected even for a plain 2-op patch.
    chip.WriteReg(1, 0x05, 0x01);

    // Modulator: attenuated almost to silence (total level near max
    // attenuation) so the carrier's own sine wave dominates the output --
    // the simplest, most predictable tone to verify a fundamental against.
    chip.WriteReg(0, 0x20 + kOp1Base, 0x21); // sustain(EGT) + multiple = 1
    chip.WriteReg(0, 0x40 + kOp1Base, 0x3F); // total level = max attenuation
    chip.WriteReg(0, 0x60 + kOp1Base, 0xFF); // attack=15, decay=15
    chip.WriteReg(0, 0x80 + kOp1Base, 0x0F); // sustain=0(max), release=15
    chip.WriteReg(0, 0xE0 + kOp1Base, 0x00); // waveform 0 = sine

    // Carrier: full volume, sine, sustained for the whole test.
    chip.WriteReg(0, 0x20 + kOp2Base, 0x21); // sustain(EGT) + multiple = 1
    chip.WriteReg(0, 0x40 + kOp2Base, 0x00); // total level = 0 (loudest)
    chip.WriteReg(0, 0x60 + kOp2Base, 0xF0); // attack=15, decay=0
    chip.WriteReg(0, 0x80 + kOp2Base, 0x0F); // sustain=0(max), release=15
    chip.WriteReg(0, 0xE0 + kOp2Base, 0x00); // waveform 0 = sine

    const double targetFreqHz = 440.0;
    const OplFnumBlock fb = HzToFnumBlock(targetFreqHz);
    const double actualFreqHz = FnumBlockToHz(fb.fnum, fb.block);

    chip.WriteReg(0, 0xC0, 0x30); // FM algorithm, no feedback, both L/R panned on
    chip.WriteReg(0, 0xA0, static_cast<uint8_t>(fb.fnum & 0xFF));
    // Key-on (bit 5) + block (bits 4-2) + fnum high 2 bits (bits 1-0).
    chip.WriteReg(0, 0xB0, static_cast<uint8_t>(0x20 | (fb.block << 2) | (fb.fnum >> 8)));

    // Render, discarding the first ~10ms so the fast-but-nonzero attack
    // envelope has settled to full volume before the analysis window.
    const int settleSamples = sampleRate / 100;
    std::vector<float> settle(static_cast<size_t>(settleSamples) * 2);
    chip.GenerateStereo(settle.data(), settleSamples);

    const int n = dsp::kFftSize;
    std::vector<float> stereo(static_cast<size_t>(n) * 2);
    chip.GenerateStereo(stereo.data(), n);

    std::vector<float> mono(n);
    float peakAbs = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        mono[i] = 0.5f * (stereo[static_cast<size_t>(i) * 2] + stereo[static_cast<size_t>(i) * 2 + 1]);
        peakAbs = std::max(peakAbs, std::abs(mono[i]));
    }

    std::vector<double> window = dsp::MakeBlackmanWindow(n);
    std::vector<double> dbfs = dsp::ComputeFrameDbfs(mono.data(), n, window);

    int peakBin = 0;
    double peakDb = dbfs[0];
    for (size_t k = 1; k < dbfs.size(); ++k)
    {
        if (dbfs[k] > peakDb) { peakDb = dbfs[k]; peakBin = static_cast<int>(k); }
    }

    const double hzPerBin = static_cast<double>(sampleRate) / n;
    const int expectedBin = static_cast<int>(std::llround(actualFreqHz / hzPerBin));

    bool ok = true;
    if (peakAbs < 1e-4f)
    {
        std::printf("FAIL: rendered audio is silent (peak abs sample = %.6f)\n", peakAbs);
        ok = false;
    }
    if (std::abs(peakBin - expectedBin) > 1)
    {
        std::printf("FAIL: peak bin = %d, expected %d (target %.2f Hz -> fnum=%d block=%d -> %.2f Hz actual)\n",
                     peakBin, expectedBin, targetFreqHz, fb.fnum, fb.block, actualFreqHz);
        ok = false;
    }

    std::printf("target=%.2fHz actual=%.2fHz(fnum=%d,block=%d) peakBin=%d(expected %d) peakDb=%.2f peakAbsSample=%.4f\n",
                targetFreqHz, actualFreqHz, fb.fnum, fb.block, peakBin, expectedBin, peakDb, peakAbs);
    return ok;
}

// Regression check for HzToFnumBlock's above-range fallback: a frequency
// past what any fnum/block combination can represent (real OPL3 tops out
// around 6208 Hz -- fnum=1023 at block=7, the largest fnum range
// available) must clamp to that actual ceiling, not silently collapse to
// some other, much lower frequency. An earlier version of the fallback
// returned block=0 instead of block=7, which for fnum=1023 is ~48.5 Hz --
// about as wrong an answer to "frequency too high" as available.
bool CheckHighFrequencyClamp()
{
    const double requestedFreqHz = 9000.0; // comfortably past the ~6208 Hz ceiling
    const OplFnumBlock fb = HzToFnumBlock(requestedFreqHz);
    const double actualFreqHz = FnumBlockToHz(fb.fnum, fb.block);

    bool ok = true;
    if (fb.block != 7 || fb.fnum != 1023)
    {
        std::printf("FAIL: expected the clamp to land on fnum=1023,block=7 (the real ceiling), got "
                     "fnum=%d,block=%d\n",
                     fb.fnum, fb.block);
        ok = false;
    }
    if (actualFreqHz < 6000.0)
    {
        std::printf("FAIL: clamped frequency %.1f Hz is nowhere near the real ~6208 Hz ceiling -- looks like "
                     "the fallback collapsed to a low block instead of clamping to the top of the range\n",
                     actualFreqHz);
        ok = false;
    }

    std::printf("highFreqClamp: requested=%.0fHz -> fnum=%d,block=%d -> actual=%.2fHz\n", requestedFreqHz, fb.fnum,
                fb.block, actualFreqHz);
    return ok;
}

// Regression check for both live workflows (FitChannelsInTurn,
// FitChannelsPerFrame): a tone entirely above kOplMaxFreqHz (~6208 Hz,
// the real hardware ceiling) is fundamentally unplayable and should be
// excluded outright, not clamped onto a channel at the ceiling -- a
// signal with ONLY such content should end up with no audible output at
// all rather than a spurious ~6208 Hz tone. Without this exclusion, an
// above-ceiling peak still gets "fit" via HzToFnumBlock's own clamp,
// which is exactly the bug this guards against.
bool CheckAboveCeilingExclusion()
{
    const int sampleRate = 44100;
    const double aboveCeilingFreqHz = 8000.0; // > kOplMaxFreqHz
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
        input[static_cast<size_t>(i)] =
            0.5f * static_cast<float>(std::sin(2.0 * kPi * aboveCeilingFreqHz * i / sampleRate));

    const double frameRateHz = 20.0;
    const int maxChannels = 4;

    bool ok = true;

    const oplfit::MultiChannelSineFit turnFit =
        oplfit::FitChannelsInTurn(input, sampleRate, frameRateHz, maxChannels);
    if (!turnFit.channels.empty())
    {
        std::printf("FAIL: FitChannelsInTurn produced %zu channel(s) for an above-ceiling-only input -- "
                    "should have excluded it entirely\n",
                    turnFit.channels.size());
        ok = false;
    }

    // Some low-level spectral leakage from the 8000 Hz tone's own window
    // sidelobes into bins just below the ceiling is normal and not what
    // this guards against -- the actual bug would clamp the tone's full
    // amplitude onto the ceiling, producing a peak in the same ballpark
    // as any other properly-fit tone in this test suite (roughly 0.1-0.5
    // for this input amplitude), not a near-noise-floor ripple.
    const oplfit::MultiChannelSineFit perFrameFit =
        oplfit::FitChannelsPerFrame(input, sampleRate, frameRateHz, maxChannels);
    float perFramePeakAbs = 0.0f;
    for (float s : perFrameFit.mixRendered) perFramePeakAbs = std::max(perFramePeakAbs, std::abs(s));
    if (perFramePeakAbs > 0.02f)
    {
        std::printf("FAIL: FitChannelsPerFrame's render peak (abs sample = %.6f) is too loud to be mere leakage "
                    "for an above-ceiling-only input -- looks like it clamped onto a channel instead of "
                    "excluding it\n",
                    perFramePeakAbs);
        ok = false;
    }

    std::printf("aboveCeilingExclusion: turnChannels=%zu perFramePeakAbsSample=%.6f\n", turnFit.channels.size(),
                perFramePeakAbs);
    return ok;
}

// End-to-end sanity check for the Phase 2 single-channel fitter: feed it a
// synthetic 440 Hz tone, confirm it finds a target trend, produces a
// non-empty frame timeline, and that rendering that timeline back through
// the real emulator is audibly non-silent and centered near the input's
// own fundamental -- proves FitSingleChannel/RenderChannel are wired
// together correctly, not that the fit is perceptually good (that needs
// ears, per the OPL3 plan's own Phase 2 verification step).
bool CheckSingleChannelFit()
{
    const int sampleRate = 44100;
    const double toneFreqHz = 440.0;
    const double durationSeconds = 1.0;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
        input[static_cast<size_t>(i)] =
            0.5f * static_cast<float>(std::sin(2.0 * kPi * toneFreqHz * i / sampleRate));

    const double frameRateHz = 50.0;
    const oplfit::FitResult fit = oplfit::FitSingleChannel(input, sampleRate, frameRateHz, 0.5, nullptr);

    if (fit.frames.empty())
    {
        std::printf("FAIL: FitSingleChannel produced no frames\n");
        return false;
    }

    const std::vector<float> rendered = oplfit::RenderChannel(fit, sampleRate);
    if (rendered.empty())
    {
        std::printf("FAIL: RenderChannel produced no samples\n");
        return false;
    }

    const int n = dsp::kFftSize;
    const int start = std::max(0, static_cast<int>(rendered.size()) / 2 - n / 2);
    std::vector<float> frame(static_cast<size_t>(n), 0.0f);
    const int available = std::min(n, static_cast<int>(rendered.size()) - start);
    for (int i = 0; i < available; ++i)
        frame[static_cast<size_t>(i)] = rendered[static_cast<size_t>(start + i)];

    float peakAbs = 0.0f;
    for (float s : frame) peakAbs = std::max(peakAbs, std::abs(s));

    const std::vector<double> window = dsp::MakeBlackmanWindow(n);
    const std::vector<double> dbfs = dsp::ComputeFrameDbfs(frame.data(), n, window);
    int peakBin = 0;
    double peakDb = dbfs[0];
    for (size_t k = 1; k < dbfs.size(); ++k)
        if (dbfs[k] > peakDb) { peakDb = dbfs[k]; peakBin = static_cast<int>(k); }

    const double hzPerBin = static_cast<double>(sampleRate) / n;
    const double peakFreqHz = peakBin * hzPerBin;

    bool ok = true;
    if (peakAbs < 1e-4f)
    {
        std::printf("FAIL: rendered channel is silent (peak abs sample = %.6f)\n", peakAbs);
        ok = false;
    }
    // Generous tolerance: the fit quantizes onto OPL's fnum/block grid and
    // may pick a modulator that shifts energy onto a sideband rather than
    // the bare fundamental, so this only checks the render landed roughly
    // in the right neighborhood, not exact frequency match.
    if (std::abs(peakFreqHz - toneFreqHz) > 100.0)
    {
        std::printf("FAIL: rendered peak frequency %.1f Hz too far from input %.1f Hz\n", peakFreqHz, toneFreqHz);
        ok = false;
    }

    std::printf("frames=%zu modMultipleIndex=%d modTotalLevel=%d modWaveform=%d carWaveform=%d peakFreqHz=%.1f peakDb=%.2f peakAbsSample=%.4f\n",
                fit.frames.size(), fit.setup.modMultipleIndex, fit.setup.modTotalLevel, fit.setup.modWaveform,
                fit.setup.carWaveform, peakFreqHz, peakDb, peakAbs);
    return ok;
}

// End-to-end sanity check for the "Make OPL3" button's combined workflow
// (FitChannelsInTurn): a two-simultaneous-tone input should produce more
// than one channel (bounded by maxChannels), a non-empty mix, and --
// the actual point of this test -- FitChannelsInTurn's incrementally-
// summed mix (each new channel's own solo render added into a running
// total, rather than re-simulating every channel together) should match
// RenderAllFramesMix's real joint simulation of the same channels
// closely, verifying against the real emulator (not just in theory) that
// the additive-superposition assumption behind that optimization holds:
// every channel here uses the same fully-disconnected, additive-mode
// timbre, so OPL3's real combined output genuinely is the sample-by-
// sample sum of each channel's own independent signal.
bool CheckFitChannelsInTurn()
{
    const int sampleRate = 44100;
    const double freqA = 300.0;
    const double freqB = 900.0;
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqA * t))
            + 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqB * t));
    }

    const double frameRateHz = 10.0;
    const int maxChannels = 4;
    const oplfit::MultiChannelSineFit fit = oplfit::FitChannelsInTurn(input, sampleRate, frameRateHz, maxChannels);

    bool ok = true;
    if (fit.channels.empty())
    {
        std::printf("FAIL: FitChannelsInTurn produced no channels\n");
        return false;
    }
    if (static_cast<int>(fit.channels.size()) > maxChannels)
    {
        std::printf("FAIL: FitChannelsInTurn produced %zu channels, more than maxChannels=%d\n",
                     fit.channels.size(), maxChannels);
        ok = false;
    }
    if (fit.mixRendered.empty())
    {
        std::printf("FAIL: FitChannelsInTurn's mixRendered is empty\n");
        ok = false;
    }

    const std::vector<float> jointMix = oplfit::RenderAllFramesMix(fit.channels, sampleRate);
    const size_t n = std::min(fit.mixRendered.size(), jointMix.size());
    double maxDiff = 0.0;
    for (size_t i = 0; i < n; ++i)
        maxDiff = std::max(maxDiff, static_cast<double>(std::abs(fit.mixRendered[i] - jointMix[i])));
    if (maxDiff > 0.01)
    {
        std::printf("FAIL: incremental-sum mix differs from joint RenderAllFramesMix by up to %.4f -- the "
                     "additive-superposition assumption behind FitChannelsInTurn's optimization doesn't hold\n",
                     maxDiff);
        ok = false;
    }

    std::printf("channelsInTurn: channels=%zu mixRenderedSamples=%zu maxMixDiff=%.5f\n", fit.channels.size(),
                fit.mixRendered.size(), maxDiff);
    return ok;
}

// Sanity check for the alternate per-frame independent workflow
// (FitChannelsPerFrame, modeled on the reference Python tool): a two-
// simultaneous-tone input should produce exactly maxChannels channels
// (this workflow always emits a fixed-size, silence-padded set, unlike
// FitChannelsInTurn), a non-empty mix whose incrementally-summed value
// matches a real joint simulation (same additive-superposition property,
// verified the same way), and -- the actual point of this workflow --
// both tones represented simultaneously at some frame despite there
// being no persistent-identity mechanism deciding which channel gets
// which tone.
bool CheckFitChannelsPerFrame()
{
    const int sampleRate = 44100;
    const double freqA = 300.0;
    const double freqB = 900.0;
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqA * t))
            + 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqB * t));
    }

    const double frameRateHz = 10.0;
    const int maxChannels = 4;
    const oplfit::MultiChannelSineFit fit =
        oplfit::FitChannelsPerFrame(input, sampleRate, frameRateHz, maxChannels);

    bool ok = true;
    if (fit.channels.size() != static_cast<size_t>(maxChannels))
    {
        std::printf("FAIL: FitChannelsPerFrame produced %zu channels, expected exactly maxChannels=%d\n",
                     fit.channels.size(), maxChannels);
        ok = false;
    }
    if (fit.mixRendered.empty())
    {
        std::printf("FAIL: FitChannelsPerFrame's mixRendered is empty\n");
        ok = false;
    }

    const std::vector<float> jointMix = oplfit::RenderAllFramesMix(fit.channels, sampleRate);
    const size_t n = std::min(fit.mixRendered.size(), jointMix.size());
    double maxDiff = 0.0;
    for (size_t i = 0; i < n; ++i)
        maxDiff = std::max(maxDiff, static_cast<double>(std::abs(fit.mixRendered[i] - jointMix[i])));
    if (maxDiff > 0.01)
    {
        std::printf("FAIL: incremental-sum mix differs from joint RenderAllFramesMix by up to %.4f\n", maxDiff);
        ok = false;
    }

    bool sawBothSimultaneously = false;
    if (!fit.channels.empty() && !fit.channels[0].frames.empty())
    {
        const size_t probeFrame = fit.channels[0].frames.size() / 2;
        double freqsAtProbe[2] = {0.0, 0.0};
        int activeCount = 0;
        for (const oplfit::AllFramesFit& ch : fit.channels)
        {
            if (activeCount >= 2 || probeFrame >= ch.frames.size())
                continue;
            const opl::FramePatch& fr = ch.frames[probeFrame].frame;
            if (fr.carTotalLevel < 63)
                freqsAtProbe[activeCount++] = FnumBlockToHz(fr.fnum, fr.block);
        }
        if (activeCount == 2)
        {
            const double lo = std::min(freqsAtProbe[0], freqsAtProbe[1]);
            const double hi = std::max(freqsAtProbe[0], freqsAtProbe[1]);
            if (std::abs(lo - freqA) < 60.0 && std::abs(hi - freqB) < 60.0)
                sawBothSimultaneously = true;
        }
    }
    if (!sawBothSimultaneously)
    {
        std::printf("FAIL: did not find both %0.f Hz and %.0f Hz represented simultaneously at the probe frame\n",
                     freqA, freqB);
        ok = false;
    }

    std::printf("channelsPerFrame: channels=%zu mixRenderedSamples=%zu maxMixDiff=%.5f\n", fit.channels.size(),
                fit.mixRendered.size(), maxDiff);
    return ok;
}

// Regression check for FitChannelsInTurn's greedy voice-allocation
// packing: a loud 300 Hz tone that's interrupted by two silent gaps
// (so Peak Trends tracks it as several separate, time-disjoint
// PeakTrend fragments, each with substantial energy of its own) plays
// alongside a quieter but continuous 3000 Hz tone. Naively ranking
// every trend by raw total energy and taking the top maxChannels would
// let two of the louder 300 Hz fragments (which never overlap each
// other in time) monopolize both available channels, leaving the
// simultaneous 3000 Hz content completely unrepresented -- exactly the
// failure mode a real recording (fragmented fundamental during pauses,
// crowding out formants/harmonics) surfaced in practice. With only 2
// channels and 3+ fragments plus the continuous tone, correct behavior
// packs the non-overlapping 300 Hz fragments onto one shared channel,
// freeing the other for the 3000 Hz tone that actually needs it.
bool CheckVoiceAllocationPacking()
{
    const int sampleRate = 44100;
    const double freqLoud = 300.0;
    const double freqQuiet = 3000.0;
    const double durationSeconds = 0.9;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples), 0.0f);
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        // 300 Hz active during [0,0.2) u [0.3,0.5) u [0.6,0.8), silent
        // (a real gap, not just quiet) in between and at the very end.
        const bool loudActive = (t < 0.2) || (t >= 0.3 && t < 0.5) || (t >= 0.6 && t < 0.8);
        float sample = 0.6f * static_cast<float>(std::sin(2.0 * kPi * freqQuiet * t));
        if (loudActive)
            sample += 0.9f * static_cast<float>(std::sin(2.0 * kPi * freqLoud * t));
        input[static_cast<size_t>(i)] = sample;
    }

    const double frameRateHz = 20.0;
    const int maxChannels = 2;
    const oplfit::MultiChannelSineFit fit = oplfit::FitChannelsInTurn(input, sampleRate, frameRateHz, maxChannels);

    bool ok = true;
    if (static_cast<int>(fit.channels.size()) != maxChannels)
    {
        std::printf("FAIL: expected exactly %d channels (one for the packed 300 Hz fragments, one for the "
                    "continuous 3000 Hz tone), got %zu\n",
                    maxChannels, fit.channels.size());
        ok = false;
    }

    // Look for a frame inside a loud-active window where one channel is
    // near 300 Hz and another is near 3000 Hz at the same time -- proof
    // the quiet continuous tone actually got its own channel rather than
    // being crowded out.
    bool sawBothSimultaneously = false;
    const int probeFrame = static_cast<int>(0.35 * frameRateHz); // inside [0.3,0.5)
    double freqsAtProbe[2] = {0.0, 0.0};
    int activeCount = 0;
    for (size_t ch = 0; ch < fit.channels.size() && ch < 2; ++ch)
    {
        const oplfit::AllFramesFit& c = fit.channels[ch];
        if (static_cast<size_t>(probeFrame) < c.frames.size())
        {
            const opl::FramePatch& fr = c.frames[static_cast<size_t>(probeFrame)].frame;
            if (fr.carTotalLevel < 63)
                freqsAtProbe[activeCount++] = FnumBlockToHz(fr.fnum, fr.block);
        }
    }
    if (activeCount == 2)
    {
        const double lo = std::min(freqsAtProbe[0], freqsAtProbe[1]);
        const double hi = std::max(freqsAtProbe[0], freqsAtProbe[1]);
        if (std::abs(lo - freqLoud) < 60.0 && std::abs(hi - freqQuiet) < 200.0)
            sawBothSimultaneously = true;
    }
    if (!sawBothSimultaneously)
    {
        std::printf("FAIL: at frame %d (t=%.2fs, both tones should be active), did not find one channel near "
                    "%.0f Hz and another near %.0f Hz simultaneously (found %d active channel(s): %.1f, %.1f)\n",
                    probeFrame, probeFrame / frameRateHz, freqLoud, freqQuiet, activeCount, freqsAtProbe[0],
                    freqsAtProbe[1]);
        ok = false;
    }

    std::printf("voiceAllocationPacking: channels=%zu activeAtProbe=%d freqs=(%.1f,%.1f)\n", fit.channels.size(),
                activeCount, freqsAtProbe[0], freqsAtProbe[1]);
    return ok;
}

// Regression check for the audibility floor: a loud 300 Hz tone plays
// alongside a MUCH quieter (~100 dB down) 5000 Hz tone. The quiet tone
// is still a genuine local spectral maximum -- nothing else is anywhere
// near 5000 Hz -- so the peak tracker legitimately finds and tracks it as
// its own trend, and frequency-banded round-robin selection would
// otherwise guarantee it a channel purely for being "the loudest thing"
// in its own (otherwise-empty) high-frequency band. But at 100 dB below
// the loud tone, it's far below what carrier total level's 63-step,
// ~0.75 dB/step range can represent on top of the channel's own loudest
// achievable output -- i.e. it can never render as anything but silence
// no matter which channel it lands on -- so it should be excluded up
// front rather than spend a whole channel producing nothing audible.
bool CheckAudibilityFloor()
{
    const int sampleRate = 44100;
    const double freqLoud = 300.0;
    const double freqInaudible = 5000.0;
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * freqLoud * t))
            + 0.000005f * static_cast<float>(std::sin(2.0 * kPi * freqInaudible * t));
    }

    const double frameRateHz = 20.0;
    const int maxChannels = 4;
    const oplfit::MultiChannelSineFit fit = oplfit::FitChannelsInTurn(input, sampleRate, frameRateHz, maxChannels);

    bool ok = true;
    if (fit.channels.size() != 1)
    {
        std::printf("FAIL: expected exactly 1 channel (the inaudible 5000 Hz tone should be excluded, not "
                    "spend a channel), got %zu\n",
                    fit.channels.size());
        ok = false;
    }

    std::printf("audibilityFloor: channels=%zu\n", fit.channels.size());
    return ok;
}

// Regression check for coherent summing of "nothing to play" channels: on
// genuinely silent input, every one of maxChannels channels in
// FitChannelsPerFrame has no real candidate for any frame and sits at its
// cosmetic SilentDefaultFb default (paired with carTotalLevel=63) for the
// whole recording. A single such channel is individually inaudible
// (~-60dB, TL=63 alone is only ~47dB of attenuation) -- but if every
// channel defaulted to the SAME frequency, they'd also share the same
// phase (identical setup, identical key-on timing) and sum COHERENTLY in
// FitChannelsPerFrame's per-channel-solo-render summation, turning that
// individually-inaudible residual into a clearly audible tone as more
// channels pile onto the same frequency. With maxChannels channels all
// silent at once (which happens now, was previously a bug demonstrated
// empirically as an 18-channel jump from -60dB to -35dB), the combined
// mix's own spectral peak should stay close to what one channel alone
// produces, not grow with channel count.
bool CheckSilentChannelsDontCoherentlySum()
{
    const int sampleRate = 44100;
    const double durationSeconds = 1.0;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);
    const std::vector<float> silence(static_cast<size_t>(totalSamples), 0.0f);

    const int maxChannels = 18;
    const oplfit::MultiChannelSineFit fit = oplfit::FitChannelsPerFrame(silence, sampleRate, 100.0, maxChannels);

    bool ok = true;
    if (static_cast<int>(fit.channels.size()) != maxChannels)
    {
        std::printf("FAIL: expected exactly maxChannels=%d channels (FitChannelsPerFrame always produces a "
                    "fixed-size set), got %zu\n",
                    maxChannels, fit.channels.size());
        ok = false;
    }

    const int n = dsp::kFftSize;
    double peakDb = -1e300;
    if (static_cast<int>(fit.mixRendered.size()) >= n)
    {
        std::vector<float> frame(fit.mixRendered.end() - n, fit.mixRendered.end());
        const std::vector<double> window = dsp::MakeBlackmanWindow(n);
        const std::vector<double> dbfs = dsp::ComputeFrameDbfs(frame.data(), n, window);
        for (double d : dbfs) peakDb = std::max(peakDb, d);
    }
    // A single silent-default channel alone measures close to -60dB; the
    // old shared-440Hz-default bug pushed 18 coherently-summed channels
    // up to about -35dB. -50dB is a comfortable line that catches the
    // bug (a real regression would land well above it) without being so
    // tight it flags harmless incoherent-summing headroom.
    if (peakDb > -50.0)
    {
        std::printf("FAIL: combined mix's spectral peak is %.2fdB -- too loud for %d individually-inaudible "
                    "silent-default channels, looks like they're coherently summing at a shared frequency again\n",
                    peakDb, maxChannels);
        ok = false;
    }

    std::printf("silentChannelsDontCoherentlySum: channels=%zu peakDb=%.2f\n", fit.channels.size(), peakDb);
    return ok;
}

// Sanity check for the Phase 3 residual loop: a two-tone input (loud 440
// Hz + quieter 1200 Hz, disjoint enough that Peak Trends tracks them as
// separate trends) should produce at least two channels, each landing
// near one of the two tones, and RenderMix should be non-silent.
bool CheckMultiChannelFit()
{
    const int sampleRate = 44100;
    const double loudFreqHz = 440.0;
    const double quietFreqHz = 1200.0;
    const double durationSeconds = 1.0;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] =
            0.5f * static_cast<float>(std::sin(2.0 * kPi * loudFreqHz * t))
            + 0.2f * static_cast<float>(std::sin(2.0 * kPi * quietFreqHz * t));
    }

    const double frameRateHz = 50.0;
    const oplfit::MultiChannelResult multi =
        oplfit::FitMultiChannel(input, sampleRate, frameRateHz, 0.5, 4, nullptr);

    if (multi.channels.size() < 2)
    {
        std::printf("FAIL: FitMultiChannel produced only %zu channel(s), expected >= 2\n", multi.channels.size());
        return false;
    }

    const std::vector<float> mix = oplfit::RenderMix(multi.channels, sampleRate);
    if (mix.empty())
    {
        std::printf("FAIL: RenderMix produced no samples\n");
        return false;
    }

    float peakAbs = 0.0f;
    for (float s : mix) peakAbs = std::max(peakAbs, std::abs(s));

    bool ok = true;
    if (peakAbs < 1e-4f)
    {
        std::printf("FAIL: rendered mix is silent (peak abs sample = %.6f)\n", peakAbs);
        ok = false;
    }

    std::printf("multiChannel: channels=%zu mixPeakAbsSample=%.4f\n", multi.channels.size(), peakAbs);
    return ok;
}

// Sanity check for the interactive "1ch all" workflow: a two-tone-in-
// sequence input (300 Hz for the first half, 600 Hz for the second)
// should produce the expected frame count, a non-silent render, AND --
// the actual point of this test -- most consecutive frames *within* each
// sustained half should end up reusing an identical timbre rather than
// re-searching independently, confirming the continuity mechanism
// (FitAllFrames's "cheap check before paying for a fresh search") is
// actually taking effect and not just running the full search every time.
bool CheckAllFramesFit()
{
    const int sampleRate = 44100;
    const double freqA = 300.0;
    const double freqB = 600.0;
    const double durationSeconds = 1.0;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);
    const int halfSamples = totalSamples / 2;

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double freq = (i < halfSamples) ? freqA : freqB;
        input[static_cast<size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * freq * i / sampleRate));
    }

    const double frameRateHz = 10.0;
    const oplfit::AllFramesFit fit = oplfit::FitAllFrames(input, sampleRate, frameRateHz, nullptr);

    bool ok = true;
    const int expectedFrames = static_cast<int>(std::ceil(durationSeconds * frameRateHz));
    if (static_cast<int>(fit.frames.size()) != expectedFrames)
    {
        std::printf("FAIL: FitAllFrames produced %zu frames, expected %d\n", fit.frames.size(), expectedFrames);
        ok = false;
    }

    const std::vector<float> rendered = oplfit::RenderAllFrames(fit, sampleRate);
    if (rendered.empty())
    {
        std::printf("FAIL: RenderAllFrames produced no samples\n");
        ok = false;
    }
    float peakAbs = 0.0f;
    for (float s : rendered) peakAbs = std::max(peakAbs, std::abs(s));
    if (peakAbs < 1e-4f)
    {
        std::printf("FAIL: rendered all-frames output is silent (peak abs sample = %.6f)\n", peakAbs);
        ok = false;
    }

    int reusedCount = 0;
    for (size_t i = 1; i < fit.frames.size(); ++i)
    {
        const opl::ChannelStaticSetup& a = fit.frames[i - 1].setup;
        const opl::ChannelStaticSetup& b = fit.frames[i].setup;
        if (a.modMultipleIndex == b.modMultipleIndex && a.modTotalLevel == b.modTotalLevel
            && a.modWaveform == b.modWaveform && a.carWaveform == b.carWaveform)
            ++reusedCount;
    }
    if (fit.frames.size() > 1 && reusedCount == 0)
    {
        std::printf("FAIL: no consecutive frames reused timbre -- continuity mechanism doesn't seem to be working\n");
        ok = false;
    }

    std::printf("allFrames: frames=%zu reusedTransitions=%d/%zu peakAbsSample=%.4f\n", fit.frames.size(),
                reusedCount, fit.frames.empty() ? size_t{0} : fit.frames.size() - 1, peakAbs);
    return ok;
}

// Sanity check for the "Add channel" workflow: a two-tone input (300 Hz +
// 900 Hz, both sustained the whole duration) fit with channel 1 alone
// only has room to cover one of them well; adding channel 2 against the
// residual (channel 1's own render, via FitAllFrames's previousMixRendered)
// should pick up meaningfully more energy than channel 1 alone -- i.e.
// the two-channel mix should be audibly "fuller", not just channel 2
// redundantly re-covering what channel 1 already had.
bool CheckIncrementalChannelFit()
{
    const int sampleRate = 44100;
    const double freqA = 300.0;
    const double freqB = 900.0;
    const double durationSeconds = 0.5;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqA * t))
            + 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqB * t));
    }

    const double frameRateHz = 10.0;
    std::vector<oplfit::AllFramesFit> channels;
    channels.push_back(oplfit::FitAllFrames(input, sampleRate, frameRateHz));
    std::vector<float> mixOne = oplfit::RenderAllFramesMix(channels, sampleRate);

    channels.push_back(oplfit::FitAllFrames(input, sampleRate, frameRateHz, nullptr, mixOne));
    const std::vector<float> mixTwo = oplfit::RenderAllFramesMix(channels, sampleRate);

    bool ok = true;
    if (channels.size() != 2 || channels[1].frames.empty())
    {
        std::printf("FAIL: second (residual) channel fit produced no frames\n");
        ok = false;
    }
    if (mixOne.empty() || mixTwo.empty())
    {
        std::printf("FAIL: RenderAllFramesMix produced no samples\n");
        return false;
    }

    auto rms = [](const std::vector<float>& v) {
        double sum = 0.0;
        for (float s : v) sum += static_cast<double>(s) * s;
        return std::sqrt(sum / std::max<size_t>(1, v.size()));
    };
    const double rmsOne = rms(mixOne);
    const double rmsTwo = rms(mixTwo);

    // The second channel should add meaningfully more energy, not just
    // redundantly reinforce (or barely touch) what channel 1 covers.
    if (rmsTwo < rmsOne * 1.15)
    {
        std::printf("FAIL: two-channel mix (rms=%.4f) isn't meaningfully louder than one-channel (rms=%.4f) -- "
                     "residual targeting doesn't seem to be adding new coverage\n",
                     rmsTwo, rmsOne);
        ok = false;
    }

    std::printf("incremental: channel2Frames=%zu rmsOne=%.4f rmsTwo=%.4f\n", channels[1].frames.size(), rmsOne,
                rmsTwo);
    return ok;
}

// Regression check for re-approximation: a SINGLE sustained tone has only
// one real feature to cover. After channel 1 locks onto it, the residual
// magnitude subtraction alone doesn't cancel perfectly (OPL's fnum/block
// quantization and 64-step TL granularity mean the rendered tone is never
// bit-for-bit identical to the original), which can leave a small
// leftover bump right where the tone already is -- exactly what could
// make a second channel re-approximate the same tone instead of
// recognizing there's nothing new left. With the already-covered
// channels passed in, the second channel should stay silent for the
// large majority of frames instead.
bool CheckOccupiedFrequencyExclusion()
{
    const int sampleRate = 44100;
    const double freqHz = 440.0;
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
        input[static_cast<size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * kPi * freqHz * i / sampleRate));

    const double frameRateHz = 10.0;
    std::vector<oplfit::AllFramesFit> channels;
    channels.push_back(oplfit::FitAllFrames(input, sampleRate, frameRateHz));
    const std::vector<float> mixOne = oplfit::RenderAllFramesMix(channels, sampleRate);

    const oplfit::AllFramesFit second =
        oplfit::FitAllFrames(input, sampleRate, frameRateHz, nullptr, mixOne, channels);

    int activeFrames = 0;
    for (const oplfit::FrameFit& frame : second.frames)
        if (frame.frame.carTotalLevel < 63) ++activeFrames;

    bool ok = true;
    // A small number of active frames is fine (e.g. right at the very
    // start before things settle), but the second channel should be
    // silent for the large majority of frames -- there's genuinely only
    // one tone here, already covered by channel 1.
    if (activeFrames > static_cast<int>(second.frames.size()) / 4)
    {
        std::printf("FAIL: second channel stayed active for %d/%zu frames on a single-tone input -- looks like "
                     "it's re-approximating the already-covered tone\n",
                     activeFrames, second.frames.size());
        ok = false;
    }

    std::printf("occupiedExclusion: secondChannelActiveFrames=%d/%zu\n", activeFrames, second.frames.size());
    return ok;
}

// Sanity check for VGM export: writes a two-channel fit out, then reads
// the raw bytes back and checks the header fields a real player/hardware
// would rely on (magic, EOF offset matching the actual file size, a
// resolvable VGM-data offset, a nonzero YMF262 clock) and that the data
// block contains at least one OPL3 register write and ends with the
// end-of-data command -- not audio correctness (already covered by the
// render-based checks above), just that the file is structurally sound.
bool CheckVgmExport()
{
    const int sampleRate = 44100;
    const double freqA = 300.0;
    const double freqB = 900.0;
    const double durationSeconds = 0.3;
    const int totalSamples = static_cast<int>(durationSeconds * sampleRate);

    std::vector<float> input(static_cast<size_t>(totalSamples));
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / sampleRate;
        input[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqA * t))
            + 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqB * t));
    }

    const double frameRateHz = 10.0;
    std::vector<oplfit::AllFramesFit> channels;
    channels.push_back(oplfit::FitAllFrames(input, sampleRate, frameRateHz));
    const std::vector<float> mixOne = oplfit::RenderAllFramesMix(channels, sampleRate);
    channels.push_back(oplfit::FitAllFrames(input, sampleRate, frameRateHz, nullptr, mixOne));

    const std::string path = "opl3_selftest_export.vgm";
    std::string err;
    if (!vgm::WriteVgmFile(path, channels, err))
    {
        std::printf("FAIL: WriteVgmFile failed: %s\n", err.c_str());
        return false;
    }

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
    {
        std::printf("FAIL: could not reopen exported VGM file\n");
        return false;
    }
    const std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0);
    std::vector<unsigned char> bytes(static_cast<size_t>(fileSize));
    ifs.read(reinterpret_cast<char*>(bytes.data()), fileSize);
    std::remove(path.c_str());

    auto readU32 = [&](size_t offset) -> uint32_t {
        return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<uint32_t>(bytes[offset + 2]) << 16) | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
    };

    bool ok = true;
    if (fileSize < 0x100 || bytes[0] != 'V' || bytes[1] != 'g' || bytes[2] != 'm' || bytes[3] != ' ')
    {
        std::printf("FAIL: missing/wrong \"Vgm \" magic\n");
        ok = false;
    }
    const uint32_t eofOffset = readU32(0x04);
    if (eofOffset + 0x04 != static_cast<uint32_t>(fileSize))
    {
        std::printf("FAIL: EOF offset field (%u) doesn't match actual file size (%lld)\n", eofOffset,
                     static_cast<long long>(fileSize));
        ok = false;
    }
    const uint32_t dataOffset = 0x34 + readU32(0x34);
    if (dataOffset < 0x100 || dataOffset >= static_cast<uint32_t>(fileSize))
    {
        std::printf("FAIL: VGM data offset (%u) doesn't point inside the file\n", dataOffset);
        ok = false;
    }
    const uint32_t clockHz = readU32(0x5C);
    if (clockHz == 0)
    {
        std::printf("FAIL: YMF262 clock field is zero\n");
        ok = false;
    }

    bool sawRegisterWrite = false;
    bool sawEndOfData = false;
    for (uint32_t i = dataOffset; i < static_cast<uint32_t>(fileSize); )
    {
        const unsigned char cmd = bytes[i];
        if (cmd == 0x5E || cmd == 0x5F) { sawRegisterWrite = true; i += 3; }
        else if (cmd == 0x61) { i += 3; }
        else if (cmd == 0x66) { sawEndOfData = true; break; }
        else { break; } // unexpected command -- stop rather than misparse
    }
    if (!sawRegisterWrite)
    {
        std::printf("FAIL: no OPL3 register write commands found in the data block\n");
        ok = false;
    }
    if (!sawEndOfData)
    {
        std::printf("FAIL: no end-of-data command found\n");
        ok = false;
    }

    std::printf("vgmExport: fileSize=%lld clockHz=%u dataOffset=%u\n", static_cast<long long>(fileSize), clockHz,
                dataOffset);
    return ok;
}

} // namespace

int main()
{
    bool ok = true;
    ok &= CheckSingleToneFundamental();
    ok &= CheckHighFrequencyClamp();
    ok &= CheckAboveCeilingExclusion();
    ok &= CheckSingleChannelFit();
    ok &= CheckMultiChannelFit();
    ok &= CheckAllFramesFit();
    ok &= CheckIncrementalChannelFit();
    ok &= CheckOccupiedFrequencyExclusion();
    ok &= CheckFitChannelsInTurn();
    ok &= CheckFitChannelsPerFrame();
    ok &= CheckVoiceAllocationPacking();
    ok &= CheckAudibilityFloor();
    ok &= CheckSilentChannelsDontCoherentlySum();
    ok &= CheckVgmExport();
    if (ok) std::printf("opl3_selftest: PASS\n");
    return ok ? 0 : 1;
}
