#include "../ChipVgmWriter.h"
#include "../Dsp.h"
#include "../OplFit.h"
#include "../SineFit.h"
#include "../VgmWriter.h"
#include "../chips/ChipTarget.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr int kSampleRate = 44100;

using chips::ChipId;
using chips::ChipTarget;
using chips::VoiceFrame;

const ChipId kAllChips[] = {
    ChipId::Opl3, ChipId::Opl2, ChipId::Opl4Fm, ChipId::Opl4Pcm, ChipId::Opl4Both, ChipId::Opm,
    ChipId::Opn, ChipId::Opna, ChipId::Opn2, ChipId::HuC6280, ChipId::Vsu, ChipId::MultiPcm,
};

std::vector<float> TwoTones(double freqA, double freqB, double seconds)
{
    const int n = static_cast<int>(seconds * kSampleRate);
    std::vector<float> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / kSampleRate;
        out[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqA * t))
            + 0.4f * static_cast<float>(std::sin(2.0 * kPi * freqB * t));
    }
    return out;
}

// Renders `voice` holding one frame steadily and returns (peak bin
// frequency, peak dBFS) of a kFftSize window after a short settle.
void MeasureSteadyTone(const ChipTarget& t, int voice, const VoiceFrame& frame, double& peakHz, double& peakDb)
{
    const double frameRate = 100.0;
    const int settle = kSampleRate / 50;
    const int n = dsp::kFftSize;
    const size_t frames = static_cast<size_t>((settle + n) / (kSampleRate / 100) + 2);
    const std::vector<float> audio = t.RenderVoice(voice, std::vector<VoiceFrame>(frames, frame), frameRate, kSampleRate);
    const std::vector<double> window = dsp::MakeBlackmanWindow(n);
    const std::vector<double> dbfs = dsp::ComputeFrameDbfs(audio.data() + settle, n, window);
    size_t best = 1;
    for (size_t k = 1; k < dbfs.size(); ++k)
        if (dbfs[k] > dbfs[best])
            best = k;
    peakHz = static_cast<double>(best) * kSampleRate / n;
    peakDb = dbfs[best];
}

bool CheckPitchAndLevel(const ChipTarget& t)
{
    bool ok = true;
    const double hzPerBin = static_cast<double>(kSampleRate) / dsp::kFftSize;
    for (int kind = 0; kind < static_cast<int>(t.Kinds().size()); ++kind)
    {
        const int voice = t.FirstVoiceOfKind(kind);
        const chips::VoiceKind& vk = t.Kinds()[static_cast<size_t>(kind)];
        for (double hz : {220.0, 1000.0, 3000.0})
        {
            if (hz < vk.minHz || hz > vk.maxHz)
                continue;
            VoiceFrame frame;
            frame.keyOn = true;
            frame.level = 0;
            if (!t.QuantizePitch(voice, hz, frame.pitch))
            {
                std::printf("FAIL [%s/%s]: %.0f Hz reported unplayable inside its own range\n", t.Name().c_str(),
                            vk.name.c_str(), hz);
                ok = false;
                continue;
            }
            const double actualHz = t.PitchToHz(voice, frame.pitch);
            if (std::abs(actualHz - hz) / hz > 0.03)
            {
                std::printf("FAIL [%s/%s]: %.1f Hz quantized to %.1f Hz\n", t.Name().c_str(), vk.name.c_str(), hz,
                            actualHz);
                ok = false;
            }
            double peakHz = 0.0, peakDb = 0.0;
            MeasureSteadyTone(t, voice, frame, peakHz, peakDb);
            if (std::abs(peakHz - actualHz) > 1.5 * hzPerBin)
            {
                std::printf("FAIL [%s/%s]: rendered peak at %.1f Hz, expected %.1f Hz\n", t.Name().c_str(),
                            vk.name.c_str(), peakHz, actualHz);
                ok = false;
            }

            // Level steps: the rendered tone should drop by roughly what
            // the level table says.
            if (hz == 1000.0)
            {
                VoiceFrame quieter = frame;
                quieter.level = static_cast<uint16_t>(vk.levelDb.size() / 3);
                double qHz = 0.0, qDb = 0.0;
                MeasureSteadyTone(t, voice, quieter, qHz, qDb);
                const double expectedDrop = -vk.levelDb[quieter.level];
                const double drop = peakDb - qDb;
                if (std::abs(drop - expectedDrop) > 1.5)
                {
                    std::printf("FAIL [%s/%s]: level %u dropped %.2f dB, table says %.2f dB\n", t.Name().c_str(),
                                vk.name.c_str(), quieter.level, drop, expectedDrop);
                    ok = false;
                }
            }
        }
        if (vk.levelDb.empty() || vk.levelDb.front() != 0.0)
        {
            std::printf("FAIL [%s/%s]: level table must start at 0 dB\n", t.Name().c_str(), vk.name.c_str());
            ok = false;
        }
        for (size_t i = 1; i < vk.levelDb.size(); ++i)
            if (!(vk.levelDb[i] < vk.levelDb[i - 1]))
            {
                std::printf("FAIL [%s/%s]: level table not strictly decreasing at %zu\n", t.Name().c_str(),
                            vk.name.c_str(), i);
                ok = false;
                break;
            }
    }
    return ok;
}

bool CheckFits(const std::shared_ptr<const ChipTarget>& t, sinefit::ChipFit& perFrameOut)
{
    bool ok = true;
    const double freqA = 300.0, freqB = 900.0;
    const std::vector<float> input = TwoTones(freqA, freqB, 0.4);
    const double frameRate = 20.0;

    perFrameOut = sinefit::FitChannelsPerFrame(t, input, kSampleRate, frameRate, t->VoiceCount());
    const sinefit::ChipFit& fit = perFrameOut;
    if (fit.tracks.size() != static_cast<size_t>(t->VoiceCount()))
    {
        std::printf("FAIL [%s]: per-frame fit made %zu tracks, expected %d\n", t->Name().c_str(), fit.tracks.size(),
                    t->VoiceCount());
        ok = false;
    }
    float peak = 0.0f;
    for (float s : fit.mixRendered)
        peak = std::max(peak, std::abs(s));
    if (peak < 1e-3f)
    {
        std::printf("FAIL [%s]: per-frame mix is silent\n", t->Name().c_str());
        ok = false;
    }

    bool sawA = false, sawB = false;
    if (!fit.tracks.empty() && !fit.tracks[0].frames.empty())
    {
        const size_t probe = fit.tracks[0].frames.size() / 2;
        for (const sinefit::VoiceTrack& track : fit.tracks)
        {
            const VoiceFrame& fr = track.frames[probe];
            if (!fr.keyOn)
                continue;
            const double hz = t->PitchToHz(track.voice, fr.pitch);
            sawA |= std::abs(hz - freqA) < 30.0;
            sawB |= std::abs(hz - freqB) < 30.0;
        }
    }
    if (!sawA || !sawB)
    {
        std::printf("FAIL [%s]: per-frame fit missed a tone at the probe frame (300 Hz %s, 900 Hz %s)\n",
                    t->Name().c_str(), sawA ? "found" : "missing", sawB ? "found" : "missing");
        ok = false;
    }

    const sinefit::ChipFit inTurn = sinefit::FitChannelsInTurn(t, input, kSampleRate, frameRate, t->VoiceCount());
    if (inTurn.tracks.empty() || inTurn.mixRendered.empty())
    {
        std::printf("FAIL [%s]: trend-based fit produced nothing\n", t->Name().c_str());
        ok = false;
    }
    return ok;
}

uint32_t ReadU32(const std::vector<uint8_t>& b, size_t at)
{
    return static_cast<uint32_t>(b[at]) | (static_cast<uint32_t>(b[at + 1]) << 8)
        | (static_cast<uint32_t>(b[at + 2]) << 16) | (static_cast<uint32_t>(b[at + 3]) << 24);
}

// Walks the command stream with the VGM spec's command lengths, so a
// malformed/unknown command anywhere is caught.
bool CheckVgm(const sinefit::ChipFit& fit)
{
    const ChipTarget& t = *fit.target;
    std::vector<uint8_t> bytes;
    std::string err;
    if (!vgm::BuildChipVgm(fit, bytes, err))
    {
        std::printf("FAIL [%s]: VGM build failed: %s\n", t.Name().c_str(), err.c_str());
        return false;
    }
    bool ok = true;
    if (bytes.size() < 0x100 || std::memcmp(bytes.data(), "Vgm ", 4) != 0)
    {
        std::printf("FAIL [%s]: bad VGM magic/size\n", t.Name().c_str());
        return false;
    }
    if (ReadU32(bytes, 0x04) + 4 != bytes.size())
    {
        std::printf("FAIL [%s]: EOF offset mismatch\n", t.Name().c_str());
        ok = false;
    }
    const chips::VgmChipInfo info = t.VgmInfo();
    // Clock field offsets per the VGM 1.71 header layout.
    static const std::map<ChipId, uint32_t> kSpecOffsets = {
        {ChipId::Opl3, 0x5C}, {ChipId::Opl2, 0x50}, {ChipId::Opl4Fm, 0x60}, {ChipId::Opl4Pcm, 0x60},
        {ChipId::Opl4Both, 0x60}, {ChipId::Opm, 0x30}, {ChipId::Opn, 0x44}, {ChipId::Opna, 0x48},
        {ChipId::Opn2, 0x2C}, {ChipId::HuC6280, 0xA4}, {ChipId::Vsu, 0xC4}, {ChipId::MultiPcm, 0x88},
    };
    if (kSpecOffsets.at(t.Id()) != info.clockHeaderOffset)
    {
        std::printf("FAIL [%s]: clock header offset 0x%X, VGM spec says 0x%X\n", t.Name().c_str(),
                    info.clockHeaderOffset, kSpecOffsets.at(t.Id()));
        ok = false;
    }
    if (ReadU32(bytes, info.clockHeaderOffset) != info.clockHz || info.clockHz == 0)
    {
        std::printf("FAIL [%s]: clock not at header offset 0x%X\n", t.Name().c_str(), info.clockHeaderOffset);
        ok = false;
    }

    size_t pos = 0x34 + ReadU32(bytes, 0x34);
    int dataBlocks = 0, writes = 0, waits = 0;
    bool ended = false;
    while (pos < bytes.size() && !ended)
    {
        const uint8_t cmd = bytes[pos];
        size_t len = 0;
        if (cmd == 0x66) { ended = true; len = 1; }
        else if (cmd == 0x61) { len = 3; ++waits; }
        else if (cmd == 0x67) { len = 7 + ReadU32(bytes, pos + 3); ++dataBlocks; }
        else if (cmd >= 0x51 && cmd <= 0x5F) { len = 3; ++writes; }
        else if (cmd == 0xB5 || cmd == 0xB9) { len = 3; ++writes; }
        else if (cmd == 0xC7 || cmd == 0xD0) { len = 4; ++writes; }
        else
        {
            std::printf("FAIL [%s]: unexpected command 0x%02X at 0x%zX\n", t.Name().c_str(), cmd, pos);
            return false;
        }
        pos += len;
    }
    if (!ended || pos != bytes.size())
    {
        std::printf("FAIL [%s]: stream doesn't end cleanly with 0x66\n", t.Name().c_str());
        ok = false;
    }
    if (dataBlocks != static_cast<int>(t.VgmDataBlocks().size()) || writes == 0 || waits == 0)
    {
        std::printf("FAIL [%s]: dataBlocks=%d writes=%d waits=%d\n", t.Name().c_str(), dataBlocks, writes, waits);
        ok = false;
    }
    std::printf("[%s] voices=%d vgmBytes=%zu writes=%d dataBlocks=%d\n", t.Name().c_str(), t.VoiceCount(),
                bytes.size(), writes, dataBlocks);
    return ok;
}

// Captures the last value written to every logical address.
class CaptureSink : public chips::RegisterSink
{
public:
    void Write(uint32_t addr, uint8_t value, bool) override { last[addr] = value; }
    std::map<uint32_t, uint8_t> last;
};

bool Expect(const char* what, const CaptureSink& s, uint32_t addr, int expected)
{
    const auto it = s.last.find(addr);
    if (it == s.last.end() || it->second != expected)
    {
        std::printf("FAIL [%s]: register 0x%X = %s, expected 0x%02X\n", what, addr,
                    it == s.last.end() ? "unwritten" : std::to_string(it->second).c_str(), expected);
        return false;
    }
    return true;
}

// Spot-checks that each target's register encoding of a known frame
// lands where the chip (as libvgm emulates it) expects it.
bool CheckRegisterEncodings()
{
    bool ok = true;
    auto applyOne = [](ChipId id, int voice, double hz, uint16_t level, CaptureSink& sink, uint32_t& pitch) {
        const auto t = chips::CreateTarget(id);
        VoiceFrame f;
        t->QuantizePitch(voice, hz, f.pitch);
        f.level = level;
        f.keyOn = true;
        pitch = f.pitch;
        auto w = t->CreateWriter(sink);
        w->Init();
        w->ApplyVoice(voice, f);
    };

    { // OPM: A4 = KC 0x4A, KF 0 at 3.579545 MHz.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::Opm, 2, 440.0, 10, s, p);
        ok &= Expect("OPM KC", s, 0x28 + 2, 0x4A);
        ok &= Expect("OPM KF", s, 0x30 + 2, 0x00);
        ok &= Expect("OPM TL C2", s, 0x60 + 0x18 + 2, 10);
        ok &= Expect("OPM key", s, 0x08, 0x40 | 2);
        ok &= Expect("OPM conn", s, 0x20 + 2, 0xC7);
    }
    { // OPN2: A4 ~= block 4 fnum 1084 at 7.670453 MHz; voice 8 = channel 6 (port 1, ch 2), op 4.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::Opn2, 8, 440.0, 5, s, p);
        const int block = static_cast<int>((p >> 11) & 7), fnum = static_cast<int>(p & 0x7FF);
        if (block != 4 || std::abs(fnum - 1084) > 1)
        {
            std::printf("FAIL [OPN2 pitch]: block %d fnum %d\n", block, fnum);
            ok = false;
        }
        ok &= Expect("OPN2 fnum hi", s, 0x100 | 0xA6, (block << 3) | (fnum >> 8));
        ok &= Expect("OPN2 fnum lo", s, 0x100 | 0xA2, fnum & 0xFF);
        ok &= Expect("OPN2 TL op4", s, 0x100 | 0x4E, 5);
        ok &= Expect("OPN2 key", s, 0x28, 0x80 | 6);
        ok &= Expect("OPN2 ch3 mode", s, 0x27, 0x40);
        ok &= Expect("OPN2 DAC off", s, 0x2B, 0x00);
    }
    { // OPN: voice 2 = channel 3 operator 1 in special mode -> 0xAD/0xA9, key bit 0x10.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::Opn, 2, 1000.0, 0, s, p);
        const int block = static_cast<int>((p >> 11) & 7), fnum = static_cast<int>(p & 0x7FF);
        ok &= Expect("OPN special hi", s, 0xAD, (block << 3) | (fnum >> 8));
        ok &= Expect("OPN special lo", s, 0xA9, fnum & 0xFF);
        ok &= Expect("OPN key", s, 0x28, 0x10 | 2);
        ok &= Expect("OPN SSG mute", s, 0x07, 0x3F);
    }
    { // HuC6280: period = 3579545 / (32 * 440) = 254.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::HuC6280, 3, 440.0, 4, s, p);
        ok &= Expect("HuC freq lo", s, (3u << 4) | 2, 254 & 0xFF);
        ok &= Expect("HuC freq hi", s, (3u << 4) | 3, 254 >> 8);
        ok &= Expect("HuC control", s, (3u << 4) | 4, 0x80 | (31 - 4));
    }
    { // VSU: F = 2048 - 5e6 / (32 * 440) = 2048 - 355 = 1693.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::Vsu, 1, 440.0, 0, s, p);
        ok &= Expect("VSU freq lo", s, 0x110 + 2, 1693 & 0xFF);
        ok &= Expect("VSU freq hi", s, 0x110 + 3, 1693 >> 8);
        ok &= Expect("VSU enable", s, 0x110 + 0, 0x80);
        ok &= Expect("VSU level", s, 0x110 + 1, 0xFF);
        ok &= Expect("VSU envelope", s, 0x110 + 4, 0xF0);
        ok &= Expect("VSU wave table", s, 0x110 + 6, 1);
    }
    { // OPL4 PCM: 44100/256 Hz is exactly 1x playback: octave 1, FN 0.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::Opl4Pcm, 5, 44100.0 / 256.0, 3, s, p);
        ok &= Expect("OPL4 FN/wave", s, 0x200 | (0x08 + 24 + 5), 0x01);
        ok &= Expect("OPL4 OCT", s, 0x200 | (0x08 + 48 + 5), 0x10);
        ok &= Expect("OPL4 TL", s, 0x200 | (0x08 + 72 + 5), (3 << 1) | 1);
        ok &= Expect("OPL4 key", s, 0x200 | (0x08 + 96 + 5), 0xA0);
        ok &= Expect("OPL4 wave", s, 0x200 | (0x08 + 5), 0x80);
        ok &= Expect("OPL4 NEW2", s, 0x105, 0x03);
    }
    { // MultiPCM: (8053975/180)/256 Hz is exactly 1x playback -> octave register 1, F 0.
        CaptureSink s; uint32_t p = 0;
        applyOne(ChipId::MultiPcm, 9, 8053975.0 / 180.0 / 256.0, 7, s, p);
        ok &= Expect("MultiPCM pitch lo", s, (9u << 8) | 2, 0x00);
        ok &= Expect("MultiPCM octave", s, (9u << 8) | 3, 0x10);
        ok &= Expect("MultiPCM TL", s, (9u << 8) | 5, (7 << 1) | 1);
        ok &= Expect("MultiPCM key", s, (9u << 8) | 4, 0x80);
    }
    return ok;
}

// The chip-neutral path on the OPL3 target must reproduce the original
// OPL3-only pipeline exactly: same register choices, same VGM bytes.
bool CheckOpl3Parity()
{
    const std::vector<float> input = TwoTones(330.0, 1250.0, 0.5);
    const double frameRate = 50.0;
    const int maxChannels = 6;
    const auto target = chips::CreateTarget(ChipId::Opl3);
    bool ok = true;

    for (int workflow = 0; workflow < 2; ++workflow)
    {
        const oplfit::MultiChannelSineFit legacy = workflow == 0
            ? oplfit::FitChannelsPerFrame(input, kSampleRate, frameRate, maxChannels)
            : oplfit::FitChannelsInTurn(input, kSampleRate, frameRate, maxChannels);
        const sinefit::ChipFit generic = workflow == 0
            ? sinefit::FitChannelsPerFrame(target, input, kSampleRate, frameRate, maxChannels)
            : sinefit::FitChannelsInTurn(target, input, kSampleRate, frameRate, maxChannels);

        const char* name = workflow == 0 ? "per-frame" : "trend-based";
        std::string err;
        const std::string legacyPath = "chip_selftest_legacy.vgm";
        if (!vgm::WriteVgmFile(legacyPath, legacy.channels, err))
        {
            std::printf("FAIL [OPL3 parity %s]: legacy export failed: %s\n", name, err.c_str());
            ok = false;
            continue;
        }
        std::ifstream ifs(legacyPath, std::ios::binary);
        const std::vector<uint8_t> legacyBytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        ifs.close();
        std::remove(legacyPath.c_str());

        std::vector<uint8_t> genericBytes;
        if (!vgm::BuildChipVgm(generic, genericBytes, err))
        {
            std::printf("FAIL [OPL3 parity %s]: generic export failed: %s\n", name, err.c_str());
            ok = false;
            continue;
        }
        if (legacyBytes != genericBytes)
        {
            std::printf("FAIL [OPL3 parity %s]: VGM bytes differ (legacy %zu bytes, generic %zu bytes)\n", name,
                        legacyBytes.size(), genericBytes.size());
            ok = false;
        }
        const size_t n = std::min(legacy.mixRendered.size(), generic.mixRendered.size());
        double maxDiff = 0.0;
        for (size_t i = 0; i < n; ++i)
            maxDiff = std::max(maxDiff, static_cast<double>(std::abs(legacy.mixRendered[i] - generic.mixRendered[i])));
        if (maxDiff > 1e-6 || legacy.mixRendered.size() != generic.mixRendered.size())
        {
            std::printf("FAIL [OPL3 parity %s]: rendered mix differs by %.6f\n", name, maxDiff);
            ok = false;
        }
        std::printf("OPL3 parity %s: %zu VGM bytes identical=%s\n", name, genericBytes.size(),
                    legacyBytes == genericBytes ? "yes" : "no");
    }
    return ok;
}

// `chip_selftest <dir>` additionally writes one VGM per target -- a 440 Hz
// + 1500 Hz two-tone fit -- into <dir>, for checking in an external VGM
// player (e.g. libvgm's vgm2wav).
void DumpVgms(const std::string& dir)
{
    const std::vector<float> input = TwoTones(440.0, 1500.0, 2.0);
    for (ChipId id : kAllChips)
    {
        const auto target = chips::CreateTarget(id);
        const sinefit::ChipFit fit = sinefit::FitChannelsPerFrame(target, input, kSampleRate, 50.0, target->VoiceCount());
        std::string name = target->Name();
        for (char& c : name)
            if (!std::isalnum(static_cast<unsigned char>(c)))
                c = '_';
        std::string err;
        const std::string path = dir + "/" + name + ".vgm";
        if (vgm::WriteChipVgmFile(path, fit, err))
            std::printf("wrote %s\n", path.c_str());
        else
            std::printf("failed %s: %s\n", path.c_str(), err.c_str());

        // One tone per voice (150 Hz + 120 Hz steps), so every voice --
        // not just the first two -- has to come out right.
        const int tones = target->VoiceCount();
        std::vector<float> multi(static_cast<size_t>(2 * kSampleRate), 0.0f);
        std::string freqList;
        for (int k = 0; k < tones; ++k)
        {
            const double hz = 150.0 + 120.0 * k;
            freqList += std::to_string(hz) + "\n";
            for (size_t i = 0; i < multi.size(); ++i)
                multi[i] += static_cast<float>(0.8 / tones * std::sin(2.0 * kPi * hz * static_cast<double>(i) / kSampleRate));
        }
        const sinefit::ChipFit multiFit =
            sinefit::FitChannelsPerFrame(target, multi, kSampleRate, 50.0, target->VoiceCount());
        const std::string multiPath = dir + "/" + name + "_multi.vgm";
        if (vgm::WriteChipVgmFile(multiPath, multiFit, err))
        {
            std::ofstream(dir + "/" + name + "_multi.txt") << freqList;
            std::printf("wrote %s\n", multiPath.c_str());
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1)
        DumpVgms(argv[1]);
    bool ok = true;
    ok &= CheckOpl3Parity();
    ok &= CheckRegisterEncodings();
    for (ChipId id : kAllChips)
    {
        const auto target = chips::CreateTarget(id);
        if (!target)
        {
            std::printf("FAIL: no target for chip id %d\n", static_cast<int>(id));
            ok = false;
            continue;
        }
        ok &= CheckPitchAndLevel(*target);
        sinefit::ChipFit fit;
        ok &= CheckFits(target, fit);
        if (!fit.tracks.empty())
            ok &= CheckVgm(fit);
    }
    if (ok) std::printf("chip_selftest: PASS\n");
    return ok ? 0 : 1;
}
