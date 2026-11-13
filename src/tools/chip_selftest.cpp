#include "../ChipVgmWriter.h"
#include "../Dsp.h"
#include "../MidiWriter.h"
#include "../chips/MidiTarget.h"
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

struct MidiEvent
{
    uint32_t tick;
    uint8_t status;
    uint8_t d1;
    uint8_t d2;
};

// Minimal SMF reader: header fields plus every channel event of every
// track (meta/sysex skipped). Returns false on any structural problem.
bool ParseSmf(const std::vector<uint8_t>& b, int& format, int& tracks, int& division, std::vector<MidiEvent>& events)
{
    auto be32 = [&](size_t at) { return (uint32_t(b[at]) << 24) | (uint32_t(b[at + 1]) << 16) | (uint32_t(b[at + 2]) << 8) | b[at + 3]; };
    auto be16 = [&](size_t at) { return (int(b[at]) << 8) | b[at + 1]; };
    if (b.size() < 14 || std::memcmp(b.data(), "MThd", 4) != 0 || be32(4) != 6)
        return false;
    format = be16(8);
    tracks = be16(10);
    division = be16(12);
    size_t pos = 14;
    for (int t = 0; t < tracks; ++t)
    {
        if (pos + 8 > b.size() || std::memcmp(&b[pos], "MTrk", 4) != 0)
            return false;
        const size_t end = pos + 8 + be32(pos + 4);
        if (end > b.size())
            return false;
        pos += 8;
        uint32_t tick = 0;
        bool sawEnd = false;
        while (pos < end)
        {
            uint32_t delta = 0;
            for (;;)
            {
                const uint8_t c = b[pos++];
                delta = (delta << 7) | (c & 0x7F);
                if (!(c & 0x80))
                    break;
            }
            tick += delta;
            const uint8_t status = b[pos++];
            if (status == 0xFF)
            {
                const uint8_t type = b[pos++];
                uint32_t len = 0;
                for (;;)
                {
                    const uint8_t c = b[pos++];
                    len = (len << 7) | (c & 0x7F);
                    if (!(c & 0x80))
                        break;
                }
                pos += len;
                if (type == 0x2F)
                    sawEnd = true;
            }
            else if (status >= 0x80 && status < 0xF0)
            {
                const uint8_t kind = status & 0xF0;
                const bool oneByte = kind == 0xC0 || kind == 0xD0;
                MidiEvent e{tick, status, b[pos], oneByte ? uint8_t(0) : b[pos + 1]};
                pos += oneByte ? 1 : 2;
                events.push_back(e);
            }
            else
            {
                return false; // running status/sysex aren't produced by our writer
            }
        }
        if (!sawEnd || pos != end)
            return false;
    }
    return pos == b.size();
}

bool CheckMidi()
{
    bool ok = true;
    // Channel priority: 1-9, 11-16, then 10.
    const int expectedChannels[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 9};
    for (int v = 0; v < 16; ++v)
        if (chips::MidiTarget::MidiChannelOf(v) != expectedChannels[v])
        {
            std::printf("FAIL [MIDI]: voice %d -> channel %d, expected %d\n", v, chips::MidiTarget::MidiChannelOf(v) + 1,
                        expectedChannels[v] + 1);
            ok = false;
        }

    const int bendRange = 12;
    const auto target = chips::CreateMidiTarget(bendRange);
    ok &= CheckPitchAndLevel(*target);

    // A steady tone, plus a slow 300 -> 1200 Hz glide (two octaves, so it
    // has to restrike past the 12-semitone bend range at least once).
    const double seconds = 1.0;
    std::vector<float> input(static_cast<size_t>(seconds * kSampleRate));
    double phase = 0.0;
    for (size_t i = 0; i < input.size(); ++i)
    {
        const double t = static_cast<double>(i) / kSampleRate;
        const double glideHz = 300.0 * std::pow(4.0, t / seconds);
        phase += 2.0 * kPi * glideHz / kSampleRate;
        input[i] = 0.4f * static_cast<float>(std::sin(2.0 * kPi * 2000.0 * t)) + 0.4f * static_cast<float>(std::sin(phase));
    }
    const double frameRate = 50.0;
    const sinefit::ChipFit fit = sinefit::FitChannelsInTurn(target, input, kSampleRate, frameRate, 16);
    std::vector<uint8_t> bytes;
    std::string err;
    if (fit.tracks.empty() || !midi::BuildMidi(fit, bytes, err))
    {
        std::printf("FAIL [MIDI]: export failed: %s\n", err.c_str());
        return false;
    }
    std::vector<uint8_t> vgmBytes;
    if (vgm::BuildChipVgm(fit, vgmBytes, err))
    {
        std::printf("FAIL [MIDI]: VGM export should refuse a MIDI fit\n");
        ok = false;
    }

    int format = 0, tracks = 0, division = 0;
    std::vector<MidiEvent> events;
    if (!ParseSmf(bytes, format, tracks, division, events))
    {
        std::printf("FAIL [MIDI]: not a well-formed SMF\n");
        return false;
    }
    if (format != 1 || division != midi::kTicksPerQuarter || tracks < 2)
    {
        std::printf("FAIL [MIDI]: format=%d tracks=%d division=%d\n", format, tracks, division);
        ok = false;
    }

    // Replay every channel and compare its sounding pitch at each frame
    // start against what the fit asked for.
    std::stable_sort(events.begin(), events.end(), [](const MidiEvent& a, const MidiEvent& b) { return a.tick < b.tick; });
    const double ticksPerFrame = midi::kTicksPerQuarter * 1e6 / midi::kTempoMicrosPerQuarter / frameRate;
    int bends = 0, restrikes = 0, checked = 0;
    for (const sinefit::VoiceTrack& track : fit.tracks)
    {
        const int ch = chips::MidiTarget::MidiChannelOf(track.voice);
        bool program = false, mono = false, rpnMsb = false, rpnLsb = false, range = false;
        int note = -1, bend = 8192, expression = 127, notesOn = 0;
        size_t e = 0;
        for (size_t f = 0; f < track.frames.size(); ++f)
        {
            const uint32_t tick = static_cast<uint32_t>(std::llround(f * ticksPerFrame));
            for (; e < events.size() && events[e].tick <= tick; ++e)
            {
                const MidiEvent& ev = events[e];
                if ((ev.status & 0x0F) != ch)
                    continue;
                switch (ev.status & 0xF0)
                {
                case 0x90: note = ev.d1; ++notesOn; break;
                case 0x80: if (ev.d1 == note) note = -1; break;
                case 0xE0: bend = ev.d1 | (ev.d2 << 7); ++bends; break;
                case 0xC0: program |= ev.d1 == chips::MidiTarget::kGmOcarinaProgram; break;
                case 0xB0:
                    mono |= ev.d1 == 126 && ev.d2 == 1;
                    rpnMsb |= ev.d1 == 101 && ev.d2 == 0;
                    rpnLsb |= ev.d1 == 100 && ev.d2 == 0;
                    range |= ev.d1 == 6 && ev.d2 == bendRange;
                    if (ev.d1 == 11) expression = ev.d2;
                    break;
                }
            }
            const VoiceFrame& fr = track.frames[f];
            if (fr.keyOn != (note >= 0))
            {
                std::printf("FAIL [MIDI]: channel %d frame %zu keyOn=%d but note=%d\n", ch + 1, f, fr.keyOn, note);
                ok = false;
                break;
            }
            if (!fr.keyOn)
                continue;
            const double sounding = note + (bend - 8192) * bendRange / 8192.0;
            const double wanted = chips::MidiTarget::PitchToNote(fr.pitch);
            if (std::abs(sounding - wanted) > 0.01 || expression != chips::MidiTarget::ExpressionOf(fr.level))
            {
                std::printf("FAIL [MIDI]: channel %d frame %zu plays %.3f (expr %d), fit wants %.3f (expr %d)\n", ch + 1,
                            f, sounding, expression, wanted, chips::MidiTarget::ExpressionOf(fr.level));
                ok = false;
                break;
            }
            ++checked;
        }
        restrikes += std::max(0, notesOn - 1);
        if (notesOn > 0 && !(program && mono && rpnMsb && rpnLsb && range))
        {
            std::printf("FAIL [MIDI]: channel %d missing setup (program=%d mono=%d rpn=%d/%d range=%d)\n", ch + 1,
                        program, mono, rpnMsb, rpnLsb, range);
            ok = false;
        }
    }
    if (bends == 0 || restrikes == 0 || checked == 0)
    {
        std::printf("FAIL [MIDI]: expected bends and a restrike on the glide (bends=%d restrikes=%d checked=%d)\n",
                    bends, restrikes, checked);
        ok = false;
    }

    // With nine simultaneous tones, channel 10 must stay unused.
    std::vector<float> nine(static_cast<size_t>(0.4 * kSampleRate), 0.0f);
    for (int k = 0; k < 9; ++k)
        for (size_t i = 0; i < nine.size(); ++i)
            nine[i] += static_cast<float>(0.08 * std::sin(2.0 * kPi * (200.0 + 150.0 * k) * i / kSampleRate));
    const sinefit::ChipFit nineFit = sinefit::FitChannelsPerFrame(target, nine, kSampleRate, frameRate, 16);
    std::vector<uint8_t> nineBytes;
    std::vector<MidiEvent> nineEvents;
    if (!midi::BuildMidi(nineFit, nineBytes, err) || !ParseSmf(nineBytes, format, tracks, division, nineEvents))
    {
        std::printf("FAIL [MIDI]: nine-tone export failed\n");
        ok = false;
    }
    else if (std::any_of(nineEvents.begin(), nineEvents.end(), [](const MidiEvent& ev) { return (ev.status & 0x0F) == 9; }))
    {
        std::printf("FAIL [MIDI]: channel 10 used with only nine voices needed\n");
        ok = false;
    }

    // At the maximum bend range the same two-octave glide never leaves its
    // note -- bends only, no restrike -- and RPN 0 carries 127.
    {
        const int wideRange = chips::MidiTarget::kMaxBendRange;
        const auto wide = chips::CreateMidiTarget(wideRange);
        const sinefit::ChipFit wideFit = sinefit::FitChannelsInTurn(wide, input, kSampleRate, frameRate, 16);
        std::vector<uint8_t> wideBytes;
        std::vector<MidiEvent> wideEvents;
        if (!midi::BuildMidi(wideFit, wideBytes, err) || !ParseSmf(wideBytes, format, tracks, division, wideEvents))
        {
            std::printf("FAIL [MIDI]: bend range %d export failed: %s\n", wideRange, err.c_str());
            return false;
        }
        int noteOns = 0;
        for (const MidiEvent& ev : wideEvents)
            if ((ev.status & 0xF0) == 0x90 && ev.d2 > 0)
                ++noteOns;
        int usedVoices = 0;
        for (const sinefit::VoiceTrack& track : wideFit.tracks)
            usedVoices += std::any_of(track.frames.begin(), track.frames.end(),
                                      [](const chips::VoiceFrame& f) { return f.keyOn; }) ? 1 : 0;
        const bool rangeSet = std::any_of(wideEvents.begin(), wideEvents.end(), [wideRange](const MidiEvent& ev) {
            return (ev.status & 0xF0) == 0xB0 && ev.d1 == 6 && ev.d2 == wideRange;
        });
        if (noteOns != usedVoices || !rangeSet)
        {
            std::printf("FAIL [MIDI]: range %d: %d note-ons for %d voices, RPN range set=%d\n", wideRange, noteOns,
                        usedVoices, rangeSet);
            ok = false;
        }
    }

    std::printf("[MIDI] bytes=%zu tracks=%d bends=%d restrikes=%d framesChecked=%d\n", bytes.size(), tracks, bends,
                restrikes, checked);
    return ok;
}

// Chain workflow: it has to behave like the per-frame one on density (a
// very short burst still gets represented) and like the trend-based one on
// identity (a sustained partial is one unbroken chain, so no re-attacks).
bool CheckChainFit()
{
    const auto target = chips::CreateTarget(ChipId::Opl3);
    const double frameRate = 100.0;
    const double seconds = 0.5;
    const int totalSamples = static_cast<int>(seconds * kSampleRate);
    std::vector<float> input(static_cast<size_t>(totalSamples), 0.0f);
    for (int i = 0; i < totalSamples; ++i)
    {
        const double t = static_cast<double>(i) / kSampleRate;
        double v = 0.4 * std::sin(2.0 * kPi * 900.0 * t);
        if (t >= 0.20 && t < 0.24) // four frames' worth -- far too short for a trend to form
            v += 0.4 * std::sin(2.0 * kPi * 300.0 * t);
        input[static_cast<size_t>(i)] = static_cast<float>(v);
    }

    bool ok = true;
    const sinefit::ChipFit fit = sinefit::FitChannelsByChains(target, input, kSampleRate, frameRate, 4);
    if (fit.tracks.empty() || fit.mixRendered.empty())
    {
        std::printf("FAIL [chains]: fit produced nothing\n");
        return false;
    }
    if (fit.tracks.size() > 4)
    {
        std::printf("FAIL [chains]: %zu tracks for a 4-voice budget\n", fit.tracks.size());
        ok = false;
    }

    // Density: both partials present together during the burst.
    const size_t probe = static_cast<size_t>(0.21 * frameRate);
    bool saw300 = false, saw900 = false;
    for (const sinefit::VoiceTrack& track : fit.tracks)
    {
        if (probe >= track.frames.size() || !track.frames[probe].keyOn)
            continue;
        const double hz = target->PitchToHz(track.voice, track.frames[probe].pitch);
        saw300 |= std::abs(hz - 300.0) < 30.0;
        saw900 |= std::abs(hz - 900.0) < 30.0;
    }
    if (!saw300 || !saw900)
    {
        std::printf("FAIL [chains]: burst frame missing a partial (300 Hz %s, 900 Hz %s)\n",
                    saw300 ? "found" : "missing", saw900 ? "found" : "missing");
        ok = false;
    }

    // Identity: the 900 Hz tone runs the whole file, so it must not be
    // chopped into a new chain every frame or so.
    if (fit.score.keyOnCount > 8)
    {
        std::printf("FAIL [chains]: %d key-ons on a two-partial input\n", fit.score.keyOnCount);
        ok = false;
    }

    // Cross-check the three workflows' scores on the same input.
    const sinefit::ChipFit perFrame = sinefit::FitChannelsPerFrame(target, input, kSampleRate, frameRate, 4);
    const sinefit::ChipFit inTurn = sinefit::FitChannelsInTurn(target, input, kSampleRate, frameRate, 4);
    const auto report = [](const char* name, const sinefit::ChipFit& f) {
        std::printf("  %-10s similarity=%.3f err=%.2fdB keyOns=%d tracks=%zu\n", name, f.score.similarity,
                    f.score.meanAbsDbError, f.score.keyOnCount, f.tracks.size());
    };
    std::printf("[chains] burst/continuity:\n");
    report("chains", fit);
    report("per-frame", perFrame);
    report("trend", inTurn);

    for (const sinefit::ChipFit* f : {&fit, &perFrame, &inTurn})
    {
        if (!(f->score.similarity > 0.0 && f->score.similarity <= 1.0) || !std::isfinite(f->score.meanAbsDbError))
        {
            std::printf("FAIL [chains]: implausible fit score\n");
            ok = false;
        }
    }

    // Mixed-kind target: an unplayable-on-FM high tone must land on a PCM
    // voice rather than being clamped into the FM range.
    const auto both = chips::CreateTarget(ChipId::Opl4Both);
    std::vector<float> wide(static_cast<size_t>(0.3 * kSampleRate), 0.0f);
    for (size_t i = 0; i < wide.size(); ++i)
    {
        const double t = static_cast<double>(i) / kSampleRate;
        wide[i] = static_cast<float>(0.4 * std::sin(2.0 * kPi * 120.0 * t) + 0.4 * std::sin(2.0 * kPi * 12000.0 * t));
    }
    const sinefit::ChipFit wideFit = sinefit::FitChannelsByChains(both, wide, kSampleRate, frameRate, 8);
    double lowHz = 0.0, highHz = 0.0;
    for (const sinefit::VoiceTrack& track : wideFit.tracks)
    {
        const chips::VoiceKind& kind = both->KindOf(track.voice);
        for (const chips::VoiceFrame& frame : track.frames)
        {
            if (!frame.keyOn)
                continue;
            const double hz = both->PitchToHz(track.voice, frame.pitch);
            if (hz < kind.minHz || hz > kind.maxHz)
            {
                std::printf("FAIL [chains]: %s voice plays %.0f Hz outside its range\n", kind.name.c_str(), hz);
                ok = false;
            }
            if (std::abs(hz - 120.0) < 20.0)
                lowHz = hz;
            if (std::abs(hz - 12000.0) < 200.0)
                highHz = hz;
        }
    }
    if (lowHz == 0.0 || highHz == 0.0)
    {
        std::printf("FAIL [chains]: mixed-kind fit missed a tone (120 Hz=%.1f, 12 kHz=%.1f)\n", lowHz, highHz);
        ok = false;
    }
    std::printf("[chains] mixed kind: 120Hz=%.1f 12kHz=%.1f tracks=%zu similarity=%.3f\n", lowHz, highHz,
                wideFit.tracks.size(), wideFit.score.similarity);
    return ok;
}

// `chip_selftest <dir>` additionally writes one VGM per target -- a 440 Hz
// + 1500 Hz two-tone fit -- into <dir>, for checking in an external VGM
// player (e.g. libvgm's vgm2wav).
void DumpVgms(const std::string& dir)
{
    const std::vector<float> input = TwoTones(440.0, 1500.0, 2.0);
    {
        const sinefit::ChipFit fit = sinefit::FitChannelsInTurn(chips::CreateMidiTarget(chips::MidiTarget::kDefaultBendRange),
                                                                input, kSampleRate, 50.0, 16);
        std::string err;
        if (midi::WriteMidiFile(dir + "/MIDI.mid", fit, err))
            std::printf("wrote %s/MIDI.mid\n", dir.c_str());
    }
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
    ok &= CheckChainFit();
    ok &= CheckMidi();
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
