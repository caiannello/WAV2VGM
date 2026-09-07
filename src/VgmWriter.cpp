#include "VgmWriter.h"

#include "OplChip.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>

namespace vgm
{

namespace
{

void WriteU32At(std::vector<uint8_t>& buf, size_t offset, uint32_t value)
{
    buf[offset + 0] = static_cast<uint8_t>(value & 0xFF);
    buf[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    buf[offset + 2] = static_cast<uint8_t>((value >> 16) & 0xFF);
    buf[offset + 3] = static_cast<uint8_t>((value >> 24) & 0xFF);
}

// VGM wait commands are always expressed in 44100 Hz-based samples, per
// the format spec, regardless of what rate any chip in the file actually
// runs at -- confirmed against the VGM spec, not assumed.
constexpr uint32_t kVgmWaitSampleRateHz = 44100;
// Standard AdLib/OPL3 crystal frequency; matches OplChip/dbopl's own
// OPLRATE constant (14318180.0 / 288.0 samples/sec internally).
constexpr uint32_t kYmf262ClockHz = 14318180;
constexpr size_t kHeaderSize = 0x100;

} // namespace

bool WriteVgmFile(const std::string& path, const std::vector<oplfit::AllFramesFit>& channels,
                   std::string& errorMessage)
{
    if (channels.empty())
    {
        errorMessage = "No channels to export.";
        return false;
    }
    if (channels.size() > static_cast<size_t>(opl::kMaxChannels))
    {
        errorMessage = "Too many channels for OPL3 (max 18).";
        return false;
    }
    const double frameRateHz = channels.front().frameRateHz;
    if (frameRateHz <= 0.0)
    {
        errorMessage = "Invalid frame rate.";
        return false;
    }

    size_t maxFrames = 0;
    for (const oplfit::AllFramesFit& ch : channels)
        maxFrames = std::max(maxFrames, ch.frames.size());
    if (maxFrames == 0)
    {
        errorMessage = "No frames to export.";
        return false;
    }

    std::vector<uint8_t> data;
    data.reserve(maxFrames * channels.size() * 16 + 64);

    // -1 = never written; only emit a register write when the value
    // actually differs from the last one written for that (bank, reg)
    // pair -- see the header comment on WriteVgmFile.
    std::array<int, 512> lastValue;
    lastValue.fill(-1);

    // Sample rate is irrelevant here -- this chip instance is only ever
    // used to capture the register writes SetupChannel/ApplyFrame make
    // (via the observer below); GenerateStereo is never called, so no
    // audio is actually synthesized.
    OplChip chip(static_cast<int>(kVgmWaitSampleRateHz));
    chip.SetWriteObserver([&](int bank, uint8_t reg, uint8_t value) {
        const size_t idx = static_cast<size_t>(bank) * 256 + reg;
        if (lastValue[idx] == static_cast<int>(value))
            return;
        lastValue[idx] = value;
        data.push_back(bank == 0 ? 0x5E : 0x5F);
        data.push_back(reg);
        data.push_back(value);
    });

    chip.WriteReg(1, 0x05, 0x01); // enable OPL3 ("new") mode

    const uint32_t waitSamplesPerFrame = static_cast<uint32_t>(
        std::clamp(std::llround(static_cast<double>(kVgmWaitSampleRateHz) / frameRateHz), 1LL, 65535LL));
    uint64_t totalSamples = 0;
    for (size_t f = 0; f < maxFrames; ++f)
    {
        for (size_t ch = 0; ch < channels.size(); ++ch)
        {
            const oplfit::AllFramesFit& fit = channels[ch];
            if (f < fit.frames.size())
            {
                opl::SetupChannel(chip, static_cast<int>(ch), fit.frames[f].setup);
                opl::ApplyFrame(chip, static_cast<int>(ch), fit.frames[f].frame);
            }
        }
        data.push_back(0x61);
        data.push_back(static_cast<uint8_t>(waitSamplesPerFrame & 0xFF));
        data.push_back(static_cast<uint8_t>((waitSamplesPerFrame >> 8) & 0xFF));
        totalSamples += waitSamplesPerFrame;
    }
    data.push_back(0x66); // end of sound data

    std::vector<uint8_t> header(kHeaderSize, 0);
    header[0] = 'V';
    header[1] = 'g';
    header[2] = 'm';
    header[3] = ' ';
    WriteU32At(header, 0x08, 0x00000171); // version 1.71 -- needed for the YMF262 clock field at 0x5C to be read
    WriteU32At(header, 0x18, static_cast<uint32_t>(totalSamples));
    WriteU32At(header, 0x34, static_cast<uint32_t>(kHeaderSize - 0x34)); // VGM data offset, relative to itself
    WriteU32At(header, 0x5C, kYmf262ClockHz);
    const uint32_t eofOffset = static_cast<uint32_t>(kHeaderSize + data.size() - 0x04);
    WriteU32At(header, 0x04, eofOffset); // EOF offset, relative to itself

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        errorMessage = "Unable to write VGM file: " + path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    ofs.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!ofs.good())
    {
        errorMessage = "Unable to write VGM file: " + path;
        return false;
    }
    return true;
}

} // namespace vgm
