#include "ChipVgmWriter.h"

#include <algorithm>
#include <fstream>
#include <unordered_map>

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

void AppendU32(std::vector<uint8_t>& buf, uint32_t value)
{
    for (int i = 0; i < 4; ++i)
        buf.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
}

constexpr uint32_t kVgmWaitSampleRateHz = 44100; // VGM waits are always in 44.1 kHz samples
constexpr size_t kHeaderSize = 0x100;

// Drops writes that repeat the last value written to the same logical
// address (unless forced), then hands the rest to the chip's encoder.
class DedupVgmSink : public chips::RegisterSink
{
public:
    DedupVgmSink(chips::VgmEncoder& encoder, std::vector<uint8_t>& data) : m_encoder(encoder), m_data(data) {}

    void Write(uint32_t addr, uint8_t value, bool force) override
    {
        auto it = m_last.find(addr);
        if (!force && it != m_last.end() && it->second == value)
            return;
        m_last[addr] = value;
        m_encoder.Encode(addr, value, m_data);
    }

private:
    chips::VgmEncoder& m_encoder;
    std::vector<uint8_t>& m_data;
    std::unordered_map<uint32_t, uint8_t> m_last;
};

} // namespace

bool BuildChipVgm(const sinefit::ChipFit& fit, std::vector<uint8_t>& out, std::string& errorMessage)
{
    if (!fit.target)
    {
        errorMessage = "No target chip.";
        return false;
    }
    if (fit.tracks.empty())
    {
        errorMessage = "No channels to export.";
        return false;
    }
    if (fit.frameRateHz <= 0.0)
    {
        errorMessage = "Invalid frame rate.";
        return false;
    }
    const chips::ChipTarget& target = *fit.target;
    size_t maxFrames = 0;
    for (const sinefit::VoiceTrack& track : fit.tracks)
    {
        if (track.voice < 0 || track.voice >= target.VoiceCount())
        {
            errorMessage = "Channel out of range for " + target.Name() + ".";
            return false;
        }
        maxFrames = std::max(maxFrames, track.frames.size());
    }
    if (maxFrames == 0)
    {
        errorMessage = "No frames to export.";
        return false;
    }

    std::vector<uint8_t> data;
    data.reserve(maxFrames * fit.tracks.size() * 16 + 64);

    for (const chips::VgmDataBlock& block : target.VgmDataBlocks())
    {
        data.push_back(0x67);
        data.push_back(0x66);
        data.push_back(block.type);
        AppendU32(data, static_cast<uint32_t>(8 + block.data.size()));
        AppendU32(data, block.memSize);
        AppendU32(data, block.offset);
        data.insert(data.end(), block.data.begin(), block.data.end());
    }

    std::unique_ptr<chips::VgmEncoder> encoder = target.CreateVgmEncoder();
    DedupVgmSink sink(*encoder, data);
    std::unique_ptr<chips::ChipWriter> writer = target.CreateWriter(sink);
    writer->Init();

    const uint32_t waitSamplesPerFrame = static_cast<uint32_t>(
        std::clamp(std::llround(static_cast<double>(kVgmWaitSampleRateHz) / fit.frameRateHz), 1LL, 65535LL));
    uint64_t totalSamples = 0;
    for (size_t f = 0; f < maxFrames; ++f)
    {
        for (const sinefit::VoiceTrack& track : fit.tracks)
            if (f < track.frames.size())
                writer->ApplyVoice(track.voice, track.frames[f]);
        data.push_back(0x61);
        data.push_back(static_cast<uint8_t>(waitSamplesPerFrame & 0xFF));
        data.push_back(static_cast<uint8_t>((waitSamplesPerFrame >> 8) & 0xFF));
        totalSamples += waitSamplesPerFrame;
    }
    data.push_back(0x66); // end of sound data

    const chips::VgmChipInfo info = target.VgmInfo();
    std::vector<uint8_t> header(kHeaderSize, 0);
    header[0] = 'V';
    header[1] = 'g';
    header[2] = 'm';
    header[3] = ' ';
    WriteU32At(header, 0x08, 0x00000171); // version 1.71: covers every supported chip's clock field
    WriteU32At(header, 0x18, static_cast<uint32_t>(totalSamples));
    WriteU32At(header, 0x34, static_cast<uint32_t>(kHeaderSize - 0x34)); // VGM data offset, relative to itself
    WriteU32At(header, info.clockHeaderOffset, info.clockHz);
    WriteU32At(header, 0x04, static_cast<uint32_t>(kHeaderSize + data.size() - 0x04)); // EOF offset, relative

    out = std::move(header);
    out.insert(out.end(), data.begin(), data.end());
    return true;
}

bool WriteChipVgmFile(const std::string& path, const sinefit::ChipFit& fit, std::string& errorMessage)
{
    std::vector<uint8_t> bytes;
    if (!BuildChipVgm(fit, bytes, errorMessage))
        return false;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        errorMessage = "Unable to write VGM file: " + path;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!ofs.good())
    {
        errorMessage = "Unable to write VGM file: " + path;
        return false;
    }
    return true;
}

} // namespace vgm
