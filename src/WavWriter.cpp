#include "WavWriter.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

namespace wavwriter
{

namespace
{

void WriteU16(std::ofstream& os, uint16_t v)
{
    os.put(static_cast<char>(v & 0xFF));
    os.put(static_cast<char>((v >> 8) & 0xFF));
}

void WriteU32(std::ofstream& os, uint32_t v)
{
    os.put(static_cast<char>(v & 0xFF));
    os.put(static_cast<char>((v >> 8) & 0xFF));
    os.put(static_cast<char>((v >> 16) & 0xFF));
    os.put(static_cast<char>((v >> 24) & 0xFF));
}

} // namespace

bool WriteMonoWavFile(const std::string& path, const std::vector<float>& samples, int sampleRate,
                       std::string& errorMessage)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        errorMessage = "Unable to write WAV file: " + path;
        return false;
    }

    const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * 2);
    ofs.write("RIFF", 4);
    WriteU32(ofs, 36 + dataBytes);
    ofs.write("WAVE", 4);
    ofs.write("fmt ", 4);
    WriteU32(ofs, 16);
    WriteU16(ofs, 1); // PCM
    WriteU16(ofs, 1); // mono
    WriteU32(ofs, static_cast<uint32_t>(sampleRate));
    WriteU32(ofs, static_cast<uint32_t>(sampleRate * 2)); // byte rate: sampleRate * channels * bytesPerSample
    WriteU16(ofs, 2);  // block align
    WriteU16(ofs, 16); // bits per sample
    ofs.write("data", 4);
    WriteU32(ofs, dataBytes);
    for (float s : samples)
    {
        const int16_t v = static_cast<int16_t>(std::clamp(s, -1.0f, 1.0f) * 32767.0f);
        ofs.put(static_cast<char>(v & 0xFF));
        ofs.put(static_cast<char>((v >> 8) & 0xFF));
    }
    return ofs.good();
}

} // namespace wavwriter
