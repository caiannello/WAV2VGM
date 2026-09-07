#include "AudioProject.h"
#include "Dsp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace {

bool parseWavHeader(std::ifstream& ifs, int& channels, int& sampleRate, int& bitsPerSample, uint32_t& dataChunkPos, uint32_t& dataChunkSize)
{
    ifs.seekg(0, std::ios::beg);
    char riff[4];
    ifs.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) return false;
    ifs.seekg(8);
    char wave[4]; ifs.read(wave,4);
    if (std::strncmp(wave, "WAVE", 4) != 0) return false;

    // read chunks until 'fmt ' and 'data'
    channels = 1; sampleRate = 44100; bitsPerSample = 16;
    dataChunkPos = 0; dataChunkSize = 0;

    ifs.seekg(12);
    while (true)
    {
        char id[4];
        ifs.read(id, 4);
        if (!ifs) break; // EOF/short read -- stop before touching stale `id`/`size`
        uint32_t size = 0; ifs.read(reinterpret_cast<char*>(&size), 4);
        if (!ifs) break;
        // size is little endian on disk; already read as little-endian machine order may vary; adjust
        // we'll compute correctly by reconstructing
        uint32_t s = static_cast<uint8_t>(reinterpret_cast<char*>(&size)[0]) |
                     (static_cast<uint8_t>(reinterpret_cast<char*>(&size)[1])<<8) |
                     (static_cast<uint8_t>(reinterpret_cast<char*>(&size)[2])<<16) |
                     (static_cast<uint8_t>(reinterpret_cast<char*>(&size)[3])<<24);
        size = s;

        std::string sid(id,4);
        if (sid == "fmt ")
        {
            std::vector<char> fmtbuf(size);
            ifs.read(fmtbuf.data(), size);
            if (size >= 16)
            {
                uint16_t audioFormat = static_cast<uint8_t>(fmtbuf[0]) | (static_cast<uint8_t>(fmtbuf[1])<<8);
                channels = static_cast<uint8_t>(fmtbuf[2]) | (static_cast<uint8_t>(fmtbuf[3])<<8);
                sampleRate = static_cast<uint8_t>(fmtbuf[4]) | (static_cast<uint8_t>(fmtbuf[5])<<8) | (static_cast<uint8_t>(fmtbuf[6])<<16) | (static_cast<uint8_t>(fmtbuf[7])<<24);
                bitsPerSample = static_cast<uint8_t>(fmtbuf[14]) | (static_cast<uint8_t>(fmtbuf[15])<<8);
            }
        }
        else if (sid == "data")
        {
            dataChunkPos = static_cast<uint32_t>(ifs.tellg());
            dataChunkSize = size;
            ifs.seekg(size, std::ios::cur);
        }
        else
        {
            ifs.seekg(size, std::ios::cur);
        }
        // align to even
        if (size % 2 == 1) ifs.seekg(1, std::ios::cur);
    }

    return dataChunkPos != 0 && dataChunkSize != 0;
}

} // namespace

AudioProject::AudioProject() = default;

bool AudioProject::ImportFromFile(const std::string& path, std::string& errorMessage,
                                   const ProgressCallback& progress)
{
    std::vector<float> newSamples;
    int rate = 44100;
    ProgressCallback loadProgress;
    if (progress)
        loadProgress = [&progress](int pct, const std::string& stage) { progress((pct * 3) / 100, stage); };
    if (!LoadWavFile(path, newSamples, rate, errorMessage, loadProgress))
        return false;

    constexpr int kTargetSampleRate = 44100;
    dsp::ProgressCallback resampleProgress;
    if (progress)
        resampleProgress = [&progress](int pct) { progress(3 + (pct * 7) / 100, "Resampling"); };
    m_samples = dsp::ResampleTo(newSamples, rate, kTargetSampleRate, resampleProgress);
    m_sampleRate = kTargetSampleRate;
    if (progress) progress(10, "Normalizing");
    dsp::NormalizePeak(m_samples);
    m_normalized = true;
    m_sourcePath = path;
    if (progress) progress(12, "Normalizing");

    ProgressCallback spectrogramProgress;
    if (progress)
        spectrogramProgress = [&progress](int pct, const std::string&) { progress(12 + (pct * 88) / 100, "Generating spectrogram"); };
    if (!GenerateSpectrogram(spectrogramProgress))
    {
        errorMessage = "Unable to generate spectrogram.";
        return false;
    }
    m_lastError.clear();
    return true;
}

const std::string& AudioProject::SourcePath() const { return m_sourcePath; }
const std::vector<float>& AudioProject::Samples() const { return m_samples; }
int AudioProject::SampleRate() const { return m_sampleRate; }
bool AudioProject::Normalized() const { return m_normalized; }
const std::vector<unsigned char>& AudioProject::SpectrogramRGB() const { return m_spectrogram_rgb; }
int AudioProject::SpectrogramWidth() const { return m_spectrogram_w; }
int AudioProject::SpectrogramHeight() const { return m_spectrogram_h; }
std::string AudioProject::LastError() const { return m_lastError; }

bool AudioProject::LoadWavFile(const std::string& path, std::vector<float>& outSamples, int& outSampleRate,
                                std::string& errorMessage, const ProgressCallback& progress)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { errorMessage = "Unable to open file: " + path; return false; }
    int channels=1, sampleRate=44100, bitsPerSample=16; uint32_t dataPos=0, dataSize=0;
    if (!parseWavHeader(ifs, channels, sampleRate, bitsPerSample, dataPos, dataSize)) { errorMessage = "Not a valid WAV file or unsupported format"; return false; }
    ifs.clear(); // parseWavHeader's chunk scan deliberately runs until a read fails at EOF, leaving failbit set
    ifs.seekg(dataPos);
    size_t frameSize = channels * (bitsPerSample/8);
    size_t frames = dataSize / frameSize;
    outSamples.clear(); outSamples.reserve(frames);
    const size_t reportStep = std::max<size_t>(1, frames / 100);
    for (size_t i=0;i<frames;i++){
        if (progress && (i % reportStep) == 0)
            progress(static_cast<int>((i * 100) / std::max<size_t>(1, frames - 1)), "Loading file");
        if (bitsPerSample==16){
            int16_t left=0,right=0;
            ifs.read(reinterpret_cast<char*>(&left),2);
            if (channels==2) ifs.read(reinterpret_cast<char*>(&right),2);
            float sample = 0.0f;
            if (channels==2) sample = (static_cast<float>(left)+static_cast<float>(right))*0.5f/32768.0f;
            else sample = static_cast<float>(left)/32768.0f;
            outSamples.push_back(sample);
        } else if (bitsPerSample==8){
            uint8_t v=0; ifs.read(reinterpret_cast<char*>(&v),1);
            float s = static_cast<float>(v)/128.0f - 1.0f;
            if (channels==2){ uint8_t r=0; ifs.read(reinterpret_cast<char*>(&r),1); s=(s + (static_cast<float>(r)/128.0f -1.0f))*0.5f; }
            outSamples.push_back(s);
        } else {
            errorMessage = "Unsupported WAV bit depth."; return false;
        }
    }
    if (progress) progress(100, "Loading file");
    outSampleRate = sampleRate;
    return true;
}

bool AudioProject::GenerateSpectrogram(const ProgressCallback& progress)
{
    m_spectrogram_rgb.clear();
    m_spectrogram_w = 0;
    m_spectrogram_h = 0;

    if (m_samples.empty())
        return true; // nothing to show yet; not an error

    const int n = dsp::kFftSize;
    const int hop = dsp::kHopSize;
    const int total = static_cast<int>(m_samples.size());

    int numFrames = (total <= n)
        ? 1
        : 1 + static_cast<int>(std::ceil(static_cast<double>(total - n) / hop));

    const int height = n / 2 + 1;
    m_spectrogram_w = numFrames;
    m_spectrogram_h = height;
    m_spectrogram_rgb.assign(static_cast<size_t>(numFrames) * height * 3, 0);

    std::vector<double> window = dsp::MakeBlackmanWindow(n);

    if (progress) progress(0, "Generating spectrogram");

    // Each column's FFT/colorization is completely independent of every
    // other column's (they read disjoint, possibly-overlapping slices of
    // the read-only m_samples buffer, and write to disjoint bytes of
    // m_spectrogram_rgb -- a fixed col picks out a unique idx for every
    // row regardless of the row-major layout, so two threads owning
    // different column ranges can never touch the same byte), so this is
    // safe to spread across hardware threads the same way OplFit.cpp's
    // PrecomputeSpectra and PeakAnalysis.cpp's ComputeStepPeaks already
    // do -- and worthwhile: a several-minute recording's dense hop
    // (consecutive columns only kHopSize samples apart) means hundreds of
    // thousands of kFftSize-point FFTs, previously run one at a time. Each
    // worker keeps its own scratch buffers (ComputeFrameDbfsInto,
    // reused across its whole column range) rather than letting every
    // column's worth of work allocate fresh ones.
    auto computeRange = [&](int startCol, int endCol) {
        std::vector<float> frameBuf(static_cast<size_t>(n));
        std::vector<std::complex<double>> complexScratch;
        std::vector<double> dbfs;
        for (int col = startCol; col < endCol; ++col)
        {
            const int start = col * hop;
            const int available = std::clamp(total - start, 0, n);
            for (int i = 0; i < available; ++i)
                frameBuf[static_cast<size_t>(i)] = m_samples[static_cast<size_t>(start + i)];
            for (int i = available; i < n; ++i)
                frameBuf[static_cast<size_t>(i)] = 0.0f;

            dsp::ComputeFrameDbfsInto(frameBuf.data(), n, window, complexScratch, dbfs);

            for (int bin = 0; bin < height; ++bin)
            {
                dsp::RGB color = dsp::DbfsToColor(dbfs[static_cast<size_t>(bin)]);
                int row = height - 1 - bin; // low frequency at the bottom row
                size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(numFrames)
                              + static_cast<size_t>(col)) * 3;
                m_spectrogram_rgb[idx + 0] = color.r;
                m_spectrogram_rgb[idx + 1] = color.g;
                m_spectrogram_rgb[idx + 2] = color.b;
            }
        }
    };

    const unsigned int hwThreads = std::max(1u, std::thread::hardware_concurrency());
    const int numThreads = static_cast<int>(std::min<unsigned int>(hwThreads, static_cast<unsigned int>(numFrames)));
    if (numThreads <= 1)
    {
        computeRange(0, numFrames);
    }
    else
    {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(numThreads));
        const int chunk = (numFrames + numThreads - 1) / numThreads;
        for (int t = 0; t < numThreads; ++t)
        {
            const int start = t * chunk;
            const int end = std::min(numFrames, start + chunk);
            if (start >= end)
                break;
            workers.emplace_back(computeRange, start, end);
        }
        for (std::thread& worker : workers)
            worker.join();
    }

    if (progress) progress(100, "Generating spectrogram");
    return true;
}
