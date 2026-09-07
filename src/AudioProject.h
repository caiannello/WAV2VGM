#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>

class AudioProject
{
public:
    // percent in [0,100]; stage is a short human-readable label ("Loading file", ...)
    using ProgressCallback = std::function<void(int percent, const std::string& stage)>;

    AudioProject();

    bool ImportFromFile(const std::string& path, std::string& errorMessage,
                         const ProgressCallback& progress = nullptr);

    const std::string& SourcePath() const;
    const std::vector<float>& Samples() const;
    int SampleRate() const;
    bool Normalized() const;

    // Spectrogram image data as RGB bytes (width*height*3)
    const std::vector<unsigned char>& SpectrogramRGB() const;
    int SpectrogramWidth() const;
    int SpectrogramHeight() const;

    std::string LastError() const;

private:
    bool LoadWavFile(const std::string& path, std::vector<float>& outSamples, int& outSampleRate,
                      std::string& errorMessage, const ProgressCallback& progress = nullptr);
    bool GenerateSpectrogram(const ProgressCallback& progress = nullptr);

    std::string m_sourcePath;
    std::vector<float> m_samples;
    std::vector<unsigned char> m_spectrogram_rgb;
    int m_spectrogram_w = 0;
    int m_spectrogram_h = 0;
    int m_sampleRate = 44100;
    bool m_normalized = false;
    std::string m_lastError;
};
