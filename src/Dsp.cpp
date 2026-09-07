#include "Dsp.h"

#include <algorithm>
#include <cmath>

namespace dsp
{

int kFftSize = 4096;
int kHopSize = 32;

namespace
{

constexpr double kPi = 3.14159265358979323846;

double Sinc(double x)
{
    if (std::abs(x) < 1e-12)
        return 1.0;
    double px = kPi * x;
    return std::sin(px) / px;
}

// 63-tap windowed-sinc lowpass FIR designed for a normalized cutoff `fc`
// (as a fraction of the sample rate the filter will run at, 0 < fc < 0.5).
std::vector<double> DesignLowpassFir(double fc, int taps)
{
    std::vector<double> h(taps);
    const double m = static_cast<double>(taps - 1);
    double sum = 0.0;
    for (int n = 0; n < taps; ++n)
    {
        double centered = n - m / 2.0;
        double ideal = 2.0 * fc * Sinc(2.0 * fc * centered);
        double hamming = 0.54 - 0.46 * std::cos(2.0 * kPi * n / m);
        h[n] = ideal * hamming;
        sum += h[n];
    }
    if (sum != 0.0)
        for (double& v : h) v /= sum;
    return h;
}

std::vector<float> ConvolveSameLength(const std::vector<float>& in, const std::vector<double>& fir,
                                       const dsp::ProgressCallback& progress)
{
    std::vector<float> out(in.size());
    const int half = static_cast<int>(fir.size() / 2);
    const int n = static_cast<int>(in.size());
    const int reportStep = std::max(1, n / 100);
    for (int i = 0; i < n; ++i)
    {
        double acc = 0.0;
        for (int k = 0; k < static_cast<int>(fir.size()); ++k)
        {
            int srcIdx = i + k - half;
            if (srcIdx >= 0 && srcIdx < n)
                acc += fir[k] * in[srcIdx];
        }
        out[i] = static_cast<float>(acc);
        if (progress && (i % reportStep) == 0)
            progress(static_cast<int>((static_cast<int64_t>(i) * 100) / std::max(1, n - 1)));
    }
    if (progress) progress(100);
    return out;
}

std::vector<float> LinearResample(const std::vector<float>& in, int srcRate, int targetRate)
{
    if (in.empty())
        return {};
    const size_t outLen = static_cast<size_t>(
        std::llround(static_cast<double>(in.size()) * targetRate / srcRate));
    std::vector<float> out(outLen);
    const double ratio = static_cast<double>(srcRate) / static_cast<double>(targetRate);
    for (size_t i = 0; i < outLen; ++i)
    {
        double srcPos = static_cast<double>(i) * ratio;
        auto idx0 = static_cast<long long>(std::floor(srcPos));
        long long idx1 = idx0 + 1;
        double frac = srcPos - static_cast<double>(idx0);
        float v0 = (idx0 >= 0 && idx0 < static_cast<long long>(in.size())) ? in[idx0] : 0.0f;
        float v1 = (idx1 >= 0 && idx1 < static_cast<long long>(in.size())) ? in[idx1] : v0;
        out[i] = static_cast<float>(v0 * (1.0 - frac) + v1 * frac);
    }
    return out;
}

} // namespace

std::vector<double> MakeBlackmanWindow(int n)
{
    std::vector<double> w(n);
    if (n == 1) { w[0] = 1.0; return w; }
    const double denom = static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i)
    {
        w[i] = 0.42
             - 0.5 * std::cos(2.0 * kPi * i / denom)
             + 0.08 * std::cos(4.0 * kPi * i / denom);
    }
    return w;
}

void FftForwardRadix2(std::vector<std::complex<double>>& a)
{
    const size_t n = a.size();
    if (n <= 1) return;

    for (size_t i = 1, j = 0; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1)
    {
        double ang = -2.0 * kPi / static_cast<double>(len);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len)
        {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j)
            {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void ComputeFrameDbfsInto(const float* frame, int n, const std::vector<double>& window,
                           std::vector<std::complex<double>>& complexScratch, std::vector<double>& outDbfs)
{
    if (static_cast<int>(complexScratch.size()) != n)
        complexScratch.resize(static_cast<size_t>(n));

    double windowSum = 0.0;
    for (int i = 0; i < n; ++i)
    {
        complexScratch[static_cast<size_t>(i)] = std::complex<double>(frame[i] * window[i], 0.0);
        windowSum += window[i];
    }
    FftForwardRadix2(complexScratch);

    const double reference = windowSum / 2.0;
    const int bins = n / 2 + 1;
    if (static_cast<int>(outDbfs.size()) != bins)
        outDbfs.resize(static_cast<size_t>(bins));
    for (int k = 0; k < bins; ++k)
    {
        double magnitude = std::abs(complexScratch[static_cast<size_t>(k)]);
        double normalized = (reference > 0.0) ? (magnitude / reference) : 0.0;
        outDbfs[static_cast<size_t>(k)] = 20.0 * std::log10(std::max(normalized, 1e-12));
    }
}

std::vector<double> ComputeFrameDbfs(const float* frame, int n, const std::vector<double>& window)
{
    std::vector<std::complex<double>> buf;
    std::vector<double> dbfs;
    ComputeFrameDbfsInto(frame, n, window, buf, dbfs);
    return dbfs;
}

void ComputeSpectrumAtTimeInto(const std::vector<float>& samples, int sampleRate, double timeSeconds,
                                std::vector<float>& frameScratch,
                                std::vector<std::complex<double>>& complexScratch, std::vector<double>& outDbfs)
{
    if (samples.empty() || sampleRate <= 0)
    {
        outDbfs.clear();
        return;
    }

    const int n = kFftSize;
    const int center = static_cast<int>(std::llround(timeSeconds * sampleRate));
    const int start = std::clamp(center - n / 2, 0, std::max(0, static_cast<int>(samples.size()) - n));

    if (static_cast<int>(frameScratch.size()) != n)
        frameScratch.assign(static_cast<size_t>(n), 0.0f);
    const int available = std::clamp(static_cast<int>(samples.size()) - start, 0, n);
    for (int i = 0; i < available; ++i)
        frameScratch[static_cast<size_t>(i)] = samples[static_cast<size_t>(start + i)];
    for (int i = available; i < n; ++i)
        frameScratch[static_cast<size_t>(i)] = 0.0f; // reused buffer -- tail may hold a previous call's data

    static const std::vector<double> window = MakeBlackmanWindow(n);
    ComputeFrameDbfsInto(frameScratch.data(), n, window, complexScratch, outDbfs);
}

std::vector<double> ComputeSpectrumAtTime(const std::vector<float>& samples, int sampleRate, double timeSeconds)
{
    std::vector<float> frameScratch;
    std::vector<std::complex<double>> complexScratch;
    std::vector<double> dbfs;
    ComputeSpectrumAtTimeInto(samples, sampleRate, timeSeconds, frameScratch, complexScratch, dbfs);
    return dbfs;
}

RGB DbfsToColor(double dbfs)
{
    struct Stop { double db; RGB color; };
    static const Stop stops[] = {
        { -115.0, {  0,   0,   0   } },
        {  -75.0, {  0,   0,   255 } },
        {  -50.0, {  255, 0,   0   } },
        {  -25.0, {  255, 255, 0   } },
        {    0.0, {  255, 255, 255 } },
    };
    constexpr int count = sizeof(stops) / sizeof(stops[0]);

    if (dbfs <= stops[0].db) return stops[0].color;
    if (dbfs >= stops[count - 1].db) return stops[count - 1].color;

    for (int i = 0; i < count - 1; ++i)
    {
        if (dbfs >= stops[i].db && dbfs <= stops[i + 1].db)
        {
            double span = stops[i + 1].db - stops[i].db;
            double t = (span != 0.0) ? (dbfs - stops[i].db) / span : 0.0;
            const RGB& a = stops[i].color;
            const RGB& b = stops[i + 1].color;
            return RGB{
                static_cast<unsigned char>(a.r + t * (b.r - a.r)),
                static_cast<unsigned char>(a.g + t * (b.g - a.g)),
                static_cast<unsigned char>(a.b + t * (b.b - a.b)),
            };
        }
    }
    return stops[count - 1].color;
}

std::vector<float> ResampleTo(const std::vector<float>& in, int srcRate, int targetRate,
                               const ProgressCallback& progress)
{
    if (in.empty() || srcRate == targetRate)
        return in;

    if (srcRate > targetRate)
    {
        double cutoffHz = 0.9 * (targetRate / 2.0);
        double fc = cutoffHz / srcRate;
        std::vector<double> fir = DesignLowpassFir(fc, 63);
        std::vector<float> filtered = ConvolveSameLength(in, fir, progress);
        return LinearResample(filtered, srcRate, targetRate);
    }

    // Upsampling: plain linear interpolation reconstructs the original
    // waveform imperfectly, leaving spectral images of the source spectrum
    // mirrored around multiples of srcRate that fall inside the new,
    // wider Nyquist band. Low-pass at the original Nyquist to remove them.
    std::vector<float> upsampled = LinearResample(in, srcRate, targetRate);
    double cutoffHz = 0.9 * (srcRate / 2.0);
    double fc = cutoffHz / targetRate;
    std::vector<double> fir = DesignLowpassFir(fc, 63);
    return ConvolveSameLength(upsampled, fir, progress);
}

void NormalizePeak(std::vector<float>& samples)
{
    if (samples.empty()) return;
    float maxAbs = 0.0f;
    for (float s : samples) maxAbs = std::max(maxAbs, std::abs(s));
    if (maxAbs > 0.0f)
        for (float& s : samples) s /= maxAbs;
}

} // namespace dsp
