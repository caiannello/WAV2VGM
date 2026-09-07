#include "../Dsp.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

bool CheckPeakBin()
{
    const int n = dsp::kFftSize;
    const int targetBin = 10;
    const double sampleRate = 44100.0;
    const double freq = targetBin * sampleRate / n; // lands exactly on bin 10

    std::vector<float> frame(n);
    for (int i = 0; i < n; ++i)
        frame[i] = static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * freq * i / sampleRate));

    std::vector<double> window = dsp::MakeBlackmanWindow(n);
    std::vector<double> dbfs = dsp::ComputeFrameDbfs(frame.data(), n, window);

    int peakBin = 0;
    double peakDb = dbfs[0];
    for (size_t k = 1; k < dbfs.size(); ++k)
    {
        if (dbfs[k] > peakDb) { peakDb = dbfs[k]; peakBin = static_cast<int>(k); }
    }

    bool ok = true;
    if (peakBin != targetBin)
    {
        std::printf("FAIL: peak bin = %d, expected %d\n", peakBin, targetBin);
        ok = false;
    }
    if (std::abs(peakDb - 0.0) > 0.5)
    {
        std::printf("FAIL: peak dBFS = %.3f, expected within +-0.5 of 0.0\n", peakDb);
        ok = false;
    }
    if (dbfs[500] > -40.0)
    {
        std::printf("FAIL: bin 500 dBFS = %.3f, expected < -40.0\n", dbfs[500]);
        ok = false;
    }

    std::printf("peak bin=%d dBFS=%.3f bin500 dBFS=%.3f\n", peakBin, peakDb, dbfs[500]);
    return ok;
}

bool CheckColorGradient()
{
    struct Case { double db; dsp::RGB expected; };
    const Case cases[] = {
        { -200.0, {0, 0, 0} },
        { -115.0, {0, 0, 0} },
        { -75.0, {0, 0, 255} },
        { -50.0, {255, 0, 0} },
        { -25.0, {255, 255, 0} },
        { 0.0, {255, 255, 255} },
        { 10.0, {255, 255, 255} },
    };
    bool ok = true;
    for (const auto& c : cases)
    {
        dsp::RGB got = dsp::DbfsToColor(c.db);
        if (got.r != c.expected.r || got.g != c.expected.g || got.b != c.expected.b)
        {
            std::printf("FAIL: DbfsToColor(%.1f) = (%d,%d,%d), expected (%d,%d,%d)\n",
                c.db, got.r, got.g, got.b, c.expected.r, c.expected.g, c.expected.b);
            ok = false;
        }
    }
    return ok;
}

} // namespace

int main()
{
    bool ok = true;
    ok &= CheckPeakBin();
    ok &= CheckColorGradient();
    if (ok) std::printf("dsp_selftest: PASS\n");
    return ok ? 0 : 1;
}
