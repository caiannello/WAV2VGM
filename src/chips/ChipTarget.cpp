#include "ChipTarget.h"
#include "ChipCommon.h"

#include <algorithm>
#include <cmath>

namespace chips
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

int ChipTarget::FirstVoiceOfKind(int kind) const
{
    for (int v = 0; v < VoiceCount(); ++v)
        if (KindIndexOf(v) == kind)
            return v;
    return -1;
}

uint16_t ChipTarget::QuantizeLevel(int voice, double dbRelativeToMax) const
{
    const VoiceKind& kind = KindOf(voice);
    const int last = static_cast<int>(kind.levelDb.size()) - 1;
    if (last <= 0)
        return 0;
    if (kind.uniformDbPerStep > 0.0)
    {
        const long long step = std::llround(-dbRelativeToMax / kind.uniformDbPerStep);
        return static_cast<uint16_t>(std::clamp<long long>(step, 0, last));
    }
    if (dbRelativeToMax >= kind.levelDb.front())
        return 0;
    if (dbRelativeToMax <= kind.levelDb.back())
        return static_cast<uint16_t>(last);
    // levelDb is strictly decreasing: first index at or below the target,
    // then whichever neighbor is closer (ties go to the quieter step,
    // matching llround's half-away-from-zero on the uniform path).
    const auto it = std::lower_bound(kind.levelDb.begin(), kind.levelDb.end(), dbRelativeToMax,
                                     [](double tableDb, double target) { return tableDb > target; });
    const int i = static_cast<int>(it - kind.levelDb.begin());
    const double above = kind.levelDb[static_cast<size_t>(i - 1)] - dbRelativeToMax;
    const double below = dbRelativeToMax - kind.levelDb[static_cast<size_t>(i)];
    return static_cast<uint16_t>(above < below ? i - 1 : i);
}

double ChipTarget::LevelToDb(int voice, uint16_t level) const
{
    const VoiceKind& kind = KindOf(voice);
    if (kind.levelDb.empty())
        return 0.0;
    return kind.levelDb[std::min<size_t>(level, kind.levelDb.size() - 1)];
}

VoiceFrame ChipTarget::SilentFrame(int voice) const
{
    // Spread across voices rather than one shared frequency: every idle
    // voice parks here, and identical idle voices would leak coherently.
    const VoiceKind& kind = KindOf(voice);
    const double hz = std::clamp(220.0 + 37.0 * voice, kind.minHz, kind.maxHz);
    VoiceFrame frame;
    QuantizePitch(voice, hz, frame.pitch);
    frame.level = static_cast<uint16_t>(kind.levelDb.empty() ? 0 : kind.levelDb.size() - 1);
    frame.keyOn = false;
    return frame;
}

std::vector<double> UniformLevelTable(int steps, double dbPerStep)
{
    std::vector<double> table(static_cast<size_t>(std::max(1, steps)));
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = -static_cast<double>(i) * dbPerStep;
    return table;
}

std::vector<double> LinearAmplitudeLevelTable(const std::vector<double>& amplitudesLoudestFirst)
{
    std::vector<double> table;
    if (amplitudesLoudestFirst.empty() || amplitudesLoudestFirst.front() <= 0.0)
        return table;
    const double ref = amplitudesLoudestFirst.front();
    for (double a : amplitudesLoudestFirst)
        table.push_back(20.0 * std::log10(std::max(a, 1e-9) / ref));
    return table;
}

std::shared_ptr<const ChipTarget> CreateTarget(ChipId id)
{
    switch (id)
    {
    case ChipId::Opl3:
    case ChipId::Opl2:
    case ChipId::Opl4Fm:
    case ChipId::Opl4Pcm:
    case ChipId::Opl4Both:
        return detail::MakeOplTarget(id);
    case ChipId::Opm:
        return detail::MakeOpmTarget();
    case ChipId::Opn:
    case ChipId::Opna:
    case ChipId::Opn2:
        return detail::MakeOpnTarget(id);
    case ChipId::HuC6280:
        return detail::MakeHuC6280Target();
    case ChipId::Vsu:
        return detail::MakeVsuTarget();
    case ChipId::MultiPcm:
        return detail::MakeMultiPcmTarget();
    }
    return nullptr;
}

namespace detail
{

std::vector<float> RenderOscillatorVoice(const ChipTarget& target, int voice, const std::vector<VoiceFrame>& frames,
                                         double frameRateHz, int sampleRate, const OscillatorModel& model)
{
    std::vector<float> out;
    if (frames.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
        return out;

    const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / frameRateHz)));
    out.assign(static_cast<size_t>(samplesPerFrame) * frames.size(), 0.0f);

    const size_t tableLen = model.table.size();
    const double releaseMul = (model.releaseDbPerSecond > 0.0)
        ? std::pow(10.0, -model.releaseDbPerSecond / 20.0 / sampleRate)
        : 0.0;

    double phase = 0.0;     // cycles, [0, 1)
    double envelope = 0.0;  // 1 while keyed on, decaying after key-off
    double heldAmp = 0.0;   // last keyed-on amplitude, held through the release
    bool keyed = false;
    size_t writePos = 0;

    for (const VoiceFrame& frame : frames)
    {
        const double hz = target.PitchToHz(voice, frame.pitch);
        const double inc = hz / sampleRate;
        if (frame.keyOn)
        {
            if (!keyed)
                phase = 0.0;
            keyed = true;
            envelope = 1.0;
            heldAmp = model.fullScale * std::pow(10.0, target.LevelToDb(voice, frame.level) / 20.0);
        }
        else
        {
            keyed = false;
        }

        for (int i = 0; i < samplesPerFrame; ++i)
        {
            if (!keyed)
            {
                envelope *= releaseMul;
                if (envelope < 1e-6)
                    envelope = 0.0;
            }
            float sample = 0.0f;
            if (envelope > 0.0)
            {
                double wave;
                if (tableLen == 0)
                {
                    wave = std::sin(2.0 * kPi * phase);
                }
                else
                {
                    const double pos = phase * static_cast<double>(tableLen);
                    const size_t idx = static_cast<size_t>(pos) % tableLen;
                    if (model.interpolate)
                    {
                        const double frac = pos - std::floor(pos);
                        const double a = model.table[idx];
                        const double b = model.table[(idx + 1) % tableLen];
                        wave = a + (b - a) * frac;
                    }
                    else
                    {
                        wave = model.table[idx];
                    }
                }
                sample = static_cast<float>(heldAmp * envelope * wave);
            }
            out[writePos++] = sample;
            phase += inc;
            phase -= std::floor(phase);
        }
    }
    return out;
}

QuantizedSine MakeQuantizedSine(int length, int amplitudeCode)
{
    QuantizedSine s;
    s.codes.resize(static_cast<size_t>(length));
    s.normalized.resize(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i)
    {
        const int code = static_cast<int>(std::lround(amplitudeCode * std::sin(2.0 * kPi * i / length)));
        s.codes[static_cast<size_t>(i)] = code;
        s.normalized[static_cast<size_t>(i)] = static_cast<float>(code) / static_cast<float>(amplitudeCode);
    }
    return s;
}

void YmPortEncoder::Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out)
{
    out.push_back(m_commands[(addr >> 8) & 1]);
    out.push_back(static_cast<uint8_t>(addr & 0xFF));
    out.push_back(value);
}

} // namespace detail

} // namespace chips
