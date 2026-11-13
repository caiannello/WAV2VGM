// VSU (Virtual Boy) target: channels 1-5 each play one 32-step, 6-bit
// sine cycle from waveform RAM (channel 6 is noise-only and stays off;
// channel 5's sweep/modulation stays disabled). Pitch f = 5 MHz /
// (32 * (2048 - F)), F 11 bits. Volume is the product of a 4-bit
// envelope value (held constant, envelope stepping disabled) and a 4-bit
// stereo level, giving the linear amplitude ((env * level) >> 3) + 1 of
// vsu.c -- so the level table is built from every distinct amplitude
// that product can reach.
//
// Waveform RAM can only be written while every channel is disabled,
// which is why the waveforms are loaded once, up front, before any
// channel is keyed on.

#include "ChipCommon.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace chips::detail
{

namespace
{

constexpr uint32_t kClockHz = 5000000;
constexpr int kChannels = 5;
constexpr int kWaveLength = 32;
constexpr int kMinPeriod = 16;  // 2048 - F; highest pitch used ~9.8 kHz
constexpr int kMaxPeriod = 2048;

// Full-amplitude channel in vsu.c: (sample - 0x20) * 29, then << 3 ->
// ~0.22 of 16-bit full scale for a +-31 sine.
constexpr double kFullScale = 0.2;

// VGM offset = byte address >> 2.
uint32_t WaveAddr(int table, int index) { return static_cast<uint32_t>(table * 0x20 + index); }
uint32_t ChanReg(int ch, int reg) { return static_cast<uint32_t>(0x100 + ch * 0x10 + reg); }
constexpr uint32_t kStopAll = 0x580 >> 2;

struct VolumeSetting { uint8_t envelope; uint8_t level; };

class VsuWriter : public ChipWriter
{
public:
    VsuWriter(RegisterSink& sink, const std::vector<int>& wave, const std::vector<VolumeSetting>& volumes)
        : m_sink(sink), m_wave(wave), m_volumes(volumes)
    {
    }

    void Init() override
    {
        m_sink.Write(kStopAll, 0x01, true);
        for (int table = 0; table < kChannels; ++table)
            for (int i = 0; i < kWaveLength; ++i)
                m_sink.Write(WaveAddr(table, i), static_cast<uint8_t>(m_wave[static_cast<size_t>(i)]), true);
        for (int ch = 0; ch < kChannels; ++ch)
        {
            m_sink.Write(ChanReg(ch, 0x00), 0x00);              // disabled, no auto-off interval
            m_sink.Write(ChanReg(ch, 0x01), 0x00);              // L/R level 0
            m_sink.Write(ChanReg(ch, 0x04), 0x00);              // envelope value 0
            m_sink.Write(ChanReg(ch, 0x05), 0x00);              // envelope stepping, sweep, modulation off
            m_sink.Write(ChanReg(ch, 0x06), static_cast<uint8_t>(ch)); // waveform table = own channel's
        }
        m_sink.Write(ChanReg(4, 0x07), 0x00); // channel 5 sweep/modulation control
    }

    void ApplyVoice(int voice, const VoiceFrame& frame) override
    {
        const int f = static_cast<int>(frame.pitch & 0x7FF);
        const VolumeSetting& vol = m_volumes[std::min<size_t>(frame.level, m_volumes.size() - 1)];
        m_sink.Write(ChanReg(voice, 0x02), static_cast<uint8_t>(f & 0xFF));
        m_sink.Write(ChanReg(voice, 0x03), static_cast<uint8_t>((f >> 8) & 0x07));
        m_sink.Write(ChanReg(voice, 0x04), static_cast<uint8_t>(vol.envelope << 4));
        m_sink.Write(ChanReg(voice, 0x01), static_cast<uint8_t>((vol.level << 4) | vol.level));
        // Writing the enable bit restarts the waveform, so it's only ever
        // written when the value actually changes (register dedup).
        m_sink.Write(ChanReg(voice, 0x00), frame.keyOn ? 0x80 : 0x00);
    }

private:
    RegisterSink& m_sink;
    const std::vector<int>& m_wave;
    const std::vector<VolumeSetting>& m_volumes;
};

class VsuEncoder : public VgmEncoder
{
public:
    void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) override
    {
        out.insert(out.end(), {0xC7, static_cast<uint8_t>((addr >> 8) & 0x7F), static_cast<uint8_t>(addr & 0xFF), value});
    }
};

class VsuTarget : public ChipTarget
{
public:
    VsuTarget()
    {
        // Every reachable linear amplitude, loudest first, with the
        // (envelope, level) pair that produces it.
        std::map<int, VolumeSetting, std::greater<int>> byAmplitude;
        for (int env = 15; env >= 1; --env)
            for (int level = 15; level >= 1; --level)
            {
                const int amp = ((env * level) >> 3) + 1;
                byAmplitude.emplace(amp, VolumeSetting{static_cast<uint8_t>(env), static_cast<uint8_t>(level)});
            }
        std::vector<double> amps;
        for (const auto& [amp, setting] : byAmplitude)
        {
            amps.push_back(static_cast<double>(amp));
            m_volumes.push_back(setting);
        }

        VoiceKind kind;
        kind.name = "Wave";
        kind.minHz = kClockHz / (32.0 * kMaxPeriod);
        kind.maxHz = kClockHz / (32.0 * kMinPeriod);
        kind.levelDb = LinearAmplitudeLevelTable(amps);
        m_kinds.push_back(kind);
        m_voiceKind.assign(kChannels, 0);

        // 6-bit unsigned samples centered on 0x20 (vsu.c outputs data - 0x20).
        const QuantizedSine sine = MakeQuantizedSine(kWaveLength, 31);
        for (int code : sine.codes)
            m_wave.push_back(32 + code);
        m_model.table = sine.normalized;
        m_model.interpolate = false;
        m_model.fullScale = kFullScale;
        m_model.releaseDbPerSecond = 0.0;
    }

    ChipId Id() const override { return ChipId::Vsu; }
    std::string Name() const override { return "VSU (Virtual Boy)"; }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz * 0.9999) || hz > kind.maxHz * 1.0001)
            return false;
        const int period = std::clamp(static_cast<int>(std::llround(kClockHz / (32.0 * hz))), kMinPeriod, kMaxPeriod);
        pitch = static_cast<uint32_t>(2048 - period);
        return true;
    }

    double PitchToHz(int, uint32_t pitch) const override
    {
        const int period = 2048 - static_cast<int>(pitch & 0x7FF);
        return kClockHz / (32.0 * period);
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<VsuWriter>(sink, m_wave, m_volumes);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_model);
    }

    VgmChipInfo VgmInfo() const override { return {0xC4, kClockHz}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override { return std::make_unique<VsuEncoder>(); }

private:
    std::vector<int> m_wave;
    std::vector<VolumeSetting> m_volumes;
    OscillatorModel m_model;
};

} // namespace

std::shared_ptr<const ChipTarget> MakeVsuTarget()
{
    return std::make_shared<VsuTarget>();
}

} // namespace chips::detail
