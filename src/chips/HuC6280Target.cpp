// HuC6280 (PC Engine PSG) target: 6 wavetable channels, each loaded with
// one 32-step, 5-bit sine cycle. Pitch f = clock / (32 * period), period
// 12 bits; volume is the channel's 5-bit level in 1.5 dB steps
// (c6280_mame.c), with global and per-channel balance left at maximum.
// Channels 5-6's noise mode and the channel 1-2 LFO stay off.

#include "ChipCommon.h"

#include <algorithm>
#include <cmath>

namespace chips::detail
{

namespace
{

constexpr uint32_t kClockHz = 3579545;
constexpr int kChannels = 6;
constexpr int kWaveLength = 32;
constexpr int kMinPeriod = 12;   // highest pitch used (~9.3 kHz); coarser steps beyond that
constexpr int kMaxPeriod = 4095;
constexpr int kLevelSteps = 31;  // channel volume 31..1 (0 is silence)
constexpr uint32_t kGlobal = 0xF0;

// One full-volume channel in c6280_mame.c: 65536/6/32 per step times a
// +-15 sample -> ~0.156 of 16-bit full scale.
constexpr double kFullScale = 0.156;

// Logical address: (channel << 4) | register, or kGlobal | register for
// the chip-wide registers (1, 8, 9).
uint32_t ChanReg(int ch, int reg) { return (static_cast<uint32_t>(ch) << 4) | static_cast<uint32_t>(reg); }

class HuCWriter : public ChipWriter
{
public:
    HuCWriter(RegisterSink& sink, const std::vector<int>& wave) : m_sink(sink), m_wave(wave) {}

    void Init() override
    {
        m_sink.Write(kGlobal | 0x01, 0xFF); // global balance: both sides max
        m_sink.Write(kGlobal | 0x08, 0x00); // LFO frequency
        m_sink.Write(kGlobal | 0x09, 0x00); // LFO off
        for (int ch = 0; ch < kChannels; ++ch)
        {
            // Dropping the DDA bit (1 -> 0) resets the waveform write
            // index; with the channel disabled, each data write then
            // advances it.
            m_sink.Write(ChanReg(ch, 0x04), 0x40, true);
            m_sink.Write(ChanReg(ch, 0x04), 0x00, true);
            for (int i = 0; i < kWaveLength; ++i)
                m_sink.Write(ChanReg(ch, 0x06), static_cast<uint8_t>(m_wave[static_cast<size_t>(i)]), true);
            m_sink.Write(ChanReg(ch, 0x05), 0xFF); // channel balance: both sides max
            if (ch >= 4)
                m_sink.Write(ChanReg(ch, 0x07), 0x00); // noise off
        }
    }

    void ApplyVoice(int voice, const VoiceFrame& frame) override
    {
        const int period = static_cast<int>(frame.pitch & 0xFFF);
        m_sink.Write(ChanReg(voice, 0x02), static_cast<uint8_t>(period & 0xFF));
        m_sink.Write(ChanReg(voice, 0x03), static_cast<uint8_t>((period >> 8) & 0x0F));
        const int volume = 31 - std::min<int>(frame.level, kLevelSteps - 1);
        m_sink.Write(ChanReg(voice, 0x04), frame.keyOn ? static_cast<uint8_t>(0x80 | volume) : 0x00);
    }

private:
    RegisterSink& m_sink;
    const std::vector<int>& m_wave;
};

// `0xB9 reg value`, with a channel-select (register 0) write inserted
// whenever a channel register targets a different channel than the last.
class HuCEncoder : public VgmEncoder
{
public:
    void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) override
    {
        const int reg = static_cast<int>(addr & 0x0F);
        const int ch = static_cast<int>((addr >> 4) & 0x0F);
        if (ch != 0x0F && ch != m_selected)
        {
            out.insert(out.end(), {0xB9, 0x00, static_cast<uint8_t>(ch)});
            m_selected = ch;
        }
        out.insert(out.end(), {0xB9, static_cast<uint8_t>(reg), value});
    }

private:
    int m_selected = -1;
};

class HuC6280Target : public ChipTarget
{
public:
    HuC6280Target()
    {
        VoiceKind kind;
        kind.name = "Wave";
        kind.minHz = kClockHz / (32.0 * kMaxPeriod);
        kind.maxHz = kClockHz / (32.0 * kMinPeriod);
        kind.levelDb = UniformLevelTable(kLevelSteps, 1.5);
        kind.uniformDbPerStep = 1.5;
        m_kinds.push_back(kind);
        m_voiceKind.assign(kChannels, 0);

        // 5-bit unsigned samples centered on 16 (the chip outputs data - 16).
        const QuantizedSine sine = MakeQuantizedSine(kWaveLength, 15);
        for (int code : sine.codes)
            m_wave.push_back(16 + code);
        m_model.table = sine.normalized;
        m_model.interpolate = false;
        m_model.fullScale = kFullScale;
        m_model.releaseDbPerSecond = 0.0;
    }

    ChipId Id() const override { return ChipId::HuC6280; }
    std::string Name() const override { return "HuC6280"; }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz * 0.9999) || hz > kind.maxHz * 1.0001)
            return false;
        const int period = std::clamp(static_cast<int>(std::llround(kClockHz / (32.0 * hz))), kMinPeriod, kMaxPeriod);
        pitch = static_cast<uint32_t>(period);
        return true;
    }

    double PitchToHz(int, uint32_t pitch) const override
    {
        const int period = std::max(1, static_cast<int>(pitch & 0xFFF));
        return kClockHz / (32.0 * period);
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<HuCWriter>(sink, m_wave);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_model);
    }

    VgmChipInfo VgmInfo() const override { return {0xA4, kClockHz}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override { return std::make_unique<HuCEncoder>(); }

private:
    std::vector<int> m_wave;
    OscillatorModel m_model;
};

} // namespace

std::shared_ptr<const ChipTarget> MakeHuC6280Target()
{
    return std::make_shared<HuC6280Target>();
}

} // namespace chips::detail
