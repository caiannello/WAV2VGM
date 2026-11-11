// OPM (YM2151) target: 8 channels, each playing one pure sine on its C2
// operator (algorithm 7, every other operator silent and keyed off).
//
// Pitch is a key code (octave + one of 12 notes) plus a 6-bit key
// fraction (1/64 semitone), looked up through the chip's own phase
// increment ROM (OpmPhaseIncRom.h) exactly as ym2151.c does. The note
// table alone tops out around 4.4 kHz, so higher frequencies also pick
// the operator's frequency multiplier (MUL 2-15), which changes pitch
// without retriggering anything.

#include "ChipCommon.h"
#include "OpmPhaseIncRom.h"

#include <algorithm>
#include <cmath>

namespace chips::detail
{

namespace
{

constexpr uint32_t kYm2151ClockHz = 3579545;
constexpr int kChannels = 8;
constexpr int kTableSize = 8 * 12 * 64; // octaves x notes x key fractions
constexpr int kC2Slot = 0x18;           // register offset of operator C2 (key-on bit 0x40)

// Per-voice output of a single full-level sine operator in libvgm's
// FM cores, as a fraction of 16-bit full scale (~-18 dBFS) -- the same
// ballpark dbopl produces for an OPL3 carrier.
constexpr double kFmFullScale = 0.125;

class OpmTarget;

class OpmWriter : public ChipWriter
{
public:
    explicit OpmWriter(RegisterSink& sink) : m_sink(sink) {}

    void Init() override
    {
        W(0x01, 0x00); // test register / LFO running
        W(0x0F, 0x00); // noise off
        W(0x18, 0x00); // LFO frequency
        W(0x19, 0x00); // AMD = 0
        W(0x19, 0x80); // PMD = 0
        W(0x1B, 0x00); // LFO waveform / CT outputs
        for (int ch = 0; ch < kChannels; ++ch)
        {
            W(0x08, static_cast<uint8_t>(ch), true); // all operators keyed off
            W(0x20 + ch, 0xC7);                      // L+R on, no feedback, algorithm 7
            W(0x38 + ch, 0x00);                      // no PMS/AMS
            for (int slot = 0; slot < 4; ++slot)
            {
                const int op = slot * 8 + ch;
                W(0x40 + op, 0x01); // DT1=0, MUL=1
                W(0x60 + op, 0x7F); // TL max (silent)
                W(0x80 + op, 0x1F); // KS=0, AR=31
                W(0xA0 + op, 0x00); // AMS-EN=0, D1R=0 (hold at peak)
                W(0xC0 + op, 0x00); // DT2=0, D2R=0
                W(0xE0 + op, 0x0F); // D1L=0, RR=15
            }
            m_keyed[ch] = false;
        }
    }

    void ApplyVoice(int voice, const VoiceFrame& frame) override
    {
        const int ch = voice;
        const int mul = static_cast<int>((frame.pitch >> 16) & 0x0F);
        const int kc = static_cast<int>((frame.pitch >> 8) & 0x7F);
        const int kf = static_cast<int>(frame.pitch & 0x3F);
        W(0x40 + kC2Slot + ch, static_cast<uint8_t>(mul));
        W(0x28 + ch, static_cast<uint8_t>(kc));
        W(0x30 + ch, static_cast<uint8_t>(kf << 2));
        W(0x60 + kC2Slot + ch, static_cast<uint8_t>(std::min<int>(frame.level, 0x7F)));
        // 0x08 is shared by every channel, so register-level dedup can't
        // tell whether *this* channel's key state changed -- track it here.
        if (frame.keyOn != m_keyed[ch])
        {
            W(0x08, static_cast<uint8_t>((frame.keyOn ? 0x40 : 0x00) | ch), true);
            m_keyed[ch] = frame.keyOn;
        }
    }

private:
    void W(int reg, uint8_t value, bool force = false) { m_sink.Write(static_cast<uint32_t>(reg), value, force); }

    RegisterSink& m_sink;
    bool m_keyed[kChannels] = {};
};

class OpmTarget : public ChipTarget
{
public:
    OpmTarget()
    {
        // Frequency of every (key code, key fraction) for MUL=1, in
        // ascending order: index = semitone * 64 + kf, semitone counted
        // from octave 0's C#.
        m_freqs.resize(kTableSize);
        for (int i = 0; i < kTableSize; ++i)
        {
            const int semitone = i / 64;
            const int oct = semitone / 12;
            const int romIndex = (semitone % 12) * 64 + (i % 64);
            m_freqs[static_cast<size_t>(i)] =
                kOpmPhaseIncRom[romIndex] * std::pow(2.0, oct - 2) * (kYm2151ClockHz / 64.0) / 1048576.0;
        }

        VoiceKind kind;
        kind.name = "FM";
        kind.minHz = m_freqs.front();
        kind.maxHz = std::min(m_freqs.back() * 15.0, 20000.0);
        kind.levelDb = UniformLevelTable(128, 0.75);
        kind.uniformDbPerStep = 0.75;
        m_kinds.push_back(kind);
        m_voiceKind.assign(kChannels, 0);

        m_model.fullScale = kFmFullScale;
        m_model.releaseDbPerSecond = 96.0 / 0.002;
    }

    ChipId Id() const override { return ChipId::Opm; }
    std::string Name() const override { return "OPM (YM2151)"; }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz) || hz > kind.maxHz)
            return false;
        for (int mul = 1; mul <= 15; ++mul)
        {
            const double base = hz / mul;
            if (base > m_freqs.back() * 1.0001)
                continue;
            const int idx = NearestIndex(std::max(base, m_freqs.front()));
            const int semitone = idx / 64;
            const int oct = semitone / 12;
            const int note12 = semitone % 12;
            const int noteCode = note12 + note12 / 3; // skips the chip's unused codes 3, 7, 11, 15
            const int kc = (oct << 4) | noteCode;
            pitch = (static_cast<uint32_t>(mul) << 16) | (static_cast<uint32_t>(kc) << 8) | static_cast<uint32_t>(idx % 64);
            return true;
        }
        return false;
    }

    double PitchToHz(int, uint32_t pitch) const override
    {
        const int mul = std::max(1, static_cast<int>((pitch >> 16) & 0x0F));
        const int kc = static_cast<int>((pitch >> 8) & 0x7F);
        const int kf = static_cast<int>(pitch & 0x3F);
        const int semitone = (kc - (kc >> 2)); // same note-code collapse ym2151.c uses
        const int idx = std::clamp(semitone * 64 + kf, 0, kTableSize - 1);
        return m_freqs[static_cast<size_t>(idx)] * mul;
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<OpmWriter>(sink);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_model);
    }

    VgmChipInfo VgmInfo() const override { return {0x30, kYm2151ClockHz}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override { return std::make_unique<YmPortEncoder>(0x54, 0x54); }

private:
    int NearestIndex(double hz) const
    {
        const auto it = std::lower_bound(m_freqs.begin(), m_freqs.end(), hz);
        if (it == m_freqs.begin())
            return 0;
        if (it == m_freqs.end())
            return kTableSize - 1;
        const int hi = static_cast<int>(it - m_freqs.begin());
        const int lo = hi - 1;
        return (std::log(hz / m_freqs[static_cast<size_t>(lo)]) <= std::log(m_freqs[static_cast<size_t>(hi)] / hz)) ? lo : hi;
    }

    std::vector<double> m_freqs;
    OscillatorModel m_model;
};

} // namespace

std::shared_ptr<const ChipTarget> MakeOpmTarget()
{
    return std::make_shared<OpmTarget>();
}

} // namespace chips::detail
