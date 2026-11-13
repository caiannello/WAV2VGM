// OPN-family targets: OPN (YM2203), OPNA (YM2608) and OPN2 (YM2612).
//
// Every FM channel uses algorithm 7 (all four operators summed straight
// to the output) with only operator 4 audible, giving one sine per
// channel -- except channel 3, which runs in its "special" mode, where
// each of its four operators gets its own frequency registers and so
// becomes an independent sine voice of its own. That gives 6 voices on
// OPN (3 channels) and 9 on OPNA/OPN2 (6 channels). The SSG (square
// wave) part of OPN/OPNA, OPNA's rhythm/ADPCM and OPN2's DAC are left
// silent.
//
// Pitch: f = fnum * (clock / prescaler) * 2^(block-1) / 2^20 (fmopn.c),
// plus the operator's frequency multiplier above the fnum/block ceiling
// (~6.6-6.9 kHz).

#include "ChipCommon.h"

#include <algorithm>
#include <cmath>

namespace chips::detail
{

namespace
{

constexpr double kFmFullScale = 0.125; // see OpmTarget.cpp

// Operator register offsets within a channel's block, in the chip's own
// (S1, S3, S2, S4) register order, indexed here by operator number.
constexpr int kSlotRegOffset[4] = {0x00, 0x08, 0x04, 0x0C}; // S1, S2, S3, S4
constexpr int kSlotKeyBit[4] = {0x10, 0x20, 0x40, 0x80};
// Channel 3 special mode: per-operator frequency registers (low byte;
// high byte/block register is +4). Operator 4 uses the channel's own.
constexpr int kSpecialFnumLow[4] = {0xA9, 0xAA, 0xA8, 0xA2};

struct OpnVoice
{
    int channel; // 0-5 (3-5 on port 1)
    int slot;    // operator 0-3 (S1-S4)
};

struct OpnVariant
{
    ChipId id;
    const char* name;
    uint32_t clockHz;
    double prescaler; // FM sample rate = clock / prescaler
    int channels;
    uint32_t clockHeaderOffset;
    uint8_t port0Command;
    uint8_t port1Command;
};

constexpr OpnVariant kOpn = {ChipId::Opn, "OPN (YM2203)", 3993600, 72.0, 3, 0x44, 0x55, 0x55};
constexpr OpnVariant kOpna = {ChipId::Opna, "OPNA (YM2608)", 7987200, 144.0, 6, 0x48, 0x56, 0x57};
constexpr OpnVariant kOpn2 = {ChipId::Opn2, "OPN2 (YM2612)", 7670453, 144.0, 6, 0x2C, 0x52, 0x53};

uint32_t PackPitch(int mul, int block, int fnum)
{
    return (static_cast<uint32_t>(mul) << 16) | (static_cast<uint32_t>(block) << 11) | static_cast<uint32_t>(fnum);
}

class OpnTarget;

class OpnWriter : public ChipWriter
{
public:
    OpnWriter(const OpnTarget& target, RegisterSink& sink);
    void Init() override;
    void ApplyVoice(int voice, const VoiceFrame& frame) override;

private:
    void W(int port, int reg, uint8_t value, bool force = false)
    {
        m_sink.Write((static_cast<uint32_t>(port) << 8) | static_cast<uint32_t>(reg), value, force);
    }
    static int KeyChannelCode(int channel) { return channel < 3 ? channel : channel + 1; }

    const OpnTarget& m_target;
    RegisterSink& m_sink;
    uint8_t m_keyMask[6] = {};
    std::vector<int> m_lastFreq; // per voice: (hi << 8) | lo last written, -1 = never
};

class OpnTarget : public ChipTarget
{
public:
    explicit OpnTarget(const OpnVariant& variant) : m_variant(variant)
    {
        m_fs = variant.clockHz / variant.prescaler;
        m_baseMaxHz = 2047.0 * m_fs * 64.0 / 1048576.0;

        for (int ch = 0; ch < variant.channels; ++ch)
        {
            if (ch == 2)
                for (int slot = 0; slot < 4; ++slot)
                    m_voices.push_back(OpnVoice{ch, slot});
            else
                m_voices.push_back(OpnVoice{ch, 3});
        }

        VoiceKind kind;
        kind.name = "FM";
        kind.minHz = 10.0;
        kind.maxHz = std::min(m_baseMaxHz * 15.0, 20000.0);
        kind.levelDb = UniformLevelTable(128, 0.75);
        kind.uniformDbPerStep = 0.75;
        m_kinds.push_back(kind);
        m_voiceKind.assign(m_voices.size(), 0);

        m_model.fullScale = kFmFullScale;
        m_model.releaseDbPerSecond = 96.0 / 0.002;
    }

    ChipId Id() const override { return m_variant.id; }
    std::string Name() const override { return m_variant.name; }
    const OpnVariant& Variant() const { return m_variant; }
    const OpnVoice& Voice(int voice) const { return m_voices[static_cast<size_t>(voice)]; }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz) || hz > kind.maxHz)
            return false;
        for (int mul = 1; mul <= 15; ++mul)
        {
            const double base = hz / mul;
            if (base > m_baseMaxHz * 1.0001)
                continue;
            for (int block = 0; block <= 7; ++block)
            {
                const double fnum = base * 1048576.0 / (m_fs * std::pow(2.0, block - 1));
                if (fnum <= 2047.5)
                {
                    const int fnumInt = std::clamp(static_cast<int>(std::llround(fnum)), 0, 2047);
                    pitch = PackPitch(mul, block, fnumInt);
                    return true;
                }
            }
        }
        return false;
    }

    double PitchToHz(int, uint32_t pitch) const override
    {
        const int mul = std::max(1, static_cast<int>((pitch >> 16) & 0x0F));
        const int block = static_cast<int>((pitch >> 11) & 7);
        const int fnum = static_cast<int>(pitch & 0x7FF);
        return fnum * m_fs * std::pow(2.0, block - 1) / 1048576.0 * mul;
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<OpnWriter>(*this, sink);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_model);
    }

    VgmChipInfo VgmInfo() const override { return {m_variant.clockHeaderOffset, m_variant.clockHz}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override
    {
        return std::make_unique<YmPortEncoder>(m_variant.port0Command, m_variant.port1Command);
    }

private:
    OpnVariant m_variant;
    double m_fs = 0.0;
    double m_baseMaxHz = 0.0;
    std::vector<OpnVoice> m_voices;
    OscillatorModel m_model;
};

OpnWriter::OpnWriter(const OpnTarget& target, RegisterSink& sink)
    : m_target(target), m_sink(sink), m_lastFreq(static_cast<size_t>(target.VoiceCount()), -1)
{
}

void OpnWriter::Init()
{
    const OpnVariant& v = m_target.Variant();
    const bool isOpn2 = v.id == ChipId::Opn2;
    const bool hasSsg = !isOpn2;
    const bool hasPan = v.id != ChipId::Opn;

    if (!isOpn2)
        W(0, 0x2D, 0x00, true); // prescaler: FM = clock / 6 (the power-on default, made explicit)
    if (hasSsg)
    {
        W(0, 0x07, 0x3F); // SSG: all tone and noise outputs off
        W(0, 0x08, 0x00);
        W(0, 0x09, 0x00);
        W(0, 0x0A, 0x00);
    }
    if (v.id == ChipId::Opna)
        W(0, 0x29, 0x80); // enable FM channels 4-6
    if (isOpn2)
        W(0, 0x2B, 0x00); // DAC off -- channel 6 back to FM
    if (v.id != ChipId::Opn)
        W(0, 0x22, 0x00); // LFO off
    W(0, 0x27, 0x40);     // channel 3 special (per-operator frequency) mode

    for (int ch = 0; ch < v.channels; ++ch)
    {
        const int port = ch / 3;
        const int cc = ch % 3;
        W(0, 0x28, static_cast<uint8_t>(KeyChannelCode(ch)), true); // all operators keyed off
        m_keyMask[ch] = 0;
        W(port, 0xB0 + cc, 0x07); // feedback 0, algorithm 7
        if (hasPan)
            W(port, 0xB4 + cc, 0xC0); // L+R on, no AMS/PMS
        for (int slot = 0; slot < 4; ++slot)
        {
            const int r = kSlotRegOffset[slot] + cc;
            W(port, 0x30 + r, 0x01); // DT=0, MUL=1
            W(port, 0x40 + r, 0x7F); // TL max (silent)
            W(port, 0x50 + r, 0x1F); // KS=0, AR=31
            W(port, 0x60 + r, 0x00); // AM off, D1R=0 (hold at peak)
            W(port, 0x70 + r, 0x00); // D2R=0
            W(port, 0x80 + r, 0x0F); // SL=0, RR=15
            W(port, 0x90 + r, 0x00); // SSG-EG off
        }
    }
}

void OpnWriter::ApplyVoice(int voice, const VoiceFrame& frame)
{
    const OpnVoice& ov = m_target.Voice(voice);
    const int port = ov.channel / 3;
    const int cc = ov.channel % 3;
    const int r = kSlotRegOffset[ov.slot] + cc;
    const int mul = static_cast<int>((frame.pitch >> 16) & 0x0F);
    const int block = static_cast<int>((frame.pitch >> 11) & 7);
    const int fnum = static_cast<int>(frame.pitch & 0x7FF);

    W(port, 0x30 + r, static_cast<uint8_t>(mul));
    W(port, 0x40 + r, static_cast<uint8_t>(std::min<int>(frame.level, 0x7F)));

    // The block/fnum-high register is only a latch until the matching
    // low-byte write, and that latch is shared, so the pair always goes
    // out together (high first) whenever either half changes.
    const uint8_t hi = static_cast<uint8_t>((block << 3) | ((fnum >> 8) & 0x07));
    const uint8_t lo = static_cast<uint8_t>(fnum & 0xFF);
    const int packed = (hi << 8) | lo;
    if (m_lastFreq[static_cast<size_t>(voice)] != packed)
    {
        const bool special = ov.channel == 2 && ov.slot != 3;
        const int loReg = special ? kSpecialFnumLow[ov.slot] : 0xA0 + cc;
        W(port, loReg + 4, hi, true);
        W(port, loReg, lo, true);
        m_lastFreq[static_cast<size_t>(voice)] = packed;
    }

    uint8_t mask = m_keyMask[ov.channel];
    if (frame.keyOn)
        mask = static_cast<uint8_t>(mask | kSlotKeyBit[ov.slot]);
    else
        mask = static_cast<uint8_t>(mask & ~kSlotKeyBit[ov.slot]);
    if (mask != m_keyMask[ov.channel])
    {
        W(0, 0x28, static_cast<uint8_t>(mask | KeyChannelCode(ov.channel)), true);
        m_keyMask[ov.channel] = mask;
    }
}

} // namespace

std::shared_ptr<const ChipTarget> MakeOpnTarget(ChipId id)
{
    switch (id)
    {
    case ChipId::Opn: return std::make_shared<OpnTarget>(kOpn);
    case ChipId::Opna: return std::make_shared<OpnTarget>(kOpna);
    default: return std::make_shared<OpnTarget>(kOpn2);
    }
}

} // namespace chips::detail
