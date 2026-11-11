// MultiPCM (Sega 315-5560 / YMW258) target: 28 slots, each looping
// sample 0 -- one 256-step, 8-bit sine cycle -- from a tiny sample ROM
// shipped in the VGM itself (data block 0x89). Pitch: playback rate =
// output rate * (1 + F/1024) * 2^octave samples/second (multipcm.c),
// volume is the 7-bit total level in 0.375 dB steps, written with the
// "direct" bit so changes take effect immediately.
//
// Registers are reached indirectly (select slot, select register, write
// data), so logical writes are addressed (slot << 8) | register and the
// VGM encoder only re-sends the slot/register selects when they change.

#include "ChipCommon.h"

#include <algorithm>
#include <cmath>

namespace chips::detail
{

namespace
{

// VGM's MultiPCM clock field predates MAME's switch to a /224 divider and
// is still read as clock / 180 = output rate (libvgm rescales it by
// 224/180 on load), so that's the rate pitch is computed against.
constexpr uint32_t kClockHz = 8053975;
constexpr double kChipRateHz = kClockHz / 180.0;
constexpr int kSlots = 28;
constexpr int kCycleLength = 256;
constexpr uint32_t kRomSize = 0x1000;
constexpr uint32_t kSampleOffset = 0x100;
constexpr int kTlSteps = 128;

// Full-level slot in multipcm.c: 8-bit sample << 8, times 1/4 at TL 0
// -> ~0.25 of 16-bit full scale.
constexpr double kFullScale = 0.25;

uint32_t SlotReg(int slot, int reg) { return (static_cast<uint32_t>(slot) << 8) | static_cast<uint32_t>(reg); }

class MultiPcmWriter : public ChipWriter
{
public:
    explicit MultiPcmWriter(RegisterSink& sink) : m_sink(sink) {}

    void Init() override
    {
        for (int slot = 0; slot < kSlots; ++slot)
        {
            m_sink.Write(SlotReg(slot, 4), 0x00); // key off
            m_sink.Write(SlotReg(slot, 0), 0x00); // pan center, no effect send
            m_sink.Write(SlotReg(slot, 2), 0x00); // pitch low / sample number bit 8
            m_sink.Write(SlotReg(slot, 3), 0x10); // octave 0 (1x)
            m_sink.Write(SlotReg(slot, 1), 0x00); // sample 0 -- loads its header
            m_sink.Write(SlotReg(slot, 5), 0xFF); // TL 0x7F, direct
        }
    }

    void ApplyVoice(int voice, const VoiceFrame& frame) override
    {
        const int f = static_cast<int>(frame.pitch & 0x3FF);
        const int shift = static_cast<int>((frame.pitch >> 10) & 0xF); // signed 4-bit octave shift
        const int octaveReg = (shift + 1) & 0xF;                        // multipcm.c subtracts 1
        m_sink.Write(SlotReg(voice, 2), static_cast<uint8_t>((f & 0x3F) << 2));
        m_sink.Write(SlotReg(voice, 3), static_cast<uint8_t>((octaveReg << 4) | ((f >> 6) & 0x0F)));
        m_sink.Write(SlotReg(voice, 5), static_cast<uint8_t>((std::min<int>(frame.level, 0x7F) << 1) | 0x01));
        // A key-on write always restarts the sample, so this relies on
        // register dedup to only emit it when the state changes.
        m_sink.Write(SlotReg(voice, 4), frame.keyOn ? 0x80 : 0x00);
    }

private:
    RegisterSink& m_sink;
};

class MultiPcmEncoder : public VgmEncoder
{
public:
    void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) override
    {
        const int slot = static_cast<int>(addr >> 8);
        const int reg = static_cast<int>(addr & 0xFF);
        if (slot != m_slot)
        {
            const int slotValue = slot + slot / 7; // slot numbering skips every 8th value
            out.insert(out.end(), {0xB5, 0x01, static_cast<uint8_t>(slotValue)});
            m_slot = slot;
        }
        if (reg != m_reg)
        {
            out.insert(out.end(), {0xB5, 0x02, static_cast<uint8_t>(reg)});
            m_reg = reg;
        }
        out.insert(out.end(), {0xB5, 0x00, value});
    }

private:
    int m_slot = -1;
    int m_reg = -1;
};

class MultiPcmTarget : public ChipTarget
{
public:
    MultiPcmTarget()
    {
        VoiceKind kind;
        kind.name = "PCM";
        kind.minHz = 1.0;
        kind.maxHz = 16000.0;
        kind.levelDb = UniformLevelTable(kTlSteps, 0.375);
        kind.uniformDbPerStep = 0.375;
        m_kinds.push_back(kind);
        m_voiceKind.assign(kSlots, 0);

        const QuantizedSine sine = MakeQuantizedSine(kCycleLength, 127);
        m_sampleCodes = sine.codes;
        m_model.table = sine.normalized;
        m_model.interpolate = true;
        m_model.fullScale = kFullScale;
        m_model.releaseDbPerSecond = 96.0 / 0.013; // RR=14 with key-rate scaling off
    }

    ChipId Id() const override { return ChipId::MultiPcm; }
    std::string Name() const override { return "MultiPCM (YMW258)"; }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz) || hz > kind.maxHz)
            return false;
        const double x = hz * kCycleLength / kChipRateHz; // = 2^shift * (1 + F/1024)
        int shift = static_cast<int>(std::floor(std::log2(x)));
        int f = static_cast<int>(std::llround((x / std::pow(2.0, shift) - 1.0) * 1024.0));
        if (f >= 1024)
        {
            f = 0;
            ++shift;
        }
        if (shift < -8 || shift > 7)
            return false;
        pitch = static_cast<uint32_t>(std::clamp(f, 0, 1023)) | (static_cast<uint32_t>(shift & 0xF) << 10);
        return true;
    }

    double PitchToHz(int, uint32_t pitch) const override
    {
        const int raw = static_cast<int>((pitch >> 10) & 0xF);
        const int shift = raw >= 8 ? raw - 16 : raw;
        const int f = static_cast<int>(pitch & 0x3FF);
        return kChipRateHz * (1.0 + f / 1024.0) * std::pow(2.0, shift) / kCycleLength;
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<MultiPcmWriter>(sink);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_model);
    }

    VgmChipInfo VgmInfo() const override { return {0x88, kClockHz}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override { return std::make_unique<MultiPcmEncoder>(); }

    std::vector<VgmDataBlock> VgmDataBlocks() const override
    {
        VgmDataBlock rom;
        rom.type = 0x89;
        rom.memSize = kRomSize;
        rom.offset = 0;
        rom.data.assign(kRomSize, 0);

        // Sample 0's 12-byte header: start (8-bit format), loop start,
        // 0x10000 - end, LFO/vibrato, AR/D1R, DL/D2R, KRS/RR, tremolo.
        const uint16_t endField = static_cast<uint16_t>(0x10000 - kCycleLength);
        uint8_t* h = rom.data.data();
        h[0] = static_cast<uint8_t>((kSampleOffset >> 16) & 0x0F);
        h[1] = static_cast<uint8_t>((kSampleOffset >> 8) & 0xFF);
        h[2] = static_cast<uint8_t>(kSampleOffset & 0xFF);
        h[3] = 0x00;
        h[4] = 0x00;
        h[5] = static_cast<uint8_t>(endField >> 8);
        h[6] = static_cast<uint8_t>(endField & 0xFF);
        h[7] = 0x00; // no LFO/vibrato
        h[8] = 0xF0; // AR=15 (instant), D1R=0 (hold)
        h[9] = 0x00; // DL=0, D2R=0
        h[10] = 0xFE; // key-rate scaling off, RR=14
        h[11] = 0x00; // no tremolo

        for (int i = 0; i < kCycleLength; ++i)
            rom.data[kSampleOffset + static_cast<size_t>(i)] = static_cast<uint8_t>(static_cast<int8_t>(m_sampleCodes[static_cast<size_t>(i)]));
        return {rom};
    }

private:
    std::vector<int> m_sampleCodes;
    OscillatorModel m_model;
};

} // namespace

std::shared_ptr<const ChipTarget> MakeMultiPcmTarget()
{
    return std::make_shared<MultiPcmTarget>();
}

} // namespace chips::detail
