// OPL-family targets: OPL3 (YMF262), OPL2 (YM3812), and OPL4 (YMF278B)
// in its FM-only, PCM-only (24 wavetable slots playing a looped sine
// sample from sample RAM), and combined modes.
//
// FM voices are rendered through the same vendored dbopl emulator the
// original OPL3 mode always used, driven by exactly the same register
// sequences (opl::SetupChannel/ApplyFrame with the pure-sine setup), so
// the OPL3 target reproduces the legacy oplfit/vgm::WriteVgmFile output
// byte for byte.

#include "ChipCommon.h"

#include "../OplChip.h"
#include "../OplPatch.h"

#include <algorithm>
#include <cmath>

namespace chips::detail
{

namespace
{

// OPL4's FM part runs at 33.8688 MHz / 684 (~49516 Hz) instead of OPL3's
// 14.31818 MHz / 288 (~49716 Hz) -- same register layout, pitch ~0.4%
// lower for the same fnum/block (libvgm ymf278b.c feeds its OPL3 core
// clock * 8 / 19). The PCM part runs at 33.8688 MHz / 768 = 44100 Hz.
constexpr double kOpl4FmBaseRateHz = 33868800.0 / 684.0;
constexpr uint32_t kYmf262ClockHz = 14318180;
constexpr uint32_t kYm3812ClockHz = 3579545;
constexpr uint32_t kYmf278bClockHz = 33868800;

// OPL4 wavetable: one 256-sample, 16-bit sine cycle in sample RAM, as
// wave number 384 (the first RAM-defined header once the header table
// is moved to RAM via register 2), header table at the start of RAM and
// sample data right after the 128 RAM headers.
constexpr int kPcmCycleLength = 256;
constexpr int kOpl4Wave = 384;
constexpr uint8_t kOpl4WaveTableHeaderReg = 0x10;  // register 2: wave table header at 4 * 512 KiB = 0x200000
constexpr uint32_t kOpl4RomSize = 0x200000;        // RAM begins after the (2 MiB) sample ROM address range
constexpr uint32_t kOpl4RamSize = 0x80000;         // MoonSound-style 512 KiB SRAM
constexpr uint32_t kOpl4SampleRamOffset = 0x600;   // right after 128 x 12-byte RAM headers
constexpr int kOpl4PcmSlots = 24;
constexpr int kOpl4TlSteps = 127;                  // TL 0x00-0x7E; 0x7F is hard silence

// Rough per-voice output levels of libvgm's cores at full volume, as a
// fraction of 16-bit full scale (see OscillatorModel::fullScale): a
// full-scale 16-bit sample at TL 0 comes out at ~-15 dB in ymf278b.c.
constexpr double kOpl4PcmFullScale = 0.177;

enum class Variant { Opl3, Opl2, Opl4Fm, Opl4Pcm, Opl4Both };

opl::ChannelStaticSetup PureSineSetup()
{
    // Same recipe as oplfit's own PureSineSetup: modulator silenced and
    // disconnected (additive mode), sine carrier.
    opl::ChannelStaticSetup setup;
    setup.modMultipleIndex = 1;
    setup.modTotalLevel = 63;
    setup.modWaveform = 0;
    setup.carMultipleIndex = 1;
    setup.carWaveform = 0;
    setup.feedback = 0;
    setup.algorithmAdditive = true;
    return setup;
}

// OPL4 PCM per-TL output level, matching ymf278b.c's vol_tab (0x40
// steps per 6 dB, linear in between -- not a true dB curve).
double Opl4TlAmplitude(int tl)
{
    const int i = tl * 4;
    return (128.0 - (i & 0x3F)) / 128.0 / static_cast<double>(1 << (i >> 6));
}

class OplTarget;

class OplWriter : public ChipWriter
{
public:
    OplWriter(const OplTarget& target, RegisterSink& sink) : m_target(target), m_sink(sink) {}
    void Init() override;
    void ApplyVoice(int voice, const VoiceFrame& frame) override;

private:
    void WritePcm(int slot, int group, uint8_t value) { m_sink.Write(0x200u | static_cast<uint32_t>(0x08 + group * 24 + slot), value); }

    const OplTarget& m_target;
    RegisterSink& m_sink;
};

// Routes logical OPL writes (bank << 8 | reg) into a live dbopl
// instance; OPL4 PCM writes (bank 2) have no dbopl equivalent and are
// dropped -- PCM voices are rendered by their own model instead.
class DboplSink : public RegisterSink
{
public:
    explicit DboplSink(OplChip& chip) : m_chip(chip) {}
    void Write(uint32_t addr, uint8_t value, bool) override
    {
        const int bank = static_cast<int>(addr >> 8);
        if (bank <= 1)
            m_chip.WriteReg(bank, static_cast<uint8_t>(addr & 0xFF), value);
    }

private:
    OplChip& m_chip;
};

// YMF278B: `0xD0 port reg value`, port 0/1 = FM banks, port 2 = PCM.
class Opl4Encoder : public VgmEncoder
{
public:
    void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) override
    {
        out.push_back(0xD0);
        out.push_back(static_cast<uint8_t>((addr >> 8) & 0x7F));
        out.push_back(static_cast<uint8_t>(addr & 0xFF));
        out.push_back(value);
    }
};

class OplTarget : public ChipTarget
{
public:
    explicit OplTarget(Variant variant) : m_variant(variant)
    {
        m_fmBaseRateHz = IsOpl4() ? kOpl4FmBaseRateHz : kOplBaseRateHz;

        int fmVoices = 0;
        int pcmVoices = 0;
        switch (variant)
        {
        case Variant::Opl3: fmVoices = opl::kMaxChannels; break;
        case Variant::Opl2: fmVoices = 9; break;
        case Variant::Opl4Fm: fmVoices = opl::kMaxChannels; break;
        case Variant::Opl4Pcm: pcmVoices = kOpl4PcmSlots; break;
        case Variant::Opl4Both: fmVoices = opl::kMaxChannels; pcmVoices = kOpl4PcmSlots; break;
        }

        if (fmVoices > 0)
        {
            VoiceKind fm;
            fm.name = "FM";
            fm.minHz = 0.0;
            fm.maxHz = FmHz(1023, 7);
            fm.levelDb = UniformLevelTable(64, 0.75);
            fm.uniformDbPerStep = 0.75;
            m_fmKind = static_cast<int>(m_kinds.size());
            m_kinds.push_back(fm);
            for (int i = 0; i < fmVoices; ++i)
                m_voiceKind.push_back(m_fmKind);
        }
        if (pcmVoices > 0)
        {
            VoiceKind pcm;
            pcm.name = "PCM";
            pcm.minHz = 1.0;
            pcm.maxHz = 20000.0;
            std::vector<double> amps;
            for (int tl = 0; tl < kOpl4TlSteps; ++tl)
                amps.push_back(Opl4TlAmplitude(tl));
            pcm.levelDb = LinearAmplitudeLevelTable(amps);
            m_pcmKind = static_cast<int>(m_kinds.size());
            m_kinds.push_back(pcm);
            m_firstPcmVoice = fmVoices;
            for (int i = 0; i < pcmVoices; ++i)
                m_voiceKind.push_back(m_pcmKind);
        }

        const QuantizedSine sine = MakeQuantizedSine(kPcmCycleLength, 32767);
        m_pcmModel.table = sine.normalized;
        m_pcmModel.interpolate = true;
        m_pcmModel.fullScale = kOpl4PcmFullScale;
        m_pcmModel.releaseDbPerSecond = 96.0 / 0.002;
        m_pcmSampleCodes = sine.codes;
    }

    ChipId Id() const override
    {
        switch (m_variant)
        {
        case Variant::Opl3: return ChipId::Opl3;
        case Variant::Opl2: return ChipId::Opl2;
        case Variant::Opl4Fm: return ChipId::Opl4Fm;
        case Variant::Opl4Pcm: return ChipId::Opl4Pcm;
        case Variant::Opl4Both: return ChipId::Opl4Both;
        }
        return ChipId::Opl3;
    }

    std::string Name() const override
    {
        switch (m_variant)
        {
        case Variant::Opl3: return "OPL3 (YMF262)";
        case Variant::Opl2: return "OPL2 (YM3812)";
        case Variant::Opl4Fm: return "OPL4 (YMF278B) FM";
        case Variant::Opl4Pcm: return "OPL4 (YMF278B) PCM";
        case Variant::Opl4Both: return "OPL4 (YMF278B) FM + PCM";
        }
        return {};
    }

    bool IsOpl4() const
    {
        return m_variant == Variant::Opl4Fm || m_variant == Variant::Opl4Pcm || m_variant == Variant::Opl4Both;
    }
    bool HasPcm() const { return m_pcmKind >= 0; }
    bool IsPcmVoice(int voice) const { return KindIndexOf(voice) == m_pcmKind; }
    int PcmSlot(int voice) const { return voice - m_firstPcmVoice; }
    Variant GetVariant() const { return m_variant; }

    double FmHz(int fnum, int block) const
    {
        return static_cast<double>(fnum) * m_fmBaseRateHz / std::pow(2.0, 20 - block);
    }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override
    {
        const VoiceKind& kind = KindOf(voice);
        if (!(hz >= kind.minHz) || hz > kind.maxHz)
            return false;
        if (IsPcmVoice(voice))
            return QuantizePcmPitch(hz, pitch);

        // Same search as HzToFnumBlock: the smallest block that keeps fnum
        // in range gives the finest resolution.
        for (int block = 0; block <= 7; ++block)
        {
            const double fnum = hz * std::pow(2.0, 20 - block) / m_fmBaseRateHz;
            if (fnum <= 1023.0)
            {
                const int fnumInt = std::clamp(static_cast<int>(std::llround(fnum)), 0, 1023);
                pitch = static_cast<uint32_t>(fnumInt | (block << 10));
                return true;
            }
        }
        pitch = 1023u | (7u << 10);
        return true;
    }

    double PitchToHz(int voice, uint32_t pitch) const override
    {
        if (IsPcmVoice(voice))
        {
            const int oct = DecodeOctave(pitch);
            const int fn = static_cast<int>(pitch & 0x3FF);
            return 44100.0 * std::pow(2.0, oct - 1) * (1024.0 + fn) / 1024.0 / kPcmCycleLength;
        }
        return FmHz(static_cast<int>(pitch & 0x3FF), static_cast<int>((pitch >> 10) & 7));
    }

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const override
    {
        return std::make_unique<OplWriter>(*this, sink);
    }

    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override
    {
        if (IsPcmVoice(voice))
            return RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, m_pcmModel);
        return RenderFmVoice(voice, frames, frameRateHz, sampleRate);
    }

    VgmChipInfo VgmInfo() const override
    {
        switch (m_variant)
        {
        case Variant::Opl3: return {0x5C, kYmf262ClockHz};
        case Variant::Opl2: return {0x50, kYm3812ClockHz};
        default: return {0x60, kYmf278bClockHz};
        }
    }

    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override
    {
        if (IsOpl4())
            return std::make_unique<Opl4Encoder>();
        if (m_variant == Variant::Opl2)
            return std::make_unique<YmPortEncoder>(0x5A, 0x5A);
        return std::make_unique<YmPortEncoder>(0x5E, 0x5F);
    }

    std::vector<VgmDataBlock> VgmDataBlocks() const override
    {
        if (!HasPcm())
            return {};

        // An empty ROM block just declares the 2 MiB sample-ROM address
        // range, so RAM (and the RAM header table) lands at 0x200000 in
        // players that don't load a real YRW801 ROM -- the sine never
        // touches ROM, so no ROM image is needed.
        VgmDataBlock rom;
        rom.type = 0x84;
        rom.memSize = kOpl4RomSize;
        rom.offset = 0;

        VgmDataBlock ram;
        ram.type = 0x87;
        ram.memSize = kOpl4RamSize;
        ram.offset = 0;
        ram.data.assign(kOpl4SampleRamOffset + static_cast<size_t>(kPcmCycleLength) * 2, 0);

        // Wave header (12 bytes) for wave 384: 16-bit format, start
        // address, loop start, negated loop end, then LFO/VIB, AR/D1R,
        // DL/D2R, RC/RR, AM.
        const uint32_t start = kOpl4RomSize + kOpl4SampleRamOffset;
        const uint16_t loopStart = 0;
        const uint16_t endNegated = static_cast<uint16_t>(0x10000 - kPcmCycleLength);
        uint8_t* h = ram.data.data();
        h[0] = static_cast<uint8_t>(0x80 | ((start >> 16) & 0x3F));
        h[1] = static_cast<uint8_t>((start >> 8) & 0xFF);
        h[2] = static_cast<uint8_t>(start & 0xFF);
        h[3] = static_cast<uint8_t>(loopStart >> 8);
        h[4] = static_cast<uint8_t>(loopStart & 0xFF);
        h[5] = static_cast<uint8_t>(endNegated >> 8);
        h[6] = static_cast<uint8_t>(endNegated & 0xFF);
        h[7] = 0x00; // LFO off, no vibrato
        h[8] = 0xF0; // AR=15 (instant), D1R=0 (hold)
        h[9] = 0x00; // DL=0, D2R=0
        h[10] = 0x0F; // RC=0, RR=15 (fastest release)
        h[11] = 0x00; // no tremolo

        for (int i = 0; i < kPcmCycleLength; ++i)
        {
            const uint16_t code = static_cast<uint16_t>(static_cast<int16_t>(m_pcmSampleCodes[static_cast<size_t>(i)]));
            const size_t at = kOpl4SampleRamOffset + static_cast<size_t>(i) * 2;
            ram.data[at] = static_cast<uint8_t>(code >> 8); // big-endian
            ram.data[at + 1] = static_cast<uint8_t>(code & 0xFF);
        }
        return {rom, ram};
    }

private:
    static int DecodeOctave(uint32_t pitch)
    {
        const int raw = static_cast<int>((pitch >> 10) & 0xF);
        return raw >= 8 ? raw - 16 : raw;
    }

    // Playback rate = 44100 * 2^(oct-1) * (1024+FN)/1024 samples/second
    // (ymf278b.c calcStep), one cycle per kPcmCycleLength samples.
    static bool QuantizePcmPitch(double hz, uint32_t& pitch)
    {
        const double x = hz * kPcmCycleLength / 44100.0; // = 2^(oct-1) * (1 + FN/1024)
        int oct = static_cast<int>(std::floor(std::log2(x))) + 1;
        double mant = x / std::pow(2.0, oct - 1);
        int fn = static_cast<int>(std::llround((mant - 1.0) * 1024.0));
        if (fn >= 1024)
        {
            fn = 0;
            ++oct;
        }
        if (oct < -7 || oct > 7)
            return false;
        fn = std::clamp(fn, 0, 1023);
        pitch = static_cast<uint32_t>(fn) | (static_cast<uint32_t>(oct & 0xF) << 10);
        return true;
    }

    std::vector<float> RenderFmVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                     int sampleRate) const
    {
        std::vector<float> mono;
        if (frames.empty() || sampleRate <= 0 || frameRateHz <= 0.0)
            return mono;

        // dbopl always runs at OPL3's internal rate; rendering OPL4 FM at
        // a proportionally adjusted dbopl output rate and treating the
        // result as `sampleRate` reproduces OPL4's own FM clock (pitch
        // and envelope timing both scale with it, as on the real chip).
        const int chipRate = IsOpl4()
            ? static_cast<int>(std::llround(sampleRate * kOplBaseRateHz / kOpl4FmBaseRateHz))
            : sampleRate;
        OplChip chip(chipRate);
        DboplSink sink(chip);
        OplWriter writer(*this, sink);
        writer.Init();

        const int samplesPerFrame = std::max(1, static_cast<int>(std::llround(sampleRate / frameRateHz)));
        mono.assign(static_cast<size_t>(samplesPerFrame) * frames.size(), 0.0f);
        std::vector<float> buf(static_cast<size_t>(samplesPerFrame) * 2);
        const bool monoChip = (m_variant == Variant::Opl2);
        size_t writePos = 0;
        for (const VoiceFrame& frame : frames)
        {
            writer.ApplyVoice(voice, frame);
            if (monoChip)
            {
                chip.GenerateMono(buf.data(), samplesPerFrame);
                for (int i = 0; i < samplesPerFrame; ++i)
                    mono[writePos + static_cast<size_t>(i)] = buf[static_cast<size_t>(i)];
            }
            else
            {
                chip.GenerateStereo(buf.data(), samplesPerFrame);
                for (int i = 0; i < samplesPerFrame; ++i)
                    mono[writePos + static_cast<size_t>(i)] =
                        0.5f * (buf[static_cast<size_t>(i) * 2] + buf[static_cast<size_t>(i) * 2 + 1]);
            }
            writePos += static_cast<size_t>(samplesPerFrame);
        }
        return mono;
    }

    Variant m_variant;
    double m_fmBaseRateHz = kOplBaseRateHz;
    int m_fmKind = -1;
    int m_pcmKind = -1;
    int m_firstPcmVoice = 0;
    OscillatorModel m_pcmModel;
    std::vector<int> m_pcmSampleCodes;
};

void OplWriter::Init()
{
    switch (m_target.GetVariant())
    {
    case Variant::Opl3:
        m_sink.Write(0x105, 0x01); // OPL3 ("new") mode
        break;
    case Variant::Opl2:
        break;
    default:
        m_sink.Write(0x105, 0x03); // NEW + NEW2: OPL3 mode plus access to the PCM part
        m_sink.Write(0x202, kOpl4WaveTableHeaderReg);
        m_sink.Write(0x2F8, 0x00); // FM mix level: 0 dB both sides
        m_sink.Write(0x2F9, 0x00); // PCM mix level: 0 dB both sides
        if (m_target.HasPcm())
        {
            for (int slot = 0; slot < kOpl4PcmSlots; ++slot)
            {
                WritePcm(slot, 4, 0x20);                                      // key off, LFO held in reset
                WritePcm(slot, 1, static_cast<uint8_t>((kOpl4Wave >> 8) & 1)); // wave bit 8, FN low = 0
                WritePcm(slot, 0, static_cast<uint8_t>(kOpl4Wave & 0xFF));    // loads wave header
                WritePcm(slot, 3, 0xFF);                                      // TL 0x7F (silent), level direct
            }
        }
        break;
    }
}

void OplWriter::ApplyVoice(int voice, const VoiceFrame& frame)
{
    if (m_target.IsPcmVoice(voice))
    {
        const int slot = m_target.PcmSlot(voice);
        const int fn = static_cast<int>(frame.pitch & 0x3FF);
        const int oct = static_cast<int>((frame.pitch >> 10) & 0xF);
        WritePcm(slot, 1, static_cast<uint8_t>(((fn & 0x7F) << 1) | ((kOpl4Wave >> 8) & 1)));
        WritePcm(slot, 2, static_cast<uint8_t>((oct << 4) | ((fn >> 7) & 0x07)));
        WritePcm(slot, 3, static_cast<uint8_t>(((frame.level & 0x7F) << 1) | 0x01));
        WritePcm(slot, 4, frame.keyOn ? 0xA0 : 0x20);
        return;
    }

    const opl::RegWriteFn write = [this](int bank, uint8_t reg, uint8_t value) {
        m_sink.Write((static_cast<uint32_t>(bank) << 8) | reg, value);
    };
    opl::SetupChannel(write, voice, PureSineSetup());
    opl::FramePatch patch;
    patch.fnum = static_cast<int>(frame.pitch & 0x3FF);
    patch.block = static_cast<int>((frame.pitch >> 10) & 7);
    patch.carTotalLevel = static_cast<uint8_t>(std::min<int>(frame.level, 63));
    patch.keyOn = frame.keyOn;
    opl::ApplyFrame(write, voice, patch);
}

} // namespace

std::shared_ptr<const ChipTarget> MakeOplTarget(ChipId id)
{
    switch (id)
    {
    case ChipId::Opl2: return std::make_shared<OplTarget>(Variant::Opl2);
    case ChipId::Opl4Fm: return std::make_shared<OplTarget>(Variant::Opl4Fm);
    case ChipId::Opl4Pcm: return std::make_shared<OplTarget>(Variant::Opl4Pcm);
    case ChipId::Opl4Both: return std::make_shared<OplTarget>(Variant::Opl4Both);
    default: return std::make_shared<OplTarget>(Variant::Opl3);
    }
}

} // namespace chips::detail
