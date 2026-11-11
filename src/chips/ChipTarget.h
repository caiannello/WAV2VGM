#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Chip-neutral description of a VGM output target: how many independent
// pure-sine "voices" it offers, how a requested frequency/loudness maps
// onto that chip's own pitch and volume registers, how to render a voice
// for preview, and how to turn the resulting register stream into VGM
// commands. The fitter (SineFit.h) and the VGM writer (ChipVgmWriter.h)
// only ever talk to this interface.
//
// Every target uses its voices as plain additive sine oscillators, so a
// chip's real output is the sample-by-sample sum of its voices -- which
// is what lets each voice be rendered (and calibrated) on its own.
namespace chips
{

enum class ChipId
{
    Opl3,
    Opl2,
    Opl4Fm,
    Opl4Pcm,
    Opl4Both,
    Opm,
    Opn,
    Opna,
    Opn2,
    HuC6280,
    Vsu,
    MultiPcm,
};

// One voice's state for one output frame. `pitch` and `level` are
// already quantized to the chip: `pitch` is a target-specific packed
// register code (see the target's PitchToHz), `level` an index into the
// voice kind's levelDb table (0 = loudest).
struct VoiceFrame
{
    uint32_t pitch = 0;
    uint16_t level = 0;
    bool keyOn = false;
};

// What a group of voices can do. levelDb[i] is the output level of
// volume step i relative to step 0, so levelDb[0] == 0 and the table is
// strictly decreasing.
struct VoiceKind
{
    std::string name;
    double minHz = 0.0;
    double maxHz = 0.0;
    std::vector<double> levelDb;
    // > 0 when levelDb[i] == -i * uniformDbPerStep exactly (the classic
    // Yamaha total-level register), letting QuantizeLevel use plain
    // rounding instead of a table search.
    double uniformDbPerStep = 0.0;
};

// Receives logical register writes. What `addr` means is up to each
// target (bank/port in the upper bits, register in the lower ones, or a
// channel/slot number for chips with indirect register access). `force`
// marks writes that must reach the chip even when they repeat the last
// value written to the same address (e.g. waveform-RAM data ports).
class RegisterSink
{
public:
    virtual ~RegisterSink() = default;
    virtual void Write(uint32_t addr, uint8_t value, bool force = false) = 0;
};

// Stateful register-stream generator for one chip instance.
class ChipWriter
{
public:
    virtual ~ChipWriter() = default;
    // One-time global setup (mode bits, waveforms, sample headers, ...).
    virtual void Init() = 0;
    virtual void ApplyVoice(int voice, const VoiceFrame& frame) = 0;
};

// Turns one logical register write into VGM command bytes.
class VgmEncoder
{
public:
    virtual ~VgmEncoder() = default;
    virtual void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) = 0;
};

// A VGM 0x67 ROM/RAM data block (types 0x80-0xBF).
struct VgmDataBlock
{
    uint8_t type = 0;
    uint32_t memSize = 0;
    uint32_t offset = 0;
    std::vector<uint8_t> data;
};

struct VgmChipInfo
{
    uint32_t clockHeaderOffset = 0;
    uint32_t clockHz = 0;
};

class ChipTarget
{
public:
    virtual ~ChipTarget() = default;

    virtual ChipId Id() const = 0;
    virtual std::string Name() const = 0;

    int VoiceCount() const { return static_cast<int>(m_voiceKind.size()); }
    const std::vector<VoiceKind>& Kinds() const { return m_kinds; }
    int KindIndexOf(int voice) const { return m_voiceKind[static_cast<size_t>(voice)]; }
    const VoiceKind& KindOf(int voice) const { return m_kinds[static_cast<size_t>(KindIndexOf(voice))]; }
    // The lowest-numbered voice of each kind -- what calibration probes.
    int FirstVoiceOfKind(int kind) const;

    // Nearest achievable pitch to `hz` on `voice`; false if `hz` is
    // outside that voice's playable range.
    virtual bool QuantizePitch(int voice, double hz, uint32_t& pitch) const = 0;
    virtual double PitchToHz(int voice, uint32_t pitch) const = 0;

    // Nearest volume step to `dbRelativeToMax` (<= 0 for anything quieter
    // than the voice's loudest output), clamped to the table's range.
    uint16_t QuantizeLevel(int voice, double dbRelativeToMax) const;
    double LevelToDb(int voice, uint16_t level) const;

    // A keyed-off frame for `voice`, parked at a voice-specific pitch --
    // see SineFit.cpp for why every voice doesn't share one.
    VoiceFrame SilentFrame(int voice) const;

    virtual std::unique_ptr<ChipWriter> CreateWriter(RegisterSink& sink) const = 0;

    // Renders `voice` alone, frame by frame (frame i covers
    // [i/frameRateHz, (i+1)/frameRateHz)), mono, at `sampleRate`.
    virtual std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                           int sampleRate) const = 0;

    virtual VgmChipInfo VgmInfo() const = 0;
    virtual std::unique_ptr<VgmEncoder> CreateVgmEncoder() const = 0;
    virtual std::vector<VgmDataBlock> VgmDataBlocks() const { return {}; }

protected:
    std::vector<VoiceKind> m_kinds;
    std::vector<int> m_voiceKind; // per voice: index into m_kinds
};

std::shared_ptr<const ChipTarget> CreateTarget(ChipId id);

// Level tables shared by several targets.
std::vector<double> UniformLevelTable(int steps, double dbPerStep);
std::vector<double> LinearAmplitudeLevelTable(const std::vector<double>& amplitudesLoudestFirst);

} // namespace chips
