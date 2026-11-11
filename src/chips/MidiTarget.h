#pragma once

#include "ChipTarget.h"

#include <memory>

// General MIDI output: up to 16 monophonic channels, each playing GM
// program 80 (Ocarina -- the closest thing to a pure sine in the GM set)
// in mono mode (CC126), with its pitch-bend range set via RPN 0 to
// `bendRangeSemitones` (1-24). Each voice's pitch is a note plus pitch
// bend; level is CC11 (Expression) on the GM 40*log10(cc/127) curve.
// Exported as a Standard MIDI File by midi::WriteMidiFile, not as VGM.
//
// Voices map to MIDI channels 1-9, 11-16, then 10 last: the fitters
// always fill the lowest-numbered free voice first, so channel 10 (drums
// on most GM synths, which may ignore the program change) is only used
// once every other channel is busy.
namespace chips
{

class MidiTarget : public ChipTarget
{
public:
    static constexpr int kChannels = 16;
    static constexpr int kMinBendRange = 1;
    static constexpr int kMaxBendRange = 24;
    static constexpr int kDefaultBendRange = 24;
    static constexpr int kGmOcarinaProgram = 79; // GM program 80, zero-based
    // Pitch codes are MIDI note numbers in 1/8192-semitone units.
    static constexpr double kPitchUnitsPerSemitone = 8192.0;

    explicit MidiTarget(int bendRangeSemitones);

    ChipId Id() const override { return ChipId::Midi; }
    std::string Name() const override { return "MIDI (GM Ocarina)"; }
    OutputFormat Format() const override { return OutputFormat::Midi; }

    int BendRangeSemitones() const { return m_bendRange; }
    // Zero-based MIDI channel (0-15) for `voice`.
    static int MidiChannelOf(int voice);
    static double PitchToNote(uint32_t pitch) { return pitch / kPitchUnitsPerSemitone; }
    // CC11 value for a level step (0 = loudest = 127).
    static int ExpressionOf(uint16_t level) { return 127 - static_cast<int>(level); }

    bool QuantizePitch(int voice, double hz, uint32_t& pitch) const override;
    double PitchToHz(int voice, uint32_t pitch) const override;

    std::unique_ptr<ChipWriter> CreateWriter(RegisterSink&) const override { return nullptr; }
    std::vector<float> RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                   int sampleRate) const override;
    VgmChipInfo VgmInfo() const override { return {}; }
    std::unique_ptr<VgmEncoder> CreateVgmEncoder() const override { return nullptr; }

private:
    int m_bendRange;
};

std::shared_ptr<const ChipTarget> CreateMidiTarget(int bendRangeSemitones);

} // namespace chips
