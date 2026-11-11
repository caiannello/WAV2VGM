#include "MidiTarget.h"
#include "ChipCommon.h"

#include <algorithm>
#include <cmath>

namespace chips
{

namespace
{

// Preview level of one voice at full expression. A GM synth's absolute
// loudness is unknowable here; this just keeps 16 voices from clipping
// the preview mix too easily.
constexpr double kPreviewFullScale = 0.1;

double NoteToHz(double note) { return 440.0 * std::pow(2.0, (note - 69.0) / 12.0); }

} // namespace

MidiTarget::MidiTarget(int bendRangeSemitones)
    : m_bendRange(std::clamp(bendRangeSemitones, kMinBendRange, kMaxBendRange))
{
    VoiceKind kind;
    kind.name = "MIDI";
    kind.minHz = NoteToHz(0.0);
    kind.maxHz = NoteToHz(127.0);
    // CC11 = 127..1 -> GM's recommended 40*log10(cc/127) gain curve.
    for (int cc = 127; cc >= 1; --cc)
        kind.levelDb.push_back(40.0 * std::log10(cc / 127.0));
    m_kinds.push_back(kind);
    m_voiceKind.assign(kChannels, 0);
}

int MidiTarget::MidiChannelOf(int voice)
{
    // Voices 0-8 -> channels 1-9, 9-14 -> 11-16, 15 -> channel 10.
    if (voice < 9)
        return voice;
    if (voice < 15)
        return voice + 1;
    return 9;
}

bool MidiTarget::QuantizePitch(int voice, double hz, uint32_t& pitch) const
{
    const VoiceKind& kind = KindOf(voice);
    if (!(hz >= kind.minHz * 0.9999) || hz > kind.maxHz * 1.0001)
        return false;
    const double note = std::clamp(69.0 + 12.0 * std::log2(hz / 440.0), 0.0, 127.0);
    pitch = static_cast<uint32_t>(std::llround(note * kPitchUnitsPerSemitone));
    return true;
}

double MidiTarget::PitchToHz(int, uint32_t pitch) const
{
    return NoteToHz(PitchToNote(pitch));
}

std::vector<float> MidiTarget::RenderVoice(int voice, const std::vector<VoiceFrame>& frames, double frameRateHz,
                                           int sampleRate) const
{
    // An ocarina patch is close to a pure sine; approximate it as one,
    // with a short release like a typical GM patch's.
    detail::OscillatorModel model;
    model.fullScale = kPreviewFullScale;
    model.releaseDbPerSecond = 96.0 / 0.05;
    return detail::RenderOscillatorVoice(*this, voice, frames, frameRateHz, sampleRate, model);
}

std::shared_ptr<const ChipTarget> CreateMidiTarget(int bendRangeSemitones)
{
    return std::make_shared<MidiTarget>(bendRangeSemitones);
}

} // namespace chips
