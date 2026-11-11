#pragma once

#include "ChipTarget.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

// Internals shared by the individual chip targets -- not part of the
// public ChipTarget interface.
namespace chips::detail
{

// How a non-emulated target's voice is rendered for preview: an
// oscillator reading one cycle of `table` (or an ideal sine if empty),
// either stepped (zero-order hold, like a wavetable chip's DAC) or
// linearly interpolated (like a PCM chip's sample interpolator), scaled
// to roughly the per-voice output level libvgm's own core for that chip
// produces (`fullScale`, as a fraction of digital full scale -- this is
// what keeps a mixed-chip target like OPL4 FM + PCM balanced). Every
// modeled chip restarts its waveform at key-on.
struct OscillatorModel
{
    std::vector<float> table;
    bool interpolate = false;
    double fullScale = 0.125;
    // Key-off fade, in dB per second; <= 0 means an instant cut.
    double releaseDbPerSecond = 0.0;
};

std::vector<float> RenderOscillatorVoice(const ChipTarget& target, int voice, const std::vector<VoiceFrame>& frames,
                                         double frameRateHz, int sampleRate, const OscillatorModel& model);

// One cycle of a sine quantized to signed `bits`-bit integers, returned
// both as the raw integer codes and normalized to [-1, 1] (by the
// largest code) for the preview model.
struct QuantizedSine
{
    std::vector<int> codes;
    std::vector<float> normalized;
};
QuantizedSine MakeQuantizedSine(int length, int amplitudeCode);

// YM-style encoder: addr = (port << 8) | reg, written as
// `commandPerPort[port] reg value`.
class YmPortEncoder : public VgmEncoder
{
public:
    YmPortEncoder(uint8_t port0Command, uint8_t port1Command) : m_commands{port0Command, port1Command} {}
    void Encode(uint32_t addr, uint8_t value, std::vector<uint8_t>& out) override;

private:
    std::array<uint8_t, 2> m_commands;
};

std::shared_ptr<const ChipTarget> MakeOplTarget(ChipId id);
std::shared_ptr<const ChipTarget> MakeOpmTarget();
std::shared_ptr<const ChipTarget> MakeOpnTarget(ChipId id);
std::shared_ptr<const ChipTarget> MakeHuC6280Target();
std::shared_ptr<const ChipTarget> MakeVsuTarget();
std::shared_ptr<const ChipTarget> MakeMultiPcmTarget();

} // namespace chips::detail
