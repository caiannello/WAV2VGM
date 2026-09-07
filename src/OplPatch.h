#pragma once

#include "OplChip.h"

#include <cstdint>

// Register-level model for driving a single 2-operator OPL3 channel over
// time: a ChannelStaticSetup chosen once (envelope shape, algorithm,
// modulator/carrier waveform and multiplier) plus a FramePatch applied
// once per output frame (frequency, carrier volume, and key-on/off) --
// see OplFit.h for how these get chosen. Total level alone can't reach
// genuine silence (max attenuation, TL=63, still measures around -60dB,
// not -inf), and a frame with nothing real to play needs to actually be
// silent, not just quiet -- especially once several/all of a fit's
// channels are simultaneously idle, which is exactly when a recording is
// genuinely quiet. So FramePatch::keyOn is a real, honored transition:
// ApplyFrame issues a genuine key-off when a channel goes idle and a
// genuine key-on when it resumes (dbopl's own Channel::WriteB0 reacts
// correctly to the bit in both directions -- confirmed by reading it,
// not assumed), rather than staying keyed on forever and leaning on
// total level alone. The fast attack/release envelope (see
// ChannelStaticSetup) keeps each transition brief.
namespace opl
{

// 18 two-operator OPL3 channels: 0-8 on VGM/OplChip "bank 0", 9-17 on
// "bank 1" -- see OplChip::WriteReg.
constexpr int kMaxChannels = 18;

// OPL's discrete frequency-multiplier table -- register field values are
// indices into this, not raw ratios (value 0 means x0.5, not x0).
constexpr double kMultipleTable[16] = {
    0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 12, 12, 15, 15
};

struct ChannelStaticSetup
{
    uint8_t modMultipleIndex = 1; // index into kMultipleTable
    uint8_t modTotalLevel = 0;    // attenuation, 0 (loudest) - 63 (silent)
    uint8_t modWaveform = 0;      // 0-7, OPL3 "new mode" waveform select

    uint8_t carMultipleIndex = 1;
    uint8_t carWaveform = 0;

    uint8_t feedback = 0; // 0-7 (ignored when algorithmAdditive is true -- feedback only applies to the modulator)

    // false = FM (the modulator's output phase-modulates the carrier);
    // true = additive: operator 1 (modulator) and operator 2 (carrier)
    // both feed the mixer directly and independently, so the carrier is
    // completely unaffected by whatever the modulator is doing --
    // register 0xC0's CNT bit. Silencing the modulator (modTotalLevel =
    // 63) in FM mode still leaves a faint, technically-nonzero amount of
    // phase modulation reaching the carrier; additive mode has none at
    // all regardless of modTotalLevel, which is what a genuinely pure,
    // uncontaminated sine carrier requires.
    bool algorithmAdditive = false;

    // Envelope is fixed for the channel's whole lifetime, not exposed
    // here: fast attack, held at full sustain via the EGT ("sustain")
    // flag in register 0x20 -- verified against opl3_selftest, whose
    // silent-output bug turned out to be exactly this flag being unset.
    // Per-frame volume comes from carrier total level (FramePatch)
    // instead, so ADSR itself never needs to vary frame to frame.
};

struct FramePatch
{
    int fnum = 0;              // 0-1023
    int block = 0;             // 0-7
    uint8_t carTotalLevel = 0; // attenuation, 0 (loudest) - 63 (silent)

    // Whether this channel should actually be sounding this frame. A
    // frame with nothing real to play should set this false (fnum/block/
    // carTotalLevel become irrelevant -- key-off silences the channel
    // regardless of them) rather than relying on carTotalLevel=63 alone,
    // which isn't genuine silence. Defaults true so callers that don't
    // care about explicit note-on/off (e.g. FitAllFrames/FitSingleChannel,
    // which play continuously) don't need to think about it.
    bool keyOn = true;
};

// Writes the operator/channel registers that make up `setup`, without
// touching key-on/frequency/carrier level -- call once, before the first
// ApplyFrame for this channel.
void SetupChannel(OplChip& chip, int channelIndex, const ChannelStaticSetup& setup);

// Writes one frame's frequency, carrier level, and key-on/off state.
// dbopl only reacts to the key-on bit's own transitions (0->1 restarts
// the waveform phase and re-enters attack; 1->0 enters release), so
// repeated frames with the same frame.keyOn value just update frequency/
// level in place with no re-trigger -- only an actual keyOn flip costs a
// (brief, given the fast attack/release envelope) transition.
void ApplyFrame(OplChip& chip, int channelIndex, const FramePatch& frame);

} // namespace opl
