#pragma once

#include "SineFit.h"

#include <cstdint>
#include <string>
#include <vector>

// Writes a sinefit::ChipFit made against the MIDI target (see
// chips/MidiTarget.h) as a Standard MIDI File, format 1: track 0 holds
// tempo/time signature, then one track per MIDI channel that actually
// plays something.
//
// Each channel starts with: Reset All Controllers, Bank 0, Program 80
// (Ocarina), Omni Off + Mono On (CC124/CC126), Volume/Pan/Expression,
// and the pitch-bend range via RPN 0 (CC101/CC100 = 0, CC6 = range,
// CC38 = 0, then RPN null). After that, a voice's pitch changes become
// pitch-bend messages around its current note, and level changes become
// CC11 (Expression). A new note is only struck when the voice keys on or
// the pitch drifts past the bend range. In that case the new note-on is
// sent before the old note-off, so a mono-mode synth plays it legato.
namespace midi
{

// Timing: 120 BPM, 960 ticks per quarter note (1920 ticks per second).
// Frame boundaries are rounded to the nearest tick, so there's no drift.
constexpr int kTicksPerQuarter = 960;
constexpr uint32_t kTempoMicrosPerQuarter = 500000;

bool BuildMidi(const sinefit::ChipFit& fit, std::vector<uint8_t>& out, std::string& errorMessage);
bool WriteMidiFile(const std::string& path, const sinefit::ChipFit& fit, std::string& errorMessage);

} // namespace midi
